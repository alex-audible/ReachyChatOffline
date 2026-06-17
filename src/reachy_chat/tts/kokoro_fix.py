"""Runtime fix for mlx-audio's Kokoro istftnet length-mismatch crash.

Symptom (observed in the end-to-end pipeline, deterministic for certain clauses):

    ValueError: [broadcast_shapes] Shapes (1, T, 1) and (1, T+Δ, 9) cannot be broadcast.
    (in kokoro/istftnet.py SineGen.__call__)

Root cause — upstream bug, see https://github.com/Blaizzy/mlx-audio/issues/786 (and #784):
mlx-audio 0.4.4 (commit aaf5ee6) replaced `mx.ceil` with Python `math.ceil` in
`tts/models/interpolate.py` to avoid worker-thread compile locks. `math.ceil` uses float64,
so a value like `988.0000000001` (from inexact `1/300`) rounds UP to 989, and `989*300 = 296700`
instead of `296400` — an off-by-one-frame error that propagates as a +Δ (=upsample_scale, e.g.
300) length mismatch between the harmonic `sine_waves` and `uv`.

We chose to **patch** rather than downgrade mlx-audio (#784): downgrading risks the emotive
engines (chatterbox-turbo, higgs v3) and the vision path that depend on 0.4.4 features.

Two idempotent monkeypatches:
  1. PRIMARY (root cause): epsilon-tolerant rounding before `math.ceil` in `interpolate()`
     (the fix suggested in #786). Fixes the length at the source for every Kokoro path.
  2. DEFENSIVE: align `sine_waves` to `uv`'s length in `SineGen.__call__` — a harmless no-op
     once (1) is active, but guards against any residual off-by-one.
"""

from __future__ import annotations

import math

import mlx.core as mx

_applied = False


def _make_fixed_interpolate(interpolate1d):
    def fixed_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
        ndim = input.ndim
        if ndim < 3:
            raise ValueError(f"Expected at least 3D input (N, C, D1), got {ndim}D")
        spatial_dims = ndim - 2
        if size is not None and scale_factor is not None:
            raise ValueError("Only one of size or scale_factor should be defined")
        if size is None and scale_factor is None:
            raise ValueError("One of size or scale_factor must be defined")
        if size is not None and not isinstance(size, (list, tuple)):
            size = [size] * spatial_dims
        if scale_factor is not None and not isinstance(scale_factor, (list, tuple)):
            scale_factor = [scale_factor] * spatial_dims
        if size is None:
            size = []
            for i in range(spatial_dims):
                val = float(input.shape[i + 2]) * float(scale_factor[i])
                rounded = round(val, 9)            # <-- #786 fix: kill float64 ceil drift
                if abs(val - rounded) < 1e-9:
                    val = rounded
                size.append(max(1, int(math.ceil(val))))
        if spatial_dims == 1:
            return interpolate1d(input, size[0], mode, align_corners)
        raise ValueError(f"Only 1D interpolation currently supported, got {spatial_dims}D")
    return fixed_interpolate


def _patched_sinegen_call(self, f0: mx.array):
    fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
    sine_waves = self._f02sine(fn) * self.sine_amp
    uv = self._f02uv(f0)
    target_t = uv.shape[1]
    t = sine_waves.shape[1]
    if t > target_t:
        sine_waves = sine_waves[:, :target_t, :]
    elif t < target_t:
        pad = mx.repeat(sine_waves[:, -1:, :], target_t - t, axis=1)
        sine_waves = mx.concatenate([sine_waves, pad], axis=1)
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    noise = noise_amp * mx.random.normal(sine_waves.shape)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


def apply_kokoro_fixes() -> None:
    """Idempotently apply the interpolate (root cause) + SineGen (defensive) patches."""
    global _applied
    if _applied:
        return
    from mlx_audio.tts.models import interpolate as interp_mod
    from mlx_audio.tts.models.kokoro import istftnet as I

    fixed = _make_fixed_interpolate(interp_mod.interpolate1d)
    interp_mod.interpolate = fixed          # source module
    I.interpolate = fixed                   # the name istftnet actually calls
    I.SineGen.__call__ = _patched_sinegen_call
    _applied = True
