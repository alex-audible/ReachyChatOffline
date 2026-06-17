"""Chatterbox (MLX, 4-bit) emotive + voice-cloning TTS for the Reachy pipeline.

STATUS (2026-06-17): **WORKING.** ``mlx-community/chatterbox-4bit`` loads cleanly via
mlx-audio's *standard* ``chatterbox`` backend and produces real, intelligible audio
(default RMS ~0.13; cloned-voice RMS ~0.04), RTF ~0.3-0.5 after warmup, TTFA ~0.7 s for a
short clause, ~1.7 GB peak GPU. Verified end-to-end: synthesize -> transcribe with
parakeet -> transcript matches the input text, for the built-in voice, a cloned voice, and
an exaggerated-emotion render.

----------------------------------------------------------------------------------------
WHY THIS REPO (and not the q4 *turbo* one that failed before)
----------------------------------------------------------------------------------------
The earlier attempt, ``mlx-community/chatterbox-turbo-mlx-q4``, shipped its S3Gen vocoder
in ResembleAI's ORIGINAL nested module layout
(``flow.decoder.estimator.down_blocks.0.0.mlp.1.weight`` ...), of which only ~40% matched
mlx-audio's re-implemented S3Gen, so the CFM decoder stayed random-init -> near-silence.

``mlx-community/chatterbox-4bit`` was converted *with mlx-audio's own converter*, so its
weights use mlx-audio's NATIVE S3Gen layout
(``s3gen.flow.decoder.estimator.down_blocks_0.resnet.block1.conv.conv.weight`` ...). The
single ``model.safetensors`` carries ``ve.*`` / ``t3.*`` / ``s3gen.*`` prefixes exactly as
the standard ``chatterbox.Model`` expects, plus ``config.json`` (``model_type:
"chatterbox"``, 4-bit affine quant), ``tokenizer.json``, and a built-in default voice in
``conds.safetensors``. So ``mlx_audio.tts.utils.load_model`` "just works" here -- no custom
weight surgery, no ``conds.safetensors`` workaround. The 8-bit / fp16 siblings
(``chatterbox-8bit`` / ``chatterbox-fp16``) share the layout and would also load; 4-bit is
the smallest + fastest and was the verified choice.

----------------------------------------------------------------------------------------
API
----------------------------------------------------------------------------------------
``ChatterboxTTS`` mirrors ``pipeline/engines.py:TTSEngine`` so it can drop into the live
pipeline, and adds zero-shot voice cloning:

  * ``set_voice(ref_audio)`` -- clone a voice from a reference WAV path / float array /
    ``mx.array`` -> conditionals; subsequent synth speaks in that voice. ``set_voice(None)``
    restores the built-in default voice.
  * ``synth_stream(text)`` -- yields float32 1-D chunks (one per sentence) so playback can
    start on the first clause.
  * ``synth(text)`` -- whole utterance as one float32 array.
  * ``sample_rate`` -- 24000.
  * ``exaggeration`` (0-~1.5) is a real emotion knob on this model -- higher = more
    expressive/animated. ``cfg_weight`` trades adherence vs. expressiveness.

Robustness contract (same as the rest of the pipeline's engines): synth NEVER raises into
the caller. On any per-sentence failure it logs and yields silence, so a bad clause can't
crash the conversation loop. Construction CAN raise (a missing model is a setup error the
coordinator should see), but the helper :func:`try_load_chatterbox` swallows that too.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# The verified-working repo. 8-bit / fp16 siblings share the layout if higher quality is
# wanted at the cost of size/speed.
DEFAULT_REPO = "mlx-community/chatterbox-4bit"
SAMPLE_RATE = 24000

# A clean ~6.5 s reference clip already in-repo, handy as a stand-in cloning voice for demos.
DEFAULT_REF_AUDIO = "audio_samples/tts_outputs/kokoro_neutral.wav"

# Split on sentence boundaries so streaming yields one chunk per sentence (low TTFA).
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _load_audio_array(ref_audio: str | Path | np.ndarray | mx.array, sr: int) -> mx.array:
    """Coerce a reference into a 1-D ``mx.array`` at ``sr`` (24 kHz).

    Accepts a file path (any rate -> resampled+mono by mlx-audio's loader), a numpy/MLX
    array already at ``sr``, or an MLX array. Raises ValueError on empty input.
    """
    from mlx_audio.utils import load_audio

    if isinstance(ref_audio, (str, Path)):
        wav = load_audio(str(ref_audio), sample_rate=sr)
    elif isinstance(ref_audio, np.ndarray):
        wav = mx.array(np.asarray(ref_audio, dtype=np.float32).reshape(-1))
    elif isinstance(ref_audio, mx.array):
        wav = ref_audio.reshape(-1)
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported ref_audio type: {type(ref_audio)!r}")
    if wav.size == 0:
        raise ValueError("reference audio is empty")
    return wav


def load_chatterbox(repo: str = DEFAULT_REPO):
    """Load the Chatterbox MLX model via mlx-audio's standard ``chatterbox`` backend.

    Downloads from the Hub on first use (incl. the shared ``mlx-community/S3TokenizerV2``
    speech tokenizer), loads the prefixed ``ve.*``/``t3.*``/``s3gen.*`` weights, the text
    tokenizer, and the built-in ``conds.safetensors`` default voice. Returns the raw
    ``Model`` (with ``.generate`` / ``.prepare_conditionals`` / ``.sample_rate``); wrap it in
    :class:`ChatterboxTTS` for the pipeline interface.

    May raise (download/Metal/missing-file) -- a genuine setup failure the caller should see.
    Use :func:`try_load_chatterbox` if you want a never-raising variant.
    """
    from mlx_audio.tts.utils import get_model_path, load_model

    p = get_model_path(repo)
    path = p[0] if isinstance(p, tuple) else p
    logger.info("Chatterbox: loading %s ...", repo)
    model = load_model(str(path))
    logger.info("Chatterbox: loaded (sr=%s, default voice=%s)",
                getattr(model, "sample_rate", SAMPLE_RATE), model._conds is not None)
    return model


class ChatterboxTTS:
    """Emotive, voice-cloning Chatterbox TTS that matches the pipeline ``TTSEngine`` contract.

    Default voice is the model's built-in (from ``conds.safetensors``). Call
    :meth:`set_voice` with a reference WAV/array to clone and speak in that voice instead.
    All synthesis is synchronous (run it via ``asyncio.to_thread`` in the async app, like the
    other engines) and never raises into the caller -- failures yield logged silence.

    Args:
        repo: HF repo id (default the verified ``chatterbox-4bit``).
        ref_audio: optional reference to clone at construction (path/array). ``None`` keeps
            the built-in default voice.
        exaggeration: emotion intensity 0..~1.5 (0.5 neutral-natural; higher = more animated).
        cfg_weight: classifier-free-guidance weight; lower (~0.3) = more expressive prosody.
        temperature: T3 sampling temperature.
    """

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        ref_audio: str | Path | np.ndarray | mx.array | None = None,
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        peak_norm: float | None = 0.97,
        model: Any | None = None,
    ):
        self.repo = repo
        self.model = model if model is not None else load_chatterbox(repo)
        self.sample_rate = int(getattr(self.model, "sample_rate", SAMPLE_RATE))
        self.sr = self.sample_rate  # alias matching pipeline TTSEngine.sr (app uses app.tts.sr)
        self.exaggeration = float(exaggeration)
        self.cfg_weight = float(cfg_weight)
        self.temperature = float(temperature)
        # Peak-normalize each chunk to this level. The turbo model renders ~2x quieter
        # (peak ~0.5 vs ~0.97) than chatterbox-4bit; this brings them to parity without
        # clipping. None disables it. Near-no-op for already-loud models.
        self.peak_norm = peak_norm
        # The built-in conditionals loaded from conds.safetensors; restored by set_voice(None).
        self._default_conds = getattr(self.model, "_conds", None)
        self._lock = threading.Lock()  # generate() mutates shared model state; serialize it.
        self.voice_name = "default"
        if ref_audio is not None:
            self.set_voice(ref_audio)

    # -- voice cloning -----------------------------------------------------------------
    def set_voice(
        self,
        ref_audio: str | Path | np.ndarray | mx.array | None,
        *,
        exaggeration: float | None = None,
    ) -> bool:
        """Clone a voice from a reference clip (zero-shot), or restore the default voice.

        ``ref_audio=None`` reverts to the model's built-in default voice. Otherwise the
        reference (path or float audio at 24 kHz / any rate if a path) is encoded into T3 +
        S3Gen conditionals and used for all subsequent synthesis. Best with ~5-10 s of clean
        speech. Returns True on success; on failure it logs, keeps the current voice, and
        returns False (never raises).
        """
        if ref_audio is None:
            with self._lock:
                self.model._conds = self._default_conds
            self.voice_name = "default"
            logger.info("Chatterbox: voice reset to built-in default")
            return True
        try:
            wav = _load_audio_array(ref_audio, self.sample_rate)
            exa = self.exaggeration if exaggeration is None else float(exaggeration)
            conds = self.model.prepare_conditionals(wav, self.sample_rate, exaggeration=exa)
            with self._lock:
                # Two backend conventions: the standard chatterbox returns the conditionals for
                # the caller to assign; the turbo backend sets self._conds INTERNALLY and returns
                # None. Only overwrite when we actually got a value back, else we'd clobber the
                # turbo's just-set conds with None (→ silent "prepare_conditionals first" errors).
                if conds is not None:
                    self.model._conds = conds
                mx.eval(self.model.parameters())
            self.voice_name = ref_audio if isinstance(ref_audio, (str, Path)) else "cloned"
            logger.info("Chatterbox: voice cloned from %s (%.2fs ref)",
                        self.voice_name, wav.size / self.sample_rate)
            return True
        except Exception as e:  # never break the caller on a bad reference
            logger.error("Chatterbox: set_voice failed (%s); keeping current voice", e)
            return False

    # -- synthesis ---------------------------------------------------------------------
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Peak-normalize to ``self.peak_norm`` (only scales UP quiet output; capped to avoid
        amplifying near-silence). No-op if ``peak_norm`` is None or the chunk is ~silent."""
        if self.peak_norm is None:
            return audio
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak < 0.02:  # near-silent: leave as-is (don't blow up noise)
            return audio
        gain = min(self.peak_norm / peak, 8.0)
        return (audio * gain).astype(np.float32)

    def _generate_one(self, sentence: str, **overrides: Any) -> np.ndarray:
        """Synthesize a single sentence with the current voice; returns 1-D float32.

        Returns a tiny silence array on any failure (logged) so streaming never crashes.
        """
        kw: dict[str, Any] = {
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "temperature": self.temperature,
            "verbose": False,
        }
        kw.update(overrides)
        try:
            with self._lock:
                chunks = []
                for seg in self.model.generate(text=sentence, **kw):
                    au = getattr(seg, "audio", seg)
                    mx.eval(au)
                    chunks.append(np.array(au).reshape(-1).astype(np.float32))
            if not chunks:
                return np.zeros(1, np.float32)
            return self._normalize(np.concatenate(chunks))
        except Exception as e:
            logger.error("Chatterbox: synth failed for %r (%s); emitting silence",
                         sentence[:60], e)
            return np.zeros(int(0.1 * self.sample_rate), np.float32)

    def synth_stream(
        self,
        text: str,
        cancel: threading.Event | None = None,
        **overrides: Any,
    ) -> Iterator[np.ndarray]:
        """Yield float32 1-D audio chunks, one per sentence, so playback starts on clause 1.

        ``overrides`` may carry per-call ``exaggeration`` / ``cfg_weight`` / ``temperature``
        (or ``ref_audio`` for a one-off voice). ``cancel`` is polled between sentences so the
        pipeline can barge-in.
        """
        text = (text or "").strip()
        if not text:
            return
        for sent in (s for s in _SENT_RE.split(text) if s.strip()):
            if cancel is not None and cancel.is_set():
                return
            audio = self._generate_one(sent, **overrides)
            if cancel is not None and cancel.is_set():
                return
            yield audio

    def synth(self, text: str, **overrides: Any) -> np.ndarray:
        """Synthesize the whole utterance into one float32 array (silence if nothing)."""
        chunks = list(self.synth_stream(text, **overrides))
        return np.concatenate(chunks) if chunks else np.zeros(1, np.float32)


def try_load_chatterbox(
    repo: str = DEFAULT_REPO,
    ref_audio: str | Path | np.ndarray | mx.array | None = None,
    **kwargs: Any,
) -> ChatterboxTTS | None:
    """Construct :class:`ChatterboxTTS`, returning ``None`` (logged) instead of raising.

    Use when the pipeline should degrade gracefully (e.g. fall back to Kokoro) if Chatterbox
    can't be loaded on this machine.
    """
    try:
        return ChatterboxTTS(repo, ref_audio=ref_audio, **kwargs)
    except Exception as e:
        logger.error("Chatterbox: load failed (%s); model unavailable", e)
        return None
