"""Chatterbox-Turbo (q4) loader + synth wrapper for the Reachy voice pipeline.

STATUS (2026-06-17): **BLOCKED — does not produce usable audio.** The wrapper below
correctly loads the q4 repo via the ``chatterbox_turbo`` backend (T3 text->token model
loads and runs at RTF ~0.6 on M3), but the **S3Gen vocoder weights in this repo do not
match the module layout of mlx-audio's S3Gen implementation**, so the audio decoder runs
on random init and emits near-silence (RMS ~0.0012, ~-58 dBFS). See "THE BLOCKER" below.

----------------------------------------------------------------------------------------
WHAT WORKS — the load incantation
----------------------------------------------------------------------------------------
``mlx-community/chatterbox-turbo-mlx-q4`` ships a TURBO file layout
(``t3_turbo_v1.safetensors``, ``model.safetensors`` [= quantized T3], ``s3gen.safetensors``,
``s3gen_meanflow.safetensors``, ``ve.safetensors``) but its ``config.json`` declares
``model_type: "chatterbox"``. Two problems with ``mlx_audio.tts.utils.load_model``:

  1. **Routing.** ``get_model_class`` actually *does* reach the ``chatterbox_turbo`` backend
     here (the repo *name* contains "turbo"), but the stock loader globs **all** ``*.safetensors``
     into one dict with no per-file prefixing, so the unprefixed keys (``tfmr.*``, ``tokenizer.*``,
     ``similarity_*``) land in "other_weights" -> "Unrecognized weight keys" and never reach
     the t3 / s3gen / ve sub-modules.
  2. **Conditionals.** ``ChatterboxTurboTTS.post_load_hook`` ends with
     ``raise FileNotFoundError("conds.safetensors not found")`` and this repo has no
     ``conds.safetensors`` — so loading aborts before generation even with a ref voice.

This wrapper fixes both by building the model directly:
  * instantiate ``ChatterboxTurboTTS({"quantization": {...}})`` (creates T3 / S3Gen(meanflow) / VE);
  * load each component file with its correct top-level prefix
    (``model.safetensors`` -> ``t3.``, ``s3gen_meanflow.safetensors`` -> ``s3gen.``,
    ``ve.safetensors`` -> ``ve.``), run the model's own ``sanitize``, then ``apply_quantization``
    (only T3 is q4 in this repo) and ``load_weights(strict=False)``;
  * load the text tokenizer + the ``mlx-community/S3TokenizerV2`` speech tokenizer
    (post_load_hook's other side effects), but **skip** the conds.safetensors requirement;
  * derive conditionals at generate time from a reference WAV via ``ref_audio=`` (zero-shot
    voice cloning) — no conds.safetensors needed.

``Model.generate(text, ref_audio=..., temperature=...)`` yields ``GenerationResult`` objects
with ``.audio`` / ``.sample_rate`` (24 kHz), matching ``benchmarks/bench_tts.py`` and the
``TTSEngine.synth_stream`` contract in ``pipeline/engines.py``. NOTE: Turbo **ignores**
``exaggeration`` / ``cfg_weight`` (the backend logs a warning and drops them) — the t3 config
has ``emotion_adv: false`` — so the "exaggeration knob" is a no-op for this model.

----------------------------------------------------------------------------------------
THE BLOCKER — S3Gen weight-layout mismatch (vocoder runs on random init)
----------------------------------------------------------------------------------------
The decoder that turns speech tokens into a waveform (``s3gen.flow.decoder.estimator.*`` in
the repo file) is exported in ResembleAI's ORIGINAL nested module layout, e.g.::

    flow.decoder.estimator.down_blocks.0.0.mlp.1.weight

whereas mlx-audio's re-implemented S3Gen expects a DIFFERENT module tree, e.g.::

    decoder.estimator.down_blocks.0.resnet.block1.block.0.conv.conv.weight

These are not reconcilable by prefix/key renaming — the block decomposition itself differs.
Measured coverage after the model's own ``sanitize`` (every meanflow / non-meanflow combo):

    S3Gen: 891 / 2183 loadable params get real weights  (~40%)
            -> the ENTIRE decoder.estimator CFM network stays at random init.
    VE:    4 / 13 params matched.

Empirical confirmation (kokoro_neutral.wav as 6.6 s reference voice):
    "Oh wow, that's amazing news..."  -> 3.36 s @ 24 kHz, RTF 0.62, but
    RMS 0.0012 / max 0.004 / ZCR 0.30  == near-silence/noise, NOT speech.

So the T3 stage works and is fast; the vocoder cannot, because its weights aren't present in
a layout this backend can consume. This is a packaging mismatch between the q4 repo
(``mlx-audio 0.2.7`` conversion, per its README) and the installed mlx-audio's
``chatterbox_turbo`` S3Gen. ``from_local`` in the backend is also unusable here (it reads only
``model.safetensors`` with ``t3.``/``s3gen.``/``ve.`` prefixes that this repo's files don't have,
and never quantizes, so it can't read the q4 T3 either).

POSSIBLE PATHS FORWARD (out of scope of "work in src/reachy_chat/tts"):
  * use a NON-q4 chatterbox-turbo MLX repo whose S3Gen matches this mlx-audio version, or
  * convert ResembleAI/chatterbox-turbo with the *installed* mlx-audio's converter so the
    S3Gen module tree matches, and capture/ship a ``conds.safetensors`` (or keep ref_audio), or
  * write an S3Gen weight remapper (flow.decoder.estimator.*  ->  decoder.estimator.resnet.*)
    — large, brittle, and effectively re-implements the converter.

Until then, ``load_chatterbox_turbo`` raises ``ChatterboxTurboUnavailable`` by default so the
pipeline fails loudly rather than speaking silence. Pass ``allow_broken_vocoder=True`` to get
the (silent) model for debugging.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_REPO = "mlx-community/chatterbox-turbo-mlx-q4"
# A clean >5 s reference clip already in-repo, usable as a stand-in cloning voice.
DEFAULT_REF_AUDIO = "audio_samples/tts_outputs/kokoro_neutral.wav"
S3TOKENIZER_REPO = "mlx-community/S3TokenizerV2"

# Files that carry real (loadable) weights, mapped to the component prefix the backend's
# sanitize()/load_weights() route by. model.safetensors is the q4-quantized T3.
_COMPONENT_FILES = {
    "model.safetensors": "t3.",
    "s3gen_meanflow.safetensors": "s3gen.",
    "ve.safetensors": "ve.",
}
# Fraction of S3Gen vocoder params that must receive real (non-init) weights for output to be
# anything other than noise. The q4 repo currently yields ~0.40; see module docstring.
_MIN_S3GEN_COVERAGE = 0.90


class ChatterboxTurboUnavailable(RuntimeError):
    """Raised when the q4 turbo repo cannot produce usable audio (vocoder layout mismatch)."""


def _resolve_repo_path(repo: str) -> Path:
    from mlx_audio.tts.utils import get_model_path

    p = get_model_path(repo)
    return Path(p[0] if isinstance(p, tuple) else p)


def _s3gen_coverage(model) -> float:
    """Fraction of S3Gen's loadable params whose values were actually loaded (not random init)."""
    from mlx.utils import tree_flatten

    init_generated = {
        "encoder.embed.pos_enc.pe",
        "encoder.up_embed.pos_enc.pe",
        "mel2wav.stft_window",
        "trim_fade",
    }
    params = {k: v for k, v in tree_flatten(model.s3gen.parameters())}
    loadable = [k for k in params if k not in init_generated]
    if not loadable:
        return 0.0
    # A param left at init is ~zero-mean random; loaded conv/linear weights are too, so we can't
    # detect "loaded" purely by stats. Instead recompute the matched set from the source files.
    return _measured_s3gen_match_fraction(model)


def _measured_s3gen_match_fraction(model) -> float:
    """Re-derive how many S3Gen params the repo file can supply, via the model's own sanitize."""
    from mlx.utils import tree_flatten

    repo_path = Path(model.local_path) if model.local_path else None
    if repo_path is None:
        return 0.0
    f = repo_path / "s3gen_meanflow.safetensors"
    if not f.exists():
        return 0.0
    weights = dict(mx.load(str(f)))
    sanitized = model.s3gen.sanitize(weights) if hasattr(model.s3gen, "sanitize") else weights
    init_generated = {
        "encoder.embed.pos_enc.pe",
        "encoder.up_embed.pos_enc.pe",
        "mel2wav.stft_window",
        "trim_fade",
    }
    params = {k for k, _ in tree_flatten(model.s3gen.parameters())}
    loadable = params - init_generated
    matched = set(sanitized) & loadable
    return len(matched) / len(loadable) if loadable else 0.0


def load_chatterbox_turbo(
    repo: str = DEFAULT_REPO,
    ref_audio: str | None = DEFAULT_REF_AUDIO,
    *,
    allow_broken_vocoder: bool = False,
):
    """Load the q4 Chatterbox-Turbo model via the ``chatterbox_turbo`` backend.

    Builds the model directly (bypassing the stock loader's all-files glob and the
    ``conds.safetensors`` requirement in ``post_load_hook``), loads the q4 T3 + S3Gen + VE
    with correct per-component prefixes, wires up the text and S3 speech tokenizers, and (if
    ``ref_audio`` is given) primes zero-shot conditionals so ``generate`` needs no extra args.

    Raises:
        ChatterboxTurboUnavailable: if the S3Gen vocoder is under-covered by this repo's
            weights (the current state — see module docstring), unless ``allow_broken_vocoder``.
    """
    from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS
    from mlx_audio.utils import apply_quantization

    repo_path = _resolve_repo_path(repo)
    config = _load_config(repo_path)
    qcfg = config.get("quantization") or config.get("quantization_config")

    model = ChatterboxTurboTTS(config)
    model.local_path = str(repo_path)

    # Assemble weights with the prefixes the backend's sanitize()/load_weights() route by.
    weights: dict[str, Any] = {}
    for filename, prefix in _COMPONENT_FILES.items():
        fpath = repo_path / filename
        if not fpath.exists():
            logger.warning("Chatterbox: expected component file missing: %s", fpath)
            continue
        for k, v in mx.load(str(fpath)).items():
            weights[prefix + k] = v

    sanitized = model.sanitize(weights) if hasattr(model, "sanitize") else weights
    if qcfg:
        apply_quantization(
            model, {"quantization": qcfg}, sanitized,
            getattr(model, "model_quant_predicate", None),
        )
    model.load_weights(list(sanitized.items()), strict=False)
    mx.eval(model.parameters())

    _load_tokenizers(model, repo_path)

    coverage = _measured_s3gen_match_fraction(model)
    if coverage < _MIN_S3GEN_COVERAGE and not allow_broken_vocoder:
        raise ChatterboxTurboUnavailable(
            f"S3Gen vocoder weights cover only {coverage:.0%} of the decoder for repo {repo!r}; "
            "the q4 turbo repo's S3Gen layout does not match this mlx-audio backend, so output "
            "would be near-silence. See src/reachy_chat/tts/chatterbox.py docstring. "
            "Pass allow_broken_vocoder=True to load anyway for debugging."
        )
    if coverage < _MIN_S3GEN_COVERAGE:
        logger.warning(
            "Chatterbox: S3Gen vocoder only %.0f%% covered — audio will be near-silence.",
            coverage * 100,
        )

    if ref_audio is not None:
        # Prime zero-shot conditionals so synth_stream/generate need no per-call ref.
        model.prepare_conditionals(ref_audio)

    return model


def _load_config(repo_path: Path) -> dict:
    import json

    with open(repo_path / "config.json", encoding="utf-8") as f:
        return json.load(f)


def _load_tokenizers(model, repo_path: Path) -> None:
    """Replicate post_load_hook's tokenizer setup without the conds.safetensors requirement."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(repo_path))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.tokenizer = tok
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Chatterbox: could not load text tokenizer: %s", e)

    try:
        from huggingface_hub import hf_hub_download

        s3_weights_path = hf_hub_download(repo_id=S3TOKENIZER_REPO, filename="model.safetensors")
        s3w = mx.load(s3_weights_path)
        if hasattr(model._s3tokenizer, "sanitize"):
            s3w = model._s3tokenizer.sanitize(s3w)
        model._s3tokenizer.load_weights(list(s3w.items()), strict=False)
        mx.eval(model._s3tokenizer.parameters())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Chatterbox: could not load S3 speech tokenizer: %s", e)


def synth_stream(
    model,
    text: str,
    *,
    ref_audio: str | None = None,
    **generate_kwargs: Any,
) -> Iterator[np.ndarray]:
    """Yield float32 1-D audio chunks for ``text`` — matches ``bench_tts``/``TTSEngine``.

    Conditionals come from ``ref_audio`` if given, else from the ref primed at load time.
    Note: Chatterbox-Turbo ignores ``exaggeration``/``cfg_weight`` (no emotion knob).
    """
    if ref_audio is not None:
        generate_kwargs.setdefault("ref_audio", ref_audio)
    for seg in model.generate(text=text, **generate_kwargs):
        au = getattr(seg, "audio", seg)
        mx.eval(au)
        yield np.array(au).reshape(-1).astype(np.float32)


def synth(model, text: str, **generate_kwargs: Any) -> np.ndarray:
    chunks = list(synth_stream(model, text, **generate_kwargs))
    return np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
