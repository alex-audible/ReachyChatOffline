"""Scratch benchmark for Chatterbox-Turbo (q4) via the chatterbox_turbo backend.

Loads the model with src/reachy_chat/tts/chatterbox.py, measures time-to-first-audio (first
clause) and RTF on M3, and writes neutral/emotive sample WAVs. Pass --force to load despite the
known S3Gen vocoder mismatch (output will be near-silence); see the chatterbox.py docstring.

Usage:
    .venv/bin/python scripts/bench_chatterbox_turbo.py [--force]

SANDBOX: needs Metal + HF cache + network (S3TokenizerV2). Re-run with
dangerouslyDisableSandbox if it hits "No Metal device" / "Operation not permitted".
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reachy_chat.tts.chatterbox import (  # noqa: E402
    DEFAULT_REF_AUDIO,
    ChatterboxTurboUnavailable,
    load_chatterbox_turbo,
    synth_stream,
)

OUT = Path("audio_samples/tts_outputs")
CLAUSE = "Oh wow, that's wonderful!"
LONG = "Let me think about that for a second. The history of robotics is genuinely fascinating, you know?"
EMOTIVE = "Oh wow, that's amazing news — I'm so happy for you! Honestly, this just made my whole day."


def split_sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def synth_timed(model, text: str) -> tuple[float, np.ndarray, int]:
    t = time.perf_counter()
    first = None
    chunks: list[np.ndarray] = []
    sr = getattr(model, "sample_rate", 24000)
    for sent in split_sentences(text):
        for chunk in synth_stream(model, sent):
            if first is None:
                first = (time.perf_counter() - t) * 1000
            chunks.append(chunk)
    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    return (first or float("nan")), audio, sr


def main(force: bool) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"loading chatterbox-turbo-q4 (ref={DEFAULT_REF_AUDIO}, force={force}) ...")
    try:
        model = load_chatterbox_turbo(allow_broken_vocoder=force)
    except ChatterboxTurboUnavailable as e:
        print("\nBLOCKED:", e)
        return 2

    # warmup
    for _ in range(2):
        synth_timed(model, CLAUSE)
    mx.eval()

    ttfas = [synth_timed(model, CLAUSE)[0] for _ in range(5)]
    t = time.perf_counter()
    _, audio, sr = synth_timed(model, LONG)
    synth_ms = (time.perf_counter() - t) * 1000
    rtf = synth_ms / (len(audio) / sr * 1000)

    _, neu, sr = synth_timed(model, EMOTIVE)
    sf.write(str(OUT / "chatterbox_turbo_neutral.wav"), neu, sr)
    # Turbo ignores exaggeration; included to show the knob is a no-op.
    emo_chunks = list(synth_stream(model, EMOTIVE, exaggeration=0.8, cfg_weight=0.3))
    emo = np.concatenate(emo_chunks) if emo_chunks else np.zeros(1, np.float32)
    sf.write(str(OUT / "chatterbox_turbo_emotive.wav"), emo, sr)

    def rms(a):
        return float(np.sqrt((a.astype(np.float64) ** 2).mean()))

    print(f"\nTTFA p50: {np.percentile(ttfas, 50):.0f} ms   RTF: {rtf:.2f}   sr: {sr}")
    print(f"peak GPU mem: {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"neutral RMS: {rms(neu):.4f}  emotive RMS: {rms(emo):.4f}  "
          f"({'AUDIBLE SPEECH' if rms(neu) > 0.02 else 'NEAR-SILENCE / broken vocoder'})")
    print(f"samples -> {OUT}/chatterbox_turbo_neutral.wav , chatterbox_turbo_emotive.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main("--force" in sys.argv))
