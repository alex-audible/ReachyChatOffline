"""Multi-engine TTS benchmark for the Reachy voice pipeline (mlx-audio backends).

Reports, per engine, the numbers that matter for the cascade:
  * **TTFA** (time-to-first-audio) = synth time for a short first CLAUSE — this is what the
    pipeline experiences: the LLM's first clause is sent to TTS and we want audio out fast.
    Budget ~110 ms p50 (architecture report). Measured warm (model loaded, ≥3 warmups).
  * **RTF** (real-time factor) = synth_time / audio_duration on a longer sentence (<1 = faster
    than real time; must be <1 to stream without underrun).
  * **peak GPU memory**.
And saves **sample WAVs** (neutral + emotive) per engine into audio_samples/tts_outputs/ so the
naturalness/emotion can be judged subjectively (TTS quality is subjective — per project spec).

Usage:
    python benchmarks/bench_tts.py                 # all ready engines
    python benchmarks/bench_tts.py kokoro orpheus  # subset
"""

from __future__ import annotations

import re
import sys
import time

import mlx.core as mx
import numpy as np
import soundfile as sf

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reachy_chat.tts import apply_kokoro_fixes  # noqa: E402

apply_kokoro_fixes()  # robustify Kokoro (mlx-audio #786) before any synth

from mlx_audio.tts.utils import get_model_path, load_model  # noqa: E402

OUT = "audio_samples/tts_outputs"

# Short first clause (TTFA), a longer line (RTF), and an emotive line (sample to judge).
CLAUSE = "Oh wow, that's wonderful!"
LONG = "Let me think about that for a second. The history of robotics is genuinely fascinating, you know?"
EMOTIVE = "Oh wow, that's amazing news — I'm so happy for you! Honestly, this just made my whole day."

# Per-engine config: repo + generate kwargs. Emotive variants where the engine supports it.
ENGINES: dict[str, dict] = {
    "kokoro": {
        "repo": "mlx-community/Kokoro-82M-bf16",
        "kwargs": {"voice": "af_heart", "lang_code": "a"},
        "emotive_text": EMOTIVE,
    },
    "chatterbox": {
        "repo": "mlx-community/chatterbox-turbo-mlx-q4",
        "kwargs": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
        # higher exaggeration = more emotion
        "emotive_kwargs": {"exaggeration": 0.8, "cfg_weight": 0.3},
        "emotive_text": EMOTIVE,
    },
    "orpheus": {
        "repo": "mlx-community/orpheus-3b-0.1-ft-bf16",
        "kwargs": {"voice": "tara"},
        # Orpheus supports inline emotion tags
        "emotive_text": "<excited> Oh wow, that's amazing news — I'm so happy for you! <laugh> "
                        "Honestly, this just made my whole day.",
    },
    "sesame": {
        "repo": "mlx-community/csm-1b",
        "kwargs": {"speaker": 0},
        "emotive_text": EMOTIVE,
    },
    "higgs": {
        "repo": "mlx-community/higgs-audio-v2-3B-mlx-q8",
        "kwargs": {},
        "emotive_text": EMOTIVE,
    },
}


def pctl(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


def split_sentences(text: str) -> list[str]:
    """Split into sentence-sized chunks — matches the production sentence-chunked streaming
    TTS path, and avoids mlx-audio's buggy multi-sentence internal batching (e.g. Kokoro)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def synth(model, text: str, kwargs: dict) -> tuple[float, np.ndarray, int]:
    """Synthesize sentence-by-sentence (production path). Return (ttfa_ms, audio, sr) where
    ttfa = time to the FIRST audio chunk of the FIRST sentence."""
    t = time.perf_counter()
    first = None
    chunks: list[np.ndarray] = []
    sr = getattr(model, "sample_rate", 24000)
    for sent in split_sentences(text):
        for seg in model.generate(text=sent, **kwargs):
            au = getattr(seg, "audio", seg)
            mx.eval(au)
            if first is None:
                first = (time.perf_counter() - t) * 1000
            sr = getattr(seg, "sample_rate", sr) or sr
            chunks.append(np.array(au).reshape(-1))
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, np.float32)
    return (first or float("nan")), audio, sr


def bench_engine(name: str, cfg: dict, n: int = 8) -> dict | None:
    repo = cfg["repo"]
    print(f"\n=== {name} ({repo}) ===")
    try:
        p = get_model_path(repo)
        p = p[0] if isinstance(p, tuple) else p
        model = load_model(p)
    except Exception as e:
        print(f"  SKIP (load failed): {type(e).__name__}: {e}")
        return None
    kw = cfg["kwargs"]
    try:
        # warmup
        for _ in range(3):
            synth(model, CLAUSE, kw)
        mx.eval()

        # TTFA over short clause
        ttfas = []
        for _ in range(n):
            ttfa, _, _ = synth(model, CLAUSE, kw)
            ttfas.append(ttfa)

        # RTF over long line
        rtfs = []
        for _ in range(3):
            t = time.perf_counter()
            _, audio, sr = synth(model, LONG, kw)
            synth_ms = (time.perf_counter() - t) * 1000
            dur_ms = len(audio) / sr * 1000
            rtfs.append(synth_ms / dur_ms if dur_ms else float("nan"))

        # Save samples (neutral clause + emotive line)
        _, neu, sr = synth(model, EMOTIVE, kw)
        sf.write(f"{OUT}/{name}_neutral.wav", neu, sr)
        em_text = cfg.get("emotive_text", EMOTIVE)
        em_kw = {**kw, **cfg.get("emotive_kwargs", {})}
        _, emo, sr = synth(model, em_text, em_kw)
        sf.write(f"{OUT}/{name}_emotive.wav", emo, sr)

        peak = mx.get_peak_memory() / 1e9
        res = {
            "engine": name, "repo": repo,
            "ttfa_p50": pctl(ttfas, 50), "ttfa_p95": pctl(ttfas, 95),
            "rtf": float(np.median(rtfs)), "peak_gb": peak, "sr": sr,
        }
        print(f"  TTFA p50/p95: {res['ttfa_p50']:.0f}/{res['ttfa_p95']:.0f} ms  "
              f"RTF: {res['rtf']:.2f}  peak: {peak:.2f} GB  sr: {sr}")
        print(f"  samples -> {OUT}/{name}_neutral.wav , {name}_emotive.wav")
        return res
    except Exception as e:
        print(f"  ERROR during synth: {type(e).__name__}: {e}")
        return None


def main(which: list[str]) -> None:
    import os
    os.makedirs(OUT, exist_ok=True)
    names = which or list(ENGINES)
    results = []
    for name in names:
        if name not in ENGINES:
            print(f"unknown engine {name}; known: {list(ENGINES)}")
            continue
        r = bench_engine(name, ENGINES[name])
        if r:
            results.append(r)
    # Summary table
    print("\n\n## TTS comparison (warm, M3)\n")
    print("| engine | TTFA p50 (ms) | TTFA p95 | RTF | peak GB | budget(≤110ms TTFA) |")
    print("|---|---:|---:|---:|---:|:--:|")
    for r in results:
        ok = "✅" if r["ttfa_p50"] <= 110 else ("🟡" if r["ttfa_p50"] <= 200 else "❌")
        print(f"| {r['engine']} | {r['ttfa_p50']:.0f} | {r['ttfa_p95']:.0f} | {r['rtf']:.2f} | {r['peak_gb']:.2f} | {ok} |")


if __name__ == "__main__":
    main(sys.argv[1:])
