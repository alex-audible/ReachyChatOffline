"""Smoke test for the reusable pipeline engines (STT -> LLM -> TTS).

Feeds a prompt WAV through STTEngine (streaming), streams clauses from LLMEngine, synthesizes
each via TTSEngine, saves the response audio, and reports first-audio latency from t0.

    python scripts/smoke_pipeline.py [prompt.wav] [--llm REPO]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from harness import find_end_of_speech  # noqa: E402
from reachy_chat.pipeline import LLMEngine, STTEngine, TTSEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", default=str(ROOT / "audio_samples/prompts/p3_vision.wav"))
    ap.add_argument("--llm", default="mlx-community/gemma-4-E2B-it-qat-4bit")
    args = ap.parse_args()

    print("Loading engines ...")
    stt = STTEngine()
    llm = LLMEngine(repo=args.llm)
    tts = TTSEngine()
    stt.warmup()
    for _ in llm.stream_clauses("Hello"):
        pass
    list(tts.synth_stream("Hello there."))

    a, sr = sf.read(args.wav, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    eos_n = int(find_end_of_speech(a, sr) * sr)

    # Stream audio up to end-of-speech into STT (simulating live capture during the turn).
    stt.start()
    frame = 512
    for i in range(0, eos_n - frame, frame):
        stt.add_frame(a[i:i + frame])

    t0 = time.perf_counter()
    transcript = stt.finalize()
    t_stt = time.perf_counter()

    audio_chunks: list[np.ndarray] = []
    clauses: list[str] = []
    first_audio = None
    for clause in llm.stream_clauses(transcript):
        clauses.append(clause)
        for chunk in tts.synth_stream(clause):
            if first_audio is None:
                first_audio = time.perf_counter()
            audio_chunks.append(chunk)
    t_done = time.perf_counter()

    out = ROOT / "audio_samples/tts_outputs/pipeline_response.wav"
    sf.write(str(out), np.concatenate(audio_chunks), tts.sr)

    print(f"\ntranscript : {transcript!r}")
    print(f"reply      : {' '.join(clauses)!r}")
    print(f"first clause: {clauses[0]!r}" if clauses else "no reply")
    print(f"\nSTT finalize     : {(t_stt - t0) * 1000:.0f} ms")
    print(f"FIRST-AUDIO (t0) : {(first_audio - t0) * 1000:.0f} ms" if first_audio else "no audio")
    print(f"full reply synth : {(t_done - t0) * 1000:.0f} ms  ({len(np.concatenate(audio_chunks)) / tts.sr:.1f}s audio)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
