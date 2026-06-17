"""Streaming STT finalize-latency benchmark (parakeet-mlx).

In the cascade, STT runs *during* the user's speech, so by end-of-speech (t0) almost all audio
is already transcribed. The number that lands in the voice-to-voice budget is the **finalize
latency**: time from feeding the last speech chunk to a stable final transcript. We also report
**per-chunk add_audio** time (must be < chunk duration → real-time keep-up, RTF<1).

Usage: python benchmarks/bench_stt_stream.py [repo]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import find_end_of_speech  # noqa: E402

from parakeet_mlx import from_pretrained  # noqa: E402

REPO = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/parakeet-tdt-0.6b-v2"
# Streaming chunk: parakeet needs a window large enough vs its frame context (tiny chunks
# underflow). 480 ms is a realistic low-latency streaming cadence. With streaming STT running
# during speech, the finalize cost at end-of-speech ≈ processing one such window.
CHUNK_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 480.0
# small right-context to minimise algorithmic look-ahead latency
CTX = (256, 8)


def pctl(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def run() -> None:
    print(f"Loading {REPO} ...")
    model = from_pretrained(REPO)
    prompts = sorted(Path("audio_samples/prompts").glob("*.wav"))

    def feed(a, sr, time_last):
        """Stream `a` in full CHUNK_MS windows; return (per_chunk_ms[], last_chunk_ms, text).
        last_chunk_ms = processing time of the final window = finalize cost at end-of-speech."""
        step = int(sr * CHUNK_MS / 1000)
        per, last_ms = [], float("nan")
        with model.transcribe_stream(context_size=CTX) as tx:
            n_full = len(a) // step
            for k in range(n_full):
                chunk = mx.array(a[k * step:(k + 1) * step])
                t = time.perf_counter()
                tx.add_audio(chunk)
                txt = tx.result.text  # forces incremental decode (realistic: partials each chunk)
                dt = (time.perf_counter() - t) * 1000
                if k == n_full - 1:
                    last_ms = dt
                else:
                    per.append(dt)
            text = tx.result.text
        return per, last_ms, text

    # warmup
    a, sr = sf.read(str(prompts[0]), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    feed(a, sr, True)
    mx.eval()

    finals, chunk_times = [], []
    print(f"\n{'prompt':16s} {'last_win':>9s} {'chunk_avg':>9s}  transcript")
    for p in prompts:
        a, sr = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        per, last_ms, text = feed(a, sr, True)
        finals.append(last_ms)
        chunk_times.extend(per)
        print(f"{p.stem:16s} {last_ms:7.0f}ms {np.mean(per) if per else 0:7.1f}ms  {text!r}")

    print("\n=== STREAMING STT (parakeet) ===")
    print(f"finalize latency p50/p95:  {pctl(finals,50):.0f} / {pctl(finals,95):.0f} ms")
    print(f"per-chunk add_audio p50:   {pctl(chunk_times,50):.1f} ms  (chunk={CHUNK_MS:.0f}ms → "
          f"RTF≈{pctl(chunk_times,50)/CHUNK_MS:.2f}, <1 = real-time)")
    print(f"budget (≤125ms finalize):  {'✅' if pctl(finals,50) <= 125 else '❌'}")


if __name__ == "__main__":
    run()
