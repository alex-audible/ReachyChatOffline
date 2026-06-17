"""Endpoint-decision latency benchmark for the VAD + Smart-Turn endpointer.

Methodology (mirrors benchmarks/harness.py):

  * ``t0`` = ground-truth end-of-speech in each prompt WAV, from
    ``harness.find_end_of_speech`` (the same clock every other benchmark uses).
  * Each prompt is streamed in **real time** in 32 ms frames (512 samples @ 16 kHz
    — Silero's only legal window), so the endpointer sees audio exactly as it would
    live. Smart-Turn runs *during* the trailing silence, just like in production.
  * **Endpoint-decision latency = (wall time the endpointer fired) - (wall time t0
    was crossed)**. This is the *perceived* turn-end lag: it includes the
    deliberate silence window (the dominant, tunable term) plus Smart-Turn compute.
    We also report the Smart-Turn *inference* cost separately (the pure compute on
    the hot path, ``EndpointEvent.decision_latency_ms``).
  * Report p50 / p95 across prompts x repeats, plus the fire-reason mix.

GPU NOTE: another process may share the Metal GPU; Silero + Smart-Turn are ONNX/CPU
so are largely insulated, but treat absolute numbers as PRELIMINARY (correctness
first). Run:  ``.venv/bin/python benchmarks/bench_endpoint.py [--no-smart-turn] [-n N]``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# load_prompt_dir() locates each prompt's t0 via harness.find_end_of_speech().
from harness import Prompt, load_prompt_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reachy_chat.audio import EndpointConfig, Endpointer  # noqa: E402
from reachy_chat.audio.vad import FRAME_SAMPLES  # noqa: E402

PROMPT_DIR = Path(__file__).resolve().parent.parent / "audio_samples" / "prompts"
SR = 16000


def pctl(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


def stream_endpoint(prompt: Prompt, ep: Endpointer, realtime: bool = True) -> dict | None:
    """Feed one prompt in real-time 512-sample frames; return the fired event's
    measured latencies, or ``None`` if it never fired."""
    ep.reset()
    audio = prompt.audio.astype(np.float32)
    eos_sample = int(prompt.eos_sec * SR)
    frame = FRAME_SAMPLES
    start = time.perf_counter()
    t0_wall: float | None = None

    for i in range(0, len(audio), frame):
        chunk = audio[i : i + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        if realtime:
            target = start + i / SR
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
        if t0_wall is None and (i + frame) >= eos_sample:
            t0_wall = time.perf_counter()
        evt = ep.process_frame(chunk)
        if evt is not None:
            fired_wall = time.perf_counter()
            if t0_wall is None:  # eos beyond end-of-file; clamp to now
                t0_wall = fired_wall
            return {
                "decision_latency_ms": (fired_wall - t0_wall) * 1000.0,
                "compute_ms": evt.decision_latency_ms,
                "reason": evt.reason,
                "probability": evt.probability,
                "silence_ms": evt.silence_ms,
            }
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=5, help="repeats per prompt (measured)")
    ap.add_argument("--warmup", type=int, default=2, help="warmup repeats per prompt")
    ap.add_argument("--no-smart-turn", action="store_true", help="VAD silence-window only")
    ap.add_argument("--eager-window-ms", type=float, default=None, help="override eager window")
    ap.add_argument("--medium-window-ms", type=float, default=None, help="override medium window")
    ap.add_argument("--max-silence-ms", type=float, default=None)
    ap.add_argument("--no-realtime", action="store_true", help="feed as fast as possible")
    args = ap.parse_args()

    prompts = load_prompt_dir(PROMPT_DIR)
    if not prompts:
        print(f"No prompts in {PROMPT_DIR}")
        return
    smart_turn = not args.no_smart_turn
    cfg = EndpointConfig()  # defaults = the tuned eagerness layer
    if args.eager_window_ms is not None:
        cfg.eager_window_ms = args.eager_window_ms
    if args.medium_window_ms is not None:
        cfg.medium_window_ms = args.medium_window_ms
    if args.max_silence_ms is not None:
        cfg.max_silence_ms = args.max_silence_ms

    print(f"Loading endpointer (smart_turn={smart_turn}) ...")
    ep = Endpointer(config=cfg, smart_turn=smart_turn)
    ep.warmup()
    st_on = ep.smart_turn is not None
    print(f"Smart-Turn active: {st_on}" + ("" if st_on or not smart_turn else "  (fell back to VAD-only)"))
    print(f"Prompts: {len(prompts)}  ground-truth t0 (eos):")
    for p in prompts:
        print(f"  {p.name:14s} dur={p.duration_sec:.2f}s  t0={p.eos_sec:.2f}s")

    realtime = not args.no_realtime
    for p in prompts:  # warmup (not measured)
        for _ in range(args.warmup):
            stream_endpoint(p, ep, realtime=realtime)

    all_latency: list[float] = []
    all_compute: list[float] = []
    reasons: dict[str, int] = {}
    misses = 0
    per_prompt: dict[str, list[float]] = {p.name: [] for p in prompts}
    per_prompt_last: dict[str, dict] = {}

    print(f"\nMeasuring (n={args.n}/prompt, realtime={realtime}) ...")
    for p in prompts:
        for _ in range(args.n):
            r = stream_endpoint(p, ep, realtime=realtime)
            if r is None:
                misses += 1
                continue
            all_latency.append(r["decision_latency_ms"])
            all_compute.append(r["compute_ms"])
            per_prompt[p.name].append(r["decision_latency_ms"])
            per_prompt_last[p.name] = r
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    print("\n=== ENDPOINT DECISION LATENCY (relative to harness t0 = end-of-speech) ===")
    print(f"{'prompt':14s} {'p50 ms':>8s} {'p95 ms':>8s}  {'reason':16s} {'P':>5s} {'sil_ms':>6s}  n")
    for name, xs in per_prompt.items():
        if xs:
            r = per_prompt_last[name]
            pr = f"{r['probability']:.2f}" if r["probability"] is not None else "  - "
            print(f"{name:14s} {pctl(xs, 50):8.0f} {pctl(xs, 95):8.0f}  "
                  f"{r['reason']:16s} {pr:>5s} {r['silence_ms']:6.0f}  {len(xs)}")
    print("-" * 60)
    print(f"{'OVERALL':14s} {pctl(all_latency, 50):8.0f} {pctl(all_latency, 95):8.0f}  {'':16s} {'':5s} {'':6s}  {len(all_latency)}")
    if all_compute:
        print(
            f"\nSmart-Turn inference (compute only): "
            f"p50={pctl(all_compute, 50):.1f} ms  p95={pctl(all_compute, 95):.1f} ms"
        )
    print(f"fire reasons: {reasons}   misses: {misses}")
    print(
        f"\nconfig: eager_window={cfg.eager_window_ms:.0f} ms (P>={cfg.eager_threshold}), "
        f"medium_window={cfg.medium_window_ms:.0f} ms (P>={cfg.turn_threshold}), "
        f"max_silence={cfg.max_silence_ms:.0f} ms, smart_turn={st_on}"
    )
    print("NOTE: latency ≈ required silence window (by confidence) + Smart-Turn compute, "
          "measured vs harness t0 (which itself includes a 120 ms hangover, so a confident "
          "eager commit can read ~0). PRELIMINARY if the Metal GPU is shared (these are CPU/ONNX).")


if __name__ == "__main__":
    main()
