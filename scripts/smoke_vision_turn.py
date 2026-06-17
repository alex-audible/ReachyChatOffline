#!/usr/bin/env python
"""Smoke test for the ReachyChatOffline vision *turn handler*.

Exercises ``VisionResponder.maybe_answer`` the way the live turn loop will:

1. a NON-visual transcript -> should return ``None`` (text LLM would handle it),
2. a VISUAL transcript -> capture a real frame (Reachy daemon camera, falling
   back to the Mac webcam, then a synthetic image), run the Gemma 4 E4B VLM, and
   print the short spoken answer.

It reports which frame source actually produced the frame and a single, ROUGH
image-prefill / inference timing (PRELIMINARY -- the Metal GPU may be contended
by another process, so this is a sanity check, not a benchmark; one inference,
no timing loops).

Usage::

    .venv/bin/python scripts/smoke_vision_turn.py

Notes:
    * Needs the Reachy daemon at localhost:8000 for the live-camera path; without
      it the script falls back to the Mac webcam, then a synthetic image.
    * Under the project's command sandbox, Metal / localhost / GStreamer access
      needs the sandbox disabled (re-run the same command with the sandbox off).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the in-repo package importable when run from the repo root without an
# editable install.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


NON_VISUAL = "What time is it in Paris right now?"
VISUAL_QUERIES = [
    "What do you see?",
    "Can you describe what's in front of you?",
    "What am I holding?",
]


def main() -> int:
    from reachy_chat.vision import VisionResponder, best_available_frame_source

    print("=" * 60)
    print("Vision turn-handler smoke test")
    print("=" * 60)

    # Build the robust frame source up front so we can report which path works
    # and reuse it across the responder (avoids opening the camera twice).
    print("\n[1] Building best-available frame source "
          "(Reachy camera -> webcam -> synthetic)...")
    frame_source = best_available_frame_source()

    responder = VisionResponder(frame_source=frame_source)

    # --- Non-visual transcript: should route to the text LLM (returns None). ---
    print(f"\n[2] Non-visual transcript: {NON_VISUAL!r}")
    ans = responder.maybe_answer(NON_VISUAL)
    if ans is None:
        print("    -> None (correct: the text LLM would handle this turn)")
    else:
        print(f"    -> UNEXPECTED non-None answer: {ans!r}")

    # --- Capture a frame once (so we can report the source + rough timing). ---
    print("\n[3] Capturing one frame...")
    cap_t0 = time.perf_counter()
    frame = frame_source.get_frame()
    cap_secs = time.perf_counter() - cap_t0
    used = frame_source.last_used.name if frame_source.last_used else "?"
    print(f"    frame source used: {used}")
    print(f"    frame size: {frame.size} (w x h), mode={frame.mode}")
    print(f"    capture wall time: {cap_secs * 1000:.0f} ms")

    # Save the captured frame for eyeballing.
    out = Path("/tmp/claude/reachy_vision_turn_frame.png")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.save(out)
        print(f"    saved frame to: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"    (could not save frame: {e})")

    # --- Demonstrate lazy-loading: VLM is NOT loaded until the first visual
    #     query. Load it explicitly here (timed separately) so the inference
    #     number below isn't polluted by the ~6 GB model load. ---
    q = VISUAL_QUERIES[0]
    print(f"\n[4] Visual transcript: {q!r}")
    print(f"    vlm_loaded before first visual query: {responder.vlm_loaded} "
          "(lazy)")
    print("    Loading VLM (one-time, slow)...")
    load_t0 = time.perf_counter()
    responder._get_vlm()  # force the lazy load so we can time inference alone
    load_secs = time.perf_counter() - load_t0
    print(f"    VLM loaded in {load_secs:.1f}s; vlm_loaded now: {responder.vlm_loaded}")

    print("    Running ONE inference "
          "(PRELIMINARY timing; GPU may be contended)...")
    t0 = time.perf_counter()
    spoken = responder.maybe_answer(q)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Question:      {q!r}")
    print(f"Frame source:  {frame_source.last_used.name if frame_source.last_used else '?'}")
    print(f"Spoken answer: {spoken!r}")
    print(
        f"\nPRELIMINARY inference wall time: {elapsed:.2f}s "
        f"(frame capture + image prefill + up to {responder.max_tokens} decoded "
        f"tokens; model load NOT included -- that was {load_secs:.1f}s above, "
        "once per session)."
    )
    print(
        "NOTE: rough, single shot, GPU possibly contended -- NOT a benchmark. "
        "The main thread does careful serial latency measurement separately."
    )

    # Show that the heuristic flags the other phrasings too (no extra inference).
    from reachy_chat.vision import is_visual_query

    print("\nis_visual_query() sanity check:")
    for t in [NON_VISUAL, *VISUAL_QUERIES]:
        print(f"    {is_visual_query(t)!s:>5}  {t!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
