#!/usr/bin/env python
"""Smoke test for the ReachyChatOffline vision path.

Generates (or loads) a simple, recognizable test image, feeds it plus a text
question through the Gemma 4 E4B QAT multimodal model, and prints the answer.

Usage:
    .venv/bin/python scripts/smoke_vision.py                 # synthetic image
    .venv/bin/python scripts/smoke_vision.py path/to/img.jpg # your own image

This does ONE correctness inference and reports ONE rough timing. It is NOT a
benchmark -- the GPU may be contended by other work, so the timing is only a
sanity check.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Make the in-repo package importable when run from the repo root without an
# editable install.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def make_test_image() -> "object":
    """Build a simple, unambiguous test image: a red circle on white."""
    from PIL import Image, ImageDraw

    size = 512
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    # A big red circle, clearly centered, so the description is easy to verify.
    margin = 120
    draw.ellipse([margin, margin, size - margin, size - margin], fill=(220, 30, 30))
    return img


def main() -> int:
    from reachy_chat.vision import VLM
    from reachy_chat.vision.frame_source import FileFrameSource

    prompt = "What do you see in this image? Describe it briefly."

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"[smoke] Using image file: {path}")
        image = FileFrameSource(path).get_frame()
        expectation = "(depends on your image)"
    else:
        print("[smoke] Generating a synthetic test image: red circle on white.")
        image = make_test_image()
        expectation = "a red circle on a white background"
        # Also save it so it can be inspected if needed.
        tmp = Path(tempfile.gettempdir()) / "reachy_smoke_vision.png"
        image.save(tmp)
        print(f"[smoke] Saved test image to: {tmp}")

    print("[smoke] Loading model (mlx-community/gemma-4-E4B-it-qat-4bit)...")
    load_start = time.perf_counter()
    vlm = VLM()
    load_secs = time.perf_counter() - load_start
    print(f"[smoke] Model loaded in {load_secs:.1f}s")

    print(f"[smoke] Prompt: {prompt!r}")
    print("[smoke] Running ONE inference (rough timing; GPU may be contended)...")
    t0 = time.perf_counter()
    answer = vlm.describe(image, prompt, max_tokens=120)
    elapsed = time.perf_counter() - t0

    print("\n========== RESULT ==========")
    print(f"Expected roughly: {expectation}")
    print(f"Model answer:\n{answer}")
    print("============================")
    print(
        f"\n[smoke] PRELIMINARY image+text inference wall time: {elapsed:.2f}s "
        f"(rough, GPU contended, single shot, includes full image prefill + "
        f"up to 120 decoded tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
