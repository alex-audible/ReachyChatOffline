"""Standalone voice-cloning demo for Chatterbox (MLX, 4-bit).

Records ~8 s of YOUR voice from the mic, clones it with
``reachy_chat.tts.chatterbox.ChatterboxTTS.set_voice``, then synthesizes a few sample
sentences in the cloned voice and writes them to ``audio_samples/tts_outputs/clone_{0,1,2}.wav``.
Prints time-to-first-audio (TTFA) and real-time factor (RTF) per sentence so you can judge
live-streaming feasibility.

USAGE
    .venv/bin/python scripts/clone_voice_demo.py            # record from mic, then clone
    .venv/bin/python scripts/clone_voice_demo.py --ref some.wav   # clone from a WAV (no mic)
    .venv/bin/python scripts/clone_voice_demo.py --seconds 10     # record 10 s instead of 8
    .venv/bin/python scripts/clone_voice_demo.py --play           # also play results aloud

MIC PERMISSION (macOS): the microphone needs a REAL terminal (Terminal.app / iTerm), not an
IDE-embedded or sandboxed shell, or macOS silently returns all-zero audio. The script checks
the recording level and warns if it looks empty. Use ``--ref FILE`` to skip the mic entirely.

SANDBOX: needs Metal + the HF cache (model + S3Tokenizer) + mic/Core Audio. If you see
"No Metal device" / "Operation not permitted", run it outside the sandbox.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reachy_chat.tts.chatterbox import ChatterboxTTS  # noqa: E402

OUT_DIR = Path("audio_samples/tts_outputs")
SENTENCES = [
    "Hello! This is my voice, cloned by Reachy in just a few seconds.",
    "Isn't it amazing? I can now say absolutely anything in your voice.",
    "Thanks for trying the demo. I hope this sounds just like you.",
]


def record_mic(seconds: float, target_sr: int) -> np.ndarray:
    """Record ``seconds`` of mono audio from the default mic and resample to ``target_sr``.

    Captures at the device's native rate (e.g. 48 kHz) then resamples with soxr -- robust to
    whatever the hardware rate is, matching the pipeline's MicStream approach.
    """
    import sounddevice as sd
    import soxr

    dev = sd.default.device[0]
    info = sd.query_devices(dev, "input")
    native = int(info["default_samplerate"])
    print(f"\nInput device: {info['name']} @ {native} Hz")
    print("\nSpeak naturally for the countdown -- read a couple of sentences, anything you like.")
    print("Tip: ~8 seconds of clean, expressive speech clones best.\n")
    input(f"Press ENTER, then start talking (recording {seconds:.0f}s)... ")
    print("- recording...", flush=True)
    rec = sd.rec(int(seconds * native), samplerate=native, channels=1, dtype="float32")
    sd.wait()
    print("- done.")
    audio = rec[:, 0]
    if native != target_sr:
        audio = soxr.resample(audio, native, target_sr).astype(np.float32)
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"  captured {len(audio)/target_sr:.1f}s  peak={peak:.3f}  rms={rms:.4f}")
    if rms < 0.005:
        print("  WARNING: recording is near-silent. On macOS the mic needs a REAL terminal and")
        print("  microphone permission. Try Terminal.app, or pass --ref FILE to skip the mic.")
    return audio


def synth_timed(tts: ChatterboxTTS, text: str):
    """Synthesize ``text``, returning (audio, ttfa_ms, rtf)."""
    t0 = time.perf_counter()
    first_ms = None
    chunks: list[np.ndarray] = []
    for chunk in tts.synth_stream(text):
        if first_ms is None:
            first_ms = (time.perf_counter() - t0) * 1000.0
        chunks.append(chunk)
    wall = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    dur = len(audio) / tts.sample_rate
    rtf = wall / dur if dur > 0 else float("nan")
    return audio, (first_ms if first_ms is not None else float("nan")), rtf


def main() -> int:
    ap = argparse.ArgumentParser(description="Chatterbox voice-cloning demo")
    ap.add_argument("--ref", type=str, default=None,
                    help="reference WAV to clone (skips the mic)")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="seconds of mic audio to record (default 8)")
    ap.add_argument("--exaggeration", type=float, default=0.5,
                    help="emotion intensity 0..~1.5 (default 0.5)")
    ap.add_argument("--play", action="store_true", help="play the cloned sentences aloud")
    args = ap.parse_args()

    import soundfile as sf

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Chatterbox (mlx-community/chatterbox-4bit)...")
    t0 = time.perf_counter()
    tts = ChatterboxTTS(exaggeration=args.exaggeration)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s (sr={tts.sample_rate} Hz)")

    # warmup so the reported TTFA/RTF reflect steady state, not first-call compile.
    print("Warming up the graph...")
    list(tts.synth_stream("Warming up the model."))

    # 1) get a reference voice
    if args.ref:
        ref = args.ref
        print(f"\nCloning from file: {ref}")
    else:
        ref = record_mic(args.seconds, tts.sample_rate)

    if not tts.set_voice(ref):
        print("\nVoice cloning FAILED (see log above). Falling back to the built-in voice.")

    # 2) synthesize the samples in the cloned voice
    print(f"\nSynthesizing {len(SENTENCES)} sentences in the cloned voice:\n")
    speaker = None
    if args.play:
        try:
            from reachy_chat.pipeline.audio_io import Speaker
            speaker = Speaker(sr=tts.sample_rate).__enter__()
        except Exception as e:
            print(f"  (could not open speaker: {e})")
            speaker = None

    saved = []
    for i, sent in enumerate(SENTENCES):
        audio, ttfa, rtf = synth_timed(tts, sent)
        out = OUT_DIR / f"clone_{i}.wav"
        sf.write(str(out), audio, tts.sample_rate)
        saved.append(out)
        print(f"  [{i}] {sent}")
        print(f"      {len(audio)/tts.sample_rate:.2f}s audio  |  TTFA {ttfa:.0f} ms  |  RTF {rtf:.2f}  ->  {out}")
        if speaker is not None:
            speaker.play(audio)

    if speaker is not None:
        speaker.wait()
        speaker.__exit__()

    print("\nDone. Cloned-voice samples written to:")
    for p in saved:
        print(f"  {p.resolve()}")
    print("\nRTF < 1.0 means faster than real-time (streamable live). TTFA is the wait before")
    print("the first audio of each sentence -- the latency the user perceives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
