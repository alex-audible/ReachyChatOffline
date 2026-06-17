"""Record a fixed sentence from the mic, resample to 16 kHz, save it for transcription analysis.

Run in a real terminal (mic permission). Reads a target sentence aloud, saves /tmp/mic_test.wav
(16 kHz mono) + prints input levels. The coordinator then transcribes that file and compares.

    python scripts/mic_test.py
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

TARGET = ("Hello Reachy, my name is Alex. What's the weather like in Sydney today, "
          "and can you set a timer for fifteen minutes?")
SECS = 9

dev = sd.default.device[0]
native = int(sd.query_devices(dev, "input")["default_samplerate"])
print(f"\nInput device: {sd.query_devices(dev, 'input')['name']} @ {native} Hz")
print("\nRead this ALOUD, clearly, at a normal pace:\n")
print(f'    "{TARGET}"\n')
input(f"Press ENTER, then read it (recording {SECS}s)… ")
print("● recording…", flush=True)
rec = sd.rec(int(SECS * native), samplerate=native, channels=1, dtype="float32")
sd.wait()
a = rec[:, 0]
a16 = soxr.resample(a, native, 16000).astype(np.float32) if native != 16000 else a
sf.write("/tmp/mic_test.wav", a16, 16000)
peak = float(np.abs(a16).max())
rms = float(np.sqrt(np.mean(a16 ** 2)))
print(f"\nsaved /tmp/mic_test.wav   peak={peak:.3f}  rms={rms:.4f}")
print("(peak < ~0.1 or rms < ~0.01 = mic too quiet — a likely STT-quality cause)")
print("Done — tell the assistant and it will transcribe /tmp/mic_test.wav.")
