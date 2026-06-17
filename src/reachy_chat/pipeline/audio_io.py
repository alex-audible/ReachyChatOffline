"""Live audio I/O via sounddevice: a 512-frame mic source and a flushable speaker.

Kept thin and dependency-isolated so the offline driver (file-fed) needs none of it.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms at 16 kHz — the frame size the audio front-end expects


class MicStream:
    """Yields 512-sample/16 kHz float32 frames from the default input device.

    Captures at the device's NATIVE rate (e.g. 48 kHz on a MacBook mic) and resamples to
    16 kHz with soxr, then re-chunks to exact 512-sample frames — robust regardless of the
    hardware rate (don't rely on CoreAudio's implicit conversion)."""

    def __init__(self, target_sr: int = SAMPLE_RATE, frame: int = FRAME, device=None):
        import sounddevice as sd
        self._sd = sd
        self.target_sr = target_sr
        self.frame = frame
        self.device = device if device is not None else sd.default.device[0]
        info = sd.query_devices(self.device, "input")
        self.native_sr = int(info["default_samplerate"])
        self.block = max(1, int(self.native_sr * 0.032))  # ~32 ms native blocks
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._resid = np.zeros(0, np.float32)
        self._stream = None

    def _cb(self, indata, frames, time_info, status):  # noqa: ANN001
        self._q.put(indata[:, 0].copy())

    def __enter__(self):
        import soxr
        # STATEFUL streaming resampler — keeps filter continuity across mic blocks. A fresh
        # one-shot soxr.resample() per 32 ms block injects edge transients every block and
        # badly degrades STT; ResampleStream avoids that.
        self._rs = (soxr.ResampleStream(self.native_sr, self.target_sr, 1, dtype="float32")
                    if self.native_sr != self.target_sr else None)
        self._stream = self._sd.InputStream(
            samplerate=self.native_sr, channels=1, dtype="float32",
            blocksize=self.block, callback=self._cb, device=self.device)
        self._stream.start()
        return self

    def frames(self):
        while True:
            block = self._q.get()
            if self._rs is not None:
                block = np.asarray(self._rs.resample_chunk(block), dtype=np.float32).reshape(-1)
            self._resid = np.concatenate([self._resid, block])
            while len(self._resid) >= self.frame:
                yield self._resid[: self.frame]
                self._resid = self._resid[self.frame :]

    def drain(self) -> None:
        """Discard any buffered/queued mic audio (e.g. captured during a turn / Reachy's TTS)."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._resid = np.zeros(0, np.float32)

    def __exit__(self, *exc):
        if self._stream:
            self._stream.stop(); self._stream.close()


class Speaker:
    """Plays float32 audio chunks at ``sr``; ``flush()`` instantly drops queued audio
    (barge-in). Runs a background writer thread feeding a sounddevice OutputStream."""

    def __init__(self, sr: int = 24000, device=None):
        import sounddevice as sd
        self._sd = sd
        self.in_sr = sr  # the TTS engine's sample rate
        self.device = device if device is not None else sd.default.device[1]
        info = sd.query_devices(self.device, "output")
        self.out_sr = int(info["default_samplerate"])  # device native rate
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream = None
        self._thread = None
        self._playing = threading.Event()
        self._pending = 0  # chunks queued but not yet written to the device
        self._lock = threading.Lock()

    def __enter__(self):
        self._stream = self._sd.OutputStream(
            samplerate=self.out_sr, channels=1, dtype="float32", device=self.device)
        self._stream.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while True:
            chunk = self._q.get()
            if chunk is None:
                break
            try:
                self._stream.write(chunk)
            except Exception:
                pass
            with self._lock:
                self._pending -= 1

    def play(self, chunk: np.ndarray) -> None:
        import soxr
        chunk = chunk.astype(np.float32)
        if self.in_sr != self.out_sr:
            chunk = soxr.resample(chunk, self.in_sr, self.out_sr).astype(np.float32)
        with self._lock:
            self._pending += 1
        self._q.put(chunk)

    def wait(self) -> None:
        """Block until all queued audio has played out (so the mic can be flushed before we
        listen again — prevents Reachy transcribing its own TTS = self-feedback loop)."""
        import time
        while True:
            with self._lock:
                done = self._pending <= 0
            if done and self._q.empty():
                break
            time.sleep(0.02)
        time.sleep(0.15)  # let the device output buffer fully drain

    def flush(self) -> None:
        """Drop all queued audio immediately (barge-in)."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._stream is not None:
            self._stream.abort()
            self._stream.start()

    def __exit__(self, *exc):
        self._q.put(None)
        if self._stream:
            self._stream.stop(); self._stream.close()
