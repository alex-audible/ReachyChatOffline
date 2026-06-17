"""Wake-word detector ("Hey Reachy") with a clean fallback.

Primary path (docs/research/04-vad-turntaking.md §4.1): **livekit-wakeword**
(``livekit.wakeword``, Apache-2.0, ONNX). It installs cleanly on Python 3.12 and
its inference backbone (bundled mel + speech-embedding ONNX) runs out of the box;
the only missing artefact for *this* keyword is a trained classifier
``hey_reachy.onnx``, produced offline by::

    pip install "livekit-wakeword[train,eval,export]"
    brew install espeak-ng ffmpeg sox portaudio
    python -m livekit.wakeword train --config hey_reachy.yaml   # -> hey_reachy.onnx (~200 KB)

Drop the resulting ``hey_reachy.onnx`` next to this file (or pass ``model_path``)
and the real conv-attention detector is used: ~2 s rolling window of 16 kHz audio
-> ``WakeWordModel.predict()`` -> per-model confidence, fired when it crosses
``threshold``.

If no trained model is present we fall back to a transparent **energy + cadence
stub** so the interaction state machine is fully exercisable end-to-end (e.g. in
tests / on a dev box without the training toolchain). The stub is NOT a real
keyword spotter — it just reacts to a short burst of speech-level energy — and logs
that it is active so nobody ships it by accident.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_MODEL = Path(__file__).with_name("hey_reachy.onnx")
WINDOW_S = 2.0  # livekit-wakeword recommends ~2 s windows


@dataclass
class WakeConfig:
    threshold: float = 0.5
    window_s: float = WINDOW_S
    # Stub-only knobs (ignored when a real model is loaded):
    stub_rms_threshold: float = 0.04
    stub_min_active_frames: int = 6


class WakeWordDetector:
    """Streaming wake-word detector. Push 512-sample frames; poll :meth:`detected`.

    Maintains its own ~2 s rolling buffer so callers can feed the same
    512-sample/16 kHz frames used everywhere else in the pipeline.
    """

    def __init__(self, config: WakeConfig | None = None, model_path: str | Path | None = None) -> None:
        self.config = config or WakeConfig()
        self._buf = np.zeros(0, dtype=np.float32)
        self._win = int(self.config.window_s * SAMPLE_RATE)
        self._model = None
        self.backend = "stub"

        path = Path(model_path) if model_path else DEFAULT_MODEL
        if path.exists():
            try:
                from livekit.wakeword import WakeWordModel

                self._model = WakeWordModel(models=[str(path)])
                self.backend = "livekit-wakeword"
                logger.info("Wake word: livekit-wakeword classifier '%s'", path.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load livekit-wakeword model %s (%s); using stub.", path, exc)
        else:
            # Backbone may still import fine; we just have no trained keyword.
            logger.warning(
                "No trained wake-word model at %s; using energy/cadence STUB "
                "(not a real keyword spotter). Train hey_reachy.onnx to enable livekit-wakeword.",
                path,
            )
        self._stub_active_frames = 0

    def push(self, frame: np.ndarray) -> bool:
        """Append a frame; return True if the wake word fired on this frame."""
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        self._buf = np.concatenate([self._buf, frame])[-self._win :]
        if self._model is not None:
            scores = self._model.predict(self._buf)
            return any(s >= self.config.threshold for s in scores.values())
        return self._stub_step(frame)

    def _stub_step(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
        if rms >= self.config.stub_rms_threshold:
            self._stub_active_frames += 1
        else:
            self._stub_active_frames = max(0, self._stub_active_frames - 1)
        if self._stub_active_frames >= self.config.stub_min_active_frames:
            self._stub_active_frames = 0
            return True
        return False

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._stub_active_frames = 0
