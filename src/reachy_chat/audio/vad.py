"""Silero VAD v6 streaming wrapper.

Silero VAD (``silero-vad`` >= 6.x, ONNX/CPU) is the first gate of the voice
pipeline: it answers "is someone speaking right now?" for every audio frame and
feeds both the endpointer (turn-end) and the barge-in detector.

Hard constraint (matches docs/research/04-vad-turntaking.md §1.1 and the project's
realtime-whisper skill note): **Silero only accepts fixed window sizes.** At
16 kHz that is exactly **512 samples** (32 ms), or 1024/1536; at 8 kHz it is 256.
Feeding any other size raises ``Provided number of samples ...``. Mic audio must
therefore be ring-buffered into exactly 512-sample blocks before reaching the
model. The model is stateful, so :meth:`SileroVAD.reset` must be called between
utterances.

This wrapper exposes two layers:

* :meth:`SileroVAD.probability` — raw speech probability for one 512-sample frame
  (~0.13 ms/frame on M3, measured).
* :meth:`SileroVAD.process_frame` — a small hysteresis state machine that turns
  the per-frame probability into :class:`SpeechState` transitions (speech start /
  speech end) using a start threshold, a (lower) end threshold and a configurable
  hangover, so brief dips below threshold mid-word do not prematurely end speech.

The wrapper deliberately does *not* own the microphone — it consumes frames pushed
to it, which keeps it usable from both the live capture loop and the offline
benchmark feeder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Silero's only legal 16 kHz window. Do not change without also changing the ring
# buffer; the model will reject anything else.
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000.0  # 32.0 ms


class SpeechState(Enum):
    """Coarse speech state derived from the VAD hysteresis machine."""

    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VADEvent:
    """Transition emitted by :meth:`SileroVAD.process_frame` (or ``None`` if no
    transition occurred on this frame)."""

    kind: str  # "start" | "end"
    timestamp: float  # seconds since the VAD was (re)started
    probability: float


@dataclass
class VADConfig:
    """Tunable thresholds for the hysteresis machine.

    ``start_threshold`` > ``end_threshold`` gives Schmitt-trigger behaviour: it
    takes a confident frame to *begin* speech but a sustained quiet stretch to
    *end* it. ``min_speech_ms`` rejects single-frame blips; ``hangover_ms`` is how
    long the probability must stay below ``end_threshold`` before we declare the
    speech segment over (this is the silence window the endpointer builds on).
    """

    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_speech_ms: float = 64.0  # >= 2 frames before we trust a speech onset
    hangover_ms: float = 200.0  # quiet must persist this long to end speech


@dataclass
class _RingState:
    in_speech: bool = False
    speech_run_ms: float = 0.0  # contiguous speech accumulated (for min_speech)
    silence_run_ms: float = 0.0  # contiguous silence accumulated (for hangover)
    elapsed_ms: float = 0.0
    pending_start_ms: float | None = None  # onset awaiting min_speech confirmation
    history: list[float] = field(default_factory=list)


class SileroVAD:
    """Streaming Silero VAD v6 wrapper (ONNX/CPU).

    Parameters
    ----------
    config:
        Threshold / hangover configuration. Defaults are sensible for clean
        16 kHz mic audio; raise ``start_threshold`` toward 0.8 while TTS is
        playing (partial ducking, see :mod:`reachy_chat.audio.interaction`).
    onnx:
        Use the ONNX runtime path (recommended; CPU, no MPS needed).
    """

    def __init__(self, config: VADConfig | None = None, onnx: bool = True) -> None:
        from silero_vad import load_silero_vad  # local import: optional dep

        self.config = config or VADConfig()
        self._model = load_silero_vad(onnx=onnx)
        self._onnx = onnx
        self._st = _RingState()
        self.state = SpeechState.SILENCE

    # -- low level -----------------------------------------------------------
    def probability(self, frame: np.ndarray) -> float:
        """Speech probability in ``[0, 1]`` for exactly one 512-sample frame.

        ``frame`` must be float32 mono in ``[-1, 1]`` with length
        :data:`FRAME_SAMPLES`. Raises ``ValueError`` otherwise (mirroring Silero's
        own strictness so bugs surface immediately rather than as garbage scores).
        """
        if frame.ndim != 1 or frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(
                f"Silero VAD requires exactly {FRAME_SAMPLES} samples @ {SAMPLE_RATE} Hz, "
                f"got shape {frame.shape}. Ring-buffer mic audio into {FRAME_SAMPLES}-sample blocks."
            )
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        import torch

        with torch.no_grad():
            return float(self._model(torch.from_numpy(frame).unsqueeze(0), SAMPLE_RATE).item())

    # -- hysteresis machine --------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> VADEvent | None:
        """Feed one 512-sample frame; advance the speech/silence state machine.

        Returns a :class:`VADEvent` on a state transition (``"start"`` /
        ``"end"``), else ``None``. ``self.state`` reflects the current state and
        :attr:`silence_ms` exposes the trailing silence the endpointer reads.
        """
        p = self.probability(frame)
        st = self._st
        st.elapsed_ms += FRAME_MS
        cfg = self.config
        event: VADEvent | None = None

        if not st.in_speech:
            if p >= cfg.start_threshold:
                # Candidate onset. Confirm only after min_speech_ms of speech so a
                # lone noisy frame cannot open a turn.
                if st.pending_start_ms is None:
                    st.pending_start_ms = st.elapsed_ms - FRAME_MS
                st.speech_run_ms += FRAME_MS
                if st.speech_run_ms >= cfg.min_speech_ms:
                    st.in_speech = True
                    self.state = SpeechState.SPEECH
                    st.silence_run_ms = 0.0
                    onset = st.pending_start_ms if st.pending_start_ms is not None else st.elapsed_ms
                    st.pending_start_ms = None
                    event = VADEvent("start", onset / 1000.0, p)
            else:
                st.speech_run_ms = 0.0
                st.pending_start_ms = None
        else:
            if p < cfg.end_threshold:
                st.silence_run_ms += FRAME_MS
                if st.silence_run_ms >= cfg.hangover_ms:
                    st.in_speech = False
                    self.state = SpeechState.SILENCE
                    # End timestamp is when silence *began*, not when hangover elapsed.
                    end_ms = st.elapsed_ms - st.silence_run_ms
                    st.silence_run_ms = 0.0
                    st.speech_run_ms = 0.0
                    event = VADEvent("end", end_ms / 1000.0, p)
            else:
                st.silence_run_ms = 0.0
                st.speech_run_ms += FRAME_MS

        st.history.append(p)
        return event

    @property
    def silence_ms(self) -> float:
        """Trailing contiguous silence (ms) while in a speech segment.

        0 while clearly speaking; grows as the user pauses. The endpointer uses
        this as its silence window. Meaningless (returns 0) when not in speech.
        """
        return self._st.silence_run_ms if self._st.in_speech else 0.0

    @property
    def in_speech(self) -> bool:
        return self._st.in_speech

    @property
    def elapsed_ms(self) -> float:
        return self._st.elapsed_ms

    def reset(self) -> None:
        """Reset both Silero's internal RNN state and our hysteresis machine.

        Call between utterances/turns (Silero is stateful — not resetting leaks
        the previous turn's context into the next one).
        """
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
        self._st = _RingState()
        self.state = SpeechState.SILENCE


def iter_frames(audio: np.ndarray, frame_samples: int = FRAME_SAMPLES) -> "list[np.ndarray]":
    """Split ``audio`` into non-overlapping ``frame_samples`` blocks, zero-padding
    the tail so the final block is full length (Silero rejects short frames)."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    n = len(audio)
    n_full = (n + frame_samples - 1) // frame_samples
    out = []
    for i in range(n_full):
        block = audio[i * frame_samples : (i + 1) * frame_samples]
        if len(block) < frame_samples:
            block = np.pad(block, (0, frame_samples - len(block)))
        out.append(block)
    return out
