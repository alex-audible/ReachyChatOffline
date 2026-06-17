"""Audio front-end: VAD, semantic endpointing, wake word, and turn/barge-in state.

Components (see docs/research/04-vad-turntaking.md and README.md):

* :mod:`reachy_chat.audio.vad` — Silero VAD v6 streaming wrapper (512-sample frames).
* :mod:`reachy_chat.audio.endpointer` — Silero silence window + Smart-Turn v3 turn-end.
* :mod:`reachy_chat.audio.wakeword` — "Hey Reachy" via livekit-wakeword (+ stub).
* :mod:`reachy_chat.audio.interaction` — IDLE/LISTENING/THINKING/SPEAKING machine
  supporting always-on and wake-word modes, with barge-in.

All components consume 16 kHz mono float32 audio in fixed 512-sample frames so they
can be driven from the live mic loop or the offline benchmark feeder alike.
"""

from __future__ import annotations

from .endpointer import EndpointConfig, Endpointer, EndpointEvent, SmartTurnModel
from .interaction import (
    Callbacks,
    InteractionConfig,
    InteractionMachine,
    Mode,
    State,
)
from .vad import FRAME_SAMPLES, SAMPLE_RATE, SileroVAD, SpeechState, VADConfig, VADEvent, iter_frames
from .wakeword import WakeConfig, WakeWordDetector

__all__ = [
    "SileroVAD",
    "VADConfig",
    "VADEvent",
    "SpeechState",
    "FRAME_SAMPLES",
    "SAMPLE_RATE",
    "iter_frames",
    "Endpointer",
    "EndpointConfig",
    "EndpointEvent",
    "SmartTurnModel",
    "WakeWordDetector",
    "WakeConfig",
    "InteractionMachine",
    "InteractionConfig",
    "State",
    "Mode",
    "Callbacks",
]
