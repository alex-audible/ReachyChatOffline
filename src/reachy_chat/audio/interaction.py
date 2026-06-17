"""Interaction state machine: VAD-gated + wake-word modes, with barge-in.

Ties the audio components (:mod:`vad`, :mod:`endpointer`, :mod:`wakeword`) into the
two interaction modes from docs/research/04-vad-turntaking.md §7.3:

* **Mode A — always-on (VAD-gated):** no wake word. Silero VAD -> Smart-Turn gates
  every utterance directly.
* **Mode B — wake word:** ``WakeWordDetector`` ("Hey Reachy") runs continuously and
  cheaply while IDLE; on a hit it opens one VAD -> Smart-Turn turn (with a short
  follow-up window for multi-turn exchanges). Same downstream path as Mode A.

States::

    IDLE ---(wake word | speech onset)---> LISTENING
    LISTENING ---(endpoint fired)--------> THINKING
    THINKING  ---(reply ready, speak)----> SPEAKING
    SPEAKING  ---(reply done)------------> IDLE / LISTENING (follow-up window)
    SPEAKING  ---(user speech = barge-in)-> LISTENING   (cancel in-flight reply)

Barge-in (docs §3): while SPEAKING, the machine runs VAD on the mic with a *raised*
start threshold ("partial ducking", §3.4) so only confident/loud speech interrupts
the robot's own TTS. On a confident detection it fires :meth:`Callbacks.on_barge_in`
— the canonical clean-cancel pattern (cancel the turn task, flush TTS playback,
clear the AEC reference, re-listen). The cancellation mechanics themselves live in
the STT/LLM/TTS layer; this module owns the *decision* and the state transition.

The machine is transport-agnostic and synchronous at the frame level: feed it
512-sample/16 kHz frames via :meth:`process_frame`. The pipeline drives side
effects (start STT, run LLM, play TTS, cancel) through the :class:`Callbacks`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .endpointer import Endpointer, EndpointEvent
from .vad import SileroVAD, VADConfig
from .wakeword import WakeConfig, WakeWordDetector

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Mode(Enum):
    ALWAYS_ON = "always_on"  # Mode A
    WAKE_WORD = "wake_word"  # Mode B


@dataclass
class Callbacks:
    """Side-effect hooks the host pipeline wires up. All optional / default no-op.

    Kept deliberately thin: the state machine decides *when*; the pipeline decides
    *how* (which STT/LLM/TTS to run, how to cancel them)."""

    on_wake: Callable[[], None] | None = None
    on_listen_start: Callable[[], None] | None = None  # begin capturing for STT
    on_endpoint: Callable[[EndpointEvent], None] | None = None  # turn ended -> run STT/LLM
    on_speak_start: Callable[[], None] | None = None  # TTS playback began
    on_barge_in: Callable[[], None] | None = None  # cancel in-flight reply + flush audio
    on_idle: Callable[[], None] | None = None

    def _call(self, name: str, *args) -> None:
        cb = getattr(self, name)
        if cb is not None:
            cb(*args)


@dataclass
class InteractionConfig:
    mode: Mode = Mode.ALWAYS_ON
    # Barge-in: while SPEAKING, raise the VAD start threshold so the robot's own
    # TTS (or quiet background talk) doesn't self-interrupt. §3.4 partial ducking.
    barge_in_start_threshold: float = 0.85
    barge_in_min_speech_ms: float = 160.0  # sustained speech before we call it barge-in
    # Wake-word mode: after a reply, stay open this long for a follow-up before
    # dropping back to needing the wake word again.
    follow_up_window_ms: float = 4000.0
    endpoint_smart_turn: bool = True


class InteractionMachine:
    """Drive IDLE -> LISTENING -> THINKING -> SPEAKING with barge-in.

    Parameters
    ----------
    config: mode + barge-in / follow-up tuning.
    callbacks: pipeline side-effect hooks.
    endpointer / wakeword: inject pre-built components (e.g. to share a warmed
        Smart-Turn model across runs); built lazily otherwise.
    """

    def __init__(
        self,
        config: InteractionConfig | None = None,
        callbacks: Callbacks | None = None,
        endpointer: Endpointer | None = None,
        wakeword: WakeWordDetector | None = None,
    ) -> None:
        self.config = config or InteractionConfig()
        self.cb = callbacks or Callbacks()
        self.state = State.IDLE
        self._elapsed_ms = 0.0
        self._speaking_since_ms = 0.0
        self._idle_since_ms = 0.0

        self.endpointer = endpointer or Endpointer(smart_turn=self.config.endpoint_smart_turn)
        if self.config.mode is Mode.WAKE_WORD:
            self.wakeword = wakeword or WakeWordDetector(WakeConfig())
        else:
            self.wakeword = None

        # Dedicated VAD for barge-in detection during SPEAKING (separate state from
        # the endpointer's VAD; uses a raised start threshold).
        self._barge_vad = SileroVAD(
            VADConfig(
                start_threshold=self.config.barge_in_start_threshold,
                min_speech_ms=self.config.barge_in_min_speech_ms,
            )
        )

    def warmup(self) -> None:
        self.endpointer.warmup()

    # -- frame pump ----------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> State:
        """Feed one 512-sample/16 kHz frame; returns the (possibly new) state.

        The frame is dispatched to whichever component is active for the current
        state (wake word / endpointer / barge-in VAD)."""
        from .vad import FRAME_MS

        self._elapsed_ms += FRAME_MS

        if self.state is State.IDLE:
            self._tick_idle(frame)
        elif self.state is State.LISTENING:
            self._tick_listening(frame)
        elif self.state is State.SPEAKING:
            self._tick_speaking(frame)
        # THINKING is driven externally (STT/LLM running); no audio gating here.
        return self.state

    def _tick_idle(self, frame: np.ndarray) -> None:
        if self.config.mode is Mode.WAKE_WORD:
            # Mode B: gate on the wake word, unless inside a follow-up window.
            in_follow_up = (
                self._idle_since_ms > 0
                and (self._elapsed_ms - self._idle_since_ms) < self.config.follow_up_window_ms
            )
            if in_follow_up:
                # Behave like Mode A for the follow-up window.
                if self.endpointer.vad.process_frame(frame) and self.endpointer.vad.in_speech:
                    self._to_listening()
                    self.endpointer.process_frame(frame)
                return
            if self.wakeword and self.wakeword.push(frame):
                self.cb._call("on_wake")
                self._to_listening()
        else:
            # Mode A: any confirmed speech onset opens a turn.
            evt = self.endpointer.vad.process_frame(frame)
            if evt and evt.kind == "start":
                self._to_listening()
                # Replay this frame into the endpointer's buffer.
                self.endpointer.process_frame(frame)

    def _tick_listening(self, frame: np.ndarray) -> None:
        evt = self.endpointer.process_frame(frame)
        if evt is not None:
            self.cb._call("on_endpoint", evt)
            self._to_thinking()

    def _tick_speaking(self, frame: np.ndarray) -> None:
        # Barge-in detection with raised threshold (partial ducking).
        evt = self._barge_vad.process_frame(frame)
        if evt is not None and evt.kind == "start":
            logger.info("Barge-in detected %.0f ms into TTS", self._elapsed_ms - self._speaking_since_ms)
            self.cb._call("on_barge_in")
            self._to_listening()
            self.endpointer.process_frame(frame)

    # -- transitions ---------------------------------------------------------
    def _to_listening(self) -> None:
        self.endpointer.reset()
        self.state = State.LISTENING
        self.cb._call("on_listen_start")

    def _to_thinking(self) -> None:
        self.state = State.THINKING

    # -- external drivers (called by the pipeline) ---------------------------
    def begin_speaking(self) -> None:
        """Pipeline calls this when TTS playback starts (THINKING -> SPEAKING)."""
        self._barge_vad.reset()
        self._speaking_since_ms = self._elapsed_ms
        self.state = State.SPEAKING
        self.cb._call("on_speak_start")

    def finish_speaking(self) -> None:
        """Pipeline calls this when TTS playback finishes (SPEAKING -> IDLE).

        In wake-word mode this opens the follow-up window; in always-on mode it
        returns to IDLE where any speech onset reopens a turn."""
        self.state = State.IDLE
        self._idle_since_ms = self._elapsed_ms
        self.endpointer.reset()
        self.cb._call("on_idle")

    def reset(self) -> None:
        self.state = State.IDLE
        self._idle_since_ms = 0.0
        self.endpointer.reset()
        self._barge_vad.reset()
        if self.wakeword is not None:
            self.wakeword.reset()
