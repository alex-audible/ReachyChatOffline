"""Semantic turn-end detection: Silero silence window + Pipecat Smart-Turn v3.

The job (docs/research/04-vad-turntaking.md §2): decide the user has *actually
finished* their turn, not just paused mid-sentence. A pure silence timeout either
cuts people off (short window) or feels sluggish (long window). Smart-Turn is an
**audio-based** semantic VAD (raw 16 kHz waveform -> P(turn complete)); pairing it
with a short silence window lets us commit fast on complete utterances and wait on
incomplete ones, while an absolute timeout guarantees the pipeline never hangs.

Decision logic — **dual-threshold "eagerness"** (docs/research/06-semantic-vad.md).
The win over a fixed silence window is *control logic*, not a new model: how
confident Smart-Turn is about completion sets how long we wait before committing.
Per frame, while VAD is in a speech segment and trailing silence has appeared:

1. Track trailing silence from the Silero hysteresis machine.
2. Start polling Smart-Turn as soon as ``eager_window_ms`` (default 100 ms) of
   silence exists — earlier than a fixed window, so confident commits happen ASAP.
3. Map P(complete) -> a required silence window:
     * P >= ``eager_threshold`` (0.85)  -> commit after only ``eager_window_ms``.
     * P >= ``turn_threshold``  (0.50)  -> commit after ``medium_window_ms`` (200).
     * P <  ``turn_threshold``          -> hold (user paused mid-thought); re-poll
       as silence grows; commit only at the ``max_silence_ms`` timeout.
   So a clearly-complete utterance fires at ~100-120 ms; an ambiguous one waits.
4. ``max_silence_ms`` (default 700 ms) is the absolute timeout safety net so an
   unsure Smart-Turn can never stall the turn.

Speculative-start hook: a confident-but-not-yet-committed poll (P crossing
``speculative_threshold``) can fire :attr:`Endpointer.on_speculative` so the
pipeline may *speculatively* start the LLM on the stabilized STT partial and
cancel-on-resume. That hook is optional and owned by the pipeline; the Endpointer
only signals "likely complete soon".

Smart-Turn v3.2 ships as an 8 MB int8 ONNX on HF ``pipecat-ai/smart-turn-v3``
(``smart-turn-v3.2-cpu.onnx``). Preprocessing follows the upstream repo exactly:
truncate/pad to the last 8 s, then ``WhisperFeatureExtractor(chunk_length=8)`` with
``do_normalize=True`` -> ``[1, 80, 800]`` log-mel; the ONNX output is already a
sigmoid probability. Measured ~10-25 ms/inference on M3 CPU.

If the model or ``transformers`` is unavailable, the endpointer degrades cleanly to
a **pure VAD silence-window** detector (``smart_turn=False``); the Smart-Turn hook
is preserved so it can be re-enabled without touching call sites.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .vad import FRAME_MS, SAMPLE_RATE, SileroVAD, VADConfig

logger = logging.getLogger(__name__)

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_WINDOW_S = 8  # model context window


@dataclass
class EndpointConfig:
    # Eagerness windows (silence required before committing, by confidence tier).
    # NOTE on safety: genuine mid-turn (inter-clause) pauses in our prompts run up to
    # ~190 ms, and Smart-Turn can briefly read the first clause as "complete" during
    # them. Two guards keep eager commits from cutting users off: (1) the eager window
    # sits *above* the longest mid-turn pause so a true turn-end — which keeps accruing
    # silence — is the only thing that reaches it; (2) ``eager_consecutive`` requires
    # sustained confidence (a short pause resolves back to speech first, resetting the
    # streak). 200 ms clears the measured pauses with margin and is still ~120-160 ms
    # faster than the old fixed 800-ms-timeout-prone path on confident utterances.
    eager_window_ms: float = 200.0  # very-confident commit
    medium_window_ms: float = 300.0  # ordinary-confident commit
    first_poll_ms: float = 80.0  # start polling Smart-Turn this early (for on_speculative)
    max_silence_ms: float = 700.0  # absolute timeout: fire regardless
    # Smart-Turn P(complete) thresholds:
    eager_threshold: float = 0.85  # >= this -> commit at eager_window_ms
    turn_threshold: float = 0.5  # >= this -> commit at medium_window_ms
    speculative_threshold: float = 0.75  # >= this -> fire on_speculative (pipeline hook)
    eager_consecutive: int = 2  # require this many consecutive confident polls to eager-commit
    smart_turn_repoll_ms: float = 60.0  # re-poll cadence while waiting
    # VAD thresholds passed through to SileroVAD (kept here so a caller configures
    # the whole endpointer from one object).
    vad: VADConfig = None  # type: ignore[assignment]

    # Back-compat: callers (and bench_endpoint.py) may pass silence_window_ms; treat
    # it as the medium window so existing call sites keep working.
    silence_window_ms: float | None = None

    def __post_init__(self) -> None:
        if self.silence_window_ms is not None:
            self.medium_window_ms = self.silence_window_ms
        if self.vad is None:
            # Large hangover is INTENTIONAL: the VAD stays "in speech" while the
            # endpointer itself tracks trailing silence (silence_run) and owns the
            # commit decision (eagerness windows + Smart-Turn, or max_silence timeout).
            # A small hangover makes the VAD emit an early "end" that resets the
            # endpointer's silence tracking → it never fires (verified: n=0).
            self.vad = VADConfig(hangover_ms=self.max_silence_ms)


@dataclass
class EndpointEvent:
    """Emitted when the endpointer commits a turn end."""

    reason: str  # "smart_turn_eager" | "smart_turn" | "timeout" | "vad_only"
    probability: float | None  # Smart-Turn P(complete), if it was consulted
    # Latency of the *decision step* that fired the endpoint (Smart-Turn inference
    # time, or ~0 for a pure-silence timeout). This is the compute cost on the hot
    # path, distinct from the silence window (which is a deliberate wait).
    decision_latency_ms: float
    silence_ms: float  # trailing silence when we fired


class SmartTurnModel:
    """Standalone Smart-Turn v3 ONNX runner (no Pipecat dependency).

    Mirrors pipecat-ai/smart-turn's ``predict_endpoint``: Whisper log-mel features
    over the last 8 s -> ONNX -> sigmoid probability of turn completion.
    """

    def __init__(self, repo: str = SMART_TURN_REPO, filename: str = SMART_TURN_FILE) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        path = hf_hub_download(repo, filename)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self._fe = WhisperFeatureExtractor(chunk_length=SMART_TURN_WINDOW_S)
        self.path = path

    def predict(self, audio: np.ndarray) -> float:
        """P(turn complete) in [0, 1] for a 16 kHz mono float32 waveform."""
        max_samples = SMART_TURN_WINDOW_S * SAMPLE_RATE
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (max_samples - len(audio), 0))  # pad at start
        inputs = self._fe(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=max_samples,
            truncation=True,
            do_normalize=True,
        )
        feat = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), axis=0)
        out = self._session.run(None, {"input_features": feat})
        return float(out[0][0].item())


class Endpointer:
    """VAD-silence + Smart-Turn turn-end detector, fed one 512-sample frame at a time.

    Usage::

        ep = Endpointer()
        ep.warmup()
        for frame in frames:               # 512 samples @16k each
            evt = ep.process_frame(frame)
            if evt:
                handle_turn_end(evt); ep.reset()

    Parameters
    ----------
    config: thresholds / windows.
    smart_turn: enable the Smart-Turn semantic stage. If ``True`` but the model
        can't load, falls back to pure VAD-silence and logs a warning.
    vad: an existing :class:`SileroVAD` to reuse (else one is created). Sharing the
        VAD with the barge-in detector avoids running Silero twice.

    Attributes
    ----------
    on_speculative: optional ``Callable[[float], None]`` the pipeline may set. Fired
        once per turn the first time Smart-Turn's P(complete) crosses
        ``speculative_threshold`` *before* the turn formally commits — the pipeline
        can speculatively start the LLM on the stabilized STT partial and cancel it
        if the user resumes. Cheap with a local LLM; saves ~150-250 ms when it holds.
    """

    def __init__(
        self,
        config: EndpointConfig | None = None,
        smart_turn: bool = True,
        vad: SileroVAD | None = None,
    ) -> None:
        self.config = config or EndpointConfig()
        self.vad = vad or SileroVAD(self.config.vad)
        self._owns_vad = vad is None
        self._audio: list[np.ndarray] = []  # rolling buffer for Smart-Turn context
        self._max_buf = SMART_TURN_WINDOW_S * SAMPLE_RATE
        self._buf_len = 0
        self._fired = False
        self._last_poll_silence_ms = 0.0
        self._spec_fired = False
        self._eager_streak = 0  # consecutive confident (>= eager_threshold) polls
        # Pipeline-owned speculative-start hook (see class docstring). Default no-op.
        self.on_speculative: "Callable[[float], None] | None" = None

        self.smart_turn: SmartTurnModel | None = None
        if smart_turn:
            try:
                self.smart_turn = SmartTurnModel()
                logger.info("Smart-Turn v3 loaded: %s", self.smart_turn.path)
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash the pipeline
                logger.warning(
                    "Smart-Turn unavailable (%s); falling back to VAD silence-window "
                    "endpointing. Re-enable by fixing the model/runtime; the hook is preserved.",
                    exc,
                )

    def warmup(self) -> None:
        """Prime the ONNX graph + feature extractor so the first real turn isn't slow."""
        if self.smart_turn is not None:
            self.smart_turn.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def _append(self, frame: np.ndarray) -> None:
        self._audio.append(frame)
        self._buf_len += len(frame)
        # Trim from the front to keep <= 8 s (Smart-Turn only sees the tail anyway).
        while self._buf_len > self._max_buf and len(self._audio) > 1:
            self._buf_len -= len(self._audio.pop(0))

    def process_frame(self, frame: np.ndarray) -> EndpointEvent | None:
        """Advance VAD, buffer audio, and decide whether the turn has ended.

        Returns an :class:`EndpointEvent` exactly once per turn (then no-ops until
        :meth:`reset`).
        """
        if self._fired:
            return None
        self._append(frame)
        self.vad.process_frame(frame)

        if not self.vad.in_speech:
            return None  # haven't started (or already ended via hangover)

        cfg = self.config
        silence_ms = self.vad.silence_ms
        if silence_ms <= 0.0:
            # Speech resumed (or never paused): reset poll cadence + eager streak so a
            # brief inter-clause pause can't accumulate toward an eager commit. Also
            # re-arm the speculative hook so it can re-signal at the *next* pause — if
            # the pipeline cancelled a speculative LLM start on resume, it can start a
            # fresh one when confidence returns at the true turn-end.
            self._last_poll_silence_ms = 0.0
            self._eager_streak = 0
            self._spec_fired = False
            return None

        # Absolute timeout safety net — fire regardless of Smart-Turn.
        if silence_ms >= cfg.max_silence_ms:
            return self._fire("timeout", None, 0.0, silence_ms)

        if self.smart_turn is None:
            # Pure VAD-silence endpointer: medium window reached -> end of turn.
            if silence_ms >= cfg.medium_window_ms:
                return self._fire("vad_only", None, 0.0, silence_ms)
            return None

        # Poll Smart-Turn once we have a minimal silence cushion (first_poll_ms),
        # then on a tight cadence. Early polling drives the speculative hook and lets
        # a confident commit fire as soon as the eager window is reached.
        if silence_ms < cfg.first_poll_ms:
            return None
        if silence_ms - self._last_poll_silence_ms < cfg.smart_turn_repoll_ms:
            return None
        self._last_poll_silence_ms = silence_ms

        audio = np.concatenate(self._audio) if self._audio else np.zeros(1, dtype=np.float32)
        t0 = time.perf_counter()
        prob = self.smart_turn.predict(audio)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._eager_streak = self._eager_streak + 1 if prob >= cfg.eager_threshold else 0

        # Speculative-start signal (pipeline hook): likely-complete but maybe not yet
        # past the required window — let the pipeline pre-warm the LLM.
        if (
            not self._spec_fired
            and self.on_speculative is not None
            and prob >= cfg.speculative_threshold
        ):
            self._spec_fired = True
            try:
                self.on_speculative(prob)
            except Exception:  # noqa: BLE001 - never let a hook break the turn
                logger.exception("on_speculative hook raised")

        # Dual-threshold eagerness: higher confidence -> shorter required window. The
        # eager tier also requires sustained confidence (``eager_consecutive`` polls)
        # so a single confident reading at a short inter-clause pause can't cut the
        # user off; a true turn-end stays confident as silence grows.
        if (
            prob >= cfg.eager_threshold
            and silence_ms >= cfg.eager_window_ms
            and self._eager_streak >= cfg.eager_consecutive
        ):
            return self._fire("smart_turn_eager", prob, latency_ms, silence_ms)
        if prob >= cfg.turn_threshold and silence_ms >= cfg.medium_window_ms:
            return self._fire("smart_turn", prob, latency_ms, silence_ms)
        return None

    def _fire(
        self, reason: str, prob: float | None, latency_ms: float, silence_ms: float
    ) -> EndpointEvent:
        self._fired = True
        return EndpointEvent(reason, prob, latency_ms, silence_ms)

    def reset(self) -> None:
        """Reset for the next turn (also resets the shared VAD)."""
        self.vad.reset()
        self._audio.clear()
        self._buf_len = 0
        self._fired = False
        self._spec_fired = False
        self._eager_streak = 0
        self._last_poll_silence_ms = 0.0
