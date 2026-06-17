"""Embodiment / presence layer — make Reachy Mini feel *alive* in conversation.

``RobotPresence`` ties the conversation state machine
(:class:`~reachy_chat.audio.interaction.State`: ``idle`` / ``listening`` /
``thinking`` / ``speaking``) to subtle, lifelike head + antenna motion. It is a
thin orchestration layer on top of the already-built, *decoupled* motion stack:

* :class:`~reachy_chat.robot.motion_queue.MotionQueue` — smooth ``goto`` /
  recorded-move execution on its own worker thread.
* :class:`~reachy_chat.robot.look_at_speaker.LookAtSpeaker` — DoA-driven (or
  manual) "turn toward the speaker" controller, also on its own thread.

Design rule (critical): **all motion is decoupled from the voice latency path.**
``on_state`` only flips an internal flag and returns immediately; an independent
background loop (:meth:`_run`) decides *what* to do and *when*, enqueuing cues
onto the motion queue. Nothing here ever blocks the caller, and if the daemon is
unreachable everything degrades to a logged no-op — the caller never crashes.

Per-state behaviour (small amplitudes, ``minjerk`` smoothing, within the motion
limits: head pitch/roll ±40°, yaw ±180°, body ±160°):

============ ===========================================================
State        Embodiment
============ ===========================================================
``listening`` Orient toward the speaker (DoA azimuth if available, else
              center), settle into a small attentive head tilt, antennas
              perked up. Look-at-speaker tracking is enabled.
``thinking``  A brief "thinking beat": glance slightly up & away, antennas
              relaxed. Tracking paused so the gaze stays pensive.
``speaking``  Re-center on the listener, then small periodic nods with
              gentle antenna motion for the duration of the turn — the
              robot looks engaged while it talks.
``idle``      Return to a relaxed neutral pose, then occasional subtle
              idle micro-motions (tiny look-arounds / antenna twitches)
              so the robot never looks frozen/dead.
============ ===========================================================
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass

from .client import HeadPose, ReachyClient, ReachyClientError
from .look_at_speaker import LookAtSpeaker, doa_angle_to_azimuth_deg
from .motion_queue import GotoCommand, MotionQueue

logger = logging.getLogger(__name__)

#: The conversation states this layer understands (strings, matching
#: :class:`reachy_chat.audio.interaction.State` values).
VALID_STATES = ("idle", "listening", "thinking", "speaking")


# --------------------------------------------------------------------------- #
# Antenna poses (radians on the wire — antennas are NOT degree-converted by the
# client). A small perk vs. a relaxed droop reads as "attentive" vs. "pensive".
# --------------------------------------------------------------------------- #
ANTENNAS_NEUTRAL = (0.0, 0.0)
ANTENNAS_PERKED = (0.35, 0.35)  # both up — alert / attentive
ANTENNAS_RELAXED = (-0.25, -0.25)  # both down — relaxed / thinking


@dataclass
class PresenceConfig:
    """Tuning knobs for :class:`RobotPresence` (degrees unless noted).

    Amplitudes are deliberately small — the goal is *tasteful and lifelike*,
    not frantic. All head values stay well inside the motion limits.
    """

    tick_hz: float = 10.0  #: background-loop rate

    # --- listening ---
    listen_tilt_deg: float = 7.0  #: attentive head roll
    listen_pitch_deg: float = -3.0  #: tiny chin-up "I'm with you"
    listen_settle_s: float = 0.6  #: goto duration when settling in

    # --- thinking ---
    think_look_up_deg: float = 14.0  #: glance up (negative pitch = up)
    think_look_away_deg: float = 16.0  #: glance to the side (yaw)
    think_settle_s: float = 0.7

    # --- speaking ---
    speak_nod_down_deg: float = 9.0  #: nod amplitude (down)
    speak_nod_up_deg: float = 3.0  #: slight rebound up
    speak_nod_duration_s: float = 0.32  #: per half-nod goto duration
    speak_nod_period_s: float = 2.4  #: seconds between nods while talking
    speak_antenna_wiggle: float = 0.18  #: antenna delta on each nod (rad)

    # --- idle ---
    idle_micro_period_s: float = 6.0  #: avg seconds between idle micro-motions
    idle_micro_yaw_deg: float = 12.0  #: max look-around yaw
    idle_micro_pitch_deg: float = 6.0  #: max look-around pitch
    idle_micro_duration_s: float = 1.2

    # --- shared ---
    return_neutral_s: float = 0.8  #: goto duration when returning to neutral
    #: how strongly DoA azimuth is trusted in listening (0=center, 1=full).
    doa_trust: float = 1.0


@dataclass
class _StateView:
    """Mutable snapshot of the requested state, guarded by a lock."""

    state: str = "idle"
    seq: int = 0  #: bumped on every state change so the loop can detect entry


class RobotPresence:
    """Conversation-state-driven embodiment controller (background loop).

    Wires conversation states to subtle head / antenna cues via the decoupled
    :class:`MotionQueue` + :class:`LookAtSpeaker`. Connect, :meth:`start`, then
    call :meth:`on_state` from the interaction driver; call :meth:`look_at` to
    point at a sound source (DoA when present, manual azimuth otherwise).

    Args:
        client: A :class:`ReachyClient` (or ``None`` to build one for
            ``base_url``).
        base_url: Daemon URL used when ``client`` is ``None``.
        config: Optional :class:`PresenceConfig` tuning.
        queue: Optional externally-owned :class:`MotionQueue` to reuse. If
            ``None``, one is created and owned (started/stopped) by this object.
        look_at: Optional externally-owned :class:`LookAtSpeaker` to reuse.
            If ``None``, one is created and owned by this object.
    """

    def __init__(
        self,
        client: ReachyClient | None = None,
        *,
        base_url: str = "http://localhost:8000",
        config: PresenceConfig | None = None,
        queue: MotionQueue | None = None,
        look_at: LookAtSpeaker | None = None,
    ) -> None:
        self._client = client or ReachyClient(base_url=base_url)
        self._cfg = config or PresenceConfig()

        # Ownership: we only start/stop helpers we created ourselves, so a host
        # app can share its own queue/look_at without us tearing them down.
        self._owns_queue = queue is None
        self._owns_look_at = look_at is None
        self._queue = queue or MotionQueue(self._client)
        self._look_at = look_at or LookAtSpeaker(self._client)

        self._view = _StateView()
        self._view_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Per-loop scheduling timers (loop-thread-local; no lock needed).
        self._next_nod_at = 0.0
        self._next_idle_at = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the presence loop (and any helpers we own). Idempotent.

        Robust to an unavailable daemon: helpers still start, the loop simply
        no-ops (logging) until the daemon comes back.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        if not self._client.ping():
            logger.warning(
                "RobotPresence: daemon not reachable at %s — running in no-op "
                "mode (motion will resume if it comes back).",
                self._client.base_url,
            )

        if self._owns_queue:
            self._queue.start()
        if self._owns_look_at:
            self._look_at.start()
            # Tracking off by default; enabled when we enter LISTENING.
            self._look_at.set_enabled(False)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="reachy-presence", daemon=True
        )
        self._thread.start()
        logger.info("RobotPresence started")

    def stop(self, timeout: float = 2.0, *, return_to_neutral: bool = True) -> None:
        """Stop the loop and the helpers we own. Best-effort, never raises."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

        if return_to_neutral:
            self._safe(lambda: self._goto_neutral(self._cfg.return_neutral_s))

        if self._owns_look_at:
            self._safe(self._look_at.stop)
        if self._owns_queue:
            self._safe(self._queue.stop)
        logger.info("RobotPresence stopped")

    def __enter__(self) -> "RobotPresence":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Public API (called by the interaction driver / host app)
    # ------------------------------------------------------------------ #
    def on_state(self, new_state: str) -> None:
        """Notify presence of a conversation-state change. Non-blocking.

        Accepts ``"idle"`` / ``"listening"`` / ``"thinking"`` / ``"speaking"``
        (also tolerates an Enum-like object with a ``.value``). Unknown states
        are ignored with a warning. This only flips a flag — the background loop
        does the work, so the caller never waits on motion.
        """
        state = getattr(new_state, "value", new_state)
        state = str(state).lower()
        if state not in VALID_STATES:
            logger.warning("RobotPresence.on_state: ignoring unknown state %r", state)
            return
        with self._view_lock:
            if state == self._view.state:
                return
            self._view.state = state
            self._view.seq += 1
        logger.debug("RobotPresence: state -> %s", state)

    def look_at(self, azimuth_deg: float) -> None:
        """Point the robot at a sound source (signed azimuth: 0=front, +=left).

        Non-blocking. Delegates to :class:`LookAtSpeaker.look_at_azimuth`, which
        splits the angle across body + head and issues a single smooth ``goto``.
        Wire real DoA here on hardware; on the bare sim DoA is ``null`` so pass a
        manual azimuth.
        """
        self._safe(lambda: self._look_at.look_at_azimuth(float(azimuth_deg)))

    @property
    def state(self) -> str:
        """The current conversation state as understood by presence."""
        with self._view_lock:
            return self._view.state

    # ------------------------------------------------------------------ #
    # Background loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        period = 1.0 / max(1.0, self._cfg.tick_hz)
        last_seq = -1
        cur_state = "idle"
        while not self._stop_event.is_set():
            with self._view_lock:
                cur_state = self._view.state
                seq = self._view.seq

            entered = seq != last_seq
            last_seq = seq

            try:
                if entered:
                    self._on_enter(cur_state)
                else:
                    self._on_tick(cur_state)
            except ReachyClientError as exc:
                logger.debug("presence cue failed (daemon issue): %s", exc)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Unexpected error in presence loop")

            self._stop_event.wait(period)

    # ------------------------------------------------------------------ #
    # State entry — fire the one-shot cue for a new state
    # ------------------------------------------------------------------ #
    def _on_enter(self, state: str) -> None:
        now = time.monotonic()
        if state == "listening":
            self._enter_listening()
        elif state == "thinking":
            self._enter_thinking()
        elif state == "speaking":
            self._enter_speaking()
            self._next_nod_at = now + self._cfg.speak_nod_period_s
        elif state == "idle":
            self._enter_idle()
            self._next_idle_at = now + self._idle_jitter()

    # ------------------------------------------------------------------ #
    # State tick — recurring behaviour while we stay in a state
    # ------------------------------------------------------------------ #
    def _on_tick(self, state: str) -> None:
        now = time.monotonic()
        if state == "speaking":
            if now >= self._next_nod_at and self._queue.pending == 0:
                self._speak_nod()
                self._next_nod_at = now + self._cfg.speak_nod_period_s
        elif state == "idle":
            if now >= self._next_idle_at and self._queue.pending == 0:
                self._idle_micro_motion()
                self._next_idle_at = now + self._idle_jitter()
        # listening / thinking are settled one-shot poses — nothing recurring,
        # except that listening keeps DoA tracking live (handled by LookAtSpeaker).

    # ------------------------------------------------------------------ #
    # Per-state cues
    # ------------------------------------------------------------------ #
    def _enter_listening(self) -> None:
        """Orient toward the speaker + attentive tilt + perked antennas."""
        cfg = self._cfg
        azimuth = self._current_azimuth_deg()

        # Enable DoA tracking so the head keeps following the speaker on
        # hardware. On the bare sim DoA is null, so tracking is a quiet no-op.
        self._safe(lambda: self._look_at.set_enabled(True))

        # A slight head tilt (roll) + tiny chin-up reads as "attentive". If we
        # have a usable azimuth, fold it into the yaw of this same goto so the
        # orient + tilt happen in one smooth motion.
        head = HeadPose(
            roll=cfg.listen_tilt_deg,
            pitch=cfg.listen_pitch_deg,
            yaw=_clamp_yaw(azimuth) if azimuth is not None else 0.0,
        )
        self._queue.enqueue(
            GotoCommand(
                head_pose=head,
                antennas=ANTENNAS_PERKED,
                duration=cfg.listen_settle_s,
            ),
            interrupt=True,
        )

    def _enter_thinking(self) -> None:
        """A 'thinking beat': glance up & away, antennas relax, pause tracking."""
        cfg = self._cfg
        self._safe(lambda: self._look_at.set_enabled(False))
        # Look up (negative pitch) and to a random side — pensive, not robotic.
        side = random.choice((-1.0, 1.0))
        head = HeadPose(
            pitch=-abs(cfg.think_look_up_deg),
            yaw=side * cfg.think_look_away_deg,
            roll=0.0,
        )
        self._queue.enqueue(
            GotoCommand(
                head_pose=head,
                antennas=ANTENNAS_RELAXED,
                duration=cfg.think_settle_s,
            ),
            interrupt=True,
        )

    def _enter_speaking(self) -> None:
        """Re-center on the listener with neutral, ready antennas."""
        cfg = self._cfg
        self._safe(lambda: self._look_at.set_enabled(False))
        self._queue.enqueue(
            GotoCommand(
                head_pose=HeadPose(),
                antennas=ANTENNAS_NEUTRAL,
                duration=cfg.return_neutral_s,
            ),
            interrupt=True,
        )

    def _enter_idle(self) -> None:
        """Relax to neutral; idle micro-motions follow via the tick."""
        self._safe(lambda: self._look_at.set_enabled(False))
        self._goto_neutral(self._cfg.return_neutral_s)

    def _speak_nod(self) -> None:
        """One gentle nod + a small antenna wiggle — looks engaged while talking."""
        cfg = self._cfg
        wig = cfg.speak_antenna_wiggle
        self._queue.enqueue(
            GotoCommand(
                head_pose=HeadPose(pitch=cfg.speak_nod_down_deg),
                antennas=(wig, -wig),
                duration=cfg.speak_nod_duration_s,
            )
        )
        self._queue.enqueue(
            GotoCommand(
                head_pose=HeadPose(pitch=-cfg.speak_nod_up_deg),
                antennas=ANTENNAS_NEUTRAL,
                duration=cfg.speak_nod_duration_s,
            )
        )
        self._queue.enqueue(
            GotoCommand(head_pose=HeadPose(), duration=cfg.speak_nod_duration_s)
        )

    def _idle_micro_motion(self) -> None:
        """A tiny, random look-around + antenna twitch so idle isn't frozen."""
        cfg = self._cfg
        yaw = random.uniform(-cfg.idle_micro_yaw_deg, cfg.idle_micro_yaw_deg)
        pitch = random.uniform(-cfg.idle_micro_pitch_deg, cfg.idle_micro_pitch_deg)
        twitch = random.uniform(-0.15, 0.15)
        self._queue.enqueue(
            GotoCommand(
                head_pose=HeadPose(yaw=yaw, pitch=pitch),
                antennas=(twitch, twitch),
                duration=cfg.idle_micro_duration_s,
            )
        )
        # Drift back toward neutral so successive micro-motions don't accumulate.
        self._queue.enqueue(
            GotoCommand(
                head_pose=HeadPose(),
                antennas=ANTENNAS_NEUTRAL,
                duration=cfg.idle_micro_duration_s,
            )
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _goto_neutral(self, duration: float) -> None:
        self._queue.enqueue(
            GotoCommand(
                head_pose=HeadPose(),
                body_yaw_deg=0.0,
                antennas=ANTENNAS_NEUTRAL,
                duration=duration,
            ),
            interrupt=True,
        )

    def _current_azimuth_deg(self) -> float | None:
        """Best-effort speaker azimuth from DoA, or ``None`` if unavailable.

        The bare simulator returns ``null`` for DoA, in which case we fall back
        to center (``None`` -> caller uses 0). Speech gating is intentionally
        *not* applied here: entering LISTENING is itself the signal to orient.
        """
        try:
            doa = self._client.get_doa()
        except ReachyClientError:
            return None
        if doa is None:
            return None
        az = doa_angle_to_azimuth_deg(doa.angle_rad) * self._cfg.doa_trust
        return az

    def _idle_jitter(self) -> float:
        """Randomised interval before the next idle micro-motion (±40%)."""
        base = self._cfg.idle_micro_period_s
        return random.uniform(base * 0.6, base * 1.4)

    @staticmethod
    def _safe(fn) -> None:
        """Run ``fn`` swallowing client/daemon errors so callers never crash."""
        try:
            fn()
        except ReachyClientError as exc:
            logger.debug("presence helper failed (daemon issue): %s", exc)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected error in presence helper")


def _clamp_yaw(value: float) -> float:
    """Keep the LISTENING orient gentle: cap the in-pose yaw to a comfy range."""
    return max(-45.0, min(45.0, value))
