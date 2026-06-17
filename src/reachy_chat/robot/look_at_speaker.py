"""DoA-driven "look at the speaker" controller.

Polls the daemon's direction-of-arrival (``/api/state/doa``) and gently turns the
robot toward whoever is speaking. It runs in its own thread and is *fully
decoupled* from the voice pipeline: head motion is allowed to trail the audio,
and nothing here is on the speech latency path.

DoA angle convention (confirmed from ``reachy_mini/media/audio_doa.py``)::

    angle is in RADIANS:  0 = left,  pi/2 = front/back,  pi = right

That is an unusual frame for a "turn toward the source" controller, which wants a
signed azimuth where 0 = front, positive = left (robot's left), negative = right.
We therefore remap the raw DoA angle into a signed azimuth before driving the
joints. The mapping is centralised in :func:`doa_angle_to_azimuth_deg` and is
easy to recalibrate against real hardware if the front/back ambiguity needs
resolving with another sensor.

Smoothing strategy (to avoid jitter):

* **dead-zone** -- ignore azimuth changes smaller than ``dead_zone_deg``.
* **exponential smoothing** -- blend new targets with the current one.
* **rate limiting** -- cap how fast the commanded azimuth may change per tick.
* **speech gating** -- only react when ``speech_detected`` is true (optional).
* azimuth is split across body yaw (coarse) and head yaw (fine) so the body
  carries large turns and the head handles the remainder, which looks natural.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

from .client import (
    BODY_YAW_LIMIT_DEG,
    HEAD_YAW_LIMIT_DEG,
    HeadPose,
    ReachyClient,
    ReachyClientError,
)

logger = logging.getLogger(__name__)


def doa_angle_to_azimuth_deg(angle_rad: float) -> float:
    """Map a raw DoA angle (radians) to a signed azimuth in degrees.

    DoA frame: ``0`` = left, ``pi/2`` = front/back, ``pi`` = right.
    Target frame: ``0`` = front, ``+`` = left, ``-`` = right.

    We treat ``pi/2`` as the front reference and measure offset from it, so:

    * ``angle = pi/2``  -> azimuth ``0``   (front)
    * ``angle = 0``     -> azimuth ``+90`` (full left)
    * ``angle = pi``    -> azimuth ``-90`` (full right)

    The front/back ambiguity inherent to a single linear array is resolved in
    favour of *front* (we never turn the robot around to face behind it from
    audio alone).
    """
    # offset from the front reference (pi/2); left side (< pi/2) -> positive.
    azimuth_rad = (math.pi / 2.0) - angle_rad
    return math.degrees(azimuth_rad)


@dataclass
class LookAtConfig:
    """Tuning knobs for :class:`LookAtSpeaker`."""

    poll_hz: float = 20.0  #: how often to poll DoA / issue targets
    dead_zone_deg: float = 6.0  #: ignore changes smaller than this
    smoothing: float = 0.25  #: EMA factor (0=frozen, 1=instant)
    max_rate_deg_per_s: float = 90.0  #: max commanded azimuth slew rate
    require_speech: bool = True  #: only react when speech is detected
    #: fraction of azimuth handled by the head (rest goes to the body).
    head_share: float = 0.5
    #: clamp the head's portion so it never strains past a comfortable range.
    max_head_yaw_deg: float = 40.0


class LookAtSpeaker:
    """Background controller that turns the robot toward the active speaker.

    Args:
        client: A connected :class:`ReachyClient`.
        config: Optional :class:`LookAtConfig` tuning.
    """

    def __init__(self, client: ReachyClient, config: LookAtConfig | None = None) -> None:
        self._client = client
        self._cfg = config or LookAtConfig()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._enabled = threading.Event()
        self._lock = threading.Lock()
        # Current smoothed/commanded azimuth (deg, front=0, left=+).
        self._cmd_azimuth_deg = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the polling thread (idempotent). Tracking is enabled by default."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._enabled.set()
            self._thread = threading.Thread(
                target=self._run, name="reachy-look-at-speaker", daemon=True
            )
            self._thread.start()
            logger.info("LookAtSpeaker started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the polling thread and wait for it to exit."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("LookAtSpeaker stopped")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable tracking without tearing down the thread."""
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()
        logger.info("LookAtSpeaker tracking %s", "enabled" if enabled else "disabled")

    @property
    def enabled(self) -> bool:
        """Whether tracking is currently active."""
        return self._enabled.is_set()

    def __enter__(self) -> "LookAtSpeaker":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Public one-shot helper
    # ------------------------------------------------------------------ #
    def look_at_azimuth(self, azimuth_deg: float) -> None:
        """Immediately aim at a given azimuth (front=0, left=+), bypassing DoA.

        Useful for the ``look_at`` tool. Splits the azimuth across body + head
        and issues a single smooth ``goto`` (so it works even mid-conversation;
        ``set_target`` would be ignored while another move runs).
        """
        body_deg, head_deg = self._split_azimuth(azimuth_deg)
        try:
            self._client.goto(
                head_pose=HeadPose(yaw=head_deg),
                body_yaw_deg=body_deg,
                duration=0.6,
            )
            with self._lock:
                self._cmd_azimuth_deg = azimuth_deg
        except ReachyClientError as exc:
            logger.warning("look_at_azimuth failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _split_azimuth(self, azimuth_deg: float) -> tuple[float, float]:
        """Split a target azimuth into (body_yaw_deg, head_yaw_deg)."""
        head_deg = max(
            -self._cfg.max_head_yaw_deg,
            min(self._cfg.max_head_yaw_deg, azimuth_deg * self._cfg.head_share),
        )
        body_deg = azimuth_deg - head_deg
        body_deg = max(-BODY_YAW_LIMIT_DEG, min(BODY_YAW_LIMIT_DEG, body_deg))
        head_deg = max(-HEAD_YAW_LIMIT_DEG, min(HEAD_YAW_LIMIT_DEG, head_deg))
        return body_deg, head_deg

    def _run(self) -> None:
        period = 1.0 / max(1.0, self._cfg.poll_hz)
        last = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            dt = now - last
            last = now

            if not self._enabled.is_set():
                self._stop_event.wait(period)
                continue

            try:
                doa = self._client.get_doa()
            except ReachyClientError:
                doa = None

            if doa is None:
                # No source / no mic array (the bare simulator returns null).
                self._stop_event.wait(period)
                continue

            if self._cfg.require_speech and not doa.speech_detected:
                self._stop_event.wait(period)
                continue

            target_az = doa_angle_to_azimuth_deg(doa.angle_rad)
            self._update_command(target_az, dt)
            self._stop_event.wait(period)

    def _update_command(self, target_az: float, dt: float) -> None:
        """Apply dead-zone, smoothing and rate-limiting, then drive set_target."""
        with self._lock:
            current = self._cmd_azimuth_deg

        # Dead-zone: ignore tiny changes to avoid micro-jitter.
        if abs(target_az - current) < self._cfg.dead_zone_deg:
            return

        # Exponential smoothing toward the target.
        smoothed = current + self._cfg.smoothing * (target_az - current)

        # Rate limit the per-tick change.
        max_step = self._cfg.max_rate_deg_per_s * max(dt, 1e-3)
        delta = max(-max_step, min(max_step, smoothed - current))
        new_az = current + delta

        body_deg, head_deg = self._split_azimuth(new_az)

        # Use set_target for smooth, high-frequency tracking. The daemon ignores
        # this while a blocking move (emotion/dance) is running, which is the
        # desired behaviour: we don't fight other motions.
        try:
            result = self._client.set_target(
                head_pose=HeadPose(yaw=head_deg), body_yaw_deg=body_deg
            )
        except ReachyClientError as exc:
            logger.debug("set_target during tracking failed: %s", exc)
            return

        if isinstance(result, dict) and result.get("status") == "ignored":
            # A blocking move is running; don't advance our command state.
            return

        with self._lock:
            self._cmd_azimuth_deg = new_az
