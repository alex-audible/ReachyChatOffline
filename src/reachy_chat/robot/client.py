"""REST client for the Reachy Mini daemon.

``ReachyClient`` is a thin, typed wrapper around the daemon's FastAPI REST API
(default base URL ``http://localhost:8000``). It speaks the exact request and
response schemas exposed by the running daemon, which were confirmed against the
live ``/openapi.json`` and by hitting the endpoints directly.

Important unit conventions (confirmed from the live daemon / SDK source):

* Head pose orientation (``roll``/``pitch``/``yaw``) is expressed in **radians**.
* ``body_yaw`` is in **radians**.
* Antenna positions are a ``(left, right)`` tuple in **radians**.
* The DoA ``angle`` is in **radians** with an unusual convention:
  ``0`` = left, ``pi/2`` = front/back, ``pi`` = right (see ``audio_doa.py``).

To stay friendly to callers and the LLM tool layer, the head-pose helpers on
this client accept **degrees** by default and convert to radians on the wire.

Motion limits enforced here (clamped, never rejected):

* head pitch / roll : +/- 40 deg
* head yaw          : +/- 180 deg
* body yaw          : +/- 160 deg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import requests

# --------------------------------------------------------------------------- #
# Motion limits (degrees). Clamping is applied before anything hits the wire.
# --------------------------------------------------------------------------- #
HEAD_PITCH_LIMIT_DEG = 40.0
HEAD_ROLL_LIMIT_DEG = 40.0
HEAD_YAW_LIMIT_DEG = 180.0
BODY_YAW_LIMIT_DEG = 160.0

#: Dataset that ships the official emotion/dance recorded moves. Confirmed
#: available on the live daemon via ``/api/move/recorded-move-datasets/list``.
DEFAULT_EMOTION_DATASET = "pollen-robotics/reachy-mini-emotions-library"

InterpolationTechnique = Literal["linear", "minjerk", "ease_in_out", "cartoon"]
MotorControlMode = Literal["enabled", "disabled", "gravity_compensation"]


def _clamp(value: float, limit: float) -> float:
    """Clamp ``value`` to ``[-limit, +limit]``."""
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class HeadPose:
    """A head pose in degrees (orientation) and metres (translation).

    All fields are optional-by-default-zero so callers can specify only what
    they care about, e.g. ``HeadPose(yaw=30)`` for a pure left turn.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def clamped(self) -> "HeadPose":
        """Return a copy with roll/pitch/yaw clamped to the motion limits."""
        return HeadPose(
            x=self.x,
            y=self.y,
            z=self.z,
            roll=_clamp(self.roll, HEAD_ROLL_LIMIT_DEG),
            pitch=_clamp(self.pitch, HEAD_PITCH_LIMIT_DEG),
            yaw=_clamp(self.yaw, HEAD_YAW_LIMIT_DEG),
        )

    def to_xyzrpy_radians(self) -> dict[str, float]:
        """Serialise to the daemon's ``XYZRPYPose`` schema (radians on the wire)."""
        c = self.clamped()
        return {
            "x": c.x,
            "y": c.y,
            "z": c.z,
            "roll": math.radians(c.roll),
            "pitch": math.radians(c.pitch),
            "yaw": math.radians(c.yaw),
        }


@dataclass(frozen=True)
class DoA:
    """Direction-of-arrival reading from the microphone array.

    Attributes:
        angle_rad: Raw daemon angle in radians (0=left, pi/2=front/back, pi=right).
        speech_detected: Whether the array currently believes speech is present.
    """

    angle_rad: float
    speech_detected: bool

    @property
    def angle_deg(self) -> float:
        """The DoA angle in degrees."""
        return math.degrees(self.angle_rad)


class ReachyClientError(RuntimeError):
    """Raised when the daemon returns an error or is unreachable."""


class ReachyClient:
    """Synchronous REST client for a running Reachy Mini daemon.

    Args:
        base_url: Daemon root URL (no trailing ``/api``). Defaults to localhost.
        timeout: Per-request timeout in seconds.
        session: Optional pre-configured ``requests.Session`` (mainly for tests).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 5.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api = f"{self.base_url}/api"
        self.timeout = timeout
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #
    def _get(self, path: str, **params: Any) -> Any:
        try:
            resp = self._session.get(
                f"{self.api}{path}",
                params={k: v for k, v in params.items() if v is not None} or None,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise ReachyClientError(f"GET {path} failed: {exc}") from exc
        return resp.json()

    def _post(self, path: str, json: Optional[dict[str, Any]] = None) -> Any:
        try:
            resp = self._session.post(
                f"{self.api}{path}", json=json, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise ReachyClientError(f"POST {path} failed: {exc}") from exc
        # Some endpoints (wake_up/goto_sleep) return empty bodies.
        if not resp.content:
            return None
        return resp.json()

    # ------------------------------------------------------------------ #
    # Health / connectivity
    # ------------------------------------------------------------------ #
    def ping(self) -> bool:
        """Return ``True`` if the daemon answers a state query."""
        try:
            self.get_present_head_pose()
            return True
        except ReachyClientError:
            return False

    # ------------------------------------------------------------------ #
    # State queries
    # ------------------------------------------------------------------ #
    def get_full_state(self, with_doa: bool = True) -> dict[str, Any]:
        """Return the daemon's full state (``GET /api/state/full``).

        Requests head pose, body yaw, antennas and (optionally) DoA in one call.
        """
        return self._get(
            "/state/full",
            with_head_pose=True,
            with_body_yaw=True,
            with_antenna_positions=True,
            with_control_mode=True,
            with_doa=with_doa,
        )

    def get_present_head_pose(self) -> HeadPose:
        """Return the current head pose as a :class:`HeadPose` (degrees)."""
        raw = self._get("/state/present_head_pose")
        return HeadPose(
            x=raw.get("x", 0.0),
            y=raw.get("y", 0.0),
            z=raw.get("z", 0.0),
            roll=math.degrees(raw.get("roll", 0.0)),
            pitch=math.degrees(raw.get("pitch", 0.0)),
            yaw=math.degrees(raw.get("yaw", 0.0)),
        )

    def get_present_body_yaw_deg(self) -> float:
        """Return current body yaw in degrees (``GET /api/state/present_body_yaw``)."""
        raw = self._get("/state/present_body_yaw")
        return math.degrees(float(raw))

    def get_doa(self) -> Optional[DoA]:
        """Return the latest direction-of-arrival, or ``None``.

        The daemon returns ``null`` when no microphone array is present or no
        source is detected (this is the case on the bare simulator).
        """
        raw = self._get("/state/doa")
        if raw is None:
            return None
        return DoA(angle_rad=float(raw["angle"]), speech_detected=bool(raw["speech_detected"]))

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #
    def goto(
        self,
        head_pose: Optional[HeadPose] = None,
        *,
        body_yaw_deg: Optional[float] = None,
        antennas: Optional[tuple[float, float]] = None,
        duration: float = 1.0,
        interpolation: InterpolationTechnique = "minjerk",
    ) -> Optional[str]:
        """Smoothly move to a target over ``duration`` seconds (``/api/move/goto``).

        Returns the move UUID (used to stop the move), or ``None``.
        """
        body: dict[str, Any] = {"duration": duration, "interpolation": interpolation}
        if head_pose is not None:
            body["head_pose"] = head_pose.to_xyzrpy_radians()
        if body_yaw_deg is not None:
            body["body_yaw"] = math.radians(_clamp(body_yaw_deg, BODY_YAW_LIMIT_DEG))
        if antennas is not None:
            body["antennas"] = list(antennas)
        result = self._post("/move/goto", body)
        if isinstance(result, dict):
            return result.get("uuid")
        return None

    def set_target(
        self,
        head_pose: Optional[HeadPose] = None,
        *,
        body_yaw_deg: Optional[float] = None,
        antennas: Optional[tuple[float, float]] = None,
    ) -> dict[str, Any]:
        """Set an instant target for high-frequency control (``/api/move/set_target``).

        Note: the daemon ignores ``set_target`` while a ``goto``/recorded move is
        running, returning ``{"status": "ignored", "reason": "move_running"}``.
        Use this only for smooth real-time control loops (e.g. look-at-speaker).
        """
        body: dict[str, Any] = {}
        if head_pose is not None:
            body["target_head_pose"] = head_pose.to_xyzrpy_radians()
        if body_yaw_deg is not None:
            body["target_body_yaw"] = math.radians(_clamp(body_yaw_deg, BODY_YAW_LIMIT_DEG))
        if antennas is not None:
            body["target_antennas"] = list(antennas)
        return self._post("/move/set_target", body) or {}

    def play_recorded_move(
        self, move_name: str, dataset: str = DEFAULT_EMOTION_DATASET
    ) -> Optional[str]:
        """Play a recorded move from a HF dataset.

        ``POST /api/move/play/recorded-move-dataset/{dataset}/{move}``.
        """
        # The path segments are URL-encoded by requests via the params? No: they
        # are path components, so build the URL with quoting handled by requests'
        # adapter is not automatic. Use requests' own quoting through a tuple.
        import urllib.parse

        ds = urllib.parse.quote(dataset, safe="")
        mv = urllib.parse.quote(move_name, safe="")
        result = self._post(f"/move/play/recorded-move-dataset/{ds}/{mv}")
        if isinstance(result, dict):
            return result.get("uuid")
        return None

    def list_recorded_moves(self, dataset: str = DEFAULT_EMOTION_DATASET) -> list[str]:
        """List the moves available in a recorded-move dataset."""
        import urllib.parse

        ds = urllib.parse.quote(dataset, safe="")
        result = self._get(f"/move/recorded-move-datasets/list/{ds}")
        return list(result) if isinstance(result, list) else []

    def wake_up(self) -> None:
        """Play the built-in wake-up move (``/api/move/play/wake_up``)."""
        self._post("/move/play/wake_up")

    def goto_sleep(self) -> None:
        """Play the built-in sleep move (``/api/move/play/goto_sleep``)."""
        self._post("/move/play/goto_sleep")

    def list_running_moves(self) -> list[str]:
        """Return UUIDs of currently running moves (``/api/move/running``)."""
        result = self._get("/move/running")
        if not isinstance(result, list):
            return []
        return [item["uuid"] for item in result if isinstance(item, dict) and "uuid" in item]

    def is_moving(self) -> bool:
        """Return ``True`` if a recorded/goto move is currently running."""
        return bool(self.list_running_moves())

    def stop_move(self, uuid: str) -> None:
        """Stop a running move by its UUID (``/api/move/stop``)."""
        self._post("/move/stop", {"uuid": uuid})

    def stop_all_moves(self) -> None:
        """Stop every currently running move."""
        for uuid in self.list_running_moves():
            try:
                self.stop_move(uuid)
            except ReachyClientError:
                pass

    # ------------------------------------------------------------------ #
    # Motors
    # ------------------------------------------------------------------ #
    def get_motor_mode(self) -> str:
        """Return the current motor control mode (``/api/motors/status``)."""
        return str(self._get("/motors/status").get("mode"))

    def set_motor_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode (``/api/motors/set_mode/{mode}``)."""
        self._post(f"/motors/set_mode/{mode}")
