"""LLM tool-call handlers for controlling Reachy Mini.

These are the functions an LLM invokes via tool/function calling. Each one
*enqueues* a command onto the decoupled :class:`~reachy_chat.robot.motion_queue.MotionQueue`
(or toggles the :class:`~reachy_chat.robot.look_at_speaker.LookAtSpeaker`
controller) and returns a short human-readable string. Nothing here blocks on
the motors, and nothing here is on the speech latency path -- motion is allowed
to trail the conversation.

The tool *schemas* (:data:`TOOL_SCHEMAS`) are modelled on the official
conversation app's tools (``move_head``, ``dance``, ``play_emotion``,
``head_tracking``) and are in OpenAI/Realtime function-calling format so they can
be handed straight to an LLM.

Usage::

    tools = RobotTools(client, queue, look_at)
    # expose tools.schemas to the LLM, then dispatch calls:
    result = tools.dispatch("nod", {})
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

from .client import HeadPose, ReachyClient
from .look_at_speaker import LookAtSpeaker
from .motion_queue import GotoCommand, MotionQueue, PlayMoveCommand

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Direction -> head delta (degrees), mirroring the reference move_head tool.
# --------------------------------------------------------------------------- #
_HEAD_DIRECTIONS: dict[str, HeadPose] = {
    "left": HeadPose(yaw=40),
    "right": HeadPose(yaw=-40),
    "up": HeadPose(pitch=-25),
    "down": HeadPose(pitch=25),
    "front": HeadPose(),
}

# --------------------------------------------------------------------------- #
# Emotion intent -> candidate recorded-move names (subset of the official map).
# resolve_emotion() picks one at random and falls back to the raw name if the
# request already is a valid move id.
# --------------------------------------------------------------------------- #
_EMOTION_INTENTS: dict[str, tuple[str, ...]] = {
    "happy": ("laughing2", "laughing1"),
    "excited": ("dance3", "dance2"),
    "loving": ("loving1",),
    "grateful": ("grateful1",),
    "success": ("success1", "success2"),
    "thinking": ("thoughtful1", "thoughtful2"),
    "attentive": ("attentive1", "attentive2"),
    "confused": ("confused1",),
    "uncertain": ("uncertain1",),
    "sad": ("sad1", "sad2", "downcast1"),
    "lonely": ("lonely1",),
    "angry": ("rage1", "irritated2", "irritated1"),
    "irritated": ("irritated1", "irritated2"),
    "disgusted": ("disgusted1",),
    "scared": ("scared1", "fear1", "anxiety1"),
    "anxious": ("anxiety1", "fear1"),
    "surprised": ("surprised1", "amazed1"),
    "amazed": ("amazed1", "surprised1"),
    "calming": ("calming1",),
    "relief": ("relief1",),
    "bored": ("boredom2", "boredom1"),
    "tired": ("exhausted1", "tired1"),
    "sleepy": ("sleep1", "exhausted1"),
    "yes": ("yes1", "understanding1"),
    "no": ("no1",),
    "welcoming": ("welcoming2", "welcoming1"),
    "greeting": ("welcoming2",),
    "goodbye": ("loving1", "welcoming2"),
    "helpful": ("helpful1", "helpful2"),
    "curious": ("curious1", "inquiring1"),
    "proud": ("proud1", "proud2", "proud3"),
}

# Dance intents map onto the dance-flavoured recorded moves in the same dataset.
_DANCE_MOVES: tuple[str, ...] = ("dance2", "dance3")


class RobotTools:
    """Dispatchable tool handlers wiring the LLM to the robot.

    Args:
        client: A connected :class:`ReachyClient` (used for emotion lookup).
        queue: The running :class:`MotionQueue`.
        look_at: The :class:`LookAtSpeaker` controller (for look_at / head_tracking).
        move_duration: Default duration (s) for goto-style tool moves.
    """

    def __init__(
        self,
        client: ReachyClient,
        queue: MotionQueue,
        look_at: LookAtSpeaker,
        move_duration: float = 1.0,
    ) -> None:
        self._client = client
        self._queue = queue
        self._look_at = look_at
        self._move_duration = move_duration
        self._available_moves: list[str] | None = None

        self._handlers: dict[str, Callable[..., str]] = {
            "move_head": self.move_head,
            "look_at": self.look_at,
            "play_emotion": self.play_emotion,
            "dance": self.dance,
            "nod": self.nod,
            "head_tracking": self.head_tracking,
        }

    # ------------------------------------------------------------------ #
    # Tool dispatch
    # ------------------------------------------------------------------ #
    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Return the OpenAI-style function schemas for all tools."""
        return TOOL_SCHEMAS

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool by name with a kwargs dict (as an LLM would call it)."""
        handler = self._handlers.get(name)
        if handler is None:
            return f"Unknown tool '{name}'."
        try:
            return handler(**(arguments or {}))
        except TypeError as exc:
            return f"Bad arguments for '{name}': {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Tool '%s' failed", name)
            return f"Tool '{name}' failed: {exc}"

    # ------------------------------------------------------------------ #
    # Individual tools
    # ------------------------------------------------------------------ #
    def move_head(self, direction: str) -> str:
        """Move the head in a named direction: left/right/up/down/front."""
        pose = _HEAD_DIRECTIONS.get(direction.lower())
        if pose is None:
            return f"Unknown direction '{direction}'. Use left, right, up, down or front."
        self._queue.enqueue(
            GotoCommand(head_pose=pose, body_yaw_deg=0.0, duration=self._move_duration)
        )
        return f"Looking {direction.lower()}."

    def look_at(self, azimuth: float) -> str:
        """Turn toward a signed azimuth in degrees (0=front, +=left, -=right)."""
        self._look_at.look_at_azimuth(float(azimuth))
        return f"Looking toward {azimuth:.0f} degrees."

    def play_emotion(self, emotion: str | None = None) -> str:
        """Play a recorded emotion by intent name (e.g. happy, sad, surprised)."""
        move = self._resolve_emotion(emotion)
        if move is None:
            return "No emotions available."
        self._queue.enqueue(PlayMoveCommand(move_name=move), interrupt=True)
        return f"Playing emotion '{move}'."

    def dance(self, move: str | None = None) -> str:
        """Play a dance move (random if unspecified)."""
        choices = [m for m in self._dataset_moves() if m in _DANCE_MOVES] or list(_DANCE_MOVES)
        chosen = move if move in choices else random.choice(choices)
        self._queue.enqueue(PlayMoveCommand(move_name=chosen), interrupt=True)
        return f"Dancing ('{chosen}')."

    def nod(self, times: int = 1) -> str:
        """Nod the head 'yes' (down-up) one or more times."""
        times = max(1, min(5, int(times)))
        for _ in range(times):
            self._queue.enqueue(GotoCommand(head_pose=HeadPose(pitch=22), duration=0.4))
            self._queue.enqueue(GotoCommand(head_pose=HeadPose(pitch=-8), duration=0.4))
        self._queue.enqueue(GotoCommand(head_pose=HeadPose(), duration=0.4))
        return f"Nodding {times} time(s)."

    def head_tracking(self, on: bool) -> str:
        """Enable or disable DoA-driven look-at-speaker tracking."""
        self._look_at.set_enabled(bool(on))
        return f"Head tracking {'on' if on else 'off'}."

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _dataset_moves(self) -> list[str]:
        """Lazily fetch and cache the available recorded-move names."""
        if self._available_moves is None:
            try:
                self._available_moves = self._client.list_recorded_moves()
            except Exception:  # pragma: no cover - network
                self._available_moves = []
        return self._available_moves

    def _resolve_emotion(self, requested: str | None) -> str | None:
        """Resolve an intent name / raw move id to an available recorded move."""
        available = self._dataset_moves()
        avail_set = set(available)

        # No request -> random available emotion.
        if not requested:
            return random.choice(available) if available else None

        key = requested.strip().lower()

        # Exact move id.
        if key in avail_set:
            return key

        # Intent -> first candidate that exists in the dataset.
        candidates = _EMOTION_INTENTS.get(key, ())
        for cand in candidates:
            if not avail_set or cand in avail_set:
                return cand

        # Fall back to any candidate (works even if listing failed).
        if candidates:
            return candidates[0]

        # Unknown intent: random available, so the robot still reacts.
        return random.choice(available) if available else None


# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI / Realtime function-calling format).
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "move_head",
        "description": "Move your head in a given direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "front"],
                }
            },
            "required": ["direction"],
        },
    },
    {
        "type": "function",
        "name": "look_at",
        "description": (
            "Turn to face a direction given as a signed azimuth in degrees: "
            "0 is straight ahead, positive is to your left, negative is to your right."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "azimuth": {
                    "type": "number",
                    "description": "Azimuth in degrees (0=front, +left, -right).",
                }
            },
            "required": ["azimuth"],
        },
    },
    {
        "type": "function",
        "name": "play_emotion",
        "description": "Play a recorded emotional animation.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "description": (
                        "Emotion intent, e.g. happy, sad, surprised, excited, "
                        "curious, grateful, welcoming. Omit for a random emotion."
                    ),
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "dance",
        "description": "Play a dance move (random if no move is given). Non-blocking.",
        "parameters": {
            "type": "object",
            "properties": {
                "move": {"type": "string", "description": "Optional dance move name."}
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "nod",
        "description": "Nod your head yes one or more times.",
        "parameters": {
            "type": "object",
            "properties": {
                "times": {
                    "type": "integer",
                    "description": "How many nods (1-5).",
                    "minimum": 1,
                    "maximum": 5,
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "head_tracking",
        "description": "Turn DoA-based look-at-the-speaker head tracking on or off.",
        "parameters": {
            "type": "object",
            "properties": {"on": {"type": "boolean"}},
            "required": ["on"],
        },
    },
]
