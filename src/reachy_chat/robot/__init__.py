"""Reachy Mini robot-integration layer for the local voice assistant.

This package talks to the Reachy Mini daemon over its REST API and provides a
clean, decoupled motion stack:

* :class:`~reachy_chat.robot.client.ReachyClient` -- typed REST wrapper.
* :class:`~reachy_chat.robot.motion_queue.MotionQueue` -- thread-safe command
  queue + control-loop worker (LLM enqueues, the loop executes smoothly).
* :class:`~reachy_chat.robot.look_at_speaker.LookAtSpeaker` -- DoA-driven
  look-at controller running in its own loop.
* :class:`~reachy_chat.robot.tools.RobotTools` -- LLM tool-call handlers.

Motion is intentionally *decoupled* from the voice pipeline: head movement may
trail the audio and is never on the speech latency path.
"""

from .client import (
    DoA,
    HeadPose,
    ReachyClient,
    ReachyClientError,
)
from .look_at_speaker import LookAtConfig, LookAtSpeaker, doa_angle_to_azimuth_deg
from .motion_queue import GotoCommand, MotionQueue, PlayMoveCommand
from .tools import TOOL_SCHEMAS, RobotTools

__all__ = [
    "DoA",
    "HeadPose",
    "ReachyClient",
    "ReachyClientError",
    "MotionQueue",
    "GotoCommand",
    "PlayMoveCommand",
    "LookAtSpeaker",
    "LookAtConfig",
    "doa_angle_to_azimuth_deg",
    "RobotTools",
    "TOOL_SCHEMAS",
]
