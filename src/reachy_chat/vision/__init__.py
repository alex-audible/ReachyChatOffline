"""Vision path for ReachyChatOffline.

Feeds a single camera frame plus a text question into the Gemma 4 E4B QAT
multimodal model (via mlx-vlm) so that "what Reachy sees" can enter the
conversation.

Public surface:
    - ``VLM`` / ``describe`` (vlm.py): load the model and run image+text -> text.
    - ``FrameSource`` / ``FileFrameSource`` / ``ReachyCameraFrameSource``
      (frame_source.py): obtain a frame as a PIL image.
"""

from .frame_source import (
    FileFrameSource,
    FrameSource,
    ReachyCameraFrameSource,
    to_pil,
)
from .vlm import DEFAULT_MODEL, VLM, describe

__all__ = [
    "DEFAULT_MODEL",
    "VLM",
    "describe",
    "FrameSource",
    "FileFrameSource",
    "ReachyCameraFrameSource",
    "to_pil",
]
