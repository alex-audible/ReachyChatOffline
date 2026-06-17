"""Vision path for ReachyChatOffline.

Feeds a single camera frame plus a text question into the Gemma 4 E4B QAT
multimodal model (via mlx-vlm) so that "what Reachy sees" can enter the
conversation.

Public surface:
    - ``VLM`` / ``describe`` (vlm.py): load the model and run image+text -> text.
    - Frame sources (frame_source.py): obtain a frame as an RGB PIL image.
      ``ReachyCameraFrameSource`` (live daemon camera), ``WebcamFrameSource``
      (Mac webcam fallback), ``SyntheticFrameSource`` (last-resort image),
      ``FileFrameSource`` (file), ``FallbackFrameSource`` /
      ``best_available_frame_source`` (robust chain).
    - Vision turn handler (responder.py): ``VisionResponder.maybe_answer`` /
      ``vision_reply`` / ``is_visual_query`` -- call before the text LLM; if it
      returns a string, speak that instead.
"""

from .frame_source import (
    FallbackFrameSource,
    FileFrameSource,
    FrameSource,
    ReachyCameraFrameSource,
    SyntheticFrameSource,
    WebcamFrameSource,
    best_available_frame_source,
    bgr_to_rgb,
    make_synthetic_frame,
    to_pil,
)
from .responder import (
    VisionResponder,
    is_visual_query,
    vision_reply,
)
from .vlm import DEFAULT_MODEL, VLM, describe

__all__ = [
    # vlm
    "DEFAULT_MODEL",
    "VLM",
    "describe",
    # frame sources
    "FrameSource",
    "FileFrameSource",
    "ReachyCameraFrameSource",
    "WebcamFrameSource",
    "SyntheticFrameSource",
    "FallbackFrameSource",
    "best_available_frame_source",
    "to_pil",
    "bgr_to_rgb",
    "make_synthetic_frame",
    # vision turn handler
    "VisionResponder",
    "vision_reply",
    "is_visual_query",
]
