"""Frame sources for the vision path.

A *frame source* yields a single still image (as a ``PIL.Image`` in RGB) that
can be handed to :func:`reachy_chat.vision.vlm.describe`.

Two sources are provided:

* :class:`FileFrameSource` -- reads a frame from an image file on disk. This is
  the path used for testing / the smoke test and is fully working now.
* :class:`ReachyCameraFrameSource` -- a documented hook for the live Reachy Mini
  camera. The daemon (``localhost:8000``) owns the camera; the SDK exposes the
  latest frame as an RGB ``numpy`` array via ``mini.media.get_frame()``. This
  class wraps that call. It is intentionally lightweight and is NOT exercised by
  the smoke test (no robot attached in this environment), but it documents the
  exact integration point so the main thread can wire it up later.

The helper :func:`to_pil` normalizes the accepted input types
(``PIL.Image`` / ``numpy`` array / path / URL / ``bytes``) into a single RGB
``PIL.Image`` so the rest of the vision path only has to deal with one type.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:  # avoid importing heavy deps at module import time
    import numpy as np
    from PIL import Image as PILImage

# Type of anything we know how to turn into a PIL image.
ImageLike = Union["PILImage.Image", "np.ndarray", str, Path, bytes]


def to_pil(image: ImageLike) -> "PILImage.Image":
    """Normalize any supported image input into an RGB ``PIL.Image``.

    Accepts:
        * a ``PIL.Image`` (returned converted to RGB),
        * a ``numpy`` array (H x W x 3 uint8 RGB, as produced by the Reachy
          camera / OpenCV-after-conversion),
        * a file path / URL / data-URI string or ``pathlib.Path``,
        * raw image ``bytes`` (e.g. an encoded JPEG/PNG).

    Note on numpy arrays: the Reachy SDK's ``media.get_frame()`` returns RGB
    uint8. If your array comes straight from OpenCV (BGR) you must convert it to
    RGB before passing it here.
    """
    from PIL import Image  # local import keeps module import cheap

    # Already a PIL image.
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # numpy array (e.g. a camera frame).
    # Detected duck-typed to avoid a hard numpy import for callers that never
    # use arrays.
    if hasattr(image, "__array_interface__") or type(image).__name__ == "ndarray":
        import numpy as np

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            # Assume float in [0, 1] or arbitrary range -> clip/scale to uint8.
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    # Raw encoded bytes.
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")

    # Path / URL / data-URI string.
    if isinstance(image, (str, Path)):
        # Reuse mlx-vlm's loader so URLs and data-URIs also work.
        from mlx_vlm.utils import load_image

        return load_image(str(image)).convert("RGB")

    raise TypeError(f"Unsupported image type for to_pil(): {type(image).__name__}")


class FrameSource(ABC):
    """Abstract source of a single still frame."""

    @abstractmethod
    def get_frame(self) -> "PILImage.Image":
        """Return the current frame as an RGB ``PIL.Image``."""
        raise NotImplementedError


class FileFrameSource(FrameSource):
    """Read a frame from an image file (used for testing / the smoke test)."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    def get_frame(self) -> "PILImage.Image":
        if not self.path.exists():
            raise FileNotFoundError(f"Frame file not found: {self.path}")
        return to_pil(self.path)


class ReachyCameraFrameSource(FrameSource):
    """Live frame from the Reachy Mini camera (documented integration hook).

    The Reachy Mini daemon owns the physical camera (see
    ``.reference/reachy_mini/docs/source/SDK/media-architecture.md``). A local
    client reads the latest decoded frame via the SDK's media manager:

        from reachy_mini import ReachyMini

        mini = ReachyMini()                 # auto-selects LOCAL backend
        frame = mini.media.get_frame()      # RGB uint8 numpy array, or None

    The conversation app polls this in a background thread roughly every 40 ms
    and keeps the latest frame (see
    ``.reference/reachy_mini_conversation_app/.../camera_worker.py``). For the
    vision path we only need the most recent frame on demand, so this class
    simply calls ``get_frame()`` when asked.

    Pass either a live ``ReachyMini`` instance (``mini=...``) or an object that
    already exposes the latest frame (``frame_provider=...``, e.g. a
    ``CameraWorker`` with ``get_latest_frame()``). This class does NOT create or
    own the robot connection -- the main app does that.

    Not exercised by the smoke test (no robot in this environment); the
    file-based path is the working one for now.
    """

    def __init__(
        self,
        mini: Optional[object] = None,
        frame_provider: Optional[object] = None,
    ) -> None:
        if mini is None and frame_provider is None:
            raise ValueError(
                "ReachyCameraFrameSource needs either `mini` (a ReachyMini) or "
                "`frame_provider` (something with get_latest_frame()/get_frame())."
            )
        self._mini = mini
        self._frame_provider = frame_provider

    def _raw_frame(self):
        """Fetch the latest raw frame (numpy RGB uint8) from whichever source."""
        if self._frame_provider is not None:
            # CameraWorker-style: get_latest_frame(); fall back to get_frame().
            getter = getattr(self._frame_provider, "get_latest_frame", None) or getattr(
                self._frame_provider, "get_frame", None
            )
            if getter is None:
                raise AttributeError(
                    "frame_provider has neither get_latest_frame() nor get_frame()"
                )
            return getter()
        # Direct ReachyMini: mini.media.get_frame()
        return self._mini.media.get_frame()

    def get_frame(self) -> "PILImage.Image":
        frame = self._raw_frame()
        if frame is None:
            raise RuntimeError(
                "Reachy camera returned no frame yet (None). The daemon media "
                "pipeline may still be warming up; retry shortly."
            )
        return to_pil(frame)
