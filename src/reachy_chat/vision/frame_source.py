"""Frame sources for the vision path.

A *frame source* yields a single still image (as a ``PIL.Image`` in RGB) that
can be handed to :func:`reachy_chat.vision.vlm.describe`.

Sources provided:

* :class:`ReachyCameraFrameSource` -- live frame from the Reachy Mini camera.
  The daemon (``localhost:8000``) owns the camera; on the same machine the SDK's
  ``LOCAL`` backend reads the latest frame over a GStreamer IPC endpoint and
  returns it as a **BGR** ``uint8`` numpy array via ``mini.media.get_frame()``.
  This class wraps that call (with warm-up polling) and converts BGR -> RGB.
* :class:`WebcamFrameSource` -- a plain ``cv2.VideoCapture`` fallback (the Mac
  webcam), for when no robot/daemon camera is available. OpenCV also returns
  BGR, so it is converted too.
* :class:`SyntheticFrameSource` -- a generated test image (last-resort fallback
  so the vision path always returns *something*).
* :class:`FileFrameSource` -- reads a frame from an image file on disk (used by
  the smoke test / for offline testing).

:func:`best_available_frame_source` builds a :class:`FallbackFrameSource` that
tries, in order: Reachy daemon camera -> Mac webcam -> synthetic image. This is
the robust ``get_frame()`` the live app should use.

The helper :func:`to_pil` normalizes the accepted input types
(``PIL.Image`` / ``numpy`` array / path / URL / ``bytes``) into a single RGB
``PIL.Image`` so the rest of the vision path only has to deal with one type.

IMPORTANT (colour order): both the Reachy SDK camera and OpenCV return **BGR**
frames. Pass such arrays through :func:`bgr_to_rgb` (or ``to_pil(arr,
bgr=True)``) before describing them, otherwise the model sees swapped colours.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # avoid importing heavy deps at module import time
    import numpy as np
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Type of anything we know how to turn into a PIL image.
ImageLike = Union["PILImage.Image", "np.ndarray", str, Path, bytes]


def bgr_to_rgb(arr: "np.ndarray") -> "np.ndarray":
    """Flip the channel order of an H x W x 3 array (BGR <-> RGB).

    Both the Reachy SDK camera and OpenCV deliver BGR; the VLM expects RGB.
    Single-channel / non-3-channel arrays are returned unchanged.
    """
    import numpy as np

    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 3:
        return a[:, :, ::-1]
    return a


def to_pil(image: ImageLike, *, bgr: bool = False) -> "PILImage.Image":
    """Normalize any supported image input into an RGB ``PIL.Image``.

    Accepts:
        * a ``PIL.Image`` (returned converted to RGB),
        * a ``numpy`` array (H x W x 3 uint8, as produced by the Reachy camera /
          OpenCV),
        * a file path / URL / data-URI string or ``pathlib.Path``,
        * raw image ``bytes`` (e.g. an encoded JPEG/PNG).

    Args:
        bgr: set ``True`` when the input is a numpy array in **BGR** channel
            order (the Reachy SDK camera and OpenCV both return BGR). The
            channels are flipped to RGB before building the PIL image. Ignored
            for non-array inputs.
    """
    from PIL import Image  # local import keeps module import cheap

    # Already a PIL image.
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # numpy array (e.g. a camera frame). Duck-typed to avoid a hard numpy import
    # for callers that never use arrays.
    if hasattr(image, "__array_interface__") or type(image).__name__ == "ndarray":
        import numpy as np

        arr = np.asarray(image)
        if bgr:
            arr = bgr_to_rgb(arr)
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


def make_synthetic_frame(size: int = 512) -> "PILImage.Image":
    """Build a simple, unambiguous test image (a red circle on white).

    Used as a last-resort frame source so the vision path always returns a valid
    RGB image even with no camera attached.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    margin = size // 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=(220, 30, 30))
    return img


class FrameSource(ABC):
    """Abstract source of a single still frame."""

    @abstractmethod
    def get_frame(self) -> "PILImage.Image":
        """Return the current frame as an RGB ``PIL.Image``.

        Implementations raise on failure; use :class:`FallbackFrameSource` (or
        :func:`best_available_frame_source`) for graceful degradation.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Short human-readable identifier for logging / the smoke test."""
        return type(self).__name__


class FileFrameSource(FrameSource):
    """Read a frame from an image file (used for testing / the smoke test)."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return f"file:{self.path}"

    def get_frame(self) -> "PILImage.Image":
        if not self.path.exists():
            raise FileNotFoundError(f"Frame file not found: {self.path}")
        return to_pil(self.path)


class SyntheticFrameSource(FrameSource):
    """A generated test image -- the last-resort fallback frame source."""

    def __init__(self, size: int = 512) -> None:
        self.size = size

    @property
    def name(self) -> str:
        return "synthetic"

    def get_frame(self) -> "PILImage.Image":
        return make_synthetic_frame(self.size)


class WebcamFrameSource(FrameSource):
    """Grab a frame from a local camera via ``cv2.VideoCapture`` (Mac webcam).

    This is the fallback used when the Reachy daemon camera is unavailable (e.g.
    no robot, or a ``--no-media`` daemon). OpenCV returns BGR frames, which are
    converted to RGB here.

    A couple of frames are read and discarded first: many webcams (and the macOS
    AVFoundation backend in particular) return a dark/half-exposed first frame
    while auto-exposure settles.
    """

    def __init__(self, index: int = 0, warmup_frames: int = 3) -> None:
        self.index = index
        self.warmup_frames = max(0, warmup_frames)

    @property
    def name(self) -> str:
        return f"webcam:{self.index}"

    def get_frame(self) -> "PILImage.Image":
        import cv2  # local import: opencv is only needed for this fallback

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open webcam (cv2.VideoCapture({self.index}))")
        try:
            frame = None
            # Read and discard a few warm-up frames, then keep the latest.
            for _ in range(self.warmup_frames + 1):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is None:
                raise RuntimeError("Webcam opened but returned no frame")
            return to_pil(frame, bgr=True)
        finally:
            cap.release()


class ReachyCameraFrameSource(FrameSource):
    """Live frame from the Reachy Mini camera (via the SDK ``LOCAL`` backend).

    The Reachy Mini daemon owns the physical camera (see
    ``.reference/reachy_mini/docs/source/SDK/media-architecture.md``). On the
    same machine the SDK reads the latest decoded frame from the daemon's
    GStreamer IPC endpoint and returns it as a **BGR** ``uint8`` numpy array via
    ``mini.media.get_frame()`` (verified against the live daemon at
    ``localhost:8000``). There is no plain REST/HTTP endpoint that returns a
    frame -- the SDK media manager is the supported path.

    Provide one of:

    * ``mini=`` -- a live ``ReachyMini`` instance whose media backend is
      ``LOCAL`` (the default auto-detected backend on-device). ``get_frame()``
      then calls ``mini.media.get_frame()``.
    * ``frame_provider=`` -- an object that already buffers the latest frame
      (e.g. the conversation app's ``CameraWorker`` with ``get_latest_frame()``,
      or anything with ``get_frame()``). Use this if the app already runs a
      camera worker thread so we don't double-read the pipeline.
    * neither -- this class will lazily construct its own ``ReachyMini`` (LOCAL
      backend, ``connection_mode="auto"``) on first use. Convenient for the
      vision path running standalone; pass ``mini=`` from the app to share the
      existing connection instead.

    The pipeline may need a moment to warm up after connecting, so
    ``get_frame()`` polls ``get_frame()`` for up to ``warmup_timeout`` seconds
    before giving up.

    Frames are returned as RGB ``PIL.Image`` (BGR -> RGB conversion is applied).
    """

    def __init__(
        self,
        mini: Optional[object] = None,
        frame_provider: Optional[object] = None,
        *,
        warmup_timeout: float = 3.0,
        poll_interval: float = 0.05,
        own_connection: bool = True,
    ) -> None:
        self._mini = mini
        self._frame_provider = frame_provider
        self._owns_mini = False
        self._own_connection = own_connection
        self.warmup_timeout = warmup_timeout
        self.poll_interval = poll_interval

    @property
    def name(self) -> str:
        if self._frame_provider is not None:
            return "reachy:frame_provider"
        return "reachy:camera"

    def _ensure_mini(self):
        """Lazily create a ReachyMini (LOCAL backend) if none was supplied."""
        if self._mini is not None:
            return self._mini
        if not self._own_connection:
            raise RuntimeError(
                "ReachyCameraFrameSource has no `mini`/`frame_provider` and "
                "own_connection=False; supply a ReachyMini or CameraWorker."
            )
        from reachy_mini import ReachyMini

        # "default" => auto-detect; LOCAL when on the same machine as the daemon.
        self._mini = ReachyMini(media_backend="default")
        self._owns_mini = True
        return self._mini

    def _raw_frame(self):
        """Fetch the latest raw frame (numpy BGR uint8) from whichever source."""
        if self._frame_provider is not None:
            getter = getattr(self._frame_provider, "get_latest_frame", None) or getattr(
                self._frame_provider, "get_frame", None
            )
            if getter is None:
                raise AttributeError(
                    "frame_provider has neither get_latest_frame() nor get_frame()"
                )
            return getter()
        mini = self._ensure_mini()
        return mini.media.get_frame()

    def get_frame(self) -> "PILImage.Image":
        import time

        deadline = time.monotonic() + self.warmup_timeout
        frame = self._raw_frame()
        while frame is None and time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            frame = self._raw_frame()
        if frame is None:
            raise RuntimeError(
                "Reachy camera returned no frame within "
                f"{self.warmup_timeout:.1f}s. The daemon may be running with "
                "--no-media, the media may be released, or the pipeline is still "
                "warming up."
            )
        # SDK camera frames are BGR -> convert to RGB.
        return to_pil(frame, bgr=True)

    def close(self) -> None:
        """Release a ReachyMini connection if this source created its own."""
        if self._owns_mini and self._mini is not None:
            try:
                self._mini.media.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


class FallbackFrameSource(FrameSource):
    """Try several frame sources in order; return the first that succeeds.

    The source that produced the frame is recorded on ``last_used`` so callers
    (e.g. the smoke test / logging) can report which path actually worked.
    """

    def __init__(self, sources: List[FrameSource]) -> None:
        if not sources:
            raise ValueError("FallbackFrameSource needs at least one source")
        self.sources = sources
        self.last_used: Optional[FrameSource] = None

    @property
    def name(self) -> str:
        if self.last_used is not None:
            return f"fallback({self.last_used.name})"
        return "fallback(?)"

    def get_frame(self) -> "PILImage.Image":
        errors = []
        for src in self.sources:
            try:
                frame = src.get_frame()
                self.last_used = src
                logger.info("Frame captured from %s", src.name)
                return frame
            except Exception as e:  # noqa: BLE001 - fall through to next source
                errors.append(f"{src.name}: {e}")
                logger.warning("Frame source %s failed: %s", src.name, e)
        raise RuntimeError(
            "All frame sources failed:\n  " + "\n  ".join(errors)
        )


def best_available_frame_source(
    *,
    mini: Optional[object] = None,
    frame_provider: Optional[object] = None,
    include_webcam: bool = True,
    include_synthetic: bool = True,
) -> FallbackFrameSource:
    """Build the robust frame source the live app should use.

    Order of preference:
        1. Reachy daemon camera (``mini`` / ``frame_provider``, else a
           lazily-created LOCAL ``ReachyMini``),
        2. Mac webcam (``cv2.VideoCapture(0)``) -- skip with
           ``include_webcam=False``,
        3. a generated synthetic image -- skip with ``include_synthetic=False``.

    Returns a :class:`FallbackFrameSource`; inspect ``.last_used.name`` after a
    ``get_frame()`` to see which path produced the frame.
    """
    sources: List[FrameSource] = [
        ReachyCameraFrameSource(mini=mini, frame_provider=frame_provider)
    ]
    if include_webcam:
        sources.append(WebcamFrameSource())
    if include_synthetic:
        sources.append(SyntheticFrameSource())
    return FallbackFrameSource(sources)
