"""Vision turn handler for the conversation loop.

The main app calls into here *before* the text LLM on each user turn. If the
user's transcript is a visual query ("what do you see?", "look at this",
"describe what's in front of you", ...), we:

1. capture a single camera frame (Reachy daemon camera, falling back to the Mac
   webcam, then a synthetic image),
2. run the Gemma 4 E4B multimodal model on (frame + question),
3. return a SHORT, spoken-style answer (1-2 sentences, no markdown).

If the transcript is *not* a visual query, we return ``None`` so the normal text
LLM handles the turn.

The VLM (~6 GB) is **lazy-loaded on first visual query**, so sessions that never
ask Reachy to look at anything don't pay for it.

Public surface:
    * :func:`is_visual_query` -- the heuristic classifier.
    * :func:`vision_reply` -- functional one-shot handler.
    * :class:`VisionResponder` -- holds the lazily-loaded VLM + frame source and
      exposes :meth:`VisionResponder.maybe_answer`.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .frame_source import FrameSource
    from .vlm import VLM

logger = logging.getLogger(__name__)

# Keep vision answers short and spoken-friendly (TTS will read them aloud).
DEFAULT_MAX_TOKENS = 64

# Instruction wrapped around the user's question so the model answers concisely
# and in a speakable style (no markdown, no lists, first person as the robot).
SPOKEN_SYSTEM_HINT = (
    "You are a friendly robot looking through your camera. Answer the user's "
    "question about what you see in one or two short, natural spoken sentences. "
    "Do not use markdown, bullet points, or asterisks. Be specific and concise."
)

# --------------------------------------------------------------------------- #
# Visual-query detection
# --------------------------------------------------------------------------- #
# Phrase patterns that strongly indicate the user wants Reachy to use its eyes.
# Matched against a lowercased, punctuation-stripped transcript.
_VISUAL_PATTERNS = [
    r"\bwhat (do|can) you see\b",
    r"\bwhat are you (seeing|looking at)\b",
    r"\bwhat'?s? (in front of|behind|around|near) (you|me|us)\b",
    r"\bwhat'?s (this|that|here|there)\b",
    r"\bwhat is (this|that)\b",
    r"\bwhat am i (holding|showing|wearing|pointing)\b",
    r"\bwhat'?s? in (my|your|the) (hand|hands)\b",
    r"\bcan you see\b",
    r"\bdo you see\b",
    r"\bare you able to see\b",
    r"\blook (at|around|over)\b",
    r"\btake a (look|picture|photo|snapshot)\b",
    r"\bdescribe (what|the|this|that|your|the scene|everything)\b",
    r"\b(tell me )?what (does it|do i|do you) look like\b",
    r"\bhow many .* (do you see|are there|can you see)\b",
    r"\bwhat colou?r\b",
    r"\bread (this|that|the|it)\b",
    r"\bwho('?s| is) (in front of|with|near|around) (you|me)\b",
    r"\bcheck (out )?(this|that|what)\b",
]
_VISUAL_RE = [re.compile(p) for p in _VISUAL_PATTERNS]

# Single keywords that, on their own, are weak signals; require them to co-occur
# with a "perception/look" verb to count (handled in is_visual_query).
_LOOK_VERBS = re.compile(r"\b(see|look|watch|view|show|describe|spot|notice)\b")
_VISUAL_NOUNS = re.compile(
    r"\b(this|that|here|there|in front|camera|picture|photo|image|scene|holding)\b"
)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s']", " ", text.lower()).strip()


def is_visual_query(transcript: str) -> bool:
    """Return ``True`` if *transcript* is asking Reachy to use its camera.

    Heuristic, intentionally lenient on the "wants to look" side -- a false
    positive merely captures a frame and lets the VLM answer; a false negative
    silently routes a visual question to the (blind) text LLM, which is worse.
    """
    if not transcript or not transcript.strip():
        return False
    norm = _normalize(transcript)

    # Strong phrase patterns.
    if any(rx.search(norm) for rx in _VISUAL_RE):
        return True

    # Weak: a perception verb co-occurring with a deictic/visual noun, e.g.
    # "show me what's there", "look here".
    if _LOOK_VERBS.search(norm) and _VISUAL_NOUNS.search(norm):
        return True

    return False


def _clean_spoken(text: str) -> str:
    """Strip markdown / list artefacts so the answer reads cleanly via TTS."""
    if not text:
        return text
    # Drop common markdown emphasis / list markers.
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    # Collapse whitespace/newlines into single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_prompt(transcript: str) -> str:
    """Compose the question sent to the VLM from the user's transcript."""
    transcript = transcript.strip()
    return f"{SPOKEN_SYSTEM_HINT}\n\nUser asked: {transcript}"


def vision_reply(
    transcript: str,
    *,
    vlm: Optional["VLM"] = None,
    frame_source: Optional["FrameSource"] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Optional[str]:
    """Answer a visual query from a real camera frame, or return ``None``.

    If *transcript* is not a visual query (per :func:`is_visual_query`), returns
    ``None`` immediately so the caller's text LLM handles the turn.

    Otherwise captures one frame from *frame_source* (default: the best
    available source -- Reachy camera -> webcam -> synthetic) and runs *vlm*
    (default: the process-wide cached :class:`VLM`, loaded on first use) to
    produce a short spoken answer.

    Args:
        transcript: the user's utterance.
        vlm: an already-loaded :class:`VLM`. If ``None``, the module-level
            cached default is used (lazy-loaded on first call).
        frame_source: a :class:`FrameSource`. If ``None``, a robust fallback
            source is built (Reachy camera -> webcam -> synthetic image).
        max_tokens: cap on the answer length (kept small for fast, spoken
            replies).

    Returns:
        A short spoken-style answer string, or ``None`` for non-visual queries.
        On capture/inference failure returns a brief spoken apology rather than
        raising, so the turn loop never crashes on a vision error.
    """
    if not is_visual_query(transcript):
        return None

    from .frame_source import best_available_frame_source
    from .vlm import get_default_vlm

    src = frame_source if frame_source is not None else best_available_frame_source()
    try:
        frame = src.get_frame()
    except Exception as e:  # noqa: BLE001
        logger.warning("Vision frame capture failed: %s", e)
        return "Sorry, I can't see anything right now; my camera isn't available."

    model = vlm if vlm is not None else get_default_vlm()
    try:
        answer = model.describe(
            frame, _build_prompt(transcript), max_tokens=max_tokens
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Vision inference failed: %s", e)
        return "Sorry, I had trouble making sense of what I'm looking at."

    return _clean_spoken(answer)


class VisionResponder:
    """Holds a lazily-loaded VLM + frame source for the conversation turn loop.

    Construct one per session and call :meth:`maybe_answer` on each user
    transcript *before* the text LLM. The model is loaded on the first visual
    query only, so non-vision sessions never pay the ~6 GB / load time.

    Example (in the app's turn loop)::

        responder = VisionResponder(mini=mini)   # mini = the live ReachyMini
        ...
        spoken = responder.maybe_answer(transcript)
        if spoken is not None:
            speak(spoken)          # vision handled this turn
        else:
            spoken = text_llm(transcript)
            speak(spoken)

    Args:
        frame_source: explicit :class:`FrameSource`. If ``None`` (default), a
            robust fallback source is built lazily on first use (Reachy daemon
            camera -> Mac webcam -> synthetic). Pass ``mini=`` to point the
            Reachy source at the app's existing connection.
        mini: a live ``ReachyMini`` to read the camera from (shares the app's
            connection instead of opening a second one). Used only when
            *frame_source* is ``None``.
        frame_provider: a camera worker exposing ``get_latest_frame()`` /
            ``get_frame()`` (e.g. the conversation app's ``CameraWorker``). Used
            only when *frame_source* is ``None``.
        model_name: override the VLM model id.
        max_tokens: answer length cap.
    """

    def __init__(
        self,
        frame_source: Optional["FrameSource"] = None,
        *,
        mini: Optional[object] = None,
        frame_provider: Optional[object] = None,
        model_name: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._frame_source = frame_source
        self._mini = mini
        self._frame_provider = frame_provider
        self._model_name = model_name
        self.max_tokens = max_tokens
        self._vlm: Optional["VLM"] = None

    @property
    def vlm_loaded(self) -> bool:
        """Whether the VLM has been loaded yet (i.e. a visual query happened)."""
        return self._vlm is not None

    def _get_frame_source(self) -> "FrameSource":
        if self._frame_source is None:
            from .frame_source import best_available_frame_source

            self._frame_source = best_available_frame_source(
                mini=self._mini, frame_provider=self._frame_provider
            )
        return self._frame_source

    def _get_vlm(self) -> "VLM":
        if self._vlm is None:
            from .vlm import DEFAULT_MODEL, VLM

            name = self._model_name or DEFAULT_MODEL
            logger.info("Lazy-loading VLM (%s) for first visual query...", name)
            self._vlm = VLM(name)
        return self._vlm

    def warmup(self) -> None:
        """Load the VLM weights now (at startup) rather than on the first visual query, so the
        conversation never stalls mid-turn to load the model. The frame source stays lazy (it's
        cheap, and we avoid holding the camera open until a visual query actually needs it)."""
        self._get_vlm()

    def maybe_answer(self, transcript: str) -> Optional[str]:
        """Return a spoken answer if *transcript* is a visual query, else ``None``.

        Lazy-loads the VLM and frame source on the first visual query. Never
        raises on capture/inference failure -- returns a brief spoken apology
        instead so the turn loop is robust.
        """
        if not is_visual_query(transcript):
            return None
        return vision_reply(
            transcript,
            vlm=self._get_vlm(),
            frame_source=self._get_frame_source(),
            max_tokens=self.max_tokens,
        )
