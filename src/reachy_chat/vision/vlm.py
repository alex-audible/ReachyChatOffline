"""Gemma 4 E4B QAT multimodal vision-language wrapper (via mlx-vlm).

This is the "vision path" for ReachyChatOffline: a single webcam frame plus a
text question go in, the model's text answer comes out, so "what Reachy sees"
can enter the conversation.

Model: ``mlx-community/gemma-4-E4B-it-qat-4bit`` (QAT int4, ~3 GB resident,
image + text input, text out). See ``docs/research/01-llm-audio-native.md``.

The mlx-vlm 0.6.x API used here (verified against the installed package and its
own ``chat.py``):

    from mlx_vlm import load, stream_generate, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load("mlx-community/gemma-4-E4B-it-qat-4bit")
    prompt = apply_chat_template(
        processor, model.config, "What do you see?", num_images=1
    )
    # image may be a PIL.Image, a path/URL string, or a list of those.
    for chunk in stream_generate(model, processor, prompt, image=pil_image):
        ...

``process_image`` inside mlx-vlm accepts a ``PIL.Image`` directly (it checks
``hasattr(img, "mode")``), so we can pass an in-memory frame without writing it
to disk -- important for the latency budget.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator, Optional

from .frame_source import ImageLike, to_pil

if TYPE_CHECKING:
    from PIL import Image as PILImage

DEFAULT_MODEL = "mlx-community/gemma-4-E4B-it-qat-4bit"

# Conservative default: a webcam "what do you see" answer is a sentence or two.
DEFAULT_MAX_TOKENS = 200


class VLM:
    """Loaded Gemma 4 multimodal model, reusable across many ``describe`` calls.

    Loading is expensive (weights into unified memory), so construct one ``VLM``
    and reuse it. The class is not thread-safe; serialize calls from a single
    worker (the assistant runs inference serially anyway).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, lazy: bool = False) -> None:
        self.model_name = model_name
        # Imported here so that importing this module is cheap and does not
        # touch Metal until a model is actually requested.
        from mlx_vlm import load

        self.model, self.processor = load(model_name, lazy=lazy)
        self.config = self.model.config

    def _build_prompt(self, prompt: str, num_images: int) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template

        return apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=num_images,
        )

    def describe(
        self,
        image: ImageLike,
        prompt: str = "What do you see?",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        **gen_kwargs,
    ) -> str:
        """Run image + text -> text and return the full answer string.

        ``image`` may be a ``PIL.Image``, a ``numpy`` RGB uint8 array (a camera
        frame), a file path / URL / data-URI string, or raw image ``bytes``.
        """
        from mlx_vlm import generate

        pil = to_pil(image)
        full_prompt = self._build_prompt(prompt, num_images=1)
        result = generate(
            self.model,
            self.processor,
            full_prompt,
            image=pil,
            max_tokens=max_tokens,
            temperature=temperature,
            **gen_kwargs,
        )
        # generate() returns a GenerationResult whose .text is the answer.
        return getattr(result, "text", str(result)).strip()

    def stream_describe(
        self,
        image: ImageLike,
        prompt: str = "What do you see?",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        **gen_kwargs,
    ) -> Iterator[str]:
        """Like :meth:`describe` but yields text chunks as they are generated.

        Useful later for streaming the vision answer into a TTS pipeline.
        """
        from mlx_vlm import stream_generate

        pil = to_pil(image)
        full_prompt = self._build_prompt(prompt, num_images=1)
        for chunk in stream_generate(
            self.model,
            self.processor,
            full_prompt,
            image=pil,
            max_tokens=max_tokens,
            temperature=temperature,
            **gen_kwargs,
        ):
            yield chunk.text


# Module-level convenience: a cached single instance so callers can do a
# one-liner ``describe(image, prompt)`` without managing the model object.
_DEFAULT_VLM: Optional[VLM] = None


def get_default_vlm(model_name: str = DEFAULT_MODEL) -> VLM:
    """Return a process-wide cached :class:`VLM` (loads on first call)."""
    global _DEFAULT_VLM
    if _DEFAULT_VLM is None or _DEFAULT_VLM.model_name != model_name:
        _DEFAULT_VLM = VLM(model_name)
    return _DEFAULT_VLM


def describe(
    image: ImageLike,
    prompt: str = "What do you see?",
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **gen_kwargs,
) -> str:
    """One-shot convenience wrapper around the cached default :class:`VLM`.

    Takes a PIL image / numpy array / path + a text prompt, returns the model's
    text answer. The first call loads the model (slow); subsequent calls reuse
    it.
    """
    return get_default_vlm(model_name).describe(
        image, prompt, max_tokens=max_tokens, **gen_kwargs
    )


def describe_timed(
    image: ImageLike,
    prompt: str = "What do you see?",
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **gen_kwargs,
) -> tuple[str, float]:
    """Like :func:`describe` but also returns wall-clock seconds for the call.

    Intended only for rough sanity timing -- the project's main thread does
    careful latency measurement separately.
    """
    vlm = get_default_vlm(model_name)
    start = time.perf_counter()
    answer = vlm.describe(image, prompt, max_tokens=max_tokens, **gen_kwargs)
    elapsed = time.perf_counter() - start
    return answer, elapsed
