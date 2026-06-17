"""Real-time conversation pipeline: streaming STT -> LLM -> TTS with overlap.

Wires the audio front-end (:mod:`reachy_chat.audio`) to the streaming engines
(:mod:`reachy_chat.pipeline.engines`) and (optionally) the robot
(:mod:`reachy_chat.robot`). Proven offline at <500 ms first-audio (EXP-E2E-1).
"""

from .engines import LLMEngine, STTEngine, TTSEngine

__all__ = ["STTEngine", "LLMEngine", "TTSEngine"]
