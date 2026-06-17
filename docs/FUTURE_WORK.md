# Things to Try in the Future

Deferred experiments and improvements, captured so they aren't lost. Ordered roughly by
value-to-effort. None block the current working pipeline.

## Emotive TTS (the open quality gap)

The hard constraint on M3: an emotive TTS must run **faster than real-time (RTF < 1)** to stream
without underrun. Today only Kokoro clears that bar, but it's flat. Candidates that are expressive
but currently too slow / not integrated:

1. **Quantize Sesame CSM-1B to int4/int8 (MLX).** CSM is regarded as very natural/emotive, but at
   fp16 on M3 it's **RTF ~1.18** (too slow to stream). Quantizing (e.g. `mlx_audio`/`mlx_lm`
   convert to 4-bit) should push RTF < 1. Repos: `mlx-community/csm-1b`, `mlx-community/csm-1b-fp16`
   (access now ungated). **Highest-value future item** — would give an emotive option.
2. **Chatterbox-Turbo** — `mlx-community/chatterbox-turbo-mlx-q4` is a **dead end** with the
   installed mlx-audio: the q4 T3 stage loads and is fast (RTF 0.38), but the repo's **S3Gen
   vocoder weights use ResembleAI's original module layout — only ~41% match mlx-audio's S3Gen**,
   so the CFM decoder stays at random init → near-silence. (Turbo also ignores
   `exaggeration`/`cfg_weight`; `emotion_adv: false`.) Wrapper at `src/reachy_chat/tts/chatterbox.py`
   documents this and raises `ChatterboxTurboUnavailable`. **To revive:** re-convert
   ResembleAI/chatterbox-turbo with the *installed* mlx-audio converter (matching S3Gen layout), or
   find a non-q4 chatterbox-turbo MLX repo built against current mlx-audio. Still the top blind-test
   engine per research, so worth a fresh conversion.
3. **Higgs Audio v2 / VibeVoice / F5-TTS** — not yet benchmarked; evaluate emotion vs RTF<1 on M3.
4. **Kokoro expressivity** — try its other voices (af_bella, af_nicole, am_michael, …) and
   sentence-level emphasis as a stopgap for more affect from the fast engine.

## Latency

5. **Speculative LLM start** on a stabilized STT partial (cancel-on-resume) — research-estimated
   ~150–250 ms saved on the full stop-talking→first-audio path. Endpointer already exposes an
   `on_speculative` hook; wire it into the pipeline. (Eagerness-gated Smart-Turn already added.)
6. **E4B decode speedup** — the `*-qat-4bit` build keeps MLP at 8-bit (~50 tok/s). Try a pure-4-bit
   or `mxfp4` E4B variant for higher tok/s (quantify the quality trade-off).
7. **Full-duplex Moshi** (`moshi_mlx`, ~250 ms raw on M4) as a documented "ultra-low-latency mode"
   Plan-B, accepting its fixed voice / weaker brain / no vision.

## STT / turn-taking

8. **Nemotron streaming 0.6B (MLX)** — research-stt's primary pick for true frame-by-frame partials;
   benchmark vs parakeet streaming finalize.
9. **Train a real "Hey Reachy" wake word** — `livekit-wakeword` is wired but currently uses an
   energy stub (needs the offline training toolchain to produce `hey_reachy.onnx`).
10. **Mac-side AEC for barge-in** (`pyaec`/SpeexDSP) — on the robot the XVF3800 does hardware AEC;
    dev on a Mac needs software AEC so the robot's own TTS doesn't self-interrupt.

## Vision

11. **Clean serial vision latency** measurement (image-prefill TTFT was ~0.8 s contended) and the
    on-robot camera path (`mini.media.get_frame()` / daemon stream) vs the file-based path.

## Audio-native LLM (revisit)

12. **Gemma 3n / Gemma 4 E-series audio input** as a STT-collapse for specific commands (not the
    main path — not Whisper-grade, 30 s cap, no streaming partials), if it ever gains streaming.
