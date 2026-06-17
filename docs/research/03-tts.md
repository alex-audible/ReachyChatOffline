# 03 — Local TTS Engine Survey (June 2026)

> Scope: fully-local, English-only, low-latency voice-to-voice for Reachy Mini on Apple
> Silicon (M3 / 24GB, working set ≲20GB). Prefer MLX via the `mlx-audio` pip package
> (Blaizzy) or MPS. Target end-to-end voice-to-voice ≤500ms, so TTS must STREAM with
> low time-to-first-audio (TTFA, ideally <150ms). HARD REQUIREMENT: natural &
> emotional/expressive speech. We want both a latency-first pick AND a quality-max pick.

## 0. Ground truth: what `mlx-audio` actually supports

Verified directly against the installed package (`mlx-audio==0.4.4`, cached under
`~/.cache/uv/.../mlx_audio`). The `mlx_audio/tts/models/` directory IS the list of
natively-supported architectures, and `mlx_audio/tts/utils.py::MODEL_REMAPPING` plus the
docs/example repo IDs confirm the canonical MLX weights.

**TTS architectures with a native model dir in mlx-audio 0.4.4:**
`bark`, `chatterbox`, `chatterbox_turbo`, `dia`, `dramabox`, `echo_tts`,
`fish_qwen3_omni`, `higgs_audio`, `higgs_audio_v3`, `indextts`, `irodori_tts`,
`kitten_tts`, `kokoro`, `kugelaudio`, `llama` (Orpheus / generic LLaMA-codec backbone),
`longcat_audiodit`, `melotts`, `moss_tts*` (delay/local/nano), `omnivoice`, `outetts`,
`pocket_tts`, `qwen3` / `qwen3_tts`, `sesame` (CSM / Marvis), `soprano`, `spark`,
`tada`, `vibevoice`, `voxcpm` / `voxcpm2`, `voxtral_tts`, `bailingmm`.

**NOT natively in mlx-audio 0.4.4** (would need PyTorch/MPS or a separate port):
Kyutai TTS / Moshi (use `moshi_mlx` package instead), **XTTS-v2** (Coqui, PyTorch/MPS
only), **Parler-TTS** (PyTorch only), **F5-TTS** (separate `f5-tts-mlx` community port,
not in mlx-audio core), **Piper** (own ONNX runtime, not MLX).

Confirmed canonical MLX repo IDs (from package source / examples):
`mlx-community/kokoro-tts` (also `prince-canuma/Kokoro-82M`),
`mlx-community/orpheus-3b-0.1-ft-bf16` (+ `mlx-community/snac_24khz` codec),
`sesame/csm-1b` (Marvis/sesame loader), `mlx-community/Chatterbox-TTS-fp16`,
`mlx-community/Dia-1.6B`, `mlx-community/higgs-audio-v2-3B-mlx-q8` (and `-q6`),
`SparkAudio/Spark-TTS-0.5B`, `OuteAI/Llama-OuteTTS-1.0-1B`,
`OpenBMB/VoxCPM` / `mlx-community/VoxCPM2-bf16`, `mlx-community/fish-audio-s2-pro`,
`mlx-community/IndexTTS`, `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16`,
`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`.

## 1. Comparison table

MLX column = supported by `mlx-audio==0.4.4` natively (✅), via a separate MLX port (◑),
or PyTorch/MPS-only / not MLX (❌). TTFA figures are GPU-class references; on M3 expect
small models (Kokoro/Piper/Kitten) to stay sub-150ms and 1–3B AR models (Orpheus, CSM,
Higgs) to land roughly 150–400ms once warm. The coordinator is measuring real M3 numbers.

| Engine | HF repo ID (canonical) | Params | mlx-audio (0.4.4)? | Streaming / TTFA | Naturalness / emotiveness | Expressivity controls | License (commercial?) |
|---|---|---|---|---|---|---|---|
| **Kokoro** | `hexgrad/Kokoro-82M` · MLX `prince-canuma/Kokoro-82M` / `mlx-community/kokoro-tts` | 82M | ✅ native (`kokoro`), gapless streaming | Yes; ~28–45ms first-audio GPU, RTF~0.03; sub-150ms on M3 | Very clean & natural; **flat/neutral affect**, weak emotion | Fixed voice packs, voice blending; **no emotion tags, no cloning** | Apache-2.0 ✅ |
| **Orpheus-TTS** | `canopylabs/orpheus-3b-0.1-ft` · MLX `mlx-community/orpheus-3b-0.1-ft-bf16` (+`mlx-community/snac_24khz`) | 3B (Llama) | ✅ native (`llama`+SNAC, `decode_stream`) | Yes; ~100–200ms streaming TTFA | **Top-tier human-like; strong emotion** | **8 emotion tags** (happy/sad/angry/fear/surprise/disgust/excite/neutral) + `<laugh>` etc.; cloning unreliable per code warning | Apache-2.0 ✅ |
| **Kyutai TTS / Moshi** | `kyutai/tts-1.6b-en_fr` / `kyutai/moshiko-mlx-bf16` | 1.6B / 7B | ◑ via `moshi_mlx` (not mlx-audio) | Yes; designed streaming, ~few-hundred-ms TTFA | Natural, conversational; moderate emotion | Voice embeddings; limited explicit emotion tags | CC-BY-4.0 ✅ |
| **Sesame CSM-1B** | `sesame/csm-1b` | 1B | ✅ native (`sesame`/`csm`/`marvis`) | Yes (frame-streaming) | Very natural conversational prosody; context-driven emotion | Context/reference audio conditioning; no explicit tags | Apache-2.0 ✅ |
| **Chatterbox (Resemble)** | `ResembleAI/chatterbox` · Turbo `ResembleAI/chatterbox-turbo` · MLX `mlx-community/Chatterbox-TTS-fp16`, `chatterbox_turbo` | ~0.5B | ✅ native (`chatterbox`, `chatterbox_turbo`) | Yes; low TTFA (turbo faster) | **Wins blind tests vs ElevenLabs (65% vs 24%)**; very natural, configurable expressiveness | **Exaggeration/intensity knob**, zero-shot voice cloning | **MIT** ✅ (esp. Turbo) |
| **XTTS-v2** | `coqui/XTTS-v2` | ~0.5B | ❌ PyTorch/MPS only | Partial streaming; ~200–400ms | Good, expressive, strong cloning | Multilingual, zero-shot cloning, some style | **Coqui CPML — non-commercial** ⚠️ |
| **Piper** | `rhasspy/piper-voices` | ~10–30M/voice | ❌ ONNX (not MLX) | Yes; extremely low TTFA (<50ms) | Robotic/utility; **low emotion** | Per-voice only; none | MIT ✅ |
| **Parler-TTS** | `parler-tts/parler-tts-large-v1` | ~880M | ❌ PyTorch only | Limited | Decent; **text-prompted style** | Natural-language style prompt (e.g. "fast, expressive") | Apache-2.0 ✅ |
| **F5-TTS** | `SWivid/F5-TTS` · MLX `lucasnewman/f5-tts-mlx` | ~330M | ◑ separate `f5-tts-mlx` port (not core) | Diffusion; higher TTFA, not great for streaming | Very natural; emotion via reference clip | Zero-shot cloning via reference audio | MIT (code) / CC-BY-NC data ⚠️ |
| **Dia (Nari)** | `nari-labs/Dia-1.6B` · MLX `mlx-community/Dia-1.6B` | 1.6B | ✅ native (`dia`) | Weak streaming; full-utterance gen | **Highly expressive/dramatic**, dialogue + nonverbals | `[S1]/[S2]` speakers, `(laughs)`/`(sighs)` nonverbal tags | Apache-2.0 ✅ |
| **Fish-Speech / OpenAudio** | `fishaudio/openaudio-s1-mini` · MLX `mlx-community/fish-audio-s2-pro` | ~0.5–4B | ✅ native (`fish_qwen3_omni`) | Yes; moderate TTFA | Natural; good emotion (S1 emotion markers) | Emotion/tone markers, cloning | CC-BY-NC (open weights) ⚠️ / Apache code |
| **Higgs Audio v2** | `bosonai/higgs-audio-v2-generation-3B-base` · MLX `mlx-community/higgs-audio-v2-3B-mlx-q8` | 3B | ✅ native (`higgs_audio`, `higgs_audio_v3`) | Yes; ~200–400ms | **Wins emotion/question blind scores**; very natural | Scene/emotion prompting, multi-speaker, cloning | Apache-2.0 (base) ✅ |
| **CosyVoice 2** | `FunAudioLLM/CosyVoice2-0.5B` · MLX `mlx-community/CosyVoice2-0.5B-S3Tokenizer` (codec) | 0.5B | ◑ tokenizer in pkg; full model community/PyTorch | Yes; low-latency streaming design | Top-3 community pick; natural & expressive | Instruction/emotion control, cloning | Apache-2.0 ✅ |
| **VoxCPM / VoxCPM2** | `OpenBMB/VoxCPM` · `openbmb/VoxCPM2` · MLX `mlx-community/VoxCPM2-bf16` | ~0.5B | ✅ native (`voxcpm`,`voxcpm2`) | Yes; low TTFA | Natural, expressive, good prosody | Voice cloning, prosody control | Apache-2.0 ✅ |
| **Qwen3-TTS** | `Qwen/Qwen3-TTS` · MLX `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | 0.6B/1.7B | ✅ native (`qwen3_tts`) | Yes; 12.5Hz codec, streaming | Natural; custom/voice-design variants | VoiceDesign + CustomVoice variants | Apache-2.0 ✅ |
| **IndexTTS / Spark / OuteTTS / Bark / MeloTTS / Voxtral-TTS / Irodori / Kitten / VibeVoice** | see §0 repo IDs | 0.1–4B | ✅ native | varies | varies (Bark expressive but slow; Kitten tiny/fast; VibeVoice long-form) | varies | mostly Apache/MIT (Bark MIT) ✅ |

## 2. r/LocalLLaMA sentiment (June 2026)

Synthesized from current community/round-up discussion (full Reddit threads were
rate-limited during this time-boxed pass; the consensus below is corroborated by
multiple 2026 round-ups citing r/LocalLLaMA blind tests):

- **Chatterbox-Turbo (Resemble, MIT)** is the headline "natural" pick: blind test
  **65.3% preferred Chatterbox-Turbo vs 24.5% ElevenLabs, 10.2% neutral**. Praised as
  "incredibly natural" with a configurable exaggeration/expressiveness knob + zero-shot
  cloning. Now a community top-3.
- **Higgs Audio v2 (BosonAI, 3B)** — "top trending TTS on Hugging Face"; community notes
  it **wins audience scores on emulating emotion and question-asking**. The go-to when
  the priority is emotional realism over latency.
- **Orpheus (Canopy, 3B Llama)** — repeatedly cited for **human-like intonation + simple
  emotion tags** and real-time SNAC streaming (~100–200ms). The favorite for an
  expressive *streaming* voice-agent stack.
- **Kokoro (82M)** — universally praised for speed/cleanliness and "#1 on TTS Arena",
  but the recurring caveat is **flat/neutral affect** — "fine for informational content,
  not character dialogue or dramatic narration."
- **CosyVoice2-0.5B** — recurring top-3 contender for natural+expressive streaming.
- Mood quote: a Redditor summarized the recent wave as *"the on-prem voice stack is
  here,"* reflecting the March–April 2026 burst of high-quality open TTS releases.

(Sources: bentoml, modal, findskill, ocdevel, murmurtts, reviewnexa round-ups citing
r/LocalLLaMA blind tests — June 2026.)

## 3. Recommendation

The hard tension: the ≤500ms voice-to-voice budget wants Kokoro-class TTFA, but the
"very natural & emotional" hard requirement is exactly where Kokoro is weakest. The
resolution is a **two-tier pick** plus a benchmark shortlist.

### (a) Latency-first pick — **Kokoro-82M**
- `prince-canuma/Kokoro-82M` (MLX: `mlx-community/kokoro-tts`), native in mlx-audio,
  gapless streaming, RTF ~0.03, sub-150ms TTFA easily achievable on M3, ~few-hundred-MB
  footprint. Apache-2.0.
- Caveat: flat affect. Acceptable only if expressivity can be sacrificed for raw latency,
  or as a fallback voice. If emotion is non-negotiable even at tier-1, substitute
  **Chatterbox-Turbo** as the latency-first pick (still small, MIT, expressive).

### (b) Quality-max pick — **Orpheus-3B** (primary) / **Higgs Audio v2** (alt)
- **Orpheus** `mlx-community/orpheus-3b-0.1-ft-bf16` (+`mlx-community/snac_24khz`): native
  in mlx-audio with real streaming (`decode_stream`), 8 explicit emotion tags, top-tier
  naturalness, ~100–200ms streaming TTFA — the best *expressive AND streamable* option.
  ~3B → roughly 3–6GB quantized, fits the ≲20GB working set alongside STT+LLM.
- **Higgs Audio v2** `mlx-community/higgs-audio-v2-3B-mlx-q8` if blind-test emotion wins
  matter most and slightly higher TTFA is acceptable.

### (c) Shortlist to benchmark on M3 (exact repo IDs + mlx-audio support + TTFA)

| Rank | Engine | Benchmark repo ID | mlx-audio 0.4.4 | Expected TTFA on M3 | Why |
|---|---|---|---|---|---|
| 1 | Kokoro-82M | `mlx-community/kokoro-tts` (`prince-canuma/Kokoro-82M`) | ✅ native | **<100ms** | latency floor / fallback |
| 2 | Chatterbox-Turbo | `mlx-community/Chatterbox-TTS-fp16` (`chatterbox_turbo`) | ✅ native | ~120–250ms | best natural+small+MIT, cloning |
| 3 | Orpheus-3B | `mlx-community/orpheus-3b-0.1-ft-bf16` (+`mlx-community/snac_24khz`) | ✅ native (streaming) | ~150–300ms | best expressive+streamable, emotion tags |
| 4 | Higgs Audio v2 | `mlx-community/higgs-audio-v2-3B-mlx-q8` | ✅ native | ~250–400ms | emotion-win, quality ceiling |
| 5 | Sesame CSM-1B | `sesame/csm-1b` | ✅ native (`sesame`) | ~200–350ms | conversational prosody, frame-streaming |

Optional 6th: **VoxCPM2** `mlx-community/VoxCPM2-bf16` (native, small, expressive) if a
smaller-than-3B expressive alternative is wanted.

**Bottom line:** ship a two-engine setup — **Kokoro** (or Chatterbox-Turbo) for the
latency-critical path and **Orpheus-3B** for the natural/emotional path — and let the M3
benchmark decide whether Orpheus's TTFA fits inside the 500ms voice-to-voice budget. All
five shortlist engines are natively supported by `mlx-audio==0.4.4`, so no PyTorch/MPS
fallback is needed for the recommended path.

## Sources
- https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models
- https://modal.com/blog/open-source-tts
- https://findskill.ai/blog/best-open-source-tts-2026/
- https://www.murmurtts.com/blog/best-local-tts-models-2026
- https://reviewnexa.com/kokoro-tts-review/
- https://blaizzy.github.io/mlx-audio/ · https://github.com/Blaizzy/mlx-audio · https://pypi.org/project/mlx-audio/
- https://bitbasti.com/blog/audio-streaming-with-orpheus · https://github.com/canopyai/Orpheus-TTS
- Ground truth: installed `mlx-audio==0.4.4` package source (`mlx_audio/tts/models/`, `utils.py`)
