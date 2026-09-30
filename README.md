# Hey Alex — Local English Speaking Coach

A private, mobile-first English speaking coach for Korean learners. Everything (conversation, error analysis, speech recognition, tutor voice) runs on this Mac.

The core loop is **talk → get corrected → find your patterns → fix them**:

1. **Talk.** Alex (the tutor) replies like a friend and never lectures. When you make a mistake, Alex casually reuses the correct form (recast).
2. **Corrections, right away.** A separate analysis pass marks each mistake in your sentence, with the fix, a one-line Korean rule, a 🔊 listen button, and 🎙 *say it again*, which checks your retry on the spot. Korean words you had to fall back on become "영어 표현" cards.
3. **Weak-point model.** Every correction is logged under a fixed category (시제, 관사, 전치사, 동사 패턴 …). Categories are scored with recency weighting. Correct uses of a weak category lower its score.
4. **Steering.** The top weak categories become the next session's *focus*. Alex is told to ask questions that force those structures (e.g. past events for 시제), without saying so. There is also a dedicated "약점 집중 훈련" topic.
5. **Spaced review.** Each correction becomes a review card (Leitner boxes: 1 → 3 → 7 → 14 → 30 → 60 days). You answer by voice or typing, and it is checked locally at the word level. Making the same mistake again in conversation resets the card.
6. **Growth.** Errors per 100 words per session, per-category trends (recent 15 turns vs the 15 before), streak, and words-per-minute from your recordings.
7. **Visual context.** A 9:16 wallpaper fills the chat view and remains fixed while messages scroll. Alex’s messages use a profile photo, sender name and white speech bubble; consecutive Alex messages show the photo and name only on the first message. The learner’s bubbles are yellow and aligned to the right. Existing scenes appear immediately; missing places and concrete subjects are generated asynchronously with the local Qwen Image 2.1 model and cached. Short continuations retain the previous background.

## Local AI

| Part | Engine | Notes |
|---|---|---|
| Conversation | Ollama `qwen3.5:4b-mlx` (`LLM_MODEL`) | Plain-text reply, ~0.5–2 s |
| Error analysis | same model (`ANALYSIS_MODEL`) | JSON-schema constrained output (category enum), then filtered in code: drops capitalization/punctuation nitpicks, quotes the learner never said, and duplicates |
| Speech → text | faster-whisper small, CPU int8 | Primed with a disfluent prompt so it keeps your grammar mistakes instead of silently fixing them |
| Tutor voice | Qwen3-TTS 1.7B CustomVoice (MLX), speakers Aiden/Ryan | ~0.2× real time once warm; falls back to macOS `say` (Samantha etc.) |
| Conversation images | Local Qwen Image 2.1 via ComfyUI, plus a reusable scene library | fixed portrait, topic-aware backgrounds, asynchronous generation and disk cache |

The 27B model was measured at 100–200 s per turn on this machine, too slow for conversation. Set `ANALYSIS_MODEL` if a faster large model becomes available.

The tutor's previous line is intentionally *not* given to the analysis model: with it, the 4B model starts "correcting" facts (yesterday → last year) instead of English.

## Run

```sh
uv sync
APP_HOST=<tailscale-ip> uv run python app/server.py   # Tailscale listener
APP_HOST=127.0.0.1      uv run python app/server.py   # loopback listener
uv run python -m pytest
```

Open `http://<mac-hostname>:18765` from a device on the tailnet. Microphone access needs a secure context (HTTPS, e.g. `tailscale serve`, or localhost). On iPhone, "홈 화면에 추가" gives a full-screen app.

Environment: `LLM_BASE_URL`, `LLM_MODEL`, `ANALYSIS_MODEL`, `TTS_ENGINE` (`auto`|`qwen`|`say`), `TTS_VOICE` (e.g. `qwen:aiden`, `say:Samantha`), `QWEN_TTS_MODEL`, `WHISPER_MODEL`, `APP_HOST`, `PORT`, `COACH_DB`.

## Code map

- `app/coach.py`: prompts, LLM calls, analysis sanitizing, retry/answer checking
- `app/store.py`: SQLite schema and migration, weak-point scoring, progress stats, Leitner review
- `app/speech.py`: Qwen3-TTS / `say` synthesis, Whisper transcription
- `app/images.py`: scene selection, conversation context, background jobs and cache
- `app/local_images.py`: local ComfyUI startup and Qwen Image 2.1 generation
- `app/server.py`: Flask routes. Turns stream over SSE: `image`, `reply`, then `analysis`
- `app/static/index.html`: the whole mobile UI (no build step)

## Privacy

The app listens only on loopback and the Tailscale IP. It has no login, so access depends on the tailnet policy; do not expose the port publicly. Chat, corrections, and review cards are stored in `coach.sqlite3`.

## Known limits

- No acoustic pronunciation scoring: feedback is based on the transcript only.
- The 4B model still occasionally over-corrects or mislabels a category. Every correction card can be judged on retry ("맞게 말했는데 잘못 들렸어요" overrides a misheard answer).
- New 432×768 (9:16) wallpapers took about 25 s each with the local model loaded in the latest run. The current picture remains visible during generation; reply and correction streaming do not wait for the image. GPU work may still compete with local speech/LLM inference.
- Alex uses the same portrait for each message group so identity stays exact. The portrait is a still image, without lip synchronization.

## Local image generation

The repository includes a Qwen-generated Alex portrait and six 432×768 wallpapers: home, cafe, forest, Kyoto, workspace and Rome. They are stored in `app/static/wallpapers/`. ComfyUI is accessed only by the Python backend. The default endpoint is `http://127.0.0.1:8188`; if unavailable, the backend starts the existing installation at `~/ComfyUI-standalone` with its own Python environment. No models are downloaded automatically. Set `COMFYUI_DIR` for another installation, `IMAGE_BASE_URL` for an already running server, or `IMAGE_AUTOSTART=0` to disable automatic startup. Server output is written to `image-model.log`.

Required ComfyUI model filenames (or override with `IMAGE_MODEL`, `IMAGE_TEXT_ENCODER`, `IMAGE_VAE`):

- `qwen_image_2.1_int8_convrot.safetensors`
- `qwen3vl_8b_int8_convrot.safetensors`
- `qwen_image_2.1_vae_bf16.safetensors`

Generated 9:16 PNGs and context manifests live in `app/static/generated/wallpapers/` and are reused across sessions and server restarts. Rome, Paris, London, Seoul, Tokyo, Kyoto and New York are recognized explicitly. Longer messages and unfamiliar important keywords outside the scene vocabulary are interpreted by the local conversation model in a background job. Pending results are polled through `/api/images/jobs/<key>`; errors keep the displayed picture. The browser discards outdated results after a newer topic update or session change. `IMAGE_TIMEOUT` controls the ComfyUI job deadline (default 240 seconds).

## Short-answer speaking guide

After corrections stream, replies of 1–7 English words can receive a separate optional example: “조금 더 길게 이렇게 말해보면 어때요?” The local language model uses the previous tutor question and the learner’s level to expand the answer while preserving its meaning. Personal details that the learner has not supplied are shown as `[placeholders]`. Greetings and short questions do not trigger this guide. Examples can be copied to the input for practice; examples without placeholders also have a listening button. Guides are saved with the turn and do not count as mistakes or create review cards.

Corrections are presented one at a time: the highest-priority correction is visible, while other corrections and the full-sentence rewrite are collapsed under “다른 교정 N개 보기”. Only visible corrections are highlighted in the learner’s message. Expanding the section shows the remaining explanations and practice controls.

The composer opens in voice mode on every session and reload. The main microphone stays centered, and an icon-only keyboard button on the left expands the optional text input above the voice controls. Sending typed text returns to the voice layout.
