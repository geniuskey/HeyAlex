# Hey Alex — Local English Speaking Coach

A private, mobile-first English speaking coach for Korean learners. Everything (conversation, error analysis, speech recognition, tutor voice) runs on this Mac.

The core loop is **talk → get corrected → find your patterns → fix them**:

1. **Talk.** Alex (the tutor) replies like a friend and never lectures. When you make a mistake, Alex casually reuses the correct form (recast).
2. **Corrections, right away.** A separate analysis pass marks each mistake in your sentence, with the fix, a one-line Korean rule, a 🔊 listen button, and 🎙 *say it again*, which checks your retry on the spot. Korean words you had to fall back on become "영어 표현" cards.
3. **Weak-point model.** Every correction is logged under a fixed category (시제, 관사, 전치사, 동사 패턴 …). Categories are scored with recency weighting. Correct uses of a weak category lower its score.
4. **Steering.** The top weak categories become the next session's *focus*. Alex is told to ask questions that force those structures (e.g. past events for 시제), without saying so. There is also a dedicated "약점 집중 훈련" topic.
5. **Spaced review.** Each correction becomes a review card (Leitner boxes: 1 → 3 → 7 → 14 → 30 → 60 days). You answer by voice or typing, and it is checked locally at the word level. Making the same mistake again in conversation resets the card.
6. **Growth.** Errors per 100 words per session, per-category trends (recent 15 turns vs the 15 before), streak, and words-per-minute from your recordings.

## Local AI

| Part | Engine | Notes |
|---|---|---|
| Conversation | Ollama `qwen3.5:4b-mlx` (`LLM_MODEL`) | Plain-text reply, ~0.5–2 s |
| Error analysis | same model (`ANALYSIS_MODEL`) | JSON-schema constrained output (category enum), then filtered in code: drops capitalization/punctuation nitpicks, quotes the learner never said, and duplicates |
| Speech → text | faster-whisper small, CPU int8 | Primed with a disfluent prompt so it keeps your grammar mistakes instead of silently fixing them |
| Tutor voice | Qwen3-TTS 1.7B CustomVoice (MLX), speakers Aiden/Ryan | ~0.2× real time once warm; falls back to macOS `say` (Samantha etc.) |

The 27B model was measured at 100–200 s per turn on this machine, too slow for conversation. Set `ANALYSIS_MODEL` if a faster large model becomes available.

The tutor's previous line is intentionally *not* given to the analysis model: with it, the 4B model starts "correcting" facts (yesterday → last year) instead of English.

## Run

```sh
cd /Users/edwin/workspaces/english-speaking-coach
uv sync
APP_HOST=100.89.133.22 uv run python app/server.py   # Tailscale listener
APP_HOST=127.0.0.1     uv run python app/server.py   # loopback listener
uv run pytest
```

Open `http://macstudio:18765` from a device on the tailnet. Microphone access needs a secure context (HTTPS, e.g. `tailscale serve`, or localhost). On iPhone, "홈 화면에 추가" gives a full-screen app.

Environment: `LLM_BASE_URL`, `LLM_MODEL`, `ANALYSIS_MODEL`, `TTS_ENGINE` (`auto`|`qwen`|`say`), `TTS_VOICE` (e.g. `qwen:aiden`, `say:Samantha`), `QWEN_TTS_MODEL`, `WHISPER_MODEL`, `APP_HOST`, `PORT`, `COACH_DB`.

## Code map

- `app/coach.py`: prompts, LLM calls, analysis sanitizing, retry/answer checking
- `app/store.py`: SQLite schema and migration, weak-point scoring, progress stats, Leitner review
- `app/speech.py`: Qwen3-TTS / `say` synthesis, Whisper transcription
- `app/server.py`: Flask routes. Turns stream over SSE: `reply` first, then `analysis`
- `app/static/index.html`: the whole mobile UI (no build step)

## Privacy

The app listens only on loopback and the Tailscale IP. It has no login, so access depends on the tailnet policy; do not expose the port publicly. Chat, corrections, and review cards are stored in `coach.sqlite3`.

## Known limits

- No acoustic pronunciation scoring: feedback is based on the transcript only.
- The 4B model still occasionally over-corrects or mislabels a category. Every correction card can be judged on retry ("맞게 말했는데 잘못 들렸어요" overrides a misheard answer).
