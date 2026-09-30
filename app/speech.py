"""Local speech.

Tutor voice: Qwen3-TTS on MLX (natural, ~0.2x real time once warm), falling back to macOS `say`.
Learner voice: faster-whisper, primed to keep the learner's mistakes instead of fixing them.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import voice_settings

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", str(Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"))
QWEN_TTS_MODEL = os.environ.get("QWEN_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit")
QWEN_INSTRUCT = "Speak clearly and naturally, like a friendly native speaker chatting with an English learner."
# Qwen3-TTS CustomVoice speakers whose native language is English.
QWEN_SPEAKERS = {"qwen:aiden": ("aiden", "Aiden (자연스러운 AI 음성)"), "qwen:ryan": ("ryan", "Ryan (자연스러운 AI 음성)")}
# Natural-sounding English voices that ship with macOS; only installed ones are offered.
SAY_PREFERRED = ["Samantha", "Ava", "Zoe", "Allison", "Evan", "Nathan", "Tom", "Susan", "Karen", "Daniel", "Moira", "Tessa", "Sandy", "Shelley", "Reed", "Flo"]
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "qwen:aiden")
SAY_RATE = 175
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto")  # auto | qwen | say

# Priming Whisper with a disfluent learner-style sentence makes it transcribe what was
# actually said (including errors and fillers) instead of silently "fixing" the grammar.
WHISPER_PROMPT = "Um, yesterday I go to the office and, uh, my boss say me the project is delay. I has many work."

_whisper: Any = None
_qwen: Any = None
_qwen_failed = False
_qwen_lock = threading.Lock()
_say_voices: list[str] | None = None


# ---------------------------------------------------------------- voices

def say_voices() -> list[str]:
    global _say_voices
    if _say_voices is None:
        try:
            out = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        english = set()
        for line in out.splitlines():
            match = re.match(r"^(.+?)\s+(en_[A-Z]{2})\s+#", line)
            if match:
                english.add(re.sub(r"\s*\(.*\)$", "", match.group(1)).strip())
        _say_voices = [v for v in SAY_PREFERRED if v in english]
    return _say_voices


def qwen_available() -> bool:
    if TTS_ENGINE == "say" or _qwen_failed:
        return False
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        return False
    return True


def voices() -> list[dict[str, str]]:
    out = [{"id": vid, "label": label} for vid, (_, label) in QWEN_SPEAKERS.items()] if qwen_available() else []
    return out + [{"id": f"say:{v}", "label": f"{v} (macOS)"} for v in say_voices()]


def resolve_voice(voice: str | None) -> str:
    ids = [v["id"] for v in voices()]
    if voice in ids:
        return voice
    if DEFAULT_VOICE in ids:
        return DEFAULT_VOICE
    return ids[0] if ids else "say:Samantha"


# ---------------------------------------------------------------- synthesis

def build_speech_command(text: str, output_path: str | Path, rate: int = SAY_RATE, voice: str = "Samantha") -> list[str]:
    return ["/usr/bin/say", "-v", voice, "-r", str(rate), "--file-format=WAVE", "--data-format=LEI16@22050", "-o", str(output_path), text]


def say_wav(text: str, voice: str) -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        wav = Path(temp_dir) / "speech.wav"
        subprocess.run(build_speech_command(text, wav, SAY_RATE, voice), check=True, timeout=40, capture_output=True)
        return wav.read_bytes()


def pcm_wav(samples: Any, sample_rate: int) -> bytes:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def load_qwen_tts_model() -> Any:
    global _qwen
    if _qwen is None:
        from mlx_audio.tts.utils import load_model

        _qwen = load_model(QWEN_TTS_MODEL)
    return _qwen


def qwen_wav(text: str, speaker: str, delivery: Any = None) -> bytes:
    import numpy as np

    with _qwen_lock:  # MLX generation is not safe to run concurrently on one model
        model = load_qwen_tts_model()
        chunks, sample_rate = [], 24000
        instruct = voice_settings.instruction(delivery) if delivery is not None else QWEN_INSTRUCT
        for result in model.generate_custom_voice(text=text, speaker=speaker, language="English", instruct=instruct):
            chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
    if not chunks:
        raise RuntimeError("Qwen3-TTS returned no audio")
    return pcm_wav(np.concatenate(chunks), sample_rate)


def synthesize_wav(text: str, voice: str, delivery: Any = None) -> bytes:
    global _qwen_failed
    voice = resolve_voice(voice)
    if voice in QWEN_SPEAKERS:
        try:
            return qwen_wav(text, QWEN_SPEAKERS[voice][0], delivery)
        except Exception as exc:  # keep the tutor audible even if the MLX model breaks
            print(f"Qwen3-TTS failed, falling back to macOS say: {type(exc).__name__}: {exc}")
            _qwen_failed = True
            voice = resolve_voice(None)
    return say_wav(text, voice.removeprefix("say:"))


def warm_up() -> None:
    """Load models in the background so the first reply is not delayed."""
    def run() -> None:
        global _qwen_failed
        if qwen_available():
            try:
                qwen_wav("Hi there!", QWEN_SPEAKERS["qwen:aiden"][0])
            except Exception as exc:
                print(f"Qwen3-TTS unavailable, using macOS say: {type(exc).__name__}: {exc}")
                _qwen_failed = True
        try:
            transcriber()
        except Exception as exc:
            print(f"Whisper warm-up failed: {exc}")

    threading.Thread(target=run, name="speech-warmup", daemon=True).start()


# ---------------------------------------------------------------- transcription

def transcriber() -> Any:
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def transcribe_file(path: str) -> dict[str, Any]:
    segments, info = transcriber().transcribe(
        path,
        language="en",
        beam_size=5,
        vad_filter=True,
        initial_prompt=WHISPER_PROMPT,
        condition_on_previous_text=False,
    )
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    # Speaking time excludes leading/trailing silence, so words-per-minute reflects fluency.
    speaking = (segments[-1].end - segments[0].start) if segments else 0.0
    return {"text": text, "duration": info.duration, "speaking_seconds": round(max(speaking, 0.0), 2)}
