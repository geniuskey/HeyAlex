from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context

import coach
import images
import speech
import store

ROOT = Path(__file__).resolve().parent
MAX_TURN_CHARS = 2000
LLM_ERRORS = (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError)
app = Flask(__name__, static_folder="static")


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response(generator) -> Response:
    return Response(stream_with_context(generator), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    try:
        available = coach.LLM_MODEL in coach.list_models()
        return jsonify(ok=True, model=coach.LLM_MODEL, model_available=available)
    except Exception as exc:
        return jsonify(ok=False, model=coach.LLM_MODEL, error=str(exc)), 503


@app.get("/api/bootstrap")
def bootstrap():
    info = store.progress()
    return jsonify(
        topics=[{"id": k, "label": v["ko"]} for k, v in coach.TOPICS.items()],
        categories=coach.CATEGORIES,
        voices=speech.voices(),
        default_voice=speech.resolve_voice(None),
        focus=info["focus"],
        review=info["review"],
        streak=info["streak"],
        today_words=info["today_words"],
    )


# ---------------------------------------------------------------- conversation

@app.post("/api/sessions")
def start_session():
    body = request.get_json(silent=True) or {}
    topic = body.get("topic") if body.get("topic") in coach.TOPICS else "free"
    level = body.get("level") if body.get("level") in coach.LEVELS else "intermediate"
    focus = store.current_focus()
    try:
        opener = coach.generate_reply([], topic, level, focus)
    except LLM_ERRORS as exc:
        return jsonify(error=f"로컬 AI에 연결하지 못했어요: {exc}"), 503
    session_id = store.create_session(topic, level, opener)
    return jsonify(id=session_id, topic=topic, level=level, opener=opener, focus=focus,
                   image=images.conversation_image(opener if images.explicit_scene(opener) in images.PLACES else "", topic,
                                                  generate=not app.config.get("TESTING")))


@app.get("/api/sessions/<int:session_id>")
def session_detail(session_id: int):
    session = store.get_session(session_id)
    if session is None:
        return jsonify(error="Session not found"), 404
    context = [session["opener"]]
    session["image"] = images.conversation_image(session["opener"] if images.explicit_scene(session["opener"]) in images.PLACES else "", session["topic"], generate=False)
    for turn in session["turns"]:
        turn["image"] = images.conversation_image(turn["user_text"], session["topic"], context, generate=False)
        session["image"] = turn["image"]
        context.extend([turn["user_text"], turn["reply"]])
    if not app.config.get("TESTING"):
        if session["turns"]:
            latest = session["turns"][-1]
            session["image"] = images.conversation_image(latest["user_text"], session["topic"], context[:-2])
        else:
            session["image"] = images.conversation_image(session["opener"] if images.explicit_scene(session["opener"]) in images.PLACES else "", session["topic"])
    return jsonify(session)


@app.post("/api/sessions/<int:session_id>/turns")
def session_turn(session_id: int):
    session = store.get_session(session_id)
    if session is None:
        return jsonify(error="대화를 찾을 수 없어요. 새 대화를 시작해 주세요."), 404
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    if not text:
        return jsonify(error="먼저 말하거나 입력해 주세요."), 400
    if len(text) > MAX_TURN_CHARS:
        return jsonify(error=f"한 번에 {MAX_TURN_CHARS:,}자 이내로 말해 주세요."), 413
    audio_seconds = body.get("audio_seconds")
    audio_seconds = float(audio_seconds) if isinstance(audio_seconds, (int, float)) and not isinstance(audio_seconds, bool) and 0 < audio_seconds < 600 else None
    focus = store.current_focus()
    history = store.conversation_history(session_id) + [{"role": "user", "content": text}]

    def events():
        yield sse("started", {})
        yield sse("image", images.conversation_image(text, session["topic"],
                  [item["content"] for item in history[:-1]],
                  generate=bool(images.explicit_scene(text)) and not images.needs_scene_interpretation(text)
                  and not app.config.get("TESTING")))
        try:
            reply = coach.generate_reply(history, session["topic"], session["level"], focus)
        except LLM_ERRORS as exc:
            yield sse("error", {"error": f"로컬 AI가 답하지 못했어요: {exc}"})
            return
        turn_id = store.add_turn(session_id, text, reply, audio_seconds)
        yield sse("reply", {"turn_id": turn_id, "reply": reply})
        # Begin unfamiliar-topic generation as soon as the reply is sent,
        # rather than waiting for corrections and the speaking guide.
        if not app.config.get("TESTING") and images.needs_scene_interpretation(text):
            yield sse("image", images.conversation_image(text, session["topic"],
                      [item["content"] for item in history[:-1]]))
        try:
            analysis = store.save_analysis(turn_id, coach.analyze(text, focus))
        except LLM_ERRORS as exc:
            yield sse("analysis_error", {"turn_id": turn_id, "error": f"교정 분석에 실패했어요: {exc}"})
            return
        yield sse("analysis", {"turn_id": turn_id, **analysis})
        previous_question = next((item["content"] for item in reversed(history[:-1]) if item["role"] == "assistant"), "")
        guide = coach.short_answer_guide(text, previous_question, session["level"])
        if guide:
            store.save_guide(turn_id, guide)
            yield sse("guide", {"turn_id": turn_id, **guide})

    return sse_response(events())


@app.post("/api/sessions/<int:session_id>/end")
def finish_session(session_id: int):
    if store.get_session(session_id) is None:
        return jsonify(error="Session not found"), 404
    report = store.end_session(session_id)
    summary = ""
    try:
        summary = coach.summarize_session(report["mistakes"])
        store.save_summary(session_id, summary)
    except LLM_ERRORS:
        pass  # the numbers are still useful without the written note
    return jsonify(**report, summary=summary, focus=store.current_focus())


@app.get("/api/images/avatar")
def avatar_image():
    return jsonify(images.avatar_image())


@app.get("/api/images/jobs/<key>")
def image_job(key):
    result = images.job_status(key)
    if result is None:
        return jsonify(status="error", error="이미지 요청이 만료됐어요."), 404
    return jsonify(result)


# ---------------------------------------------------------------- progress and review

@app.get("/api/progress")
def progress():
    return jsonify(store.progress())


@app.get("/api/review")
def review_queue():
    return jsonify(cards=store.due_cards())


@app.post("/api/cards/<int:card_id>/attempt")
def card_attempt(card_id: int):
    card = store.get_card(card_id)
    if card is None:
        return jsonify(error="Card not found"), 404
    text = str((request.get_json(silent=True) or {}).get("text", "")).strip()
    if not text:
        return jsonify(error="답을 말하거나 입력해 주세요."), 400
    result = coach.check_answer(text, card["improved"], card["original"])
    graded = store.grade_card(card_id, result["correct"])
    return jsonify(**result, expected=card["improved"], card=graded)


@app.post("/api/cards/<int:card_id>/grade")
def card_grade(card_id: int):
    """Manual override, e.g. when speech recognition misheard a correct answer."""
    if store.get_card(card_id) is None:
        return jsonify(error="Card not found"), 404
    correct = bool((request.get_json(silent=True) or {}).get("correct"))
    return jsonify(card=store.grade_card(card_id, correct))


# ---------------------------------------------------------------- speech

@app.post("/api/speech")
def synthesize():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    if not text or len(text) > 2000:
        return jsonify(error="Speech text must contain 1–2,000 characters."), 400
    # Playback speed is applied by the browser (pitch-preserving), so every engine behaves the same.
    try:
        audio = speech.synthesize_wav(text, str(body.get("voice") or ""))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return jsonify(error=f"Local speech synthesis failed: {exc}"), 503
    return send_file(BytesIO(audio), mimetype="audio/wav", download_name="speech.wav")


@app.post("/api/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if audio is None:
        return jsonify(error="녹음 파일이 없어요."), 400
    content = audio.read(25 * 1024 * 1024 + 1)
    if not content:
        return jsonify(error="녹음이 비어 있어요."), 400
    if len(content) > 25 * 1024 * 1024:
        return jsonify(error="녹음이 너무 길어요 (25MB 제한)."), 413
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(content)
            f.flush()
            result = speech.transcribe_file(f.name)
    except Exception as exc:
        return jsonify(error=f"음성 인식에 실패했어요: {exc}"), 503
    if not result["text"]:
        return jsonify(error="말소리를 알아듣지 못했어요. 마이크에 조금 더 가까이 말해 보세요."), 422
    return jsonify(result)


store.init_db()

if __name__ == "__main__":
    speech.warm_up()
    app.run(host=os.environ.get("APP_HOST", "100.89.133.22"), port=int(os.environ.get("PORT", "18765")), debug=False, threaded=True)
