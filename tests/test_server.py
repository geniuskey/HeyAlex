import io
import json
import sqlite3
import time
import wave

import pytest

import coach
import server
import speech
import store


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.sqlite3")
    store.init_db()
    monkeypatch.setattr(coach, "short_answer_guide", lambda *_args: None)


@pytest.fixture
def client():
    server.app.config.update(TESTING=True)
    return server.app.test_client()


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


# ---------------------------------------------------------------- analysis sanitizing

def test_sanitize_drops_capitalization_and_ungrounded_corrections():
    raw = {
        "corrections": [
            {"original": "i'm", "improved": "I'm", "category": "naturalness", "severity": "error", "explain_ko": "대문자"},
            {"original": "I goes", "improved": "I went", "category": "tense", "severity": "error", "explain_ko": "x"},
            {"original": "go", "improved": "went", "category": "tense", "severity": "error", "explain_ko": "과거형"},
            {"original": "go", "improved": "went", "category": "tense", "severity": "error", "explain_ko": "중복"},
        ],
        "natural": "yesterday i go there",
        "good": [],
    }
    result = coach.sanitize_analysis(raw, "Yesterday i go there", [])
    assert [c["improved"] for c in result["corrections"]] == ["went"]
    assert result["natural"] == ""  # identical after normalization


def test_sanitize_forces_expression_gap_for_korean_and_filters_good_uses():
    raw = {
        "corrections": [{"original": "친근하게", "improved": "in a friendly way", "category": "word_choice", "severity": "unnatural", "explain_ko": "친근하게는 in a friendly way"}],
        "natural": "I don't know how to talk in a friendly way.",
        "good": [{"category": "tense", "text": "don't know"}, {"category": "article", "text": "not said"}],
    }
    result = coach.sanitize_analysis(raw, "i don't know how to talk 친근하게", ["tense", "article"])
    assert result["corrections"][0]["category"] == "expression_gap"
    assert result["corrections"][0]["severity"] == "error"
    assert result["good"] == [{"category": "tense", "text": "don't know"}]


def test_analysis_message_lists_korean_words_and_focus():
    msg = coach.analysis_user_message("I want to 친근하게 말하다 talk", ["tense"])
    assert "친근하게 말하다" in msg and "Focus categories: tense" in msg


def test_analysis_uses_json_schema_with_category_enum(monkeypatch):
    captured = {}

    def fake_chat(messages, **kwargs):
        captured.update(kwargs)
        return json.dumps({"corrections": [], "natural": "", "good": []})

    monkeypatch.setattr(coach, "local_chat", fake_chat)
    coach.analyze("hello", [])
    assert captured["fmt"]["properties"]["corrections"]["items"]["properties"]["category"]["enum"] == list(coach.CATEGORIES)
    assert captured["temperature"] <= 0.2


def test_chat_payload_turns_thinking_off():
    payload = coach.build_chat_payload([{"role": "user", "content": "Hi"}], fmt="json")
    assert payload["think"] is False and payload["stream"] is False and payload["format"] == "json"


def test_reply_prompt_steers_toward_focus():
    prompt = coach.reply_system_prompt("day", "beginner", ["tense"])
    assert "past events" in prompt and "A2" in prompt
    assert "weak points" not in coach.reply_system_prompt("day", "beginner", [])


def test_clean_reply_strips_markdown_and_speaker():
    assert coach.clean_reply("Alex: **Nice!** How was it?") == "Nice! How was it?"
    assert coach.clean_reply('{"reply": "Hi there"}') == "Hi there"


# ---------------------------------------------------------------- answer checking

@pytest.mark.parametrize(
    "answer, expected, original, correct",
    [
        ("I work at an IT company", "an IT company", "a IT company", True),
        ("I work at a IT company", "an IT company", "a IT company", False),
        ("He told me that", "told me", "said me", True),
        ("he said me that", "told me", "said me", False),
        ("I've been working here for five years", "for 5 years", "since 5 years", True),
        ("Yesterday I went there", "went", "go", True),
        ("", "went", "go", False),
    ],
)
def test_check_answer(answer, expected, original, correct):
    assert coach.check_answer(answer, expected, original)["correct"] is correct


# ---------------------------------------------------------------- learner model

def make_turn(session_id, text, corrections, good=()):
    turn_id = store.add_turn(session_id, text, "ok", None)
    return store.save_analysis(turn_id, {"corrections": list(corrections), "natural": "", "good": list(good)})


def corr(category, original="go", improved="went", severity="error"):
    return {"original": original, "improved": improved, "category": category, "severity": severity, "explain_ko": "설명"}


def test_repeated_mistakes_become_focus_and_correct_use_lowers_score():
    sid = store.create_session("free", "intermediate", "Hi!")
    for i in range(3):
        make_turn(sid, f"I go there {i}", [corr("tense", improved=f"went {i}")])
    make_turn(sid, "a IT company", [corr("article", "a IT", "an IT")])
    assert store.current_focus()[0] == "tense"
    tense = next(w for w in store.weak_points() if w["category"] == "tense")
    assert tense["status"] == "focus" and tense["recent"] == 3
    before = tense["score"]
    make_turn(sid, "I went there", [], [{"category": "tense", "text": "went"}])
    after = next(w for w in store.weak_points() if w["category"] == "tense")["score"]
    assert after < before


def test_expression_gaps_are_tracked_but_not_used_for_steering():
    sid = store.create_session("free", "intermediate", "Hi!")
    for i in range(4):
        make_turn(sid, f"친근하게 {i}", [corr("expression_gap", f"친근하게 {i}", f"friendly {i}")])
    assert store.current_focus() == []
    assert store.weak_points()[0]["category"] == "expression_gap"


def test_same_mistake_again_resets_existing_card():
    sid = store.create_session("free", "intermediate", "Hi!")
    first = make_turn(sid, "I go", [corr("tense")])["corrections"][0]["card_id"]
    store.grade_card(first, True)
    store.grade_card(first, True)
    second = make_turn(sid, "I go again", [corr("tense")])["corrections"][0]["card_id"]
    card = store.get_card(second)
    assert second == first and card["box"] == 0 and card["lapses"] == 1


def test_leitner_scheduling():
    sid = store.create_session("free", "intermediate", "Hi!")
    card_id = make_turn(sid, "I go", [corr("tense")])["corrections"][0]["card_id"]
    now = time.time()
    assert store.grade_card(card_id, True, now)["due"] == pytest.approx(now + store.DAY)
    assert store.grade_card(card_id, True, now)["due"] == pytest.approx(now + 3 * store.DAY)
    wrong = store.grade_card(card_id, False, now)
    assert wrong["box"] == 0 and wrong["due"] == pytest.approx(now + store.RETRY_DELAY)
    assert store.due_cards() == []  # not due until the retry delay passes


def test_history_uses_plain_replies_not_json():
    sid = store.create_session("day", "intermediate", "How was your day?")
    store.add_turn(sid, "It was good", "Nice! What did you do?", None)
    assert store.conversation_history(sid) == [
        {"role": "assistant", "content": "How was your day?"},
        {"role": "user", "content": "It was good"},
        {"role": "assistant", "content": "Nice! What did you do?"},
    ]


def test_legacy_turns_table_is_migrated(tmp_path, monkeypatch):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY, created REAL NOT NULL, user_text TEXT NOT NULL, reply TEXT NOT NULL, feedback TEXT NOT NULL)")
        conn.execute("INSERT INTO turns(created, user_text, reply, feedback) VALUES(1, 'hi', 'hello', '{}')")
    monkeypatch.setattr(store, "DB_PATH", path)
    store.init_db()
    assert store.progress()["turns"] == 1


# ---------------------------------------------------------------- API

def test_full_session_flow(client, monkeypatch):
    monkeypatch.setattr(coach, "generate_reply", lambda history, topic, level, focus: "Oh, you went there yesterday? What did you see?" if history else "Hi! How was your day?")
    monkeypatch.setattr(coach, "analyze", lambda text, focus: {"corrections": [corr("tense")], "natural": "I went there yesterday.", "good": []})
    monkeypatch.setattr(coach, "summarize_session", lambda mistakes: "과거 시제를 연습하세요.")

    start = client.post("/api/sessions", json={"topic": "day", "level": "beginner"}).get_json()
    assert start["opener"] == "Hi! How was your day?" and start["level"] == "beginner"

    response = client.post(f"/api/sessions/{start['id']}/turns", json={"text": "I go there yesterday", "audio_seconds": 2.5})
    evs = events(response.get_data(as_text=True))
    assert [e for e, _ in evs] == ["started", "image", "reply", "analysis"]
    assert evs[1][1]["scene"] == "home"
    assert evs[1][1]["url"] == "/static/wallpapers/home.png"
    card_id = evs[3][1]["corrections"][0]["card_id"]

    session = client.get(f"/api/sessions/{start['id']}").get_json()
    assert session["turns"][0]["image"]["scene"] == "home"
    assert session["turns"][0]["analysis"]["corrections"][0]["card_id"] == card_id

    assert client.get("/api/review").get_json()["cards"][0]["id"] == card_id
    attempt = client.post(f"/api/cards/{card_id}/attempt", json={"text": "I went there yesterday"}).get_json()
    assert attempt["correct"] is True and attempt["card"]["box"] == 1

    report = client.post(f"/api/sessions/{start['id']}/end").get_json()
    assert report["turns"] == 1 and report["errors"] == 1 and report["summary"] == "과거 시제를 연습하세요."
    assert report["wpm"] == round(4 / (2.5 / 60))

    progress = client.get("/api/progress").get_json()
    assert progress["turns"] == 1 and progress["streak"] == 1
    assert progress["weak_points"][0]["category"] == "tense"


def test_turn_emits_image_before_generating_tutor_reply(client, monkeypatch):
    sid = store.create_session("free", "intermediate", "Hi!")
    generated = False

    def reply(*_args):
        nonlocal generated
        generated = True
        return "Sounds good!"

    monkeypatch.setattr(coach, "generate_reply", reply)
    response = client.post(f"/api/sessions/{sid}/turns", json={"text": "I went hiking"}, buffered=False)
    chunks = iter(response.response)
    try:
        assert "event: started" in next(chunks).decode()
        image_event = next(chunks).decode()
    finally:
        response.close()

    assert "event: image" in image_event
    assert not generated


def test_static_scene_asset_is_served(client):
    response = client.get("/static/scenes/cafe.png")
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_turn_reports_llm_failure_as_event(client, monkeypatch):
    sid = store.create_session("free", "intermediate", "Hi!")

    def boom(*_args):
        raise ValueError("model offline")

    monkeypatch.setattr(coach, "generate_reply", boom)
    evs = events(client.post(f"/api/sessions/{sid}/turns", json={"text": "hello"}).get_data(as_text=True))
    assert evs[-1][0] == "error" and "model offline" in evs[-1][1]["error"]


def test_analysis_failure_still_keeps_reply(client, monkeypatch):
    sid = store.create_session("free", "intermediate", "Hi!")

    def bad_analysis(*_args):
        raise ValueError("bad json")

    monkeypatch.setattr(coach, "generate_reply", lambda *a: "Cool!")
    monkeypatch.setattr(coach, "analyze", bad_analysis)
    evs = events(client.post(f"/api/sessions/{sid}/turns", json={"text": "hello"}).get_data(as_text=True))
    assert [e for e, _ in evs] == ["started", "image", "reply", "analysis_error"]


def test_turn_validation(client):
    sid = store.create_session("free", "intermediate", "Hi!")
    assert client.post(f"/api/sessions/{sid}/turns", json={"text": " "}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/turns", json={"text": "x" * 2001}).status_code == 413
    assert client.post("/api/sessions/999/turns", json={"text": "hi"}).status_code == 404


def test_bootstrap_and_index(client):
    assert client.get("/").status_code == 200
    data = client.get("/api/bootstrap").get_json()
    assert data["categories"]["tense"] == "시제" and any(t["id"] == "focus" for t in data["topics"])


# ---------------------------------------------------------------- speech

def test_say_command_writes_wav_directly():
    command = speech.build_speech_command("Let's talk.", "/tmp/out.wav", 180, "Samantha")
    assert command == ["/usr/bin/say", "-v", "Samantha", "-r", "180", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", "/tmp/out.wav", "Let's talk."]


def test_qwen_tts_uses_english_speaker_and_returns_wav(monkeypatch):
    captured = {}

    class FakeModel:
        def generate_custom_voice(self, **kwargs):
            captured.update(kwargs)
            yield type("Chunk", (), {"audio": [0.0, 0.5, -0.5], "sample_rate": 24000})()
            yield type("Chunk", (), {"audio": [0.1, 0.0, -0.1], "sample_rate": 24000})()

    monkeypatch.setattr(speech, "load_qwen_tts_model", lambda: FakeModel())
    monkeypatch.setattr(speech, "qwen_available", lambda: True)
    audio = speech.synthesize_wav("Hello there.", "qwen:aiden")
    with wave.open(io.BytesIO(audio), "rb") as w:
        assert w.getframerate() == 24000 and w.getnframes() == 6 and w.getsampwidth() == 2
    assert captured["speaker"] == "aiden" and captured["language"] == "English"


def test_qwen_failure_falls_back_to_say(monkeypatch):
    def broken():
        raise RuntimeError("metal error")

    monkeypatch.setattr(speech, "load_qwen_tts_model", broken)
    monkeypatch.setattr(speech, "_qwen_failed", False)
    monkeypatch.setattr(speech, "qwen_available", lambda: not speech._qwen_failed)
    monkeypatch.setattr(speech, "say_voices", lambda: ["Samantha"])
    monkeypatch.setattr(speech, "say_wav", lambda text, voice: f"say:{voice}".encode())
    assert speech.synthesize_wav("Hi", "qwen:aiden") == b"say:Samantha"


def test_speech_endpoint(client, monkeypatch):
    monkeypatch.setattr(speech, "synthesize_wav", lambda text, voice, delivery: b"RIFF")
    assert client.post("/api/speech", json={"text": ""}).status_code == 400
    ok = client.post("/api/speech", json={"text": "hi", "voice": "whatever"})
    assert ok.status_code == 200 and ok.data == b"RIFF"


def test_transcribe_requires_audio(client):
    assert client.post("/api/transcribe").status_code == 400


def test_image_job_status_and_fixed_panel(client, monkeypatch):
    import images
    monkeypatch.setattr(images, 'job_status', lambda key: {'status': 'pending'} if key == 'queued' else None)
    assert client.get('/api/images/jobs/queued').get_json() == {'status': 'pending'}
    assert client.get('/api/images/jobs/missing').status_code == 404
    assert client.get('/api/images/avatar').get_json()['status'] == 'ready'
    markup = client.get('/').get_data(as_text=True)
    assert markup.index('id="sceneImage"') < markup.index('id="messages"')
    assert "class: 'message-avatar'" in markup
    assert "class: 'sender-name'" in markup
