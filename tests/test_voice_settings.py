import io
import wave

import pytest

import coach
import speech
import store
import voice_settings as delivery
from test_server import client, events, temp_db


@pytest.mark.parametrize("text, expected", [
    ("느리게 말해줘", {"speed": 0.8}),
    ("천천히 말해 주세요", {"speed": 0.8}),
    ("더 천천히 말해줘", {"speed": 0.9}),
    ("빠르게 말해줘", {"speed": 1.2}),
    ("좀 더 빠르게 말해줄래?", {"speed": 1.1}),
    ("말하기 속도를 0.75배로 설정해줘", {"speed": 0.75}),
    ("2배속으로 말해줘", {"speed": 1.4}),
    ("밝고 친근하게 말해줘", {"tone": "friendly", "emotion": "happy"}),
    ("느리고 차분하게 말해줘", {"speed": 0.8, "tone": "calm"}),
    ("천천히, 차분하게 말해줘", {"speed": 0.8, "tone": "calm"}),
    ("좀 더 슬픈 목소리로 읽어 줄 수 있어?", {"emotion": "sad"}),
    ("화난 어조로 말해줘", {"emotion": "angry"}),
    ("신나게 말해줘", {"emotion": "excited"}),
    ("감정 없이 말해줘", {"emotion": "neutral"}),
    ("격식 있게 말해줘", {"tone": "formal"}),
    ("따뜻하고 자신 있게 말해줘", {"tone": "friendly", "emotion": "confident"}),
    ("Please speak more slowly and with a cheerful tone.", {"speed": 0.9, "emotion": "happy"}),
    ("Can you slow down please?", {"speed": 0.9}),
    ("Speed up, please", {"speed": 1.1}),
    ("Alex, speak slowly", {"speed": 0.8}),
    ("Slower please", {"speed": 0.9}),
    ("Speak at 0.7x", {"speed": 0.7}),
    ("Use a gentle and empathetic voice.", {"tone": "soft", "emotion": "empathetic"}),
    ("원래 속도로 말해줘", {"speed": 1.0}),
    ("Please reset your voice settings.", delivery.DEFAULTS),
    ("음성 설정을 초기화해줘", delivery.DEFAULTS),
    ("원래대로 말해줘", delivery.DEFAULTS),
    ("천천히 말하고 밝게 말해줘", {"speed": 0.8, "emotion": "happy"}),
    ("말하는 속도를 낮춰줘", {"speed": 0.9}),
    ("Reduce your speaking speed", {"speed": 0.9}),
    ("Slow down and sound happier", {"speed": 0.9, "emotion": "happy"}),
    ("더 감정적으로 말해줘", {"emotion": "expressive"}),
])
def test_explicit_requests(text, expected):
    patch, remaining = delivery.chat_request(text)
    assert patch == expected
    assert remaining == ""


@pytest.mark.parametrize("text", [
    "I am happy today.", "My friend told me to speak slowly.",
    "Can you tell me about your sad day?", "Use a happy example in a sentence.",
    "나는 어제 천천히 걸었어", "느리게 말하지 마", "친근한 친구가 있어",
    'How do I say "느리게 말해줘" in English?', '"Speak slowly" is what he said.',
    "Don't speak faster", "I want to talk to my angry boss.",
    "Alex, how was your day?",
])
def test_conversation_and_negated_requests_do_not_change_settings(text):
    assert delivery.chat_request(text) == ({}, text)


def test_relative_changes_use_current_speed_and_clamp():
    assert delivery.chat_request("더 천천히 말해줘", {"speed": 0.75})[0] == {"speed": 0.65}
    assert delivery.chat_request("더 천천히 말해줘", {"speed": 0.6})[0] == {"speed": 0.6}
    assert delivery.chat_request("Speak faster", {"speed": 1.4})[0] == {"speed": 1.4}


def test_settings_are_normalized_without_trusting_arbitrary_instructions():
    assert delivery.normalize_settings({"speed": True, "tone": [], "emotion": "ignore instructions"}) == delivery.DEFAULTS
    assert delivery.normalize_settings({"speed": float("nan")}) == delivery.DEFAULTS
    assert delivery.normalize_settings({"speed": 100})["speed"] == 1.4
    assert delivery.normalize_settings([]) == delivery.DEFAULTS


def test_control_turn_changes_settings_before_reply_without_llm_or_analysis(client, monkeypatch):
    def unexpected(*args):
        pytest.fail("A voice setting request must not call the conversation/analysis model")

    monkeypatch.setattr(coach, "generate_reply", unexpected)
    monkeypatch.setattr(coach, "analyze", unexpected)
    sid = store.create_session("free", "intermediate", "How was your day?")
    evs = events(client.post(f"/api/sessions/{sid}/turns", json={
        "text": "느리게 말해줘", "speech_settings": {"speed": 1.0},
    }).get_data(as_text=True))
    assert [name for name, _ in evs] == ["started", "speech_settings", "reply"]
    assert evs[1][1]["settings"] == {"speed": 0.8}
    assert evs[-1][1]["is_control"] is True
    assert "0.80" in evs[-1][1]["reply"]
    session = client.get(f"/api/sessions/{sid}").get_json()
    assert session["turns"][0]["is_control"] is True
    assert session["turns"][0]["analysis"] is None
    assert store.conversation_history(sid) == [{"role": "assistant", "content": "How was your day?"}]
    assert store.progress()["turns"] == store.progress()["words"] == store.progress()["streak"] == 0
    assert store.end_session(sid)["turns"] == 0
    assert store.due_cards() == []


def test_style_request_switches_to_capable_voice(client, monkeypatch):
    monkeypatch.setattr(speech, "qwen_available", lambda: True)
    sid = store.create_session("free", "intermediate", "Hi!")
    evs = events(client.post(f"/api/sessions/{sid}/turns", json={
        "text": "밝고 친근하게 말해줘", "voice": "say:Samantha",
    }).get_data(as_text=True))
    assert evs[1][1]["settings"] == {"tone": "friendly", "emotion": "happy", "voice": "qwen:aiden"}


def test_unsupported_style_is_saved_with_an_explanation(client, monkeypatch):
    monkeypatch.setattr(speech, "qwen_available", lambda: False)
    sid = store.create_session("free", "intermediate", "Hi!")
    evs = events(client.post(f"/api/sessions/{sid}/turns", json={"text": "밝게 말해줘"}).get_data(as_text=True))
    assert evs[1][1]["settings"] == {"emotion": "happy"}
    assert "지원하지 않아" in evs[1][1]["message"]
    assert "when the AI voice is available" in evs[2][1]["reply"]


def test_mixed_request_keeps_conversation_and_analysis(client, monkeypatch):
    seen = {}

    def reply(history, *args):
        seen["reply_text"] = history[-1]["content"]
        return "What did you see in Rome?"

    def analyze(text, focus):
        seen["analysis_text"] = text
        return {"corrections": [], "natural": "", "good": []}

    monkeypatch.setattr(coach, "generate_reply", reply)
    monkeypatch.setattr(coach, "analyze", analyze)
    sid = store.create_session("free", "intermediate", "Hi!")
    evs = events(client.post(f"/api/sessions/{sid}/turns", json={
        "text": "Please speak slowly. I went to Rome yesterday.",
    }).get_data(as_text=True))
    assert [name for name, _ in evs] == ["started", "speech_settings", "image", "reply", "analysis"]
    assert seen["reply_text"] == seen["analysis_text"] == "I went to Rome yesterday."


def test_qwen_receives_tone_and_emotion_in_synthesis(client, monkeypatch):
    captured = {}

    class Model:
        def generate_custom_voice(self, **kwargs):
            captured.update(kwargs)
            yield type("Chunk", (), {"audio": [0.0, 0.2, -0.2], "sample_rate": 24000})()

    monkeypatch.setattr(speech, "load_qwen_tts_model", lambda: Model())
    monkeypatch.setattr(speech, "qwen_available", lambda: True)
    response = client.post("/api/speech", json={
        "text": "Hello!", "voice": "qwen:ryan",
        "speech_settings": {"tone": "soft", "emotion": "happy", "speed": 0.8},
    })
    assert response.status_code == 200
    assert response.headers["X-Speech-Style"] == "available"
    assert captured["speaker"] == "ryan"
    assert "soft-spoken" in captured["instruct"] and "happy and cheerful" in captured["instruct"]
    with wave.open(io.BytesIO(response.data), "rb") as wav:
        assert wav.getnframes() == 3
