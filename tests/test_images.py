import pytest

import images


@pytest.mark.parametrize(
    "text, topic, expected",
    [
        ("I had coffee at a cafe after work.", "free", "cafe"),
        ("I went hiking in the forest last weekend.", "free", "nature"),
        ("I wrote code on my laptop today.", "free", "technology"),
        ("주말에 기타를 연습했어.", "free", "music"),
        ("I bought apples at the market.", "free", "shopping"),
        ("Yeah, it was nice.", "travel", "travel"),
        ("No, not really.", "free", "home"),
    ],
)
def test_select_scene_from_conversation(text, topic, expected):
    assert images.select_scene(text, topic) == expected


def test_image_payload_has_the_matching_static_asset():
    assert images.image_for_turn("I had coffee at a cafe.", "free") == {
        "scene": "cafe",
        "url": "/static/wallpapers/cafe.png",
        "alt": "비 오는 날 창가 카페",
    }


def test_specific_place_takes_precedence_over_generic_words():
    assert images.select_scene('I ate pasta on my trip to Rome.', 'travel') == 'rome'
    assert images.select_scene('로마에서 커피를 마셨어.', 'free') == 'rome'


def test_short_continuation_keeps_rome_background():
    result = images.conversation_image('Yes, absolutely.', 'free', ['I visited Rome.', 'What did you see?'], generate=False)
    assert result['scene'] == 'rome'


def test_missing_scene_is_generated_off_the_reply_path(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(images, 'CACHE', tmp_path)
    started = threading.Event()
    release = threading.Event()

    def generate(prompt, destination):
        started.set()
        assert release.wait(3)
        destination.write_bytes(b'png')

    monkeypatch.setattr(images.local_images, 'generate', generate)
    payload = images.conversation_image('Paris', 'free')
    try:
        assert payload['status'] == 'pending'
        assert started.wait(1)
        assert images.job_status('paris')['status'] == 'pending'
    finally:
        release.set()
    images._jobs['paris'].result(timeout=2)
    assert images.job_status('paris')['scene'] == 'paris'
    assert images.conversation_image('Paris')['status'] == 'ready'
    assert images.conversation_image('Paris')['url'] == '/static/generated/wallpapers/paris.png'
    images._jobs.pop('paris', None)


def test_cached_custom_scene_survives_short_answer_and_reload(tmp_path, monkeypatch):
    import hashlib
    import json
    monkeypatch.setattr(images, 'CACHE', tmp_path)
    text = 'I visited Barcelona last summer.'
    previous = ['Hello!']
    key = hashlib.sha256(json.dumps([text, 'free', previous], ensure_ascii=False).encode()).hexdigest()[:24]
    payload = {'scene': 'scene-barcelona', 'url': '/static/generated/barcelona.png', 'alt': '바르셀로나'}
    (tmp_path / f'{key}.json').write_text(json.dumps(payload))
    assert images.conversation_image(text, 'free', previous, generate=False)['scene'] == 'scene-barcelona'
    assert images.conversation_image('Yes!', 'free', previous + [text, 'How was it?'], generate=False)['scene'] == 'scene-barcelona'


def test_failed_job_allows_retry(monkeypatch):
    from concurrent.futures import Future
    future = Future()
    future.set_exception(RuntimeError('offline'))
    monkeypatch.setitem(images._jobs, 'failed', future)
    assert images.job_status('failed')['status'] == 'error'
    assert images.job_status('failed') is None
