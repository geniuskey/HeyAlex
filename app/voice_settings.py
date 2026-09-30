"""Explicit chat requests for persistent playback speed and TTS delivery."""

from __future__ import annotations

import math
import re
from typing import Any

DEFAULTS = {"speed": 1.0, "tone": "natural", "emotion": "neutral"}
MIN_SPEED, MAX_SPEED = 0.6, 1.4
# Labels are shared with the settings UI; instructions go only to the speech model.
TONES = {
    "natural": ("자연스럽게", "Use a natural, conversational tone."),
    "friendly": ("친근하게", "Use a warm, friendly, welcoming tone."),
    "calm": ("차분하게", "Use a calm, composed, relaxed tone."),
    "soft": ("부드럽게", "Use a gentle, soft-spoken tone."),
    "formal": ("격식 있게", "Use a polite, professional, formal tone."),
    "energetic": ("활기차게", "Use a lively, energetic, animated tone."),
}
EMOTIONS = {
    "neutral": ("중립", "Keep the emotional delivery neutral."),
    "happy": ("밝고 즐겁게", "Sound happy and cheerful, with a smile in your voice."),
    "excited": ("신나게", "Sound excited and enthusiastic."),
    "sad": ("슬프게", "Sound sad and subdued."),
    "angry": ("화나게", "Express anger through a firm, frustrated delivery."),
    "empathetic": ("공감하며", "Sound empathetic, caring, and reassuring."),
    "confident": ("자신 있게", "Sound confident and assured."),
    "expressive": ("감정 풍부하게", "Use emotionally expressive delivery and varied intonation suited to the meaning of the text."),
}


def normalize_settings(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    speed = value.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed):
        speed = 1.0
    return {
        "speed": round(max(MIN_SPEED, min(MAX_SPEED, speed)), 2),
        "tone": value.get("tone") if isinstance(value.get("tone"), str) and value["tone"] in TONES else "natural",
        "emotion": value.get("emotion") if isinstance(value.get("emotion"), str) and value["emotion"] in EMOTIONS else "neutral",
    }


def options() -> dict[str, list[dict[str, str]]]:
    return {key: [{"id": k, "label": v[0]} for k, v in values.items()]
            for key, values in (("tones", TONES), ("emotions", EMOTIONS))}


ATTRIBUTES = [
    ("speed", "slow", r"천천히|느리(?:게|고)|느린|느려|\b(?:slowly|slower|slow)\b"),
    ("speed", "fast", r"빠르(?:게|고)|빠른|빨리|빠르게|\b(?:quickly|faster|fast)\b"),
    ("tone", "natural", r"자연스럽게|자연스러운|\b(?:natural|naturally)\b"),
    ("tone", "friendly", r"친근(?:하게|한|하고)|친절(?:하게|한|하고)|다정(?:하게|한|하고)|따뜻(?:하게|한|하고)|친구처럼|\b(?:friendly|friendlier|warm|warmly|casual|casually)\b"),
    ("tone", "calm", r"차분(?:하게|한|하고)|침착(?:하게|한|하고)|\b(?:calm|calmly|relaxed)\b"),
    ("tone", "soft", r"부드럽(?:게|고)|부드러운|상냥(?:하게|한|하고)|나긋나긋하게|\b(?:soft|softly|gentle|gently)\b"),
    ("tone", "formal", r"격식\s*(?:있게|있는)|정중하게|정중한|공손하게|공손한|\b(?:formal|formally|polite|politely|professional)\b"),
    ("tone", "energetic", r"활기차게|활기찬|힘차게|힘찬|\b(?:energetic|energetically|lively)\b"),
    ("emotion", "neutral", r"감정\s*없이|무감정으로|중립적으로|중립적인|담담하게|\b(?:neutral|neutrally|emotionless)\b"),
    ("emotion", "happy", r"밝(?:게|은|고)|행복(?:하게|한|하고)|기쁘(?:게|고)|기쁜|즐겁(?:게|고)|즐거운|명랑(?:하게|한|하고)|\b(?:happy|happier|happily|cheerful|cheerfully|bright|brightly)\b"),
    ("emotion", "excited", r"신나게|신나는|신난|들뜬|흥분한|\b(?:excited|excitedly|enthusiastic|enthusiastically)\b"),
    ("emotion", "sad", r"슬프게|슬픈|슬퍼하는|\b(?:sad|sadly|subdued)\b"),
    ("emotion", "angry", r"화난|화나게|화가\s*난|화내는|분노한|\b(?:angry|angrily|frustrated)\b"),
    ("emotion", "empathetic", r"공감하며|공감하는|공감해주는|위로하듯|위로하는|\b(?:empathetic|empathetically|reassuring|caring)\b"),
    ("emotion", "confident", r"자신\s*있게|자신감\s*있게|자신감\s*있는|\b(?:confident|confidently|assured)\b"),
    ("emotion", "expressive", r"감정적으로|감정\s*풍부하게|감정을\s*담아서|\b(?:expressive|expressively|emotionally)\b"),
    ("speed", 1.0, r"(?:보통|정상|원래|기본)\s*속도(?:로)?|\b(?:normal|default|original|regular)\s+(?:speed|pace)\b"),
    ("tone", "natural", r"(?:보통|원래|기본)\s*(?:어조|톤)|\b(?:normal|default|original)\s+tone\b"),
    ("emotion", "neutral", r"(?:보통|원래|기본)\s*감정|\b(?:normal|default|original)\s+emotion\b"),
]
NUMBER_RATE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:배(?:속)?|×|x\b)", re.I)
KO_REQUEST = re.compile(r"(?:말해|읽어|대답해|이야기해|얘기해|해|바꿔|변경해|설정해|조절해|돌려|되돌려)(?:\s*(?:줘|주세요|줄래|주실래|주면\s*좋겠어|줄\s*수\s*있어))?(?:요)?\s*[?!.]*$|(?:낮춰|높여|줄여|올려)(?:\s*(?:줘|주세요|줄래))?(?:요)?\s*[?!.]*$")
EN_REQUEST = re.compile(r"^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:speak|talk|read|say|sound|be|use|set|change|make|reset|reduce|decrease|increase|raise|slow\s+down|speed\s+up)\b", re.I)
# Only a closed vocabulary may remain after delivery attributes are removed. This
# prevents quoted/reported requests and ordinary stories from changing settings.
FILLER = re.compile(r"말하는\s*속도|말하기\s*속도|말해주고|말하고|읽고|대답하고|말투|목소리|음성|어조|속도|감정|설정|톤|알렉스|좀|조금|아주|매우|훨씬|더|으로|처럼|대로|하게|있게|로|를|을|은|는|도|와|과|하고|그리고|의|게|고|에|\b(?:alex|please|a|an|the|your|my|voice|tone|emotion|settings|speed|pace|speaking|delivery|read|it|this|aloud|to|at|with|in|and|more|much|very|really|little|bit|slightly|back)\b|[\s,!?。:]+", re.I)
RESET = re.compile(r"(?:음성|목소리|말투|말하기|설정).*?(?:초기화|리셋|기본|원래)|^(?:원래대로|기본으로)|\breset\b|\b(?:default|original|normal)\s+(?:voice|settings)\b", re.I)


def parse_clause(text: str, current: dict[str, Any]) -> dict[str, Any]:
    text = text.strip().lower()
    text = re.sub(r"^(?:alex|알렉스)(?:야|아)?[\s,:]+", "", text)
    korean_request = KO_REQUEST.search(text)
    english_request = EN_REQUEST.search(text)
    shorthand = re.fullmatch(r"(?:more\s+)?(?:slowly|slower|faster)(?:\s+please)?[.!?]*|(?:좀\s*)?(?:더\s*)?(?:천천히|느리게|빠르게|빨리)[.!?]*", text)
    if not (korean_request or english_request or shorthand):
        return {}
    if RESET.search(text) and not re.search(r"속도|어조|감정|\b(?:speed|pace|tone|emotion)\b", text):
        # A reset must itself be a direct instruction, not a mention of one.
        reset_rest = RESET.sub("", text)
        reset_rest = KO_REQUEST.sub("", reset_rest)
        reset_rest = EN_REQUEST.sub("", reset_rest)
        if not FILLER.sub("", reset_rest).strip("."):
            return dict(DEFAULTS)

    patch: dict[str, Any] = {}
    rest = EN_REQUEST.sub(" ", text)
    if english_request and re.search(r"slow\s+down|speed\s+up", english_request.group()):
        patch["speed"] = current["speed"] + (-0.1 if "slow" in english_request.group() else 0.1)
    if re.search(r"속도.*(?:낮춰|줄여)|\b(?:reduce|decrease)\b.*\b(?:speed|pace)\b", text):
        patch["speed"] = current["speed"] - 0.1
    elif re.search(r"속도.*(?:높여|올려)|\b(?:increase|raise)\b.*\b(?:speed|pace)\b", text):
        patch["speed"] = current["speed"] + 0.1
    rest = re.sub(r"\band\s+(?:please\s+)?(?:speak|sound|talk|be|use)\b", " ", rest)
    for key, value, pattern in ATTRIBUTES:
        if re.search(pattern, rest, re.I):
            if key == "speed" and isinstance(value, str):
                relative = bool(re.search(r"더|조금|\b(?:slower|faster|more|slightly)\b", text))
                value = current["speed"] + (-0.1 if value == "slow" else 0.1) if relative else (0.8 if value == "slow" else 1.2)
            patch[key] = value
            rest = re.sub(pattern, " ", rest, flags=re.I)
    rate = NUMBER_RATE.search(rest)
    if rate:
        patch["speed"] = float(rate.group(1))
        rest = NUMBER_RATE.sub(" ", rest)
    if not patch:
        return {}
    rest = KO_REQUEST.sub(" ", rest)
    rest = EN_REQUEST.sub(" ", rest)
    if FILLER.sub("", rest).strip("."):
        return {}
    return {key: normalize_settings({**current, **patch})[key] for key in patch}


def chat_request(text: str, settings: Any = None) -> tuple[dict[str, Any], str]:
    """Return changed fields and conversation text with explicit requests removed."""
    current = normalize_settings(settings)
    original_text = text
    text = re.sub(r"^(?:alex|알렉스)(?:야|아)?[\s,:]+", "", text, flags=re.I)
    whole = parse_clause(text, current)
    if whole:
        return whole, ""
    patch: dict[str, Any] = {}
    remaining = []
    # A decimal speed such as 0.75x is kept intact. Mixed conversation sentences
    # continue through the usual reply/analysis path.
    clauses = re.split(r"((?<!\d)[.!?。]+\s*|[.!?。]+(?=\s|$)|\n+|[,;]\s*|\s+and\s+(?=(?:please\s+)?(?:speak|talk|tell|ask|use|set|make|I\b)))", text, flags=re.I)
    for i in range(0, len(clauses), 2):
        clause = clauses[i]
        if not clause.strip():
            continue
        change = parse_clause(clause, current)
        if change:
            patch.update(change)
            current.update(change)
        else:
            remaining.append(clause + (clauses[i + 1] if i + 1 < len(clauses) else ""))
    rest = re.sub(r"\s+and\s*$", "", "".join(remaining)).strip()
    return patch, (rest if patch else original_text)


def confirmation(patch: dict[str, Any]) -> str:
    parts = []
    if "speed" in patch:
        parts.append(f"말하기 속도 {patch['speed']:.2f}배")
    if "tone" in patch:
        parts.append(f"어조 {TONES[patch['tone']][0]}")
    if "emotion" in patch:
        parts.append(f"감정 {EMOTIONS[patch['emotion']][0]}")
    return "음성 설정을 바꿨어요: " + " · ".join(parts)


def spoken_confirmation(patch: dict[str, Any]) -> str:
    parts = []
    if "speed" in patch:
        parts.append(f"a speaking speed of {patch['speed']:.2f} times normal")
    if "tone" in patch:
        parts.append(f"a {patch['tone']} tone")
    if "emotion" in patch:
        parts.append(f"a {patch['emotion']} emotional delivery")
    return "Sure. I'll use " + ", and ".join(parts) + "."


def instruction(settings: Any) -> str:
    settings = normalize_settings(settings)
    return " ".join(("Speak clearly and naturally to an English learner.",
                     TONES[settings["tone"]][1], EMOTIONS[settings["emotion"]][1]))
