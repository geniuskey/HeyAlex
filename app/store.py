"""SQLite storage plus the learner model: weak points, progress, and spaced review."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from coach import CATEGORIES, normalize

DB_PATH = Path(os.environ.get("COACH_DB", Path(__file__).resolve().parent.parent / "coach.sqlite3"))

DAY = 86400.0
# Leitner boxes: how long to wait before reviewing a card again after each success.
REVIEW_INTERVALS_DAYS = [0, 1, 3, 7, 14, 30, 60]
MASTERED_BOX = 4
RETRY_DELAY = 5 * 60
# Categories the tutor cannot steer a conversation toward.
NOT_STEERABLE = {"expression_gap", "naturalness"}
FOCUS_HALF_LIFE_TURNS = 12


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY, created REAL NOT NULL, ended REAL,
                topic TEXT NOT NULL, level TEXT NOT NULL, opener TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY, created REAL NOT NULL, user_text TEXT NOT NULL, reply TEXT NOT NULL, feedback TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mistakes (
                id INTEGER PRIMARY KEY, turn_id INTEGER NOT NULL, session_id INTEGER, created REAL NOT NULL,
                category TEXT NOT NULL, severity TEXT NOT NULL, original TEXT NOT NULL, improved TEXT NOT NULL,
                explain_ko TEXT NOT NULL, context TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS good_uses (
                id INTEGER PRIMARY KEY, turn_id INTEGER NOT NULL, created REAL NOT NULL, category TEXT NOT NULL, text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, mistake_id INTEGER NOT NULL,
                box INTEGER NOT NULL DEFAULT 0, due REAL NOT NULL, reps INTEGER NOT NULL DEFAULT 0,
                lapses INTEGER NOT NULL DEFAULT 0, last_reviewed REAL
            );
            CREATE INDEX IF NOT EXISTS mistakes_turn ON mistakes(turn_id);
            CREATE INDEX IF NOT EXISTS cards_due ON cards(due);
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(turns)")}
        for name, decl in (("session_id", "INTEGER"), ("words", "INTEGER NOT NULL DEFAULT 0"), ("audio_seconds", "REAL"), ("analyzed", "INTEGER NOT NULL DEFAULT 0")):
            if name not in columns:
                conn.execute(f"ALTER TABLE turns ADD COLUMN {name} {decl}")


def word_count(text: str) -> int:
    return len([w for w in normalize(text).split() if w.isascii()])


# ---------------------------------------------------------------- sessions and turns

def create_session(topic: str, level: str, opener: str) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO sessions(created, topic, level, opener) VALUES(?,?,?,?)", (time.time(), topic, level, opener))
        return int(cur.lastrowid)


def get_session(session_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        turns = conn.execute("SELECT id, created, user_text, reply, feedback, audio_seconds, analyzed FROM turns WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
    out = []
    for t in turns:
        analysis = json.loads(t["feedback"]) if t["analyzed"] else None
        out.append({"id": t["id"], "user_text": t["user_text"], "reply": t["reply"], "analysis": analysis, "audio_seconds": t["audio_seconds"]})
    return {**dict(row), "turns": out}


def conversation_history(session_id: int, limit_turns: int = 8) -> list[dict[str, str]]:
    session = get_session(session_id)
    if session is None:
        return []
    messages = [{"role": "assistant", "content": session["opener"]}] if session["opener"] else []
    turns = session["turns"]
    if len(turns) > limit_turns:
        messages = []
        turns = turns[-limit_turns:]
    for t in turns:
        messages += [{"role": "user", "content": t["user_text"]}, {"role": "assistant", "content": t["reply"]}]
    return messages


def add_turn(session_id: int, user_text: str, reply: str, audio_seconds: float | None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO turns(created, user_text, reply, feedback, session_id, words, audio_seconds, analyzed) VALUES(?,?,?,?,?,?,?,0)",
            (time.time(), user_text, reply, "{}", session_id, word_count(user_text), audio_seconds),
        )
        return int(cur.lastrowid)


def save_analysis(turn_id: int, analysis: dict[str, Any]) -> dict[str, Any]:
    """Store the analysis, log mistakes, and create/refresh review cards. Returns analysis with card ids."""
    now = time.time()
    with connect() as conn:
        turn = conn.execute("SELECT session_id, user_text FROM turns WHERE id = ?", (turn_id,)).fetchone()
        corrections = []
        for c in analysis["corrections"]:
            cur = conn.execute(
                "INSERT INTO mistakes(turn_id, session_id, created, category, severity, original, improved, explain_ko, context) VALUES(?,?,?,?,?,?,?,?,?)",
                (turn_id, turn["session_id"], now, c["category"], c["severity"], c["original"], c["improved"], c["explain_ko"], turn["user_text"]),
            )
            mistake_id = int(cur.lastrowid)
            key = normalize(c["original"]) + " => " + normalize(c["improved"])
            existing = conn.execute("SELECT id FROM cards WHERE key = ?", (key,)).fetchone()
            if existing:
                # Same mistake again: back to the first box, and show the latest context.
                conn.execute("UPDATE cards SET mistake_id = ?, box = 0, due = ?, lapses = lapses + 1 WHERE id = ?", (mistake_id, now, existing["id"]))
                card_id = existing["id"]
            else:
                card_id = conn.execute("INSERT INTO cards(key, mistake_id, due) VALUES(?,?,?)", (key, mistake_id, now)).lastrowid
            corrections.append({**c, "card_id": card_id})
        for g in analysis["good"]:
            conn.execute("INSERT INTO good_uses(turn_id, created, category, text) VALUES(?,?,?,?)", (turn_id, now, g["category"], g["text"]))
        result = {**analysis, "corrections": corrections}
        conn.execute("UPDATE turns SET feedback = ?, analyzed = 1 WHERE id = ?", (json.dumps(result, ensure_ascii=False), turn_id))
    return result


def end_session(session_id: int) -> dict[str, Any]:
    with connect() as conn:
        conn.execute("UPDATE sessions SET ended = COALESCE(ended, ?) WHERE id = ?", (time.time(), session_id))
        turns = conn.execute("SELECT words, audio_seconds FROM turns WHERE session_id = ?", (session_id,)).fetchall()
        mistakes = [dict(r) for r in conn.execute("SELECT category, severity, original, improved, explain_ko FROM mistakes WHERE session_id = ? ORDER BY id", (session_id,))]
        good = [dict(r) for r in conn.execute("SELECT g.category, g.text FROM good_uses g JOIN turns t ON t.id = g.turn_id WHERE t.session_id = ?", (session_id,))]
    words = sum(t["words"] for t in turns)
    by_cat: dict[str, int] = {}
    for m in mistakes:
        by_cat[m["category"]] = by_cat.get(m["category"], 0) + 1
    spoken = [t for t in turns if t["audio_seconds"]]
    wpm = round(sum(t["words"] for t in spoken) / (sum(t["audio_seconds"] for t in spoken) / 60)) if spoken and sum(t["audio_seconds"] for t in spoken) > 0 else None
    return {
        "turns": len(turns),
        "words": words,
        "errors": sum(1 for m in mistakes if m["severity"] == "error"),
        "mistakes": mistakes,
        "good": good,
        "categories": sorted(({"category": k, "label": CATEGORIES.get(k, k), "count": v} for k, v in by_cat.items()), key=lambda x: -x["count"]),
        "wpm": wpm,
    }


def save_summary(session_id: int, summary: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))


# ---------------------------------------------------------------- learner model

def weak_points(window_turns: int = 60) -> list[dict[str, Any]]:
    """Score each category by recency-weighted mistakes minus recent correct uses."""
    with connect() as conn:
        turn_ids = [r["id"] for r in conn.execute("SELECT id FROM turns WHERE analyzed = 1 ORDER BY id DESC LIMIT ?", (window_turns,))]
        if not turn_ids:
            return []
        age = {tid: i for i, tid in enumerate(turn_ids)}
        marks = ",".join("?" * len(turn_ids))
        mistakes = conn.execute(f"SELECT turn_id, category, severity FROM mistakes WHERE turn_id IN ({marks})", turn_ids).fetchall()
        good = conn.execute(f"SELECT turn_id, category FROM good_uses WHERE turn_id IN ({marks})", turn_ids).fetchall()
        totals = {r["category"]: r["n"] for r in conn.execute("SELECT category, COUNT(*) n FROM mistakes GROUP BY category")}
    stats: dict[str, dict[str, Any]] = {}

    def entry(cat: str) -> dict[str, Any]:
        return stats.setdefault(cat, {"category": cat, "label": CATEGORIES.get(cat, cat), "score": 0.0, "recent": 0, "previous": 0, "good_recent": 0, "total": totals.get(cat, 0)})

    for m in mistakes:
        e, a = entry(m["category"]), age[m["turn_id"]]
        e["score"] += (1.0 if m["severity"] == "error" else 0.6) * 0.5 ** (a / FOCUS_HALF_LIFE_TURNS)
        if a < 15:
            e["recent"] += 1
        elif a < 30:
            e["previous"] += 1
    for g in good:
        e, a = entry(g["category"]), age[g["turn_id"]]
        e["score"] -= 0.5 * 0.5 ** (a / FOCUS_HALF_LIFE_TURNS)
        if a < 15:
            e["good_recent"] += 1
    for e in stats.values():
        e["score"] = round(max(0.0, e["score"]), 2)
        if e["recent"] >= 3 or (e["recent"] >= 2 and e["recent"] > e["previous"]):
            e["status"] = "focus"
        elif e["previous"] > e["recent"] or (e["good_recent"] > e["recent"]):
            e["status"] = "improving"
        elif e["recent"] == 0:
            e["status"] = "stable"
        else:
            e["status"] = "watch"
    return sorted(stats.values(), key=lambda e: (-e["score"], -e["total"]))


def current_focus(limit: int = 2) -> list[str]:
    return [w["category"] for w in weak_points() if w["category"] not in NOT_STEERABLE and w["score"] >= 0.8][:limit]


def progress() -> dict[str, Any]:
    with connect() as conn:
        days = [r["d"] for r in conn.execute("SELECT DISTINCT date(created, 'unixepoch', 'localtime') d FROM turns ORDER BY d DESC")]
        totals = conn.execute("SELECT COUNT(*) turns, COALESCE(SUM(words), 0) words FROM turns").fetchone()
        sessions = conn.execute(
            """
            SELECT s.id, s.created, s.topic, COUNT(DISTINCT t.id) turns, COALESCE(SUM(t.words), 0) words,
                   (SELECT COUNT(*) FROM mistakes m WHERE m.session_id = s.id AND m.severity = 'error') errors,
                   (SELECT COUNT(*) FROM mistakes m WHERE m.session_id = s.id) mistakes
            FROM sessions s JOIN turns t ON t.session_id = s.id
            GROUP BY s.id ORDER BY s.id DESC LIMIT 14
            """
        ).fetchall()
        examples: dict[str, list[dict[str, str]]] = {}
        for r in conn.execute("SELECT category, original, improved, explain_ko FROM mistakes ORDER BY id DESC LIMIT 300"):
            bucket = examples.setdefault(r["category"], [])
            if len(bucket) < 4:
                bucket.append({"original": r["original"], "improved": r["improved"], "explain_ko": r["explain_ko"]})
        cards = conn.execute("SELECT COUNT(*) total, SUM(due <= ?) due, SUM(box >= ?) mastered FROM cards", (time.time(), MASTERED_BOX)).fetchone()
        today_words = conn.execute("SELECT COALESCE(SUM(words), 0) w FROM turns WHERE date(created, 'unixepoch', 'localtime') = date('now', 'localtime')").fetchone()["w"]
    streak, day = 0, date.today()
    day_set = set(days)
    if day.isoformat() not in day_set:
        day -= timedelta(days=1)  # a streak survives until the end of today
    while day.isoformat() in day_set:
        streak += 1
        day -= timedelta(days=1)
    weak = weak_points()
    for w in weak:
        w["examples"] = examples.get(w["category"], [])
    return {
        "streak": streak,
        "days_active": len(days),
        "turns": totals["turns"],
        "words": totals["words"],
        "today_words": today_words,
        "sessions": [
            {**dict(s), "date": datetime.fromtimestamp(s["created"]).strftime("%m/%d"), "time": datetime.fromtimestamp(s["created"]).strftime("%H:%M"), "error_rate": round(100 * s["errors"] / s["words"], 1) if s["words"] else None}
            for s in reversed(sessions)
        ],
        "weak_points": weak,
        "focus": current_focus(),
        "review": {"total": cards["total"] or 0, "due": cards["due"] or 0, "mastered": cards["mastered"] or 0},
    }


# ---------------------------------------------------------------- spaced review

def due_cards(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.box, c.reps, c.lapses, m.category, m.original, m.improved, m.explain_ko, m.context
            FROM cards c JOIN mistakes m ON m.id = c.mistake_id
            WHERE c.due <= ? ORDER BY c.box, c.due LIMIT ?
            """,
            (time.time(), limit),
        ).fetchall()
    return [{**dict(r), "label": CATEGORIES.get(r["category"], r["category"])} for r in rows]


def get_card(card_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT c.*, m.original, m.improved, m.explain_ko, m.category FROM cards c JOIN mistakes m ON m.id = c.mistake_id WHERE c.id = ?", (card_id,)).fetchone()
    return dict(row) if row else None


def grade_card(card_id: int, correct: bool, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    card = get_card(card_id)
    if card is None:
        raise KeyError(card_id)
    if correct:
        box = min(card["box"] + 1, len(REVIEW_INTERVALS_DAYS) - 1)
        due = now + REVIEW_INTERVALS_DAYS[box] * DAY
        lapses = card["lapses"]
    else:
        box, due, lapses = 0, now + RETRY_DELAY, card["lapses"] + 1
    with connect() as conn:
        conn.execute("UPDATE cards SET box = ?, due = ?, reps = reps + 1, lapses = ?, last_reviewed = ? WHERE id = ?", (box, due, lapses, now, card_id))
    return {"id": card_id, "box": box, "due": due, "mastered": box >= MASTERED_BOX}
