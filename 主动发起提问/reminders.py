"""时间事件识别与提醒管理。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from api import deepseek_chat
from config import (
    BIRTHDAY_GREETING_SYSTEM,
    EVENT_EXTRACTION_SYSTEM,
    EVENT_REMINDER_SYSTEM,
    REMINDERS_PATH,
)


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def load_reminders() -> dict[str, Any]:
    """返回 {'birthdays': [...], 'events': [...], 'last_reminded': {...}}。"""
    if not REMINDERS_PATH.exists():
        return {"birthdays": [], "events": [], "last_reminded": {}}
    try:
        return json.loads(REMINDERS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"birthdays": [], "events": [], "last_reminded": {}}


def save_reminders(data: dict[str, Any]) -> None:
    REMINDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REMINDERS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_dt(
    date_str: str,
    time_str: str | None,
    fallback_year: int | None = None,
) -> datetime | None:
    """
    将 date_str（YYYY-MM-DD 或 MM-DD）和 time_str（HH:MM）合并为 datetime。
    MM-DD 格式使用当前年；若该日期已过，fallback 到下一年。
    """
    now = _now_dt()
    year = fallback_year or now.year

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        year_str, month_str, day_str = date_str.split("-")
        year = int(year_str)
        month, day = int(month_str), int(day_str)
    elif re.match(r"^\d{2}-\d{2}$", date_str):
        month_str, day_str = date_str.split("-")
        month, day = int(month_str), int(day_str)
        candidate = now.replace(year=year, month=month, day=day)
        if candidate < now:
            year += 1
    else:
        return None

    if time_str and re.match(r"^\d{2}:\d{2}$", time_str):
        hour_str, minute_str = time_str.split(":")
        hour, minute = int(hour_str), int(minute_str)
    else:
        hour, minute = 9, 0  # 默认上午9点

    try:
        return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
    except ValueError:
        return None


def _next_occurrence(dt: datetime) -> datetime:
    """生日（MM-DD）：计算最近一次到来（今年或明年）。"""
    now = _now_dt()
    this_yr = now.replace(month=dt.month, day=dt.day)
    if this_yr > now:
        return this_yr
    return this_yr.replace(year=this_yr.year + 1)


def _is_due(
    dt: datetime,
    reminded_key: str,
    last_reminded: dict[str, float],
) -> bool:
    """判断是否到达提醒时间窗口（now ~ now+REMINDER_AHEAD_SECONDS）。"""
    from config import REMINDER_AHEAD_SECONDS
    now = _now_dt()
    due_window = now + timedelta(seconds=REMINDER_AHEAD_SECONDS)
    if not (now <= dt <= due_window):
        return False
    last_ts = last_reminded.get(reminded_key, 0.0)
    return (now.timestamp() - last_ts) > REMINDER_AHEAD_SECONDS


# ──── 事件提取 ────────────────────────────────────────────────────────────────

def extract_events_from_text(user_line: str) -> list[dict[str, Any]]:
    """调用模型从用户发言中提取生日/具体时间事件。"""
    reply = deepseek_chat(
        [
            {"role": "system", "content": EVENT_EXTRACTION_SYSTEM},
            {"role": "user", "content": f"用户发言：{user_line}"},
        ],
        temperature=0.1,
    )
    if not reply:
        return []
    try:
        obj = json.loads(reply.strip())
        events = obj.get("events", [])
        if not isinstance(events, list):
            return []
        return [
            e
            for e in events
            if isinstance(e, dict)
            and e.get("type") in ("birthday", "event")
            and e.get("date")
        ]
    except json.JSONDecodeError:
        return []


# ──── 提醒查询 ────────────────────────────────────────────────────────────────

def check_due_reminders() -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str | None, str]],
]:
    """
    返回 (待祝福生日列表, 待提醒事件列表)。
    生日项 = (description, date_str, raw_desc)
    事件项 = (description, date_str, time_str, raw_desc)
    """
    birthdays_greet: list[tuple[str, str, str]] = []
    events_remind: list[tuple[str, str, str | None, str]] = []
    reminders = load_reminders()
    last_reminded: dict[str, float] = reminders.get("last_reminded", {})

    for bday in reminders.get("birthdays", []):
        desc = (bday.get("description") or "生日").strip()
        date_str = bday.get("date", "")
        time_str = bday.get("time")
        fallback_year = bday.get("year")
        dt = _parse_dt(date_str, time_str, fallback_year=fallback_year)
        if dt is None:
            continue
        next_dt = _next_occurrence(dt)
        key = f"bday:{date_str}"
        if _is_due(next_dt, key, last_reminded):
            birthdays_greet.append((desc, date_str, desc))

    for evt in reminders.get("events", []):
        desc = (evt.get("description") or "日程").strip()
        date_str = evt.get("date", "")
        time_str: str | None = evt.get("time")
        dt = _parse_dt(date_str, time_str)
        if dt is None:
            continue
        key = f"evt:{date_str}:{time_str or '00:00'}"
        if _is_due(dt, key, last_reminded):
            events_remind.append((desc, date_str, time_str, desc))

    return birthdays_greet, events_remind


def mark_reminded(keys: list[str]) -> None:
    """将指定 key 标记为已提醒。"""
    reminders = load_reminders()
    now_ts = _now_dt().timestamp()
    last = reminders.setdefault("last_reminded", {})
    for k in keys:
        last[k] = now_ts
    save_reminders(reminders)


# ──── 添加记录 ───────────────────────────────────────────────────────────────

def add_birthday(date_str: str, description: str, year: int | None = None) -> None:
    """追加一条生日记录（去重）。"""
    reminders = load_reminders()
    for b in reminders["birthdays"]:
        if b.get("date") == date_str and b.get("description") == description:
            return
    entry: dict[str, Any] = {
        "type": "birthday",
        "date": date_str,
        "description": description,
        "added_ts": _now_dt().timestamp(),
    }
    if year:
        entry["year"] = year
    reminders["birthdays"].append(entry)
    save_reminders(reminders)
    print(f"已记录生日：{description}（{date_str}）")


def add_event(date_str: str, time_str: str | None, description: str) -> None:
    """追加一条一次性事件记录（去重）。"""
    reminders = load_reminders()
    for e in reminders["events"]:
        if (
            e.get("date") == date_str
            and e.get("time") == time_str
            and e.get("description") == description
        ):
            return
    entry: dict[str, Any] = {
        "type": "event",
        "date": date_str,
        "time": time_str,
        "description": description,
        "added_ts": _now_dt().timestamp(),
    }
    reminders["events"].append(entry)
    save_reminders(reminders)
    time_note = f" {time_str}" if time_str else "（未指定时间）"
    print(f"已记录事件：{description}（{date_str}{time_note}）")


# ──── 祝福/提醒语生成 ─────────────────────────────────────────────────────────

def generate_birthday_greeting(description: str) -> str | None:
    return deepseek_chat(
        [{"role": "system", "content": BIRTHDAY_GREETING_SYSTEM}],
        temperature=0.8,
    )


def generate_event_reminder(description: str, date_str: str) -> str | None:
    return deepseek_chat(
        [
            {"role": "system", "content": EVENT_REMINDER_SYSTEM},
            {"role": "user", "content": f"事件：{description}，时间：{date_str}"},
        ],
        temperature=0.7,
    )
