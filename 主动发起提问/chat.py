"""单轮多轮对话：用户输入处理、话题边界判断、显式结束检测。"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from api import deepseek_chat
from config import (
    CHAT_SESSION_SYSTEM,
    SESSION_COOLDOWN_SECONDS,
    SESSION_IDLE_TIMEOUT_SECONDS,
    TOPIC_EXIT_SYSTEM,
)
from reminders import add_birthday, add_event, extract_events_from_text
from store import save_store


# ──── 带超时的 input ─────────────────────────────────────────────────────────

_INPUT_TIMEOUT = SESSION_IDLE_TIMEOUT_SECONDS


def _input_with_timeout(prompt: str) -> tuple[bool, str]:
    """
    在独立线程中执行 input()，主线程等待超时。
    返回 (timed_out, user_line)。
    """
    result: dict[str, Any] = {"value": ""}

    def _read():
        try:
            result["value"] = input(prompt)
        except Exception:
            pass

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=_INPUT_TIMEOUT)
    if t.is_alive():
        # 线程仍在运行，说明超时了
        print()  # 换行，防止光标停留在 prompt 后面
        return True, ""
    return False, result["value"]


# ──── 显式结束语检测 ─────────────────────────────────────────────────────────

_ROUND_END_EXACT_EN = frozenset({"exit", "quit", "/end", "q", "bye"})
_ROUND_END_EXACT_CN = frozenset({"结束", "拜拜", "回见", "再见"})
_ROUND_END_CONTAINS = (
    "不想聊天",
    "不想聊了",
    "不想再聊",
    "不想说了",
    "不想讲了",
    "不想继续",
    "不聊了",
    "不讲了",
    "不说了",
    "别聊了",
    "别说了",
    "别问了",
    "先这样",
    "就这样吧",
    "到这儿吧",
    "到此为止",
    "结束聊天",
    "不用聊了",
    "不要聊了",
    "先挂了",
    "有空再聊",
    "改天再聊",
    "改天聊",
    "不想谈",
    "先忙了",
    "不打扰了",
)


def _is_round_end_keyword(user_line: str) -> bool:
    t = user_line.strip()
    if not t:
        return False
    low = t.lower()
    if low in _ROUND_END_EXACT_EN:
        return True
    if t in _ROUND_END_EXACT_CN:
        return True
    for phrase in _ROUND_END_CONTAINS:
        if phrase in t:
            return True
    return False


# ──── 话题边界模型 ───────────────────────────────────────────────────────────

from config import USE_TOPIC_EXIT_MODEL as _USE_TOPIC_EXIT_MODEL


def should_end_by_topic_model(user_line: str, opening: str) -> bool:
    if not _USE_TOPIC_EXIT_MODEL:
        return False
    reply = deepseek_chat(
        [
            {"role": "system", "content": TOPIC_EXIT_SYSTEM},
            {"role": "user", "content": f"开场白：{opening}\n用户说：{user_line}"},
        ],
        temperature=0.1,
    )
    if not reply:
        return False
    first = reply.strip().splitlines()[0].strip().upper()
    return bool(re.match(r"^END\b", first))


# ──── API 消息构建 ───────────────────────────────────────────────────────────

def _api_messages_for_session(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = [{"role": "system", "content": CHAT_SESSION_SYSTEM}]
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


# ──── 完整多轮对话 ────────────────────────────────────────────────────────────

def run_full_chat_session(
    messages: list[dict[str, Any]],
    opening: str,
    last_session_end_ts: float | None,
) -> tuple[list[dict[str, Any]], float | None]:
    """
    本轮多次对话：开场 → 等待用户输入 → 判断结束条件 → 循环 → 结束时写入文件。
    返回 (messages, new_last_session_end_ts)。
    """
    messages = list(messages)
    opening = opening.strip()
    messages.append({"role": "assistant", "content": opening})
    save_store(messages, last_session_end_ts)

    print(f"\n[助手] {opening}\n")
    print(
        f"（本轮多轮对话；显式结束语如「不想聊了」「结束」，"
        f"或模型判定你想结束当前话题，或超过 {SESSION_IDLE_TIMEOUT_SECONDS}s 无回应时会结束本轮；"
        f"结束后需冷却 {SESSION_COOLDOWN_SECONDS}s 才可能再次主动提问）\n"
    )

    while True:
        timed_out, user_line = _input_with_timeout("[用户] ")
        if timed_out:
            ended_ts = time.time()
            messages.append(
                {"role": "assistant", "content": f"（用户超过 {SESSION_IDLE_TIMEOUT_SECONDS}s 无回应，本轮结束。）"}
            )
            save_store(messages, ended_ts)
            print(f"（用户超时 {SESSION_IDLE_TIMEOUT_SECONDS}s，本轮结束。）\n")
            return messages, ended_ts
        user_line = user_line.strip()

        if _is_round_end_keyword(user_line):
            break

        if not user_line:
            print(f"（请输入内容，或输入 结束 / 不想聊了 等，超时 {SESSION_IDLE_TIMEOUT_SECONDS}s 将自动结束）")
            continue

        if should_end_by_topic_model(user_line, opening):
            messages.append({"role": "user", "content": user_line})
            save_store(messages, last_session_end_ts)
            print("（已判定结束本轮话题/会话）\n")
            break

        # ── 事件提取：识别用户发言中的生日 / 具体时间 ──
        extracted = extract_events_from_text(user_line)
        for ev in extracted:
            t = ev.get("type", "event")
            desc = (ev.get("description") or "").strip()
            date_str = (ev.get("date") or "").strip()
            time_str = ev.get("time")
            if t == "birthday":
                add_birthday(date_str, desc)
            else:
                add_event(date_str, time_str, desc)

        messages.append({"role": "user", "content": user_line})
        save_store(messages, last_session_end_ts)

        api_msgs = _api_messages_for_session(messages)
        reply = deepseek_chat(api_msgs, temperature=0.7)
        if not reply or not reply.strip():
            print("模型无回复，结束本轮。")
            break
        reply = reply.strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"\n[助手] {reply}\n")
        save_store(messages, last_session_end_ts)

    ended_ts = time.time()
    save_store(messages, ended_ts)
    return messages, ended_ts
