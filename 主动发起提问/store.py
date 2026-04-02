"""聊天记录持久化：读取、写入、格式化。"""

from __future__ import annotations

import json
import time
from typing import Any

from config import CHAT_HISTORY_PATH


def load_store() -> tuple[list[dict[str, Any]], bool, float | None]:
    """
    返回 (messages, file_existed, last_session_end_ts)。
    last_session_end_ts 为 None 表示尚无「上一轮结束」记录（不挡冷却）。
    """
    if not CHAT_HISTORY_PATH.exists():
        return [], False, None
    try:
        raw = CHAT_HISTORY_PATH.read_text(encoding="utf-8")
        if not raw.strip():
            return [], True, None
        obj = json.loads(raw)
        msgs = obj.get("messages")
        if not isinstance(msgs, list):
            msgs = []
        ts = obj.get("last_session_end_ts")
        last_end: float | None = None
        if ts is not None:
            try:
                last_end = float(ts)
            except (TypeError, ValueError):
                last_end = None
        return msgs, True, last_end
    except (json.JSONDecodeError, OSError) as e:
        print(f"读取历史失败，将视为空: {e}")
        return [], True, None


def save_store(messages: list[dict[str, Any]], last_session_end_ts: float | None) -> None:
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"messages": messages}
    if last_session_end_ts is not None:
        out["last_session_end_ts"] = last_session_end_ts
    CHAT_HISTORY_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_history_file_exists() -> None:
    if not CHAT_HISTORY_PATH.exists():
        save_store([], None)


def messages_nonempty(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(m, dict) and (m.get("content") or "").strip()
        for m in messages
    )


def format_history_for_prompt(messages: list[dict[str, Any]]) -> str:
    if not messages_nonempty(messages):
        return "(暂无聊天记录)"
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = {"user": "用户", "assistant": "助手", "system": "系统"}.get(role, role)
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(暂无聊天记录)"


def cooldown_blocks(last_session_end_ts: float | None) -> bool:
    from config import SESSION_COOLDOWN_SECONDS
    if last_session_end_ts is None:
        return False
    elapsed = time.time() - last_session_end_ts
    return elapsed < SESSION_COOLDOWN_SECONDS


def seconds_until_cooldown_ok(last_session_end_ts: float | None) -> float:
    from config import SESSION_COOLDOWN_SECONDS
    if last_session_end_ts is None:
        return 0.0
    need = SESSION_COOLDOWN_SECONDS - (time.time() - last_session_end_ts)
    return max(0.0, need)
