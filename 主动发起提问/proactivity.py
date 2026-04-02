"""主动提问判断与开场白生成。"""

from __future__ import annotations

import re
from typing import Any

from api import deepseek_chat
from config import DECISION_SYSTEM, QUESTION_SYSTEM
from store import format_history_for_prompt


def parse_decision(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip().splitlines()[0].strip().lower()
    if t.startswith("true") or t == "是" or t == "yes":
        return True
    if t.startswith("false") or t == "否" or t == "no":
        return False
    if re.search(r"\btrue\b", t):
        return True
    return False


def should_ask_question(messages: list[dict[str, Any]]) -> bool:
    history_text = format_history_for_prompt(messages)
    reply = deepseek_chat(
        [
            {"role": "system", "content": DECISION_SYSTEM},
            {
                "role": "user",
                "content": f"以下为历史聊天记录：\n{history_text}\n\n是否现在由助手主动提问？",
            },
        ],
        temperature=0.3,
    )
    return parse_decision(reply)


def generate_proactive_question(messages: list[dict[str, Any]]) -> str | None:
    history_text = format_history_for_prompt(messages)
    msgs = [
        {"role": "system", "content": QUESTION_SYSTEM},
        {
            "role": "user",
            "content": f"历史聊天记录：\n{history_text}\n\n请结合上下文生成一句主动提问。",
        },
    ]
    return deepseek_chat(msgs, temperature=0.8)
