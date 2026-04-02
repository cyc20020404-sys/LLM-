"""
AI 主动发起提问主入口。

文件结构：
  config.py      - 配置常量与 System Prompt
  api.py         - DeepSeek HTTP 调用
  store.py       - 聊天记录读写
  proactivity.py - 主动提问判断与开场白生成
  reminders.py   - 时间事件识别与提醒管理
  chat.py        - 本轮多轮对话（用户输入、话题边界、显式结束）
  api_talk.py    - 本文件：主循环 cycle_once + main

依赖: pip install httpx
环境变量: DEEPSEEK_API_KEY（必填）
可选: DEEPSEEK_API_URL, DEEPSEEK_MODEL, PROACTIVE_CHAT_HISTORY,
      PROACTIVE_INTERVAL_SECONDS（默认 10）,
      PROACTIVE_SESSION_COOLDOWN_SECONDS（默认 21）,
      PROACTIVE_FORCE_AFTER_COOLDOWN（默认 1：冷却结束后强制提问）,
      PROACTIVE_REMINDER_AHEAD_SECONDS（默认 60）,
      USE_TOPIC_EXIT_MODEL（默认 1：启用话题边界模型判断）
"""

from __future__ import annotations

import time

from chat import run_full_chat_session
from config import (
    CHAT_HISTORY_PATH,
    FORCE_ASK_AFTER_COOLDOWN,
    INTERVAL_SECONDS,
    SESSION_COOLDOWN_SECONDS,
)
from proactivity import generate_proactive_question, should_ask_question
from reminders import (
    check_due_reminders,
    generate_birthday_greeting,
    generate_event_reminder,
    mark_reminded,
)
from store import (
    cooldown_blocks,
    ensure_history_file_exists,
    load_store,
    messages_nonempty,
    seconds_until_cooldown_ok,
)


def cycle_once() -> None:
    """
    聊天触发判断循环（主流程）。

    ① 文件不存在 → 创建 → 触发提问
    ② 记录为空   → 触发提问
    ③ 记录非空 + 未过冷却 → 不触发，返回
    ④ 记录非空 + 已过冷却 → 触发提问（强制或模型判断）

    【提醒系统】在任何触发前先检查到期提醒：
      生日 → 生成祝福语；事件 → 生成提醒语；
      优先作为本轮开场白，触发后标记已提醒。
    """
    messages, file_existed, last_end = load_store()

    # ── 优先：检查到期提醒 ───────────────────────────────────────────────────
    birthdays, events = check_due_reminders()
    reminder_keys: list[str] = []
    reminder_opening: str | None = None

    if birthdays:
        desc = birthdays[0][2]
        greeting = generate_birthday_greeting(desc)
        reminder_opening = greeting.strip() if greeting else f"生日快乐，{desc}！🎂"
        date_str = birthdays[0][1]
        reminder_keys.append(f"bday:{date_str}")
        for b in birthdays[1:]:
            reminder_keys.append(f"bday:{b[1]}")

    elif events:
        desc = events[0][0]
        date_str = events[0][1]
        time_str = events[0][2]
        reminder_text = generate_event_reminder(desc, date_str)
        reminder_opening = (
            reminder_text.strip()
            if reminder_text
            else f"提醒：{desc}（{date_str}）"
        )
        reminder_keys.append(f"evt:{date_str}:{time_str or '00:00'}")
        for ev in events[1:]:
            reminder_keys.append(f"evt:{ev[1]}:{ev[2] or '00:00'}")

    # ── 触发判断（左列）──────────────────────────────────────────────────────
    if not file_existed:
        ensure_history_file_exists()
        messages = []
        last_end = None
        print("未找到聊天记录文件，已创建并触发提问。")
        trigger = True

    elif not messages_nonempty(messages):
        print("聊天记录为空，触发提问。")
        trigger = True

    else:
        if cooldown_blocks(last_end):
            wait = int(seconds_until_cooldown_ok(last_end)) + 1
            print(
                f"距离上一轮结束未满 {SESSION_COOLDOWN_SECONDS}s（约还需 {wait}s），"
                "跳过本轮触发判断。"
            )
            return

        if FORCE_ASK_AFTER_COOLDOWN:
            trigger = True
            print(
                f"冷却已满 {SESSION_COOLDOWN_SECONDS}s，强制触发提问（未调用「是否提问」模型）。"
            )
        else:
            trigger = should_ask_question(messages)
            print(f"模型判断是否提问: {trigger}")

    if not trigger:
        return

    # ── 确定开场白 ───────────────────────────────────────────────────────────
    if reminder_opening is not None:
        question = reminder_opening
        if birthdays:
            print("【提醒】检测到今日生日，发送祝福。")
        else:
            print(f"【提醒】检测到到期事件，发送提醒。")
    else:
        question = generate_proactive_question(messages)
        if not question:
            print("生成提问失败，本轮跳过。")
            return

    # ── 本轮多轮对话 ─────────────────────────────────────────────────────────
    messages, _ = run_full_chat_session(messages, question, last_end)

    # ── 标记已提醒 ───────────────────────────────────────────────────────────
    if reminder_keys:
        mark_reminded(reminder_keys)

    print("本轮已结束，记录已写入；等待下一轮检查。")


def main() -> None:
    force_note = (
        "冷却结束后强制提问；"
        if FORCE_ASK_AFTER_COOLDOWN
        else "冷却结束后由模型判断是否提问；"
    )
    print(
        f"【聊天触发判断循环】每 {INTERVAL_SECONDS}s 检查一次；"
        f"上一轮结束后须隔 {SESSION_COOLDOWN_SECONDS}s 才可能触发；"
        f"{force_note}\n"
        f"历史文件: {CHAT_HISTORY_PATH}"
    )
    while True:
        try:
            cycle_once()
        except KeyboardInterrupt:
            print("已退出。")
            break
        except Exception as e:
            print(f"本轮异常: {e}")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
