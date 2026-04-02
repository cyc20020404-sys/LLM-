"""配置文件：环境变量读取、路径常量、System Prompt 模板。"""

from __future__ import annotations

import os
from pathlib import Path

# ──── 路径 ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY = SCRIPT_DIR / "proactive_chat_history.json"
DEFAULT_REMINDERS = SCRIPT_DIR / "reminders.json"

# ──── DeepSeek（已弃用，保留兼容）───────────────────────────────────────────

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "")

# ──── 本地模型（transformers + gguf，无需 llama-cpp-python）─────────────────────
LOCAL_MODEL_PATH = "D:\\PythonProjects\\Sentiment-Analysis2\\LLM-\\model\\qwen2-3b-q4_k_m\\qwen2-3b-q4_k_m.gguf"
# GGUF 模型文件路径（必填），如 D:/LLM/Qwen2-0.5B.Q3_K_M.gguf
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", LOCAL_MODEL_PATH).strip()
# 推理设备：cpu / cuda（无独显请保持 cpu）
LOCAL_MODEL_N_CTX = int(os.getenv("LOCAL_MODEL_N_CTX", "2048"))
LOCAL_MODEL_N_GPU_LAYERS = int(os.getenv("LOCAL_MODEL_N_GPU_LAYERS", "0"))
LOCAL_MODEL_DEVICE = os.getenv("LOCAL_MODEL_DEVICE", "cpu").strip()
# 生成参数
LOCAL_MODEL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MODEL_MAX_NEW_TOKENS", "256"))
LOCAL_MODEL_TEMPERATURE_DEFAULT = float(os.getenv("LOCAL_MODEL_TEMPERATURE_DEFAULT", "0.7"))
LOCAL_MODEL_TEMPERATURE_LOW = float(os.getenv("LOCAL_MODEL_TEMPERATURE_LOW", "0.1"))
LOCAL_MODEL_TEMPERATURE_HIGH = float(os.getenv("LOCAL_MODEL_TEMPERATURE_HIGH", "0.8"))

# ──── 主动提问循环 ───────────────────────────────────────────────────────────

CHAT_HISTORY_PATH = Path(os.getenv("PROACTIVE_CHAT_HISTORY", str(DEFAULT_HISTORY)))
INTERVAL_SECONDS = int(os.getenv("PROACTIVE_INTERVAL_SECONDS", "10"))
SESSION_COOLDOWN_SECONDS = int(os.getenv("PROACTIVE_SESSION_COOLDOWN_SECONDS", "21"))
SESSION_IDLE_TIMEOUT_SECONDS = int(os.getenv("PROACTIVE_SESSION_IDLE_TIMEOUT_SECONDS", "30"))
FORCE_ASK_AFTER_COOLDOWN = os.getenv("PROACTIVE_FORCE_AFTER_COOLDOWN", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
USE_TOPIC_EXIT_MODEL = os.getenv("USE_TOPIC_EXIT_MODEL", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

# ──── 提醒系统 ────────────────────────────────────────────────────────────────

REMINDERS_PATH = Path(os.getenv("PROACTIVE_REMINDERS_PATH", str(DEFAULT_REMINDERS)))
REMINDER_AHEAD_SECONDS = int(os.getenv("PROACTIVE_REMINDER_AHEAD_SECONDS", "60"))

# ──── System Prompt ──────────────────────────────────────────────────────────

DECISION_SYSTEM = (
    "你是对话节奏助手。根据下方历史记录判断：当前是否适合由助手主动发起一个新问题"
    "（例如长时间无互动、话题可自然延续、或需要关心用户）。\n"
    "只输出一个词：True 或 False。不要其它文字。"
)

QUESTION_SYSTEM = (
    "你是主动关心用户的助手。仔细阅读历史聊天记录，结合上下文生成**一句**自然、简短、"
    "有针对性的主动提问或新话题；避免泛泛而谈。\n"
    "只输出这一句话，不要前缀说明。"
)

CHAT_SESSION_SYSTEM = (
    "你是主动关心用户的助手。根据当前对话自然、简短、共情地回复；可适当追问或给建议。"
    "保持口语化，单次回复不要过长。"
)

TOPIC_EXIT_SYSTEM = (
    "你是会话边界判断器。下面「开场白」是本轮助手主动发起的主题。\n"
    "看用户最新一句：若表示不想继续聊、结束对话、先告一段落、换话题、拒绝回应，"
    "或与开场主题明显无关且暗示不想继续闲聊，只输出 END；\n"
    "若仍在正常接话、回应或展开当前话题，只输出 CONTINUE。\n"
    "只输出一个词：END 或 CONTINUE。不要其它文字。"
)

BIRTHDAY_GREETING_SYSTEM = (
    "用户今天生日！请生成一句温馨、真诚、简短的生日快乐祝福。"
    "不要前缀说明，直接输出一句话。"
)

EVENT_REMINDER_SYSTEM = (
    "以下事件即将到来或已经到达提醒时间。请根据事件内容，生成一句自然、简短、"
    "友好的提醒，语气像朋友间的善意提示，不要生硬。\n"
    "只输出这一句提醒，不要前缀说明。"
)

EVENT_EXTRACTION_SYSTEM = (
    "你是时间与日程信息提取助手。仔细阅读用户的发言，识别其中涉及的具体时间点或日期。\n"
    "识别以下两类信息：\n"
    "1. 生日：包含「生日」「出生日期」「几号生日」等关键词的描述。\n"
    "2. 一次性具体时间事件：指定了具体日期和时间的行为，如会议、约会、预约、吃药、"
    "运动、课程、任务截止等（如「明天上午10点开会」「下周三下午3点去爬山」）。\n\n"
    "忽略以下情况：\n"
    "- 模糊时间描述（改天、有空、以后再说、尽快）\n"
    "- 日常重复性极高的事件（每天几点起床这类）\n"
    "- 过去的时间点（昨天、上周等已过期的时间描述）\n\n"
    "如果发言中没有识别到任何生日或具体时间事件，输出 JSON：\n"
    '{"events": []}\n\n'
    "如果识别到事件，按以下 JSON 格式输出（数组，可包含多个事件）：\n"
    '{\n  "events": [\n    {\n      "type": "birthday | event",\n      "description": "事件简述，10字以内",\n      "date": "YYYY-MM-DD 或 MM-DD（生日只用月日）",\n      "time": "HH:MM（可选，24小时制，未指定则填 null）"\n    }\n  ]\n}\n\n'
    "只输出 JSON，不要任何解释文字。"
)
