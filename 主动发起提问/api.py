"""本地模型调用封装（Ollama）。"""

from __future__ import annotations

import os

import ollama

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2-3b-emo")
TEMPERATURE_DEFAULT = float(os.getenv("LOCAL_MODEL_TEMPERATURE_DEFAULT", "0.7"))
TEMPERATURE_LOW = float(os.getenv("LOCAL_MODEL_TEMPERATURE_LOW", "0.1"))
TEMPERATURE_HIGH = float(os.getenv("LOCAL_MODEL_TEMPERATURE_HIGH", "0.8"))


def _map_temperature(temperature: float | None) -> float:
    if temperature is None:
        return TEMPERATURE_DEFAULT
    if temperature <= 0.15:
        return TEMPERATURE_LOW
    if temperature >= 0.75:
        return TEMPERATURE_HIGH
    return temperature


def deepseek_chat(
    messages: list[dict[str, str]],
    temperature: float | None = None,
) -> str | None:
    """
    与原 DeepSeek 接口兼容：输入消息列表，返回模型生成的文本内容。

    与 main.py 中的 ollama.chat() 用法完全一致：
        ollama.chat(
            model="qwen2-3b-emo",   # 模型名
            messages=[...],          # 消息列表
        )
    区别在于这里统一了 temperature 参数并做了异常处理。
    """
    temp = _map_temperature(temperature)

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": temp,
            },
        )
    except Exception as e:
        print(f"[Ollama] 请求异常: {e}")
        return None

    if not response or "message" not in response:
        return None

    content = response["message"].get("content", "")
    return content.strip() or None
