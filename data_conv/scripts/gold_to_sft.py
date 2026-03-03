#!/usr/bin/env python3
"""
将 gold_data_2000.jsonl（instruction-output 格式）转为 SFT 所需的 messages 格式。
输出：gold_data_sft.jsonl，可直接用于 train.py 的 SFT 训练。
"""
import json
import os


def gold_to_messages(instruction: str, output: str) -> dict:
    """单条 gold_data 转为 messages 格式，注入小团团 system 人设。"""
    return {
        "messages": [
            {
                "role": "system",
                "content": "你是小团团，一个活泼温柔、像朋友一样聊天的AI助手。请用轻松自然的语气回答，可带emoji和网络用语。"
            },
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    in_path = os.path.join(data_dir, "gold_data_2000.jsonl")
    out_path = os.path.join(data_dir, "gold_data_sft.jsonl")

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"未找到输入文件: {in_path}")

    lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                instruction = (obj.get("instruction") or "").strip()
                output = (obj.get("output") or "").strip()
                if not instruction or not output:
                    continue
                msg_obj = gold_to_messages(instruction, output)
                lines.append(json.dumps(msg_obj, ensure_ascii=False))
            except json.JSONDecodeError:
                continue

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已转换 {len(lines)} 条 → {out_path}")


if __name__ == "__main__":
    main()
