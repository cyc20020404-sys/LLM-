#!/usr/bin/env python3
"""
将 messages 格式的 train jsonl 转为 KTO 格式：
  prompt:     对话上下文（到当前 user 为止的若干轮 {role, content}）
  completion: {role: "assistant", content: "..."}
  label:      true（好回复）或 false（坏回复）
每条原始 (user, assistant) 对会生成 2 条 KTO：一条 label=true + 一条 label=false。
用于 KTO 训练以缓解不截断、think 泄露、重复等问题。
"""
import argparse
import json
import os
import random
import re

# ---------- 复用 DPO 脚本的 rejected 合成逻辑 ----------

TRAILING_ENGLISH = [
    " You are valuable and strong.",
    " Take care of you today.",
    " You matter so much more than you know.",
    " You are doing great!",
    " </assistant>",
]

SUGGESTION_NOISE = [
    "》》》建议使用深呼吸或冥想放松心情。》》》建议尝试着不愉快的事放在心里。",
    "》》》建议使用深呼吸。》》》建议尝试着不愉快。",
    "》》》建议给自己一点小奖励，奖励自己今天已经做得很好。》》》建议给自己一些空间。",
]


def strip_think(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def has_think(content: str) -> bool:
    return "<think>" in content and "</think>" in content


def reject_by_think_only(clean: str, raw: str) -> str | None:
    if has_think(raw) and strip_think(raw) == clean:
        return raw
    return None


def reject_by_repetition(clean: str) -> str:
    parts = re.split(r"[。！？\n]", clean)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return clean + random.choice(TRAILING_ENGLISH)
    first = parts[0] + "。"
    repeat = 2 + random.randint(0, 1)
    repeated = first * repeat
    rest = "".join(p + "。" for p in parts[1:3] if p) if len(parts) > 1 else ""
    return (repeated + " " + rest).strip() + random.choice(TRAILING_ENGLISH)


def reject_by_suggestion(clean: str) -> str:
    return clean + random.choice(SUGGESTION_NOISE) + random.choice(TRAILING_ENGLISH)


def reject_by_trailing_tag(clean: str) -> str:
    return clean + random.choice(TRAILING_ENGLISH)


def reject_by_off_topic(clean: str) -> str:
    if "。" in clean:
        before, _ = clean.split("。", 1)
        head = (before + "。").strip()
    else:
        head = clean
    return head + "我该怎么释放压力？" + random.choice(SUGGESTION_NOISE) + random.choice(TRAILING_ENGLISH)


def build_rejected(chosen: str, raw_assistant: str, seed: int) -> str:
    rng = random.Random(seed)
    if has_think(raw_assistant) and rng.random() < 0.4:
        cand = reject_by_think_only(chosen, raw_assistant)
        if cand:
            return cand
    strategies = [
        reject_by_repetition,
        reject_by_repetition,
        reject_by_suggestion,
        reject_by_trailing_tag,
        reject_by_off_topic,
    ]
    fn = rng.choice(strategies)
    return fn(chosen)


# ---------- KTO 转换逻辑 ----------

def _build_context(messages: list[dict], end_index: int, include_system: bool) -> list[dict]:
    ctx = []
    for k in range(end_index + 1):
        m = messages[k]
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            if include_system:
                ctx.append({"role": "system", "content": content})
            continue
        if role in ("user", "assistant"):
            ctx.append({"role": role, "content": content})
    return ctx


def messages_to_kto(messages: list[dict], include_system: bool = False) -> list[dict]:
    """
    每条 (user, assistant) 对 → 2 条 KTO：
      label=true  → completion 为去 think 的干净回复
      label=false → completion 为规则合成的坏回复
    """
    out = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = (m.get("role") or "").strip().lower()
        if role == "system":
            i += 1
            continue
        if role == "user":
            user_content = (m.get("content") or "").strip()
            j = i + 1
            while j < len(messages) and (messages[j].get("role") or "").strip().lower() != "assistant":
                j += 1
            if j < len(messages):
                assistant_content = (messages[j].get("content") or "").strip()
                if not user_content and not assistant_content:
                    i = j + 1
                    continue
                chosen_text = strip_think(assistant_content)
                if not chosen_text:
                    i = j + 1
                    continue
                context = _build_context(messages, i, include_system)
                seed = hash((user_content, assistant_content, len(out)))
                rejected_text = build_rejected(chosen_text, assistant_content, seed)
                # label=true: 好回复
                out.append({
                    "prompt": context,
                    "completion": {"role": "assistant", "content": chosen_text},
                    "label": True,
                })
                # label=false: 坏回复
                out.append({
                    "prompt": context,
                    "completion": {"role": "assistant", "content": rejected_text},
                    "label": False,
                })
            i = j + 1 if j < len(messages) else i + 1
        else:
            i += 1
    return out


def convert_file(in_path: str, out_path: str, include_system: bool = False) -> int:
    count = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages")
            if not msgs or not isinstance(msgs, list):
                continue
            for kto in messages_to_kto(msgs, include_system=include_system):
                f_out.write(json.dumps(kto, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="messages jsonl → KTO prompt/completion/label jsonl")
    ap.add_argument("input", nargs="?", default="train.jsonl", help="输入 messages jsonl 路径")
    ap.add_argument("-o", "--output", default="train_kto.jsonl", help="输出 KTO jsonl 路径")
    ap.add_argument("--include-system", action="store_true", help="将 system 加入 prompt")
    ap.add_argument("--merge", action="store_true", help="合并项目内多个 train.jsonl 再输出")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    if args.merge:
        inputs = [os.path.join(root_dir, "train.jsonl")]
        extra = [
            os.path.join(root_dir, "data_conv", "datagirl_train.jsonl"),
            os.path.join(root_dir, "data_conv", "music", "train.jsonl"),
            os.path.join(root_dir, "data_conv", "film", "train.jsonl"),
            os.path.join(root_dir, "data_conv", "travel", "train.jsonl"),
        ]
        for p in extra:
            if os.path.isfile(p):
                inputs.append(p)
        out_path = os.path.join(root_dir, args.output)
        total = 0
        seen = set()
        with open(out_path, "w", encoding="utf-8") as f_out:
            for in_path in inputs:
                if not os.path.isfile(in_path):
                    continue
                with open(in_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msgs = obj.get("messages")
                        if not msgs or not isinstance(msgs, list):
                            continue
                        for kto in messages_to_kto(msgs, include_system=args.include_system):
                            key = (tuple((m.get("role"), m.get("content")) for m in kto["prompt"]),
                                   kto["completion"]["content"], kto["label"])
                            if key in seen:
                                continue
                            seen.add(key)
                            f_out.write(json.dumps(kto, ensure_ascii=False) + "\n")
                            total += 1
        print(f"合并写出 {total} 条 KTO -> {out_path}")
    else:
        in_path = args.input
        if not os.path.isabs(in_path):
            in_path = os.path.join(root_dir, in_path)
        out_path = args.output
        if not os.path.isabs(out_path):
            out_path = os.path.join(root_dir, out_path)
        n = convert_file(in_path, out_path, include_system=args.include_system)
        print(f"写出 {n} 条 KTO -> {out_path}")


if __name__ == "__main__":
    main()
