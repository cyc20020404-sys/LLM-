#!/usr/bin/env python3
"""
将 gold_data_labeled.jsonl 转为 DPO 训练格式。

输入：instruction / output / intent
输出：messages / chosen / rejected（与 train_dpo.py 期望格式一致）

rejected 采用 intent 相关的「官方生硬」模板合成，与 chosen 的活泼温柔风格形成对比，
用于 DPO 偏好学习。
"""
import argparse
import json
import os
import re

# 各 intent 对应的「官方生硬」风格 rejected 模板（多句随机选，增加多样性）
REJECTED_TEMPLATES = {
    "吐槽抱怨": [
        "了解到您遇到了一些不便。建议您保持积极心态，适当调节情绪。如有持续困扰可寻求朋友或专业人士支持。",
        "您描述的情况属于生活中常见的挫折体验。建议您适当放松，通过运动或爱好来分散注意力。",
        "职场/生活中的压力较为常见。建议您合理规划时间，劳逸结合，必要时可与上级或同事沟通。",
    ],
    "求助安慰": [
        "了解到您目前情绪不佳。建议您通过运动、与朋友交流或休息来调节心情。如情况持续可考虑寻求专业心理咨询。",
        "情绪波动是正常的心理现象。建议您保持规律作息，适当倾诉，必要时寻求专业人士帮助。",
        "您的感受值得被重视。建议您先让自己放松下来，做一些喜欢的事，如需支持可向信任的人倾诉。",
    ],
    "恋爱情感": [
        "情感问题需要理性看待。建议您保持平常心，适时与对方沟通，如有困扰可向信任的人倾诉。",
        "人际与情感是人生的重要课题。建议您先理清自己的感受，再决定如何应对。",
        "感情之事不宜急于下结论。建议您给自己一点时间，必要时可与朋友或专业人士聊聊。",
    ],
    "追星娱乐": [
        "追星是个人爱好，建议适度投入。注意平衡娱乐与生活其他方面，保持理性消费。",
        "娱乐活动有助于放松心情，但建议合理安排时间，避免影响学业或工作。",
        "您的兴趣值得尊重。建议在享受娱乐的同时，也关注生活中的其他重要事项。",
    ],
    "职场学业": [
        "工作/学业压力较为常见。建议您合理规划时间，必要时与导师或上级沟通，保持劳逸结合。",
        "了解到您在学习/工作中遇到了一些挑战。建议您分解任务、循序渐进，必要时寻求他人支持。",
        "职场与学业中的困难需要积极面对。建议您制定可行计划，一步步推进。",
    ],
    "美食生活": [
        "很高兴您分享了用餐体验。饮食方面建议注意营养均衡，适量为宜，保持健康习惯。",
        "美食是生活的一部分。建议您享受美食的同时，也注意饮食的多样性和适度。",
        "感谢分享。如有饮食或健康方面的需求，可参考专业建议进行调整。",
    ],
    "轻松闲聊": [
        "感谢您的分享。如有其他需求，欢迎随时告知。",
        "了解到您分享了上述内容。我作为智能助手会尽力为您提供帮助。",
        "感谢信任。如有具体问题或需要支持的地方，请随时说明。",
    ],
    "分享喜悦": [
        "很高兴听到您的好消息。愿您保持积极心态，享受生活的美好时刻。",
        "感谢分享您的喜悦。正向情绪对身心健康有益，祝您一切顺利。",
        "为您感到高兴。愿这份快乐持续伴随您。",
    ],
}


def strip_think(content: str) -> str:
    """去掉 <think>...</think> 及其内容。"""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def build_rejected(intent: str, instruction: str, seed: int) -> str:
    """根据 intent 选取官方生硬风格模板作为 rejected。"""
    import random
    rng = random.Random(seed)
    templates = REJECTED_TEMPLATES.get(intent, REJECTED_TEMPLATES["轻松闲聊"])
    return rng.choice(templates)


def convert_line(obj: dict, line_idx: int) -> dict | None:
    """单条 gold 数据转为 DPO 格式。"""
    instruction = (obj.get("instruction") or "").strip()
    output = (obj.get("output") or "").strip()
    intent = (obj.get("intent") or "轻松闲聊").strip()

    if not instruction or not output:
        return None

    chosen_text = strip_think(output)
    if not chosen_text:
        return None

    seed = hash((instruction, output, line_idx))
    rejected_text = build_rejected(intent, instruction, seed)

    return {
        "messages": [{"role": "user", "content": instruction}],
        "chosen": {"role": "assistant", "content": chosen_text},
        "rejected": {"role": "assistant", "content": rejected_text},
    }


def main():
    ap = argparse.ArgumentParser(description="gold_data_labeled.jsonl → DPO 格式")
    ap.add_argument(
        "-i", "--input",
        default="intent_classifier/gold_data_labeled.jsonl",
        help="输入 gold jsonl 路径",
    )
    ap.add_argument(
        "-o", "--output",
        default="data_conv/data/gold_train_dpo.jsonl",
        help="输出 DPO jsonl 路径",
    )
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))  # 项目根目录
    in_path = args.input if os.path.isabs(args.input) else os.path.join(root_dir, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(root_dir, args.output)

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"未找到输入文件: {in_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    count = 0
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            dpo = convert_line(obj, idx)
            if dpo is None:
                continue
            f_out.write(json.dumps(dpo, ensure_ascii=False) + "\n")
            count += 1

    print(f"写出 {count} 条 DPO 数据 -> {out_path}")
    print("可用于 train_dpo.py，需将 DPO_DATA_FILE 改为 data_conv/data/gold_train_dpo.jsonl 或合并到 train_dpo.jsonl")


if __name__ == "__main__":
    main()
