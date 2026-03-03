#!/usr/bin/env python3
"""
基于 gold_data_2000.jsonl 的意图规则标注。
将 instruction 文本打上意图标签，用于训练 BERT 意图分类模型。
不消耗 LLM Token，纯规则+关键词，适合冷启动。
"""
import json
import os
from collections import Counter
from typing import Optional

# 意图定义（与 streamlit 情感机器人场景匹配）
INTENTS = [
    "吐槽抱怨",   # 职场压力、生活琐事、社死、尴尬
    "分享喜悦",   # 开心的事、小确幸、感动
    "恋爱情感",   # 暗恋、前任、crush、恋爱
    "追星娱乐",   # 追星、爱豆、演唱会
    "职场学业",   # 工作、实习、答辩、论文
    "美食生活",   # 美食、外卖、食堂、奶茶
    "求助安慰",   # 需要安慰、求抱抱、情绪低落
    "轻松闲聊",   # 日常分享、无特定目的
]

# 关键词匹配（按优先级，先匹配到的优先；同一 intent 内关键词 OR）
# 易混场景：吐槽(职场吐槽) vs 职场(纯工作)，求助(情绪) vs 恋爱(心碎/前任)
KEYWORDS = {
    "吐槽抱怨": [
        "吐槽", "救命", "会谢", "社死", "尴尬", "烦死", "崩溃", "心梗",
        "真的会谢", "瞳孔地震", "咱就是说", "裂开", "窒息", "破防", "无语",
        "笑死", "气死", "委屈", "扎心", "好气", "心凉",
        "内卷", "磕头", "摸鱼", "摆烂", "离谱", "太坑", "坑爹", "离谱",
        "行为艺术", "玄学", "中招",
    ],
    "分享喜悦": [
        "开心", "感动", "幸福", "圆满", "进步", "可爱", "甜", "暖",
        "笑出声", "太懂", "绝了", "好类", "感动落泪", "超开心",
        "赢了", "抢到", "中奖", "成功", "闪闪发光",
    ],
    "恋爱情感": [
        "暗恋", "前任", "crush", "恋爱", "对象", "喜欢", "心动", "表白",
        "他回", "他发", "他点赞", "他记得", "男朋友", "女友", "官宣",
        "冷战", "分手", "失恋", "心碎", "傲娇", "恋爱脑", "表白墙",
    ],
    "追星娱乐": [
        "追星", "爱豆", "演唱会", "应援", "抢票", "偶像", "超话", "站姐",
        "小卡", "灯牌", "打投", "蹲直播", "接机", "粉圈",
    ],
    "职场学业": [
        "老板", "实习", "答辩", "论文", "小组", "作业", "导师", "公司",
        "团建", "需求", "PPT", "debug", "代码", "周报", "会议", "HR",
        "高数", "挂科", "选课", "考试", "简历", "面试", "转正",
    ],
    "美食生活": [
        "外卖", "食堂", "奶茶", "美食", "探店", "火锅", "泡面", "鸡腿",
        "关东煮", "煎饼", "沙拉", "辣条", "零食", "点单", "蛋饼", "菠萝包",
    ],
    "求助安慰": [
        "抱抱", "心疼", "累", "怎么办", "难过", "伤心", "想哭", "委屈",
        "好累", "压力", "焦虑", "慌", "紧张", "害怕", "迷茫", "无助",
        "撑不住", "心态崩", "顶不住", "扛不住", "想不开", "蹲路边",
    ],
    "轻松闲聊": [],  # 兜底，无特定关键词时归入此类
}


def _match_intent(text: str) -> Optional[str]:
    """规则匹配：返回第一个匹配到的意图，若多个则取优先级最高的。"""
    text_lower = text.strip().lower()
    matched = []
    for intent, keywords in KEYWORDS.items():
        if not keywords:
            continue
        for kw in keywords:
            if kw in text_lower:
                matched.append(intent)
                break
    if not matched:
        return "轻松闲聊"
    # 简单优先级：求助安慰 > 恋爱情感 > 追星 > 职场 > 吐槽 > 美食 > 分享 > 闲聊
    priority = {
        "求助安慰": 0, "恋爱情感": 1, "追星娱乐": 2, "职场学业": 3,
        "吐槽抱怨": 4, "美食生活": 5, "分享喜悦": 6, "轻松闲聊": 7,
    }
    return min(matched, key=lambda x: priority.get(x, 99))


def label_gold_data(
    in_path: str,
    out_path: str,
    label_key: str = "intent",
) -> dict:
    """
    对 gold_data jsonl 进行意图标注，输出带 intent 字段的 jsonl。
    返回统计信息。
    """
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"未找到输入文件: {in_path}")

    labeled = []
    counter = Counter()
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst = (obj.get("instruction") or "").strip()
            if not inst:
                continue
            intent = _match_intent(inst)
            obj[label_key] = intent
            labeled.append(obj)
            counter[intent] += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in labeled:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    stats = {
        "total": len(labeled),
        "distribution": dict(counter),
    }
    return stats


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_script_dir)
    in_path = os.path.join(_root, "data_conv", "data", "gold_data_2000.jsonl")
    out_path = os.path.join(_root, "intent_classifier", "gold_data_labeled.jsonl")

    stats = label_gold_data(in_path, out_path)
    print("意图标注完成:")
    print(f"  总数: {stats['total']}")
    for k, v in sorted(stats["distribution"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\n已保存到: {out_path}")
