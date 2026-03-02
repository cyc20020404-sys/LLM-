#!/usr/bin/env python3
"""
基于 BERT 的意图分类模型训练。
使用 gold_data_labeled.jsonl 训练轻量级分类器，推理时不消耗 LLM Token，速度极快。
"""
import json
import os
from dataclasses import dataclass

# 使用 transformers + 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
LABELED_PATH = os.path.join(_ROOT, "intent_classifier", "gold_data_labeled.jsonl")
OUTPUT_DIR = os.path.join(_ROOT, "intent_classifier", "intent_model")
INTENTS = [
    "吐槽抱怨", "分享喜悦", "恋爱情感", "追星娱乐",
    "职场学业", "美食生活", "求助安慰", "轻松闲聊",
]
ID2LABEL = {i: lbl for i, lbl in enumerate(INTENTS)}
LABEL2ID = {lbl: i for i, lbl in enumerate(INTENTS)}
NUM_LABELS = len(INTENTS)

# 使用中文 BERT，体积小、推理快
MODEL_NAME = "bert-base-chinese"


def load_labeled_data(path: str) -> tuple[list[str], list[int]]:
    """加载标注数据，返回 (texts, label_ids)。"""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inst = (obj.get("instruction") or "").strip()
            intent = obj.get("intent", "轻松闲聊")
            if not inst or intent not in LABEL2ID:
                continue
            texts.append(inst)
            labels.append(LABEL2ID[intent])
    return texts, labels


def main():
    if not os.path.isfile(LABELED_PATH):
        raise FileNotFoundError(
            f"未找到标注文件: {LABELED_PATH}\n请先运行: python intent_classifier/label_rules.py"
        )

    print("加载标注数据...")
    texts, labels = load_labeled_data(LABELED_PATH)
    print(f"  样本数: {len(texts)}")

    print("加载 Tokenizer 与模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    dataset = Dataset.from_dict({
        "text": texts,
        "label": labels,
    })
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")

    # 划分训练/验证
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # 可改为 5+ 以提升鲁棒性
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print("开始训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存 label 映射，推理时使用
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, ensure_ascii=False)

    print(f"模型已保存到: {OUTPUT_DIR}")
    print(f"验证集评估: {trainer.evaluate()}")


if __name__ == "__main__":
    main()
