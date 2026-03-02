"""
意图识别模型推理封装。
加载训练好的 BERT 分类器，对用户输入进行意图预测，不消耗 LLM Token。
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch


class IntentClassifier:
    """基于 BERT 的意图分类器。"""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if model_dir is None:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(_root, "intent_classifier", "intent_model")
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._id2label = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()

        label_path = os.path.join(self.model_dir, "label_map.json")
        if os.path.isfile(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._id2label = data.get("id2label", {})
        else:
            self._id2label = {str(i): lbl for i, lbl in enumerate(
                ["吐槽抱怨", "分享喜悦", "恋爱情感", "追星娱乐",
                 "职场学业", "美食生活", "求助安慰", "轻松闲聊"])}

    def predict(self, text: str) -> tuple[str, float]:
        """
        预测单条文本的意图。
        Returns:
            (intent_label, confidence)
        """
        self._load()
        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)
            pred_id = probs.argmax(dim=-1).item()
            conf = probs[0, pred_id].item()
        label = self._id2label.get(str(pred_id), "轻松闲聊")
        return label, conf

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """批量预测，适合一次处理多条。"""
        self._load()
        inputs = self._tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)
            pred_ids = probs.argmax(dim=-1)
            confs = probs.gather(1, pred_ids.unsqueeze(1)).squeeze(1)
        results = []
        for pid, c in zip(pred_ids.tolist(), confs.tolist()):
            label = self._id2label.get(str(pid), "轻松闲聊")
            results.append((label, c))
        return results


# 全局单例，懒加载
_classifier: Optional[IntentClassifier] = None


def get_intent_classifier(model_dir: Optional[str] = None) -> IntentClassifier:
    """获取意图分类器单例。"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier(model_dir=model_dir)
    return _classifier


def predict_intent(text: str) -> tuple[str, float]:
    """便捷函数：预测单条文本意图。"""
    return get_intent_classifier().predict(text)
