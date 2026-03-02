"""
意图识别模块：基于 BERT 的轻量级分类器，不消耗 LLM Token。
"""
from .intent_model import (
    IntentClassifier,
    get_intent_classifier,
    predict_intent,
)

__all__ = ["IntentClassifier", "get_intent_classifier", "predict_intent"]
