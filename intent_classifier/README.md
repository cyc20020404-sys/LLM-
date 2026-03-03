# 意图识别模型 (Intent Classifier)

基于 gold_data 2000 条数据训练的 **RoBERTa-wwm 意图分类器**，用于在调用大模型前快速判断用户意图，**不消耗 LLM Token**，推理速度极快。

## 意图类别

| 类别 | 说明 |
|------|------|
| 吐槽抱怨 | 职场压力、生活琐事、社死、尴尬、内卷、摆烂 |
| 分享喜悦 | 开心的事、小确幸、感动 |
| 恋爱情感 | 暗恋、前任、crush、恋爱 |
| 追星娱乐 | 追星、爱豆、演唱会 |
| 职场学业 | 工作、实习、答辩、论文 |
| 美食生活 | 美食、外卖、食堂、奶茶 |
| 求助安慰 | 需要安慰、求抱抱、情绪低落 |
| 轻松闲聊 | 日常分享、无特定目的 |

## 使用流程

### 1. 标注数据（规则标注，零成本）

```bash
python intent_classifier/label_rules.py
```

输出：`intent_classifier/gold_data_labeled.jsonl`

可选：`python intent_classifier/label_rules.py --review` 导出每类样本供人工复核，修正误标后再训练。

### 2. 训练 RoBERTa 模型

```bash
pip install -r intent_classifier/requirements.txt
python intent_classifier/train_intent.py
```

输出：`intent_classifier/intent_model/`（模型权重 + tokenizer + label_map.json）

训练改进：stratified split、F1 指标、混淆矩阵、chinese-roberta-wwm-ext 基座。

### 3. 推理

```python
from intent_classifier import predict_intent

label, conf = predict_intent("公司团建玩你画我猜，内卷俩字我画了三轮...")
# ('吐槽抱怨', 0.92)
```

## 集成到 Streamlit

可在 `streamlit_app.py` 中按意图做路由或调整 prompt，例如：

```python
from intent_classifier import predict_intent

intent, conf = predict_intent(user_input)
if intent == "求助安慰":
    # 注入更多共情类提示词
    ...
```

## 依赖

```
transformers>=4.30
torch>=2.0
datasets>=2.14
scikit-learn>=1.0
```
