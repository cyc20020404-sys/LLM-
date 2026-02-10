# DPO 训练数据（messages / chosen / rejected）

用于 DPO（Direct Preference Optimization）训练，缓解模型**不截断、泄露 think 标签、重复**等问题。

## 格式说明

每行一条 JSON：

- **messages**：对话上下文，到当前回复为止的若干轮 `{role, content}`（最后一轮为 user）
- **chosen**：`{role: "assistant", content: "..."}`，期望回复（已去掉 `<think>...</think>`）
- **rejected**：`{role: "assistant", content: "..."}`，规则合成的“坏回复”（含 think、重复、》》》建议、尾随英文等）

## 生成方式

从现有 **messages 格式**的 `train.jsonl` 生成 `train_dpo.jsonl`：

```bash
cd /root/autodl-tmp/emention_bot

# 仅从根目录 train.jsonl 生成
python3 data_conv/messages_to_dpo_jsonl.py train.jsonl -o train_dpo.jsonl

# 合并根目录 + data_conv/datagirl_train.jsonl + music/film/travel 的 train.jsonl 后生成（去重）
python3 data_conv/messages_to_dpo_jsonl.py --merge -o train_dpo.jsonl
```

可选参数：

- `--include-system`：把 system 内容拼进 prompt（默认仅 user）
- `-o`：输出路径，默认 `train_dpo.jsonl`

## 使用

将 `train_dpo.jsonl` 放入你的 DPO 训练流程（如 `trl.DPOTrainer`），按 `messages` / `chosen` / `rejected` 字段加载即可。
