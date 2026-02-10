import os
# 关键：彻底禁用容易报错的加速插件，改用稳定的下载模式
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# 若下载/训练太慢，可先在终端开启加速再运行本脚本：
#   source /etc/network_turbo && export HF_ENDPOINT=https://hf-mirror.com
#   python train_dpo.py

# ── 1. 导入 & Patch ──────────────────────────────────────────────
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
PatchDPOTrainer()  # 必须在 import DPOTrainer 之前调用

import torch
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import TrainerCallback, TrainerControl, TrainerState

# ── 2. 模型与参数 ────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))

# 直接基于预训练基座做 DPO（之前 SFT/LoRA 效果不佳，不再使用 merged_model）
model_name = "unsloth/deepseek-r1-distill-qwen-7b-unsloth-bnb-4bit"
print(f"使用预训练基座做 DPO: {model_name}")

max_seq_length = 2048  # DPO 需要同时编码 chosen + rejected，适当缩短

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# ── 3. LoRA 配置 ─────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 节省 30% 显存
    random_state=3407,
    max_seq_length=max_seq_length,
)

# ── 4. 加载 DPO 数据集 ──────────────────────────────────────────
DPO_DATA_FILE = "train_dpo.jsonl"

_data_path = os.path.join(_script_dir, DPO_DATA_FILE)
if not os.path.isfile(_data_path):
    raise FileNotFoundError(f"未找到 DPO 数据文件：{_data_path}")
_data_paths = [_data_path]
print(f"将加载 DPO 数据文件：{_data_paths}")

raw_dataset = load_dataset("json", data_files={"train": _data_paths}, split="train")
print(f"原始样本数：{len(raw_dataset)}")


# ── 5. 预处理：转为 DPOTrainer 期望的格式 ────────────────────────
# 关键：统一注入 system 人设，让模型明确学到「小团团」的 chosen 风格
DPO_SYSTEM_PROMPT = (
    "你是小团团，一个活泼温柔、像朋友一样聊天的AI助手。"
    "请用轻松自然的语气回答，可带emoji和网络用语，避免官方、生硬、模板化的表达。"
)

# trl DPOTrainer 期望 prompt / chosen / rejected 三列，
# 其中 chosen 和 rejected 为完整对话（含 prompt + response）的 messages 列表。
def preprocess_dpo(example):
    """
    原始格式:
      messages: [{role, content}, ...]   # 上下文（到 user 为止）
      chosen:   {role, content}          # 好回复
      rejected: {role, content}          # 坏回复
    转为 DPOTrainer 格式（统一注入 system 人设）:
      prompt:   [system, ...msgs]        # system + 上下文
      chosen:   [system, ...msgs, chosen]
      rejected: [system, ...msgs, rejected]
    """
    msgs = example["messages"]
    chosen_msg = example["chosen"]
    rejected_msg = example["rejected"]

    if isinstance(msgs, dict):
        msgs = [msgs]
    base = [{"role": "system", "content": DPO_SYSTEM_PROMPT}] + list(msgs)

    prompt = list(base)
    chosen = list(base) + [chosen_msg]
    rejected = list(base) + [rejected_msg]

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


dataset = raw_dataset.map(
    preprocess_dpo,
    remove_columns=raw_dataset.column_names,
    desc="预处理 DPO 数据",
)
print(f"预处理后样本数：{len(dataset)}")

# ── 5.5 Loss 早停回调：loss 降至阈值时停止，防止过拟合 ─────────────────
LOSS_STOP_THRESHOLD = 0.000001  # loss 降至此时停止，此时模型权重较好，避免过拟合

class EarlyStoppingLossCallback(TrainerCallback):
    """当 loss 降至 LOSS_STOP_THRESHOLD 及以下时停止训练，避免过拟合。"""
    def __init__(self, threshold: float = LOSS_STOP_THRESHOLD):
        self.threshold = threshold
        self._last_loss = None

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return control
        loss = float(logs["loss"])
        self._last_loss = loss
        if loss <= self.threshold:
            print(f"\n[早停] loss={loss:.4f} 已降至阈值 {self.threshold}，停止训练防止过拟合")
            control.should_training_stop = True
        return control

# ── 6. DPO 训练器 ───────────────────────────────────────────────
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,          # LoRA 时无需单独加载 ref_model，隐式使用基座
    train_dataset=dataset,
    processing_class=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=512,
    args=DPOConfig(
        beta=0.2,                             # 增大以强化 chosen/rejected 偏好信号
        per_device_train_batch_size=2,        # DPO 显存开销比 SFT 大
        gradient_accumulation_steps=2,        # 有效 batch = 2 * 2 = 4
        warmup_ratio=0.1,
        num_train_epochs=15,                  # 上限轮次，由 loss 早停控制实际停止
        learning_rate=1e-4,                   # 加大学习率，强化活泼语气学习
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,                     # 每步打印 loss，便于早停检测
        optim="adamw_8bit",
        seed=42,
        output_dir=os.path.join(_script_dir, "outputs_dpo"),
        save_strategy="epoch",
        save_total_limit=2,
    ),
    callbacks=[EarlyStoppingLossCallback(threshold=LOSS_STOP_THRESHOLD)],
)

# ── 7. 启动训练 ─────────────────────────────────────────────────
print("--- DPO 训练启动 ---")
print(f"  beta={dpo_trainer.args.beta}, lr={dpo_trainer.args.learning_rate}")
print(f"  epochs={dpo_trainer.args.num_train_epochs}, "
      f"batch={dpo_trainer.args.per_device_train_batch_size} × "
      f"grad_accum={dpo_trainer.args.gradient_accumulation_steps}")
print(f"  早停阈值: loss <= {LOSS_STOP_THRESHOLD} 时停止（防止过拟合）")
dpo_trainer.train()

# ── 8. 保存 LoRA 适配器（推理时直接加载，无需合并）──────────────────
_lora_dir = os.path.join(_script_dir, "lora_model_dpo")
model.save_pretrained(_lora_dir)
tokenizer.save_pretrained(_lora_dir)
print(f"DPO LoRA 适配器已保存到 {_lora_dir}")
print("Streamlit 可直接从 lora_model_dpo 加载（基础模型 + LoRA），无需合并步骤。")
