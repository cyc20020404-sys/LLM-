import os
# 关键：彻底禁用容易报错的加速插件，改用稳定的下载模式
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# 若下载/训练太慢，可先在终端开启加速再运行本脚本：
#   source /etc/network_turbo && export HF_ENDPOINT=https://hf-mirror.com
#   python train.py 

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 模型与参数 (5090 32GB 显存很充裕)
model_name = "unsloth/deepseek-r1-distill-qwen-7b-unsloth-bnb-4bit"
max_seq_length = 4096  # 5090 可以支持更长对话

# 2. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 3. LoRA 配置 (增加秩到 32，让陪伴机器人的语气更细腻)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
)


# 1. 定义格式化函数：支持 messages 格式（system/user/assistant）
def formatting_prompts_func(examples):
    # 兼容单条与批量：单条时 examples["messages"] 为 [msg, msg, ...]，批量时为 [[msg, msg], ...]
    raw = examples["messages"]
    if not raw:
        return []
    convs = [raw] if isinstance(raw[0], dict) else raw
    texts = []
    for messages in convs:
        parts = []
        for msg in messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        texts.append("\n".join(parts))
    return texts

# 2. 数据集配置：可配置多个 json/jsonl 文件，会合并后用于训练
_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = [
    "train.jsonl",           # 主训练集
    # "train_extra.jsonl",   # 可追加更多文件
    # "data/emotion_qa.jsonl",
]


def _resolve_data_path(p: str) -> str:
    """将相对路径解析为基于脚本目录的绝对路径。"""
    path = p.strip()
    if not os.path.isabs(path):
        path = os.path.join(_script_dir, path)
    return path


# 解析并过滤存在的文件
_data_paths = [_resolve_data_path(f) for f in DATA_FILES]
_data_paths = [p for p in _data_paths if os.path.isfile(p)]
if not _data_paths:
    raise FileNotFoundError(
        f"未找到任何数据文件。请检查 DATA_FILES 中的路径是否正确：{DATA_FILES}"
    )
print(f"将加载以下数据文件（共 {len(_data_paths)} 个）：{_data_paths}")

# 加载并合并多个文件为单一 train split
dataset = load_dataset("json", data_files={"train": _data_paths}, split="train")
print(f"合并后样本数：{len(dataset)}")

# 3. 配置针对 5090 优化的微调器
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func = formatting_prompts_func, # 这里的函数现在返回的是列表
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 4,   # 5090 32GB 显存，算力非常充裕
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,                   # 数据集较小，先进行快速迭代测试
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = os.path.join(_script_dir, "outputs"),
    ),
)
# 6. 启动
print("--- 训练启动，人设注入中 ---")
trainer.train()

# 7. 保存 LoRA 适配器（便于后续继续微调或切换）
_lora_dir = os.path.join(_script_dir, "lora_model")
model.save_pretrained(_lora_dir)
tokenizer.save_pretrained(_lora_dir)
print(f"LoRA 适配器已保存到 {_lora_dir}")

# 8. 合并 LoRA 并保存完整参数（单文件推理用）
_merged_dir = os.path.join(_script_dir, "merged_model")
model = FastLanguageModel.for_inference(model)
model.save_pretrained_merged(_merged_dir, tokenizer, save_method="merged_16bit")
print(f"完整参数（合并后）已保存到 {_merged_dir}")