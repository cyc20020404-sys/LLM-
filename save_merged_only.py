"""仅合并 LoRA 并保存到 merged_model（训练已完成时用，无需重新训练）"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from unsloth import FastLanguageModel

_script_dir = os.path.dirname(os.path.abspath(__file__))
_lora_dir = os.path.join(_script_dir, "lora_model")
_merged_dir = os.path.join(_script_dir, "merged_model")
max_seq_length = 4096

print("--- 从 lora_model 加载（含基础模型 + 适配器）---")
model, tokenizer = FastLanguageModel.from_pretrained(
    _lora_dir,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
model = FastLanguageModel.for_inference(model)
print("--- 保存合并后的完整模型 (merged_16bit) ---")
model.save_pretrained_merged(_merged_dir, tokenizer, save_method="merged_16bit")
print(f"已保存到 {_merged_dir}")
