import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 彻底离线
os.environ["TRANSFORMERS_OFFLINE"] = "1"

base_path = "/root/autodl-tmp/official_model" # 指向标准版
lora_path = "/root/autodl-tmp/lora_model"
save_path = "/root/autodl-tmp/final_merged_f16"

print("--- 正在加载 15GB 官方模型 (请耐心等待，约 3 分钟) ---")
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    device_map="cpu", # 内存够大，用 CPU 最稳
    local_files_only=True
)

print("--- 正在合并 LoRA 陪伴人设 ---")
model = PeftModel.from_pretrained(model, lora_path)
merged_model = model.merge_and_unload()

print(f"--- 正在保存全量 F16 模型 ---")
merged_model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(save_path)

print("--- 恭喜！全量模型合并成功，且没有任何量化残留 ---")