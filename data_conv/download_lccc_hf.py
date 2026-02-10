#!/usr/bin/env python3
"""
从 Hugging Face 下载 LCCC 到 data_conv/LCCC。
需可访问 Hugging Face（或设置 HF_ENDPOINT 镜像）。
"""
import os
import json

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装: pip install datasets")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "LCCC")
    os.makedirs(out_dir, exist_ok=True)

    # LCCC-base 较小，先下 base；若要 large 可改为 'large'
    print("正在从 Hugging Face 加载 LCCC-base（可能需要几分钟）...")
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset("silver/lccc", "base", split=split)
        except Exception as e:
            print(f"加载 {split} 失败: {e}")
            continue
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                # 格式按 LCCC 原始字段写入，若需转为 messages 格式可再写转换脚本
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"已写入 {out_path}，共 {len(ds)} 条")
    print(f"LCCC 已保存到 {out_dir}")

if __name__ == "__main__":
    main()
