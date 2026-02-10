#!/usr/bin/env python3
"""
将 data_conv/datagirl.xlsx 转为与 train 对应的 jsonl 格式。
每行：{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""
import json
import os
import pandas as pd

# 与项目根 train.jsonl 一致的系统提示
SYSTEM_PROMPT = "你是一个语气温柔、贴心的情感陪伴机器人，名字叫小团团。"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(script_dir, "datagirl.xlsx")
    out_path = os.path.join(script_dir, "datagirl_train.jsonl")

    df = pd.read_excel(xlsx_path, header=0)
    # 去掉表头说明行、空行
    df = df.dropna(subset=["user_intent", "语音专属content"])
    df["user_intent"] = df["user_intent"].astype(str).str.strip()
    df["语音专属content"] = df["语音专属content"].astype(str).str.strip()
    df = df[(df["user_intent"] != "") & (df["语音专属content"] != "")]
    # 去掉列名当数据的那一行
    df = df[~df["user_intent"].str.contains("用户真实问法", na=False)]
    df = df[~df["语音专属content"].str.contains("知识库标准答案", na=False)]

    lines = []
    for _, row in df.iterrows():
        user = (row["user_intent"] or "").strip()
        assistant = (row["语音专属content"] or "").strip()
        if not user or not assistant:
            continue
        obj = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已写入 {len(lines)} 条 → {out_path}")

if __name__ == "__main__":
    main()
