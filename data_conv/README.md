# data_conv - 数据与格式转换

## 目录结构

```
data_conv/
├── data/          # 数据文件（jsonl、xlsx 等）
│   ├── train.jsonl
│   ├── train_dpo.jsonl
│   ├── train_kto.jsonl
│   ├── gold_data_2000.jsonl
│   ├── gold_data_sft.jsonl
│   ├── gold_train_dpo.jsonl
│   ├── film/, music/, travel/  # 领域数据
│   └── LCCC/      # 下载的 LCCC 数据集
└── scripts/       # 格式转换脚本
    ├── gold_to_dpo.py        # gold_data_labeled → DPO
    ├── gold_to_sft.py        # gold_data_2000 → SFT messages
    ├── excel_to_train_jsonl.py
    ├── messages_to_dpo_jsonl.py
    ├── messages_to_kto_jsonl.py
    └── download_lccc_*.py
```

## 转换脚本说明

- **gold_to_dpo.py**：将 intent_classifier 的 gold_data_labeled.jsonl 转为 DPO 格式，输出到 `data/gold_train_dpo.jsonl`
- **gold_to_sft.py**：将 gold_data_2000.jsonl 转为 SFT messages 格式
- **messages_to_dpo_jsonl.py** / **messages_to_kto_jsonl.py**：messages 格式 → DPO/KTO 格式
