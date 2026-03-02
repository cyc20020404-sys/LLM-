# LLM- 陪伴机器人（emention_bot）

[![GitHub](https://img.shields.io/badge/GitHub-cyc20020404--sys%2FLLM--blue)](https://github.com/cyc20020404-sys/LLM-)

面向对话场景的 **陪伴型 AI 助手**，集成情感机器人、意图识别、RAG 知识库与图 RAG 烹饪助手，支持 LoRA/DPO 微调与 Streamlit 一站式对话界面。

---

## 一、项目概览

| 模块 | 说明 |
|------|------|
| **情感机器人** | 基于 DeepSeek-R1-Distill-Qwen 7B，支持 Gold SFT / DPO 微调，GGUF 本地推理 |
| **意图识别** | BERT 轻量分类器，识别吐槽/分享喜悦/恋爱情感等 8 类意图，按意图增强 prompt，不消耗 LLM Token |
| **RAG 知识库** | FAISS 向量 + BM25 混合检索，支持 PDF/Word/Txt/MD 文档加载 |
| **图 RAG** | Neo4j + Milvus + DeepSeek API，智能路由与烹饪领域问答 |

---

## 二、目录结构

```
emention_bot/
├── README.md                 # 本说明
├── requirements.txt          # 项目依赖
├── streamlit_app.py          # Streamlit 对话主应用
├── start_streamlit.py        # 启动脚本
├── train.py                  # SFT 微调（Unsloth + LoRA）
├── train_dpo.py              # DPO 微调
├── data_conv/                # 数据转换
│   ├── gold_to_sft.py        # gold_data → messages 格式
│   ├── messages_to_dpo_jsonl.py
│   ├── messages_to_kto_jsonl.py
│   └── excel_to_train_jsonl.py
├── rag_modules/              # RAG 检索（FAISS + BM25）
│   ├── knowledge_base.py     # 文档加载、分块、FAISS 索引
│   └── retriever.py          # 混合检索器
├── intent_classifier/        # 意图识别
│   ├── train_intent.py       # 训练 BERT 分类器
│   ├── label_rules.py        # 规则标注
│   ├── intent_model.py       # 推理封装
│   └── gold_data_labeled.jsonl
├── graph_rag_deepseek/       # 图 RAG 烹饪助手（Neo4j + Milvus + DeepSeek）
│   ├── main.py
│   ├── config.py
│   ├── docker-compose.yml    # Neo4j + Milvus
│   ├── data/cypher/          # 菜品图数据
│   └── rag_modules/
├── knowledge_base/            # RAG 文档目录（可自定义）
├── rag_index/                # FAISS 索引缓存（自动生成）
└── （训练输出）lora_model/, merged_model/, merged_model_dpo/, outputs/  # 见 .gitignore
```

---

## 三、快速开始

### 3.1 安装依赖

```bash
cd emention_bot
pip install -r requirements.txt
```

国内环境建议设置 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt
```

### 3.2 启动 Streamlit

```bash
python start_streamlit.py
```

或：

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### 3.3 训练数据格式

- **SFT**（`train.py`）：`data_conv/train.jsonl`，每行 `{"messages": [{"role":"system/user/assistant", "content":"..."}]}`
- **DPO**（`train_dpo.py`）：`data_conv/train_dpo.jsonl`，每行 `{"messages": [...], "chosen": {...}, "rejected": {...}}`

---

## 四、核心模块说明

### 4.1 情感机器人

| 功能 | 说明 |
|------|------|
| **SFT 微调** | `python train.py`，基于 `train.jsonl` 和 `gold_data_sft.jsonl`，输出 LoRA 到 `lora_model/` |
| **DPO 微调** | `python train_dpo.py`，基于 `train_dpo.jsonl`，输出到 `lora_model_dpo/`、`merged_model_dpo/` |
| **对话界面** | Streamlit 可选：Gold SFT 模型、DPO 模型、基础模型，支持 GGUF 本地推理 |

**模型选项**（按优先级）：
- Gold SFT：`merged_model/` 或 GGUF（`my_emotional_bot.Q4_K_M.gguf`）
- DPO：`merged_model_dpo/`（推荐）或 `lora_model_dpo/`

### 4.2 意图识别

按用户意图注入 prompt，不调用 LLM，推理极快。

**流程：**

1. 规则标注：`python intent_classifier/label_rules.py` → `gold_data_labeled.jsonl`
2. 训练：`python intent_classifier/train_intent.py` → `intent_classifier/intent_model/`
3. 推理：`from intent_classifier import predict_intent` → `(label, confidence)`

**意图类别**：吐槽抱怨、分享喜悦、恋爱情感、追星娱乐、职场学业、美食生活、求助安慰、轻松闲聊。

### 4.3 RAG 知识库

- 文档格式：PDF、Word、Txt、Markdown
- 检索：FAISS 向量 + BM25 关键词，RRF 融合
- 索引目录：`knowledge_base/`，缓存到 `rag_index/`

### 4.4 图 RAG 烹饪助手

独立子项目，详见 [graph_rag_deepseek/README.md](graph_rag_deepseek/README.md)。

- **依赖**：Neo4j、Milvus、DeepSeek API
- **启动**：`docker compose up -d`（在 `graph_rag_deepseek/` 下）→ `python main.py`
- **配置**：复制 `.env.example` 为 `.env`，填写 `DEEPSEEK_API_KEY`

---

## 五、数据准备

### 5.1 Gold 数据转 SFT 格式

```bash
python data_conv/gold_to_sft.py
# 输入：gold_data_2000.jsonl（instruction-output）
# 输出：gold_data_sft.jsonl（messages）
```

### 5.2 对话数据转 DPO

```bash
python data_conv/messages_to_dpo_jsonl.py [输入.jsonl] -o train_dpo.jsonl
```

### 5.3 Excel 转训练数据

```bash
python data_conv/excel_to_train_jsonl.py datagirl.xlsx -o train.jsonl
```

---

## 六、依赖与可选安装

| 用途 | 依赖 |
|------|------|
| 基础 | `requirements.txt` |
| 图 RAG | `pip install -r graph_rag_deepseek/requirements.txt` |
| 意图识别训练 | `pip install -r intent_classifier/requirements.txt` |
| HuggingFace 不可用 | `pip install modelscope`（Unsloth 走 ModelScope） |
| GGUF 推理 | `pip install llama-cpp-python`（GPU 版见下文） |

---

## 七、GGUF 使用 GPU（解决 CPU 占满）

在 AutoDL 等环境加载 GGUF 时若 CPU 占满、GPU 几乎不用，需安装 **CUDA 版** `llama-cpp-python`：

```bash
pip uninstall -y llama-cpp-python
source /etc/network_turbo   # AutoDL 学术加速
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --timeout 600
```

**GLIBCXX_3.4.30 报错**：用 `python start_streamlit.py` 启动（脚本会优先使用系统 libstdc++），或执行：

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

---

## 八、常见问题

| 问题 | 处理 |
|------|------|
| Neo4j Bolt incomplete handshake | `start_streamlit.py` 已设置 `no_proxy=127.0.0.1`；SSH 隧道需用反向隧道 `-R`，见 graph_rag README |
| HuggingFace 下载超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com`，或安装 `modelscope` 并 `UNSLOTH_USE_MODELSCOPE=1` |
| 意图识别不可用 | 需先运行 `train_intent.py` 生成 `intent_model/`，或关闭 Streamlit 中的「意图识别」开关 |

---

## 九、License

本项目仅供学习与交流使用。
