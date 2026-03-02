# 图 RAG 烹饪助手（DeepSeek + Milvus + Neo4j）

位于 **emention_bot/graph_rag_deepseek/**，在 emention_bot 项目下独立实现的升级版 RAG：智能查询路由、传统混合检索（Milvus + BM25）、图 RAG 检索（Neo4j 菜品图），生成与路由均使用 **DeepSeek API**。不修改同项目下原有 `rag_modules/` RAG。

## 环境要求

- Python 3.10+
- Docker 与 Docker Compose（用于 Neo4j + Milvus）
- 网络可访问 DeepSeek API、HuggingFace 或镜像（嵌入模型 BAAI/bge-small-zh-v1.5）

## 快速开始

### 1. 配置 API 密钥

**勿将 API key 提交到 git。**

```bash
cd emention_bot/graph_rag_deepseek
cp .env.example .env
# 编辑 .env，将 DEEPSEEK_API_KEY=your_deepseek_api_key 改为你的真实 key
```

### 2. 启动 Neo4j 与 Milvus

在本目录（`emention_bot/graph_rag_deepseek/`）下执行：

```bash
docker compose up -d
```

首次启动会自动导入 `data/cypher/` 中的 C9 菜品图数据到 Neo4j。等待约 1～2 分钟，确认服务就绪：

- Neo4j: http://localhost:7474（浏览器可打开）
- Neo4j Bolt: localhost:7687
- Milvus: localhost:19530

**本机没有完整项目时**：需先把远程的 `emention_bot/graph_rag_deepseek` 拷到本机（至少含 `docker-compose.yml` 和 `data/cypher/`），再在本机该目录执行 `docker compose up -d`，否则本机起的是别的 compose，没有 Neo4j。

**确认本机 7687 / 19530 已监听**（在本地 PowerShell 或 CMD 执行）：
- `netstat -an | findstr "7687"`、`netstat -an | findstr "19530"`：看到 `LISTENING` 即正常。
- Neo4j：浏览器打开 http://localhost:7474 能打开即 7687 可用。
- Milvus：`Test-NetConnection localhost -Port 19530`（PowerShell）显示 `TcpTestSucceeded : True` 即正常。

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

若在 AutoDL 等国内环境，可先设置 HuggingFace 镜像再安装：

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt
```

### 4. 运行交互式问答

```bash
python main.py
```

程序会依次：初始化 Neo4j / Milvus 连接、加载或构建知识库、进入交互循环。输入问题后会自动做智能路由（传统检索 / 图 RAG / 组合），并用 DeepSeek 生成回答。输入 `quit` 退出，`stats` 查看统计，`rebuild` 重建知识库。

## 目录结构

```
emention_bot/graph_rag_deepseek/
├── .env              # 本地配置（勿提交），含 DEEPSEEK_API_KEY
├── .env.example      # 示例，复制为 .env 后填写
├── config.py         # Neo4j / Milvus / DeepSeek 配置
├── main.py           # 入口
├── requirements.txt
├── README.md
├── docker-compose.yml    # Neo4j + Milvus（etcd + minio + standalone）
├── data/
│   └── cypher/           # C9 菜品图：nodes.csv, relationships.csv, neo4j_import.cypher
└── rag_modules/
    ├── graph_data_preparation.py
    ├── graph_indexing.py
    ├── graph_rag_retrieval.py
    ├── hybrid_retrieval.py
    ├── intelligent_query_router.py
    ├── milvus_index_construction.py
    └── generation_integration.py   # DeepSeek API
```

## 配置说明

- **Neo4j**：默认 `bolt://localhost:7687`，用户名 `neo4j`，密码 `all-in-rag`（与 C9 一致）。可通过环境变量 `NEO4J_URI`、`NEO4J_PASSWORD` 覆盖。
- **Milvus**：默认 `localhost:19530`。可通过 `MILVUS_HOST`、`MILVUS_PORT` 覆盖。
- **DeepSeek**：仅通过环境变量 `DEEPSEEK_API_KEY` 配置；模型与 base_url 在 `config.py` 中为 `deepseek-chat` 与 `https://api.deepseek.com/v1`。

## 常见问题

- **SSH 隧道方向（必读）**：Neo4j 在本机（如 Windows）时，要让远程（AutoDL）访问本机 Neo4j，必须用 **反向隧道**，且方向不能反。应在 **本机（Neo4j 所在机器）** 执行 `ssh -R`，把 **远程** 的 7687 转发到 **本机** 7687；不能在本机用 `-L` 把本机端口转到远程，也不能在远程执行“转发到本机”的命令（远程通常无法 SSH 回你本机）。正确示例（在本机执行）：
  ```bash
  ssh -o Compression=no -p <端口> -R 7687:127.0.0.1:7687 -R 19530:127.0.0.1:19530 root@<AutoDL连接地址>
  ```
  保持该终端不关，再在 AutoDL 上连接 `127.0.0.1:7687` 即可。
- **Neo4j Bolt "incomplete handshake response" / Bolt 握手超时**：先确认隧道方向正确（见上）。若方向无误仍失败：① 本机先 `python test_neo4j_bolt.py --local` 确认 Neo4j 正常；② 隧道加 `-o Compression=no`；③ 若仍失败，改用 **Neo4j Aura Free**（云托管）：在 [neo4j.com/cloud](https://neo4j.com/cloud) 创建免费实例，设置 `NEO4J_URI=neo4j+s://xxx.databases.neo4j.io` 和 `NEO4J_PASSWORD`，无需隧道。
- **连接 Milvus 失败**：确认 `docker compose up -d` 后 Milvus 健康（如 `docker compose ps`），再等约 30 秒后重试。
- **Neo4j 导入失败**：确认 `data/cypher/` 下存在 `nodes.csv`、`relationships.csv`、`neo4j_import.cypher`，且 neo4j-init 容器已执行完成（可查看 `docker compose logs neo4j-init`）。
- **DeepSeek 报错**：检查 `.env` 中 `DEEPSEEK_API_KEY` 是否正确、网络是否可访问 `api.deepseek.com`。
