# 图 RAG 烹饪助手包（Neo4j + Milvus + DeepSeek）
# 最早清除代理，避免 Bolt 经隧道时被劫持导致 incomplete handshake
import os
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,.local"
for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)
