"""
图 RAG 系统配置：Neo4j + Milvus + DeepSeek API
API 密钥仅通过环境变量 DEEPSEEK_API_KEY 配置，勿写入本文件或提交仓库。
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GraphRAGConfig:
    """图 RAG 系统配置类"""

    # Neo4j（与 C9 一致，便于复用菜品图数据）
    # 这里默认使用 127.0.0.1，避免某些环境下 localhost 优先解析到 ::1（IPv6）导致连接失败
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # Milvus
    # 同理，默认走 IPv4 回环，配合 SSH 反向转发更稳定
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5

    # 嵌入模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # DeepSeek API（从环境变量读取 key，勿写死在代码中）
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"

    # 检索与生成
    top_k: int = 5
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图与分块
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2

    def __post_init__(self):
        # 允许通过环境变量覆盖
        if os.getenv("NEO4J_URI"):
            self.neo4j_uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_PASSWORD"):
            self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        if os.getenv("MILVUS_HOST"):
            self.milvus_host = os.getenv("MILVUS_HOST")
        if os.getenv("MILVUS_PORT"):
            self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphRAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "neo4j_database": self.neo4j_database,
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "milvus_collection_name": self.milvus_collection_name,
            "milvus_dimension": self.milvus_dimension,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_graph_depth": self.max_graph_depth,
        }


DEFAULT_CONFIG = GraphRAGConfig()
