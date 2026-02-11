"""
检索模块 —— 混合检索（FAISS 向量 + BM25 关键词）+ RRF 重排。
"""

import logging
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器：FAISS 向量检索 + BM25 关键词检索，通过 RRF 融合排序。"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document], bm25_k: int = 10):
        """
        Args:
            vectorstore: 已构建的 FAISS 向量存储
            chunks: 用于构建 BM25 索引的文档块列表
            bm25_k: BM25 检索器返回的候选数量
        """
        self.vectorstore = vectorstore
        self.vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": bm25_k},
        )
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=bm25_k)
        logger.info(
            "HybridRetriever 初始化完成（向量 + BM25，候选各 %d）", bm25_k
        )

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索：分别调用向量检索和 BM25，然后 RRF 重排取 Top-K。

        Args:
            query: 用户查询
            top_k: 最终返回数量

        Returns:
            按相关性排序的 Document 列表
        """
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    # ---------------------------------------------------------------- RRF 重排
    @staticmethod
    def _rrf_rerank(
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion（RRF）算法：

            score(doc) = sum( 1 / (k + rank_i) )  对每个检索源 i

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25 检索结果
            k: RRF 平滑参数

        Returns:
            按融合分数降序排列的 Document 列表
        """
        doc_scores: dict = {}
        doc_objects: dict = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_objects[did] for did in sorted_ids if did in doc_objects]


# ---------------------------------------------------------------- 格式化工具
def format_context(docs: List[Document], max_chars: int = 2000) -> str:
    """
    将检索到的文档块格式化为可注入 prompt 的纯文本。

    Args:
        docs: 检索到的 Document 列表
        max_chars: 上下文最大字符数

    Returns:
        格式化后的字符串，可直接拼入 system prompt
    """
    if not docs:
        return ""

    parts: List[str] = []
    total = 0
    for doc in docs:
        source = doc.metadata.get("file_name", doc.metadata.get("source", "未知"))
        text = doc.page_content.strip()
        block = f"[来源: {source}]\n{text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)
