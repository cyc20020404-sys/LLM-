"""
知识库模块 —— 多格式文件加载、文本分块、FAISS 向量索引构建与持久化。

支持的文件格式：PDF、Word（.docx）、Markdown、纯文本，以及
unstructured 兜底加载的其他常见格式。
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------

def _load_pdf(path: str) -> List[Document]:
    """使用 PyMuPDF 加载 PDF"""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        return PyMuPDFLoader(path).load()
    except ImportError:
        logger.warning("pymupdf 未安装，跳过 PDF: %s", path)
        return []


def _load_docx(path: str) -> List[Document]:
    """使用 docx2txt 加载 Word 文档"""
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(path).load()
    except ImportError:
        logger.warning("docx2txt 未安装，跳过 Word: %s", path)
        return []


def _load_markdown(path: str) -> List[Document]:
    """直接读取 Markdown 文件"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": path})]


def _load_text(path: str) -> List[Document]:
    """加载纯文本文件"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": path})]


def _load_unstructured(path: str) -> List[Document]:
    """使用 UnstructuredFileLoader 兜底加载"""
    try:
        from langchain_community.document_loaders import UnstructuredFileLoader
        return UnstructuredFileLoader(path).load()
    except Exception as e:
        logger.warning("Unstructured 加载失败 %s: %s", path, e)
        return []


# 后缀 -> 加载函数映射
# 注意：.doc 为老版 Word 二进制格式，docx2txt 只支持 .docx，故 .doc 用 Unstructured 兜底
_LOADER_MAP = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_unstructured,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".txt": _load_text,
    ".text": _load_text,
}


def load_documents(dir_path: str) -> List[Document]:
    """
    遍历目录，按后缀自动选择加载器，返回 Document 列表。

    Args:
        dir_path: 知识库文件夹路径

    Returns:
        所有文件解析后的 Document 列表
    """
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.is_dir():
        logger.warning("知识库目录不存在: %s", dir_path)
        return []

    documents: List[Document] = []
    for file_path in sorted(dir_path_obj.rglob("*")):
        if not file_path.is_file():
            continue
        # 跳过隐藏文件和 README
        if file_path.name.startswith(".") or file_path.name.upper() == "README.MD":
            continue

        suffix = file_path.suffix.lower()
        loader_fn = _LOADER_MAP.get(suffix, _load_unstructured)

        try:
            docs = loader_fn(str(file_path))
            # 统一填充 source 元数据（使用相对于知识库目录的路径）
            rel = file_path.relative_to(dir_path_obj)
            for doc in docs:
                doc.metadata["source"] = str(rel)
                doc.metadata["file_name"] = file_path.name
            documents.extend(docs)
            logger.info("已加载 %s (%d 段)", rel, len(docs))
        except Exception as e:
            logger.warning("加载失败 %s: %s", file_path, e)

    logger.info("知识库共加载 %d 个文件段落", len(documents))
    return documents


# ---------------------------------------------------------------------------
# 文本分块
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    对加载的文档进行分块。Markdown 文件按标题结构分块，其他文件使用
    RecursiveCharacterTextSplitter。

    Args:
        documents: 原始 Document 列表
        chunk_size: 通用分块大小
        chunk_overlap: 分块重叠长度

    Returns:
        分块后的 Document 列表
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )

    all_chunks: List[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "")
        is_md = source.lower().endswith((".md", ".markdown"))

        if is_md:
            # Markdown 按标题结构分块
            md_chunks = md_splitter.split_text(doc.page_content)
            for chunk in md_chunks:
                chunk.metadata.update(doc.metadata)
            # 如果 Markdown 分块后某块仍然过大，再用通用分块器切割
            for chunk in md_chunks:
                if len(chunk.page_content) > chunk_size * 1.5:
                    sub_chunks = text_splitter.split_documents([chunk])
                    all_chunks.extend(sub_chunks)
                else:
                    all_chunks.append(chunk)
        else:
            # 其他格式：通用递归分块
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

    # 为每个 chunk 编号
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    logger.info("分块完成，共 %d 个 chunk", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# FAISS 向量索引
# ---------------------------------------------------------------------------

def build_faiss_index(
    chunks: List[Document],
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
):
    """
    构建 FAISS 向量索引。

    Args:
        chunks: 分块后的 Document 列表
        embedding_model: HuggingFace embedding 模型名

    Returns:
        (FAISS vectorstore, HuggingFaceEmbeddings)
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    logger.info("FAISS 索引构建完成，包含 %d 个向量", len(chunks))
    return vectorstore, embeddings


def save_faiss_index(vectorstore, save_path: str):
    """将 FAISS 索引保存到磁盘"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_path)
    logger.info("FAISS 索引已保存到 %s", save_path)


def load_faiss_index(save_path: str, embeddings):
    """从磁盘加载 FAISS 索引"""
    from langchain_community.vectorstores import FAISS

    if not Path(save_path).exists():
        return None
    try:
        vs = FAISS.load_local(
            save_path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info("FAISS 索引已从 %s 加载", save_path)
        return vs
    except Exception as e:
        logger.warning("加载 FAISS 索引失败: %s", e)
        return None


# ---------------------------------------------------------------------------
# 目录指纹：检测文件变化
# ---------------------------------------------------------------------------

def _dir_fingerprint(dir_path: str) -> str:
    """
    计算目录指纹（基于文件列表 + 大小 + mtime），用于判断是否需要重建索引。
    """
    entries = []
    for fp in sorted(Path(dir_path).rglob("*")):
        if not fp.is_file() or fp.name.startswith(".") or fp.name.upper() == "README.MD":
            continue
        st = fp.stat()
        entries.append(f"{fp}|{st.st_size}|{int(st.st_mtime)}")
    return hashlib.md5("\n".join(entries).encode()).hexdigest()


def need_rebuild(dir_path: str, index_path: str) -> bool:
    """
    检测 knowledge_base/ 目录是否有变化，需要重建索引。

    比较方式：将目录指纹存入 index_path/.fingerprint，
    启动时对比当前指纹是否一致。
    """
    fp_file = os.path.join(index_path, ".fingerprint")
    current_fp = _dir_fingerprint(dir_path)

    if not os.path.isfile(fp_file):
        return True

    with open(fp_file, "r") as f:
        saved_fp = f.read().strip()

    return current_fp != saved_fp


def save_fingerprint(dir_path: str, index_path: str):
    """保存当前目录指纹"""
    Path(index_path).mkdir(parents=True, exist_ok=True)
    fp_file = os.path.join(index_path, ".fingerprint")
    with open(fp_file, "w") as f:
        f.write(_dir_fingerprint(dir_path))
