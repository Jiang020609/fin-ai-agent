"""RAG 检索服务 — 基于 LangChain + ChromaDB + OpenAI Embeddings"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

logger = logging.getLogger(__name__)

# ---------- 路径配置 ----------
_BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_DATA_DIR = _BASE_DIR / "data"
_CHROMA_DIR = _BASE_DIR / "chroma_data"

# ---------- 单例 ----------
_vectorstore: Chroma | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    """获取 OpenAI Embedding 模型实例。"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def _load_documents() -> list:
    """从 data/ 目录加载所有 Markdown 和文本文件。"""
    documents = []

    if not _DATA_DIR.exists():
        logger.warning("Data directory not found: %s", _DATA_DIR)
        return documents

    # 加载 Markdown 和文本文件
    file_patterns = ["**/*.md", "**/*.txt"]
    all_files: list[Path] = []
    for pattern in file_patterns:
        all_files.extend(_DATA_DIR.glob(pattern))

    for txt_file in all_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = txt_file.name
            documents.extend(docs)
        except Exception as e:
            logger.warning("Failed to load %s: %s", txt_file.name, e)

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def _chunk_documents(documents: list) -> list:
    """对文档进行分块处理。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """构建或加载向量数据库。

    如果 chroma_data/ 已存在且 force_rebuild=False，直接加载；
    否则重新从 data/ 目录读取文档、分块、向量化并持久化。
    """
    global _vectorstore
    embeddings = _get_embeddings()

    # 尝试加载已有数据库
    if not force_rebuild and _CHROMA_DIR.exists():
        try:
            _vectorstore = Chroma(
                persist_directory=str(_CHROMA_DIR),
                embedding_function=embeddings,
                collection_name="financial_knowledge",
            )
            count = _vectorstore._collection.count()
            if count > 0:
                logger.info("Loaded existing ChromaDB with %d documents", count)
                return _vectorstore
        except Exception as e:
            logger.warning("Failed to load existing ChromaDB: %s", e)

    # 重新构建
    logger.info("Building vector store from documents...")
    documents = _load_documents()
    if not documents:
        raise RuntimeError(f"No documents found in {_DATA_DIR}")

    chunks = _chunk_documents(documents)

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(_CHROMA_DIR),
        collection_name="financial_knowledge",
    )
    logger.info("Vector store built and persisted to %s", _CHROMA_DIR)
    return _vectorstore


def get_vectorstore() -> Chroma:
    """获取向量数据库实例（懒加载）。"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


def get_relevant_context(query: str, top_k: int = 4) -> str:
    """检索与查询最相关的文本片段。

    Args:
        query: 用户问题
        top_k: 返回的最相关片段数量

    Returns:
        拼接后的参考资料文本；若无匹配则返回空字符串
    """
    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return ""

    if not results:
        return ""

    # 过滤掉相关度太低的结果（阈值 0.3）
    filtered = [(doc, score) for doc, score in results if score >= 0.3]

    if not filtered:
        logger.info("No results above relevance threshold for: %s", query)
        return ""

    # 格式化输出
    context_parts = []
    for i, (doc, score) in enumerate(filtered, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[参考资料 {i}] (来源: {source}, 相关度: {score:.2f})\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
