"""RAG 检索服务 — 混合检索 (BM25 + Vector) + LLM 重排序 + PDF 解析"""

import json
import logging
import os
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os as _os
_os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # 关闭 Chroma telemetry

from langchain_chroma import Chroma

load_dotenv()

logger = logging.getLogger(__name__)

# ---------- 路径配置 ----------
_BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_DATA_DIR = _BASE_DIR / "data"
_CHROMA_DIR = _BASE_DIR / "chroma_data"
_BM25_INDEX_PATH = _CHROMA_DIR / "bm25_index.pkl"
_BM25_CHUNKS_PATH = _CHROMA_DIR / "bm25_chunks.pkl"

# ---------- 单例 ----------
_vectorstore: Chroma | None = None
_bm25_index = None  # BM25Okapi | None
_bm25_chunks: list[Document] | None = None


# ---------- 配置读取 ----------
def _get_hybrid_weight() -> float:
    return float(os.getenv("HYBRID_WEIGHT", "0.5"))


def _is_rerank_enabled() -> bool:
    return os.getenv("RERANK_ENABLED", "true").lower() in ("true", "1", "yes")


def _get_rerank_top_k() -> int:
    return int(os.getenv("RERANK_TOP_K", "3"))


# ---------- Embedding ----------
def _get_embeddings() -> OpenAIEmbeddings:
    """获取 Embedding 模型实例，支持 OpenRouter 等兼容 API。"""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=base_url,
    )


# ================================================================
#  PDF 解析
# ================================================================

def _table_to_markdown(table: list[list[str | None]]) -> str:
    """将 pdfplumber 提取的表格（二维列表）转为 Markdown 表格。"""
    if not table or not table[0]:
        return ""
    # 清洗 None
    cleaned = [[cell or "" for cell in row] for row in table]
    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = [("| " + " | ".join(row) + " |") for row in cleaned[1:]]
    return "\n".join([header, separator, *rows])


def _load_pdf(pdf_path: Path) -> list[Document]:
    """使用 pdfplumber 逐页提取文本 + 表格，每页生成一个 Document。"""
    import pdfplumber

    docs: list[Document] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                parts: list[str] = []

                # 提取文本
                text = page.extract_text()
                if text:
                    parts.append(text)

                # 提取表格
                tables = page.extract_tables()
                for tbl in tables:
                    md_table = _table_to_markdown(tbl)
                    if md_table:
                        parts.append(md_table)

                if parts:
                    content = "\n\n".join(parts)
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "type": "pdf",
                        },
                    ))
    except Exception as e:
        logger.warning("Failed to load PDF %s: %s", pdf_path.name, e)

    return docs


# ================================================================
#  文档加载 & 分块
# ================================================================

def _load_documents() -> list[Document]:
    """从 data/ 目录加载所有 Markdown、文本和 PDF 文件。"""
    documents: list[Document] = []

    if not _DATA_DIR.exists():
        logger.warning("Data directory not found: %s", _DATA_DIR)
        return documents

    # 加载 Markdown 和文本文件
    text_patterns = ["**/*.md", "**/*.txt"]
    text_files: list[Path] = []
    for pattern in text_patterns:
        text_files.extend(_DATA_DIR.glob(pattern))

    for txt_file in text_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = txt_file.name
            documents.extend(docs)
        except Exception as e:
            logger.warning("Failed to load %s: %s", txt_file.name, e)

    # 加载 PDF 文件
    for pdf_file in _DATA_DIR.glob("**/*.pdf"):
        pdf_docs = _load_pdf(pdf_file)
        documents.extend(pdf_docs)
        if pdf_docs:
            logger.info("Loaded PDF %s: %d pages", pdf_file.name, len(pdf_docs))

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def _chunk_documents(documents: list[Document]) -> list[Document]:
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


# ================================================================
#  BM25 索引
# ================================================================

def _tokenize_for_bm25(text: str) -> list[str]:
    """使用 jieba 分词，适配中英文混合金融文本。"""
    import jieba

    # jieba 分词后过滤空白 token
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip()]


def _build_bm25_index(chunks: list[Document]):
    """构建 BM25 索引并持久化。"""
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize_for_bm25(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(corpus)

    # 持久化
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(_BM25_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("BM25 index built with %d chunks and persisted", len(chunks))
    return bm25, chunks


def _load_bm25_index():
    """加载已持久化的 BM25 索引。"""
    if not _BM25_INDEX_PATH.exists() or not _BM25_CHUNKS_PATH.exists():
        return None, None

    try:
        with open(_BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)
        with open(_BM25_CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Loaded existing BM25 index with %d chunks", len(chunks))
        return bm25, chunks
    except Exception as e:
        logger.warning("Failed to load BM25 index: %s", e)
        return None, None


# ================================================================
#  构建 / 加载 向量库 + BM25
# ================================================================

def build_vectorstore(force_rebuild: bool = False) -> Chroma | None:
    """构建或加载向量数据库 + BM25 索引。

    即使向量库构建失败（如 embedding API 不可用），也会构建 BM25 索引，
    确保系统至少有关键词检索能力。
    """
    global _vectorstore, _bm25_index, _bm25_chunks

    # 1. 尝试加载已有 BM25 索引
    _bm25_index, _bm25_chunks = _load_bm25_index()

    # 2. 尝试加载/构建向量库
    try:
        embeddings = _get_embeddings()

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
                    # BM25 已在上面加载
                    if _bm25_index is None:
                        documents = _load_documents()
                        if documents:
                            chunks = _chunk_documents(documents)
                            _bm25_index, _bm25_chunks = _build_bm25_index(chunks)
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

        # 同步构建 BM25
        _bm25_index, _bm25_chunks = _build_bm25_index(chunks)
        return _vectorstore

    except Exception as e:
        logger.error("Vector store initialization failed: %s", e)
        # 向量库失败，但仍尝试构建 BM25（纯关键词检索兜底）
        if _bm25_index is None:
            try:
                documents = _load_documents()
                if documents:
                    chunks = _chunk_documents(documents)
                    _bm25_index, _bm25_chunks = _build_bm25_index(chunks)
                    logger.info("BM25-only mode: vector search unavailable, keyword search active (%d chunks)", len(chunks))
            except Exception as bm25_err:
                logger.error("BM25 build also failed: %s", bm25_err)
        return None


def get_vectorstore() -> Chroma:
    """获取向量数据库实例（懒加载）。"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


def is_rag_available() -> bool:
    """检查 RAG 系统是否可用（向量检索或 BM25 至少一个可用）。"""
    return _vectorstore is not None or _bm25_index is not None


# ================================================================
#  检索：BM25 / Vector / RRF 融合
# ================================================================

def _bm25_search(query: str, top_k: int = 10) -> list[tuple[Document, float]]:
    """BM25 关键词检索。"""
    if _bm25_index is None or _bm25_chunks is None:
        return []

    tokens = _tokenize_for_bm25(query)
    scores = _bm25_index.get_scores(tokens)

    # 取 top-k
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in indexed:
        if score > 0:
            results.append((_bm25_chunks[idx], float(score)))
    return results


def _vector_search(query: str, top_k: int = 10) -> list[tuple[Document, float]]:
    """向量语义检索。"""
    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(query, k=top_k)
        return [(doc, score) for doc, score in results]
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []


def _reciprocal_rank_fusion(
    bm25_results: list[tuple[Document, float]],
    vector_results: list[tuple[Document, float]],
    hybrid_weight: float,
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion 融合 BM25 和向量检索结果。

    score = w * 1/(k+rank_bm25) + (1-w) * 1/(k+rank_vector)
    """
    # 用 page_content hash 做去重 key
    def _doc_key(doc: Document) -> int:
        return hash(doc.page_content)

    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    w = hybrid_weight

    # BM25 排名贡献
    for rank, (doc, _score) in enumerate(bm25_results, 1):
        key = _doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + w * (1.0 / (k + rank))

    # Vector 排名贡献
    for rank, (doc, _score) in enumerate(vector_results, 1):
        key = _doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 - w) * (1.0 / (k + rank))

    # 按 RRF 分数排序
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [(doc_map[key], rrf_scores[key]) for key in sorted_keys]


# ================================================================
#  LLM 重排序
# ================================================================

def _rerank_with_llm(
    query: str,
    candidates: list[tuple[Document, float]],
    top_k: int = 3,
) -> list[tuple[Document, float]]:
    """用 LLM 对候选文档打分 (0-10)，选出 top-k。"""
    from app.services.llm import chat_completion

    if not candidates:
        return []

    # 构建候选列表（每条截取前 300 字）
    snippets = []
    for i, (doc, _score) in enumerate(candidates):
        text = doc.page_content[:300]
        snippets.append(f"[{i}] {text}")
    candidates_text = "\n\n".join(snippets)

    system_prompt = (
        "你是一个文档相关性评估专家。根据用户查询，为每个候选文档打分（0-10），"
        "10 表示高度相关，0 表示完全不相关。\n"
        "只返回一个 JSON 数组，包含每个文档的分数，顺序与输入一致。\n"
        "例如：[8, 3, 9, 1, 5]"
    )
    user_message = f"查询: {query}\n\n候选文档:\n{candidates_text}"

    try:
        response = chat_completion(system_prompt, user_message)
        # 提取 JSON 数组
        match = re.search(r'\[[\d\s,]+\]', response)
        if not match:
            logger.warning("Rerank: failed to parse LLM response, using fusion order")
            return candidates[:top_k]

        scores = json.loads(match.group())
        if len(scores) != len(candidates):
            logger.warning("Rerank: score count mismatch (%d vs %d), using fusion order",
                           len(scores), len(candidates))
            return candidates[:top_k]

        # 按 LLM 分数排序
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info("Rerank scores: %s", scores)
        return scored[:top_k]

    except Exception as e:
        logger.warning("Rerank failed (%s), falling back to fusion order", e)
        return candidates[:top_k]


# ================================================================
#  主入口：get_relevant_context()
# ================================================================

def get_relevant_context(query: str, top_k: int = 4) -> str:
    """检索与查询最相关的文本片段。

    Pipeline: BM25(top-10) + Vector(top-10) → RRF 融合 → LLM 重排序 → 格式化输出

    Args:
        query: 用户问题
        top_k: 返回的最相关片段数量

    Returns:
        拼接后的参考资料文本；若无匹配则返回空字符串
    """
    hybrid_weight = _get_hybrid_weight()

    # Step 1: 双路检索
    bm25_results = _bm25_search(query, top_k=10)

    # 向量检索（仅在 vectorstore 可用时执行）
    vector_results = []
    if _vectorstore is not None:
        vector_results = _vector_search(query, top_k=10)

    # Step 2: 融合
    if bm25_results and vector_results:
        fused = _reciprocal_rank_fusion(bm25_results, vector_results, hybrid_weight)
        logger.info("Hybrid search: BM25=%d, Vector=%d, Fused=%d",
                     len(bm25_results), len(vector_results), len(fused))
    elif bm25_results:
        # BM25-only 模式（向量库不可用）
        fused = bm25_results
        logger.info("BM25-only search: %d results (vector unavailable)", len(fused))
    elif vector_results:
        # 纯向量 + 阈值过滤
        fused = [(doc, score) for doc, score in vector_results if score >= 0.3]
        logger.info("Vector-only search: %d results", len(fused))
    else:
        fused = []

    if not fused:
        logger.info("No results for query: %s", query)
        return ""

    # Step 3: LLM 重排序（可选）
    if _is_rerank_enabled() and len(fused) > 1:
        rerank_top_k = _get_rerank_top_k()
        final = _rerank_with_llm(query, fused, top_k=rerank_top_k)
    else:
        final = fused[:top_k]

    if not final:
        return ""

    # Step 4: 格式化输出（含页码和相关度）
    context_parts = []
    for i, (doc, score) in enumerate(final, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "-")
        context_parts.append(
            f"[参考资料 {i}] (来源: {source}, 页码: {page}, 相关度: {score:.2f})\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
