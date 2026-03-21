"""FastAPI 应用入口"""

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.chat import router as chat_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预加载知识库向量索引。"""
    try:
        from app.services.rag import build_vectorstore
        build_vectorstore()
        logger.info("Knowledge base loaded successfully")
    except Exception as e:
        logger.error("Knowledge base FAILED to load: %s — RAG queries will fallback to web search", e)
    yield

app = FastAPI(
    title="金融资产问答系统",
    description="基于 LLM 的金融数据问答 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
async def health():
    try:
        from app.services.rag import is_rag_available
        rag_ok = is_rag_available()
    except Exception:
        rag_ok = False
    return {"status": "ok", "rag_available": rag_ok}


@app.get("/api/metrics")
async def get_metrics():
    """返回系统运行指标快照。"""
    from app.services.metrics import metrics
    return metrics.snapshot()
