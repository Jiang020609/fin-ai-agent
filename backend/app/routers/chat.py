"""聊天 API 路由"""

import hashlib
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import process_question, process_question_stream
from app.services.metrics import metrics
from app.services.session import session_manager
from app.utils.sanitize import sanitize_input

router = APIRouter(prefix="/api", tags=["chat"])


def _history_to_dicts(request: ChatRequest) -> list[dict] | None:
    """将 ChatRequest.history 转为 LLM messages 格式。"""
    if not request.history:
        return None
    return [{"role": h.role, "content": h.content} for h in request.history[-6:]]


def _resolve_session_id(request: ChatRequest) -> str:
    """获取或生成 session_id。"""
    if request.session_id:
        return request.session_id
    # 从 question + 时间戳哈希生成
    raw = f"{request.question}:{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户提问，返回结构化回答 + 图表数据。"""
    question = sanitize_input(request.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        metrics.incr("chat_requests")
        start = time.time()

        session_id = _resolve_session_id(request)
        session = session_manager.get_or_create(session_id)

        result = process_question(question, history=_history_to_dicts(request), session=session)
        metrics.timing("chat_latency", (time.time() - start) * 1000)
        return ChatResponse(**result)
    except Exception as e:
        metrics.incr("chat_errors")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式响应端点。"""
    question = sanitize_input(request.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    history = _history_to_dicts(request)
    session_id = _resolve_session_id(request)
    session = session_manager.get_or_create(session_id)

    def event_generator():
        metrics.incr("chat_stream_requests")
        start = time.time()
        try:
            for event in process_question_stream(question, history=history, session=session):
                event_type = event.get("event", "token")
                data = event.get("data", "")
                payload = json.dumps(data, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {payload}\n\n"
            metrics.timing("chat_stream_latency", (time.time() - start) * 1000)
        except Exception as e:
            metrics.incr("chat_stream_errors")
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
