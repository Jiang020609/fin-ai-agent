"""聊天 API 路由"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import process_question

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户提问，返回结构化回答 + 图表数据。"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = process_question(request.question)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
