"""Pydantic 数据模型"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChartPoint(BaseModel):
    date: str
    close: float
    volume: int


class ThoughtStepSchema(BaseModel):
    """Agent 思考链单步记录 — 前端可据此展示推理过程。"""
    step: str
    result: str
    detail: dict = {}


class ChatResponse(BaseModel):
    text_response: str
    chart_data: list[ChartPoint] | None = None
    intent: str
    ticker: str | None = None
    rag_used: bool | None = None
    steps: list[ThoughtStepSchema] = []
