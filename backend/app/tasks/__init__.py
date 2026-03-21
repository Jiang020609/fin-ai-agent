"""Task Registry — 按意图名查找 Task 实例。"""

from app.tasks.base import BaseTask
from app.tasks.market_data import MarketDataTask
from app.tasks.market_reasoning import MarketReasoningTask
from app.tasks.knowledge_rag import KnowledgeRagTask
from app.tasks.compare_assets import CompareAssetsTask

TASK_REGISTRY: dict[str, BaseTask] = {
    "market_data": MarketDataTask(),
    "market_reasoning": MarketReasoningTask(),
    "knowledge_rag": KnowledgeRagTask(),
    "compare": CompareAssetsTask(),
}
