"""知识库问答任务。"""

from app.tasks.base import BaseTask


class KnowledgeRagTask(BaseTask):
    name = "knowledge_rag"
    required_tools = ["rag", "web_search"]

    def run(self, question: str, **kwargs) -> dict:
        from app.services.orchestrator import StepContext, build_plan, execute_plan
        import app.services.steps  # noqa: F401
        steps = kwargs.get("steps", [])
        history = kwargs.get("history")
        plan = build_plan("knowledge_rag", [])
        ctx = StepContext(question=question, history=history, plan=plan, steps=steps)
        return execute_plan(ctx)

    def run_stream(self, question: str, **kwargs):
        from app.services.orchestrator import StepContext, build_plan, execute_plan_stream
        import app.services.steps  # noqa: F401
        steps = kwargs.get("steps", [])
        history = kwargs.get("history")
        plan = build_plan("knowledge_rag", [])
        ctx = StepContext(question=question, history=history, plan=plan, steps=steps)
        yield from execute_plan_stream(ctx)
