"""市场原因分析任务。"""

from app.tasks.base import BaseTask


class MarketReasoningTask(BaseTask):
    name = "market_reasoning"
    required_tools = ["market_api", "web_search"]

    def run(self, question: str, **kwargs) -> dict:
        from app.services.orchestrator import StepContext, build_plan, execute_plan
        import app.services.steps  # noqa: F401
        ticker = kwargs["ticker"]
        steps = kwargs.get("steps", [])
        history = kwargs.get("history")
        plan = build_plan("market_reasoning", [ticker])
        ctx = StepContext(question=question, history=history, plan=plan,
                          ticker=ticker, tickers=[ticker], steps=steps)
        return execute_plan(ctx)

    def run_stream(self, question: str, **kwargs):
        from app.services.orchestrator import StepContext, build_plan, execute_plan_stream
        import app.services.steps  # noqa: F401
        ticker = kwargs["ticker"]
        steps = kwargs.get("steps", [])
        history = kwargs.get("history")
        plan = build_plan("market_reasoning", [ticker])
        ctx = StepContext(question=question, history=history, plan=plan,
                          ticker=ticker, tickers=[ticker], steps=steps)
        yield from execute_plan_stream(ctx)
