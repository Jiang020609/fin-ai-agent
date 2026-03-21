"""多资产对比任务。"""

from app.tasks.base import BaseTask


class CompareAssetsTask(BaseTask):
    name = "compare"
    required_tools = ["market_api"]

    def run(self, question: str, **kwargs) -> dict:
        from app.services.orchestrator import StepContext, build_plan, execute_plan
        import app.services.steps  # noqa: F401
        tickers = kwargs["tickers"]
        steps = kwargs.get("steps", [])
        history = kwargs.get("history")
        plan = build_plan("compare", tickers)
        ctx = StepContext(question=question, history=history, plan=plan,
                          ticker=tickers[0], tickers=tickers, steps=steps)
        return execute_plan(ctx)

    def run_stream(self, question: str, **kwargs):
        from app.services.orchestrator import StepContext, build_plan, execute_plan_stream
        import app.services.steps  # noqa: F401
        tickers = kwargs["tickers"]
        steps = kwargs.get("steps", [])
        history = kwargs.get("history")
        plan = build_plan("compare", tickers)
        ctx = StepContext(question=question, history=history, plan=plan,
                          ticker=tickers[0], tickers=tickers, steps=steps)
        yield from execute_plan_stream(ctx)
