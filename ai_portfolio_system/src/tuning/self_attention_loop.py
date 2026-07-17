"""Self-Attention iterative improvement loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.base import AgentResult
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.rag_agent import RAGAgent
from src.core.config import Config, get_config
from src.core.llm import LLM, get_llm
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Iteration:
    step: int
    result: AgentResult
    eval_score: float
    eval_metadata: dict[str, Any] = field(default_factory=dict)


class SelfAttentionTuner:
    """Self-Attention loop: evaluate -> refine prompt/strategy -> retry up to N times."""

    def __init__(self, llm: LLM | None = None, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.llm = llm or get_llm(self.config)
        self.evaluator = EvaluatorAgent(self.llm, self.config)
        self.max_iterations = int(self.config.get("tuning.max_iterations", 3))
        self.score_threshold = float(self.config.get("tuning.score_threshold", 0.85))

    def run(self, agent: RAGAgent, query: str) -> AgentResult:
        """Run the agent iteratively until score threshold or max iterations."""
        best: AgentResult | None = None
        best_score = -1.0
        history: list[Iteration] = []

        instructions = ""
        for step in range(1, self.max_iterations + 1):
            logger.info("Self-Attention iteration %d/%d", step, self.max_iterations)
            result = agent.run(query, extra_instructions=instructions)
            eval_result = self.evaluator.evaluate(query, result)
            score = eval_result.score or 0.0

            iteration = Iteration(step=step, result=result, eval_score=score, eval_metadata=eval_result.metadata)
            history.append(iteration)

            if score > best_score:
                best = result
                best_score = score

            if score >= self.score_threshold:
                logger.info("Score %.2f reached threshold, stopping early.", score)
                break

            # Build refinement instructions for next round.
            issues = "；".join(eval_result.metadata.get("issues", []))
            suggestions = "；".join(eval_result.metadata.get("suggestions", []))
            instructions = (
                f"上一轮评估得分 {score:.2f}，未达到 {self.score_threshold:.2f}。"
                f"请重点关注以下问题：{issues or '无明显问题'}。"
                f"建议：{suggestions or '继续优化回答质量'}。"
            )

        if best is not None:
            best.iterations = len(history)
            best.metadata["self_attention_history"] = [
                {"step": i.step, "score": i.eval_score} for i in history
            ]
            best.score = best_score
        return best or result
