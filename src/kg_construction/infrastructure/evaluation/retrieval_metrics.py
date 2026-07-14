"""Retrieval and GraphRAG metrics."""

from __future__ import annotations


class RetrievalMetrics:
    """Compute retrieval and GraphRAG quality metrics from optional judgments."""

    name = "retrieval_metrics"

    def evaluate(
        self,
        *,
        retrieved: list[str] | None = None,
        relevant: set[str] | None = None,
        useful_paths: int = 0,
        retrieved_paths: int = 0,
        correct_multi_hop_answers: int = 0,
        total_multi_hop_questions: int = 0,
        correct_citations: int = 0,
        total_answers: int = 0,
        faithful_answers: int = 0,
    ) -> dict[str, float]:
        retrieved = retrieved or []
        relevant = relevant or set()
        recall_at_k = (
            len(set(retrieved) & relevant) / len(relevant)
            if relevant
            else 0.0
        )
        return {
            "retrieval_recall_at_k": recall_at_k,
            "path_relevance": useful_paths / retrieved_paths if retrieved_paths else 0.0,
            "multi_hop_success_rate": (
                correct_multi_hop_answers / total_multi_hop_questions
                if total_multi_hop_questions
                else 0.0
            ),
            "citation_accuracy": correct_citations / total_answers if total_answers else 0.0,
            "answer_faithfulness": faithful_answers / total_answers if total_answers else 0.0,
        }


__all__ = ["RetrievalMetrics"]
