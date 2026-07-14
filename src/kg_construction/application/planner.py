"""Technique planner that chooses cheap reliable methods before expensive ones."""

from __future__ import annotations

from kg_construction.application.registry import TechniqueDescriptor


class TechniquePlanner:
    """Sort enabled construction methods by reliability/cost priority."""

    priority_order = {
        "deterministic": 10,
        "ontology": 15,
        "rules": 20,
        "nlp": 25,
        "ml": 30,
        "embedding": 35,
        "llm": 40,
        "human": 50,
        "evaluation": 60,
    }

    def plan(self, techniques: list[TechniqueDescriptor]) -> list[TechniqueDescriptor]:
        return sorted(
            techniques,
            key=lambda item: (item.priority, self.priority_order.get(item.method_type, 99), item.name),
        )


__all__ = ["TechniquePlanner"]
