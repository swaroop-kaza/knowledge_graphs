"""Comparative research example, intentionally separate from construction."""

from __future__ import annotations

import json

from kg_construction.research import evaluate_techniques


def main() -> None:
    """Compare techniques without treating a weighted rank as publication proof."""

    report = evaluate_techniques(
        "examples/data_to_kg/sample.txt",
        source_type="text",
        techniques="regex_entity_extractor,hybrid_rules_gazetteer_regex",
    )
    print(
        json.dumps(
            {
                "research_only": True,
                "best_weighted_technique": report.best_technique,
                "candidate_count": len(report.candidates),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
