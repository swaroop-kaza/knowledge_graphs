"""Regenerate and verify the modelling-correctness report.

Runs the gold structured and unstructured cases through the real pipeline, writes a
published report, and exits non-zero if any modelling guarantee regresses -- so CI
verifies categorical dedup, foreign-key edges, and entity coverage on every change.

    make modelling-correctness
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from kg_construction.application.modelling_correctness import (
    ModellingCase,
    evaluate_structured_case,
    evaluate_unstructured_case,
)

_FIXTURES = Path("tests/fixtures/gold_standards")
_REPORTS = Path("reports")


def _run_cases() -> list[ModellingCase]:
    structured = json.loads((_FIXTURES / "modelling_correctness_structured.json").read_text(encoding="utf-8"))
    unstructured = json.loads((_FIXTURES / "modelling_correctness_unstructured.json").read_text(encoding="utf-8"))
    return [
        evaluate_structured_case(
            case_id=structured["case_id"],
            records=structured["records"],
            table_name=structured["table_name"],
            record_type=structured["record_type"],
            gold_graph=structured["gold_graph"],
        ),
        evaluate_unstructured_case(
            case_id=unstructured["case_id"],
            text=unstructured["text"],
            gold_entities=unstructured["gold_entities"],
            minimum_entity_coverage=unstructured["minimum_entity_coverage"],
        ),
    ]


def _markdown(cases: list[ModellingCase], overall: bool) -> str:
    lines = [
        "# Modelling Correctness Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Runs the gold modelling cases through the real pipeline and verifies the output "
        "conforms to the expected graph (RML/R2RML conformance style). CI fails on regression.",
        "",
        f"**Overall: {'PASS' if overall else 'FAIL'}**",
        "",
    ]
    for case in cases:
        lines.append(f"## {case.case_id} ({case.kind}) - {'PASS' if case.passed else 'FAIL'}")
        lines.append("")
        lines.append("| Check | Result |")
        lines.append("| --- | :---: |")
        for name, passed in case.checks.items():
            lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
        lines.append("")
        lines.append("Metrics: " + ", ".join(f"`{key}={value:g}`" for key, value in sorted(case.metrics.items())))
        lines.append("")
    lines.append("Regenerate with `make modelling-correctness`.")
    return "\n".join(lines) + "\n"


def main() -> int:
    cases = _run_cases()
    overall = all(case.passed for case in cases)
    _REPORTS.mkdir(exist_ok=True)
    payload = {"generated": date.today().isoformat(), "overall_passed": overall, "cases": [case.model_dump(mode="json") for case in cases]}
    (_REPORTS / "modelling_correctness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (_REPORTS / "modelling_correctness.md").write_text(_markdown(cases, overall), encoding="utf-8")
    print(f"modelling-correctness: overall={'PASS' if overall else 'FAIL'}; cases={len(cases)}")
    for case in cases:
        failed = [name for name, passed in case.checks.items() if not passed]
        if failed:
            print(f"  {case.case_id}: FAILED {failed}", file=sys.stderr)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
