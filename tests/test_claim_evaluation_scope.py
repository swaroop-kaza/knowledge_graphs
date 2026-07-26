from __future__ import annotations

import pytest

from kg_construction.domain.claim import (
    DocumentRef,
    Evidence,
    EvidenceKind,
    Mention,
    Provenance,
    TextSpan,
)
from kg_construction.domain.models import SourceRecord
from kg_construction.evaluation.claim_evaluations import (
    ClaimEvaluationSuite,
)
from kg_construction.unstructured.pipeline import PipelineResult


def _result(
    text: str,
    predictions: tuple[tuple[int, int, str], ...],
) -> PipelineResult:
    source = SourceRecord.from_text(
        text,
        source_name="held-out://partial-entity-annotations.txt",
    )
    document = DocumentRef.from_source(source)
    provenance = Provenance(
        extractor="scope-test",
        provider="scope-test",
        model_version="scope-test-v1",
        configuration_id="scope-test-v1",
        confidence=1.0,
    )
    mentions = tuple(
        Mention(
            document_version=document.version,
            evidence=Evidence.build(
                document,
                span=TextSpan(
                    start=start,
                    end=end,
                    text=text[start:end],
                ),
                kind=EvidenceKind.MENTION,
            ),
            surface=text[start:end],
            entity_type=entity_type,
            provenance=provenance,
        )
        for start, end, entity_type in predictions
    )
    return PipelineResult(
        document=document,
        document_text=text,
        segments=(),
        mentions=mentions,
        canonical_entities=(),
        claims=(),
        promoted=(),
        quarantined=(),
    )


def _gold(
    text: str,
    annotations: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "gold_entities": [
            {"name": surface, "type": entity_type}
            for surface, entity_type in annotations
        ],
        "gold_mentions": [
            {
                "start": text.index(surface),
                "end": text.index(surface) + len(surface),
                "text": surface,
                "type": entity_type,
            }
            for surface, entity_type in annotations
        ],
    }


def test_out_of_scope_entity_types_do_not_reduce_precision() -> None:
    text = "Ada Lovelace joined Acme in 2020."
    result = _result(
        text,
        (
            (text.index("Ada Lovelace"), text.index("Ada Lovelace") + 12, "Person"),
            (text.index("Acme"), text.index("Acme") + 4, "Organization"),
            (text.index("2020"), text.index("2020") + 4, "Date"),
        ),
    )

    scores = ClaimEvaluationSuite().evaluate(
        result,
        _gold(text, (("Ada Lovelace", "Person"),)),
    )

    for metric in ("entity", "entity_span"):
        score = scores[metric]
        assert score.predicted_count == 1
        assert score.matched == 1
        assert score.precision == 1.0
        assert score.recall == 1.0


def test_same_scope_fragment_and_wrong_type_remain_false_positives() -> None:
    text = "Ada Lovelace joined Acme."
    result = _result(
        text,
        (
            (text.index("Ada"), text.index("Ada") + 3, "Person"),
            (
                text.index("Ada Lovelace"),
                text.index("Ada Lovelace") + 12,
                "Organization",
            ),
            (text.index("Acme"), text.index("Acme") + 4, "Organization"),
        ),
    )

    scores = ClaimEvaluationSuite().evaluate(
        result,
        _gold(
            text,
            (
                ("Ada Lovelace", "Person"),
                ("Acme", "Organization"),
            ),
        ),
    )

    for metric in ("entity", "entity_span"):
        score = scores[metric]
        assert score.predicted_count == 3
        assert score.matched == 1
        assert score.precision == pytest.approx(1 / 3)
        assert score.recall == 0.5


def test_missing_gold_mentions_lower_recall_after_scope_filtering() -> None:
    text = "Ada met Grace on Monday."
    result = _result(
        text,
        (
            (text.index("Ada"), text.index("Ada") + 3, "Person"),
            (text.index("Monday"), text.index("Monday") + 6, "Date"),
        ),
    )

    scores = ClaimEvaluationSuite().evaluate(
        result,
        _gold(
            text,
            (
                ("Ada", "Person"),
                ("Grace", "Person"),
            ),
        ),
    )

    for metric in ("entity", "entity_span"):
        score = scores[metric]
        assert score.predicted_count == 1
        assert score.matched == 1
        assert score.precision == 1.0
        assert score.recall == 0.5
        assert score.f1 == pytest.approx(2 / 3)
