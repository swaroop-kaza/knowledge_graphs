"""Regression tests for sentence-scoped model relationship formation."""

from kg_construction.domain.claim import DocumentRef
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    ExactSpanOutput,
    RelationInference,
)
from kg_construction.infrastructure.extractors.model_backed.relations import (
    _relation_evidence,
)
from kg_construction.unstructured.pipeline import RuleSegmenter


def _relation(
    text: str,
    *,
    subject: str,
    object_: str,
) -> RelationInference:
    subject_start = text.index(subject)
    object_start = text.index(object_)
    return RelationInference(
        subject_span=ExactSpanOutput(
            start=subject_start,
            end=subject_start + len(subject),
            text=subject,
        ),
        predicate="created",
        object_span=ExactSpanOutput(
            start=object_start,
            end=object_start + len(object_),
            text=object_,
        ),
        confidence=0.99,
    )


def test_relation_arguments_in_one_exact_sentence_receive_that_evidence() -> None:
    text = "Ada created the Engine. Bob observed it."
    source = SourceRecord.from_text(text, source_name="memory://same-sentence")
    document = DocumentRef.from_source(source)
    segments = RuleSegmenter().segment(document, text)

    evidence = _relation_evidence(
        _relation(text, subject="Ada", object_="Engine"),
        segments=segments,
    )

    assert evidence is not None
    assert evidence.span.text == "Ada created the Engine."


def test_cross_sentence_relation_has_no_publishable_evidence_unit() -> None:
    text = "Ada denied the claim. The Engine appeared later."
    source = SourceRecord.from_text(text, source_name="memory://cross-sentence")
    document = DocumentRef.from_source(source)
    segments = RuleSegmenter().segment(document, text)

    evidence = _relation_evidence(
        _relation(text, subject="Ada", object_="Engine"),
        segments=segments,
    )

    assert evidence is None
