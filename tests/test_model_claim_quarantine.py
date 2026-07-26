"""Fail-closed per-claim quarantine for mandatory structured generators."""

import json

import pytest

from kg_construction.infrastructure.extractors.model_backed.claims import (
    LocalOllamaQwenBackend,
    _align_valid_quoted_claims,
)
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    HydratedServiceModel,
    ProviderOutputError,
)


def _prompt(text: str) -> str:
    return "Extract grounded claims.\nINPUT=" + json.dumps(
        {"document": {"text": text}},
        sort_keys=True,
    )


def _valid_claim(text: str) -> dict[str, object]:
    return {
        "kind": "relation",
        "subject_span": {"start": 0, "end": 3, "text": "Ada"},
        "predicate": "worked_at",
        "predicate_span": {"start": 4, "end": 13, "text": "worked at"},
        "object_span": {"start": 14, "end": 18, "text": "Acme"},
        "evidence_spans": [
            {"start": 0, "end": len(text), "text": text},
        ],
        "confidence": 0.95,
    }


def _invalid_claim(text: str) -> dict[str, object]:
    return {
        "kind": "relation",
        "subject_span": {"start": 0, "end": 3, "text": "Bob"},
        "predicate": "worked_at",
        "predicate_span": {"start": 4, "end": 13, "text": "worked at"},
        "object_span": {"start": 14, "end": 18, "text": "Acme"},
        "evidence_spans": [
            {"start": 0, "end": len(text), "text": text},
        ],
        "confidence": 0.95,
    }


def test_independently_valid_claim_survives_and_rejection_is_audited() -> None:
    text = "Ada worked at Acme."

    response, _alignment_count, diagnostics = _align_valid_quoted_claims(
        {"claims": [_valid_claim(text), _invalid_claim(text)]},
        prompt=_prompt(text),
    )

    assert len(response.claims) == 1
    assert response.claims[0].subject_span.text == "Ada"
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("claim[1] ")


def test_quarantine_never_turns_all_invalid_claims_into_empty_success() -> None:
    text = "Ada worked at Acme."

    with pytest.raises(ProviderOutputError, match="no model claim survived"):
        _align_valid_quoted_claims(
            {"claims": [_invalid_claim(text)]},
            prompt=_prompt(text),
        )


def test_quarantine_rejects_unknown_top_level_keys() -> None:
    text = "Ada worked at Acme."

    with pytest.raises(ProviderOutputError, match="exactly one top-level"):
        _align_valid_quoted_claims(
            {"claims": [_valid_claim(text)], "ignored": True},
            prompt=_prompt(text),
        )


class _OllamaTransport:
    def __init__(self, content: str) -> None:
        self.content = content

    def list_models(self, base_url: str) -> dict[str, str]:
        del base_url
        return {"qwen3:test": f"sha256:{'a' * 64}"}

    def chat(self, base_url: str, payload: object) -> object:
        del base_url, payload
        return {"message": {"content": self.content}}


def test_ollama_returns_valid_subset_only_after_bounded_whole_output_attempt() -> None:
    text = "Ada worked at Acme."
    backend = LocalOllamaQwenBackend(
        HydratedServiceModel(
            name="ollama_structured_extractor",
            service="ollama",
            model="qwen3:test",
            digest=f"sha256:{'a' * 64}",
        ),
        transport=_OllamaTransport(
            json.dumps(
                {"claims": [_valid_claim(text), _invalid_claim(text)]}
            )
        ),
        max_retries=0,
    )

    response = backend.generate_json(
        prompt=_prompt(text),
        json_schema={"type": "object"},
    )

    assert len(response.claims) == 1
    assert backend.last_attempt_count == 1
    assert backend.last_rejected_claim_count == 1
    assert backend.last_rejected_claim_diagnostics[0].startswith("claim[1] ")
