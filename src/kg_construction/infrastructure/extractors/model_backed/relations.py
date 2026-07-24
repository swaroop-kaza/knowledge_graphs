"""Grounded relation claims from the mandatory GLiNER RelEx snapshot."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.claim import (
    Claim,
    ClaimKind,
    DocumentRef,
    Evidence,
    EvidenceKind,
    Mention,
    Provenance,
    TextSpan,
)
from kg_construction.domain.models import stable_id
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    ExactSpanOutput,
    HydratedArtifact,
    OfflineArtifactError,
    ProviderOutputError,
    RelationInference,
    RelationInferenceBackend,
    enforce_offline_model_environment,
    require_local_directory,
)

GLINER_RELEX_REPO_ID = "knowledgator/gliner-relex-large-v1.0"


class RelationProviderConfig(BaseModel):
    """Immutable provenance and threshold policy for a relation provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractor: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LocalGlinerRelExBackend:
    """Joint GLiNER NER/RelEx inference from one hydrated local directory."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        entity_labels: Sequence[str],
        relation_labels: Sequence[str],
        entity_threshold: float = 0.4,
        adjacency_threshold: float = 0.55,
        relation_threshold: float = 0.7,
        flat_ner: bool = False,
    ) -> None:
        _require_relex_artifact(artifact)
        self.artifact = artifact
        self.entity_labels = _unique_nonempty(entity_labels, label_kind="entity")
        raw_relations = _unique_nonempty(relation_labels, label_kind="relation")
        canonical_relations = tuple(_canonical_predicate(item) for item in raw_relations)
        if len(canonical_relations) != len(set(canonical_relations)):
            raise ValueError(
                "GLiNER relation labels must not collapse to the same canonical predicate"
            )
        self.relation_labels = raw_relations
        self._predicate_by_canonical_label = dict(
            zip(canonical_relations, canonical_relations, strict=True)
        )
        self.entity_threshold = _threshold(entity_threshold, name="entity_threshold")
        self.adjacency_threshold = _threshold(
            adjacency_threshold,
            name="adjacency_threshold",
        )
        self.relation_threshold = _threshold(
            relation_threshold,
            name="relation_threshold",
        )
        self.flat_ner = flat_ner
        self._model: Any | None = None

    def infer(self, text: str) -> tuple[RelationInference, ...]:
        model = self._load()
        raw_result = model.inference(
            texts=[text],
            labels=list(self.entity_labels),
            relations=list(self.relation_labels),
            threshold=self.entity_threshold,
            adjacency_threshold=self.adjacency_threshold,
            relation_threshold=self.relation_threshold,
            flat_ner=self.flat_ner,
            multi_label=False,
            return_relations=True,
        )
        if not isinstance(raw_result, tuple) or len(raw_result) != 2:
            raise ProviderOutputError(
                "GLiNER RelEx returned neither the required entity/relation tuple"
            )
        raw_relation_batches = raw_result[1]
        if (
            not isinstance(raw_relation_batches, Sequence)
            or isinstance(raw_relation_batches, (str, bytes))
            or len(raw_relation_batches) != 1
        ):
            raise ProviderOutputError(
                "GLiNER RelEx returned an invalid single-document relation batch"
            )
        raw_relations = raw_relation_batches[0]
        if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
            raise ProviderOutputError("GLiNER RelEx returned a non-list relation result")
        return tuple(
            _mapping_to_relation_inference(
                raw,
                text=text,
                predicate_by_label=self._predicate_by_canonical_label,
            )
            for raw in raw_relations
        )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        require_local_directory(
            self.artifact.local_path,
            model_name=self.artifact.model_version,
        )
        enforce_offline_model_environment()
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - exercised by hard-pass environment
            raise ProviderOutputError("the mandatory gliner package is unavailable") from exc
        model = GLiNER.from_pretrained(
            str(self.artifact.local_path),
            local_files_only=True,
        )
        if hasattr(model, "to"):
            model = model.to(self.artifact.device)
        if not callable(getattr(model, "inference", None)):
            raise ProviderOutputError(
                f"{self.artifact.model_version} does not expose GLiNER RelEx inference"
            )
        self._model = model
        return model


class GlinerRelExClaimExtractor:
    """Convert joint GLiNER relation predictions into exact, provenanced claims."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        entity_labels: Sequence[str],
        relation_labels: Sequence[str],
        entity_threshold: float = 0.4,
        adjacency_threshold: float = 0.55,
        relation_threshold: float = 0.7,
        flat_ner: bool = False,
        backend: RelationInferenceBackend | None = None,
    ) -> None:
        _require_relex_artifact(artifact)
        clean_entity_labels = _unique_nonempty(entity_labels, label_kind="entity")
        if "other" not in {item.casefold() for item in clean_entity_labels}:
            clean_entity_labels = (*clean_entity_labels, "other")
        clean_relation_labels = _unique_nonempty(
            relation_labels,
            label_kind="relation",
        )
        self.allowed_predicates = frozenset(
            _canonical_predicate(item) for item in clean_relation_labels
        )
        self.config = RelationProviderConfig(
            extractor="gliner_relex_claim_extractor",
            provider="gliner_relex",
            model_version=artifact.model_version,
            configuration_id=stable_id(
                "gliner-relex-claims",
                "schema-v1",
                entity_threshold,
                adjacency_threshold,
                relation_threshold,
                flat_ner,
                *clean_entity_labels,
                *clean_relation_labels,
            ),
            minimum_confidence=relation_threshold,
        )
        self.provider_id = self.config.provider
        self.backend = backend or LocalGlinerRelExBackend(
            artifact,
            entity_labels=clean_entity_labels,
            relation_labels=clean_relation_labels,
            entity_threshold=entity_threshold,
            adjacency_threshold=adjacency_threshold,
            relation_threshold=relation_threshold,
            flat_ner=flat_ner,
        )

    def extract(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[Claim, ...]:
        if not document.matches_text(text):
            raise ProviderOutputError("relation input does not match its document snapshot")
        for segment in segments:
            if not segment.is_exact_for(document, text):
                raise ProviderOutputError(
                    "relation input contains evidence from another or inexact snapshot"
                )

        best: dict[tuple[object, ...], RelationInference] = {}
        for output in self.backend.infer(text):
            _require_exact_relation(output, text=text, provider=self.provider_id)
            if output.predicate not in self.allowed_predicates:
                raise ProviderOutputError(
                    "gliner_relex returned predicate outside the configured relation schema: "
                    f"{output.predicate!r}"
                )
            if output.confidence < self.config.minimum_confidence:
                continue
            key = (
                output.subject_span.start,
                output.subject_span.end,
                output.subject_span.text,
                output.predicate,
                output.object_span.start,
                output.object_span.end,
                output.object_span.text,
            )
            previous = best.get(key)
            if previous is None or output.confidence > previous.confidence:
                best[key] = output

        claims = [
            self._to_claim(
                output,
                document=document,
                text=text,
                segments=segments,
                mentions=mentions,
            )
            for output in sorted(
                best.values(),
                key=lambda item: (
                    item.subject_span.start,
                    item.object_span.start,
                    item.predicate,
                ),
            )
        ]
        return tuple(claims)

    def _to_claim(
        self,
        output: RelationInference,
        *,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
    ) -> Claim:
        evidence = _relation_evidence(
            output,
            document=document,
            text=text,
            segments=segments,
        )
        subject_mention = _matching_mention(output.subject_span, mentions)
        object_mention = _matching_mention(output.object_span, mentions)
        claim = Claim(
            document=document,
            kind=ClaimKind.RELATION,
            subject=output.subject_span.text,
            predicate=output.predicate,
            object=output.object_span.text,
            subject_mention_id=(
                subject_mention.mention_id if subject_mention is not None else None
            ),
            object_mention_id=(object_mention.mention_id if object_mention is not None else None),
            evidence=(evidence,),
            provenance=Provenance(
                extractor=self.config.extractor,
                provider=self.config.provider,
                model_version=self.config.model_version,
                configuration_id=self.config.configuration_id,
                confidence=output.confidence,
            ),
        )
        if not claim.has_complete_provenance(text):
            raise ProviderOutputError(
                "gliner_relex produced a claim without exact evidence/provenance"
            )
        return claim


def _mapping_to_relation_inference(
    raw: object,
    *,
    text: str,
    predicate_by_label: Mapping[str, str],
) -> RelationInference:
    if not isinstance(raw, Mapping):
        raise ProviderOutputError("GLiNER RelEx returned a non-object relation")
    try:
        subject = _mapping_to_exact_span(raw["head"], argument="head")
        object_ = _mapping_to_exact_span(raw["tail"], argument="tail")
        raw_predicate = str(raw["relation"]).strip()
        canonical_label = _canonical_predicate(raw_predicate)
        predicate = predicate_by_label[canonical_label]
        confidence = float(raw["score"])
        output = RelationInference(
            subject_span=subject,
            predicate=predicate,
            object_span=object_,
            confidence=confidence,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderOutputError(
            f"GLiNER RelEx returned an invalid relation object: {raw!r}"
        ) from exc
    _require_exact_relation(output, text=text, provider="gliner_relex")
    return output


def _mapping_to_exact_span(raw: object, *, argument: str) -> ExactSpanOutput:
    if not isinstance(raw, Mapping):
        raise ProviderOutputError(f"GLiNER RelEx {argument} is not an object")
    return ExactSpanOutput(
        start=int(raw["start"]),
        end=int(raw["end"]),
        text=str(raw["text"]),
    )


def _relation_evidence(
    output: RelationInference,
    *,
    document: DocumentRef,
    text: str,
    segments: tuple[Evidence, ...],
) -> Evidence:
    start = min(output.subject_span.start, output.object_span.start)
    end = max(output.subject_span.end, output.object_span.end)
    containing_segments = [
        item for item in segments if item.span.start <= start and end <= item.span.end
    ]
    if containing_segments:
        return min(
            containing_segments,
            key=lambda item: (
                item.span.end - item.span.start,
                item.span.start,
                item.evidence_id,
            ),
        )
    span = TextSpan(start=start, end=end, text=text[start:end])
    return Evidence.build(document, span=span, kind=EvidenceKind.PASSAGE)


def _matching_mention(
    span: ExactSpanOutput,
    mentions: tuple[Mention, ...],
) -> Mention | None:
    matching = [
        item
        for item in mentions
        if item.span.start == span.start and item.span.end == span.end and item.surface == span.text
    ]
    if not matching:
        return None
    return min(matching, key=lambda item: (item.entity_type, item.mention_id))


def _require_exact_relation(
    output: RelationInference,
    *,
    text: str,
    provider: str,
) -> None:
    for argument, span in (
        ("subject", output.subject_span),
        ("object", output.object_span),
    ):
        if not span.is_exact_in(text):
            raise ProviderOutputError(
                f"{provider} returned an inexact {argument} span "
                f"[{span.start}, {span.end}): {span.text!r}"
            )


def _require_relex_artifact(artifact: HydratedArtifact) -> None:
    if artifact.repo_id != GLINER_RELEX_REPO_ID:
        raise OfflineArtifactError(
            "gliner_relex must use locked repository "
            f"{GLINER_RELEX_REPO_ID!r}, got {artifact.repo_id!r}"
        )
    if artifact.backend != "gliner" or artifact.task != "relation-extraction":
        raise OfflineArtifactError(
            "gliner_relex requires backend='gliner' and task='relation-extraction'"
        )
    require_local_directory(artifact.local_path, model_name=artifact.model_version)


def _unique_nonempty(labels: Sequence[str], *, label_kind: str) -> tuple[str, ...]:
    cleaned = tuple(dict.fromkeys(item.strip() for item in labels if item.strip()))
    if not cleaned:
        raise ValueError(f"GLiNER RelEx requires at least one {label_kind} label")
    return cleaned


def _canonical_predicate(label: str) -> str:
    canonical = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not canonical or not canonical[0].isalpha():
        raise ValueError(f"relation label cannot form a predicate: {label!r}")
    return canonical


def _threshold(value: float, *, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


__all__ = [
    "GLINER_RELEX_REPO_ID",
    "GlinerRelExClaimExtractor",
    "LocalGlinerRelExBackend",
    "RelationProviderConfig",
]
