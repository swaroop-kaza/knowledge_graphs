"""Mandatory GLiNER, Transformers, and spaCy mention detectors."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.claim import (
    DocumentRef,
    Evidence,
    EvidenceKind,
    Mention,
    Provenance,
    TextSpan,
)
from kg_construction.domain.models import stable_id
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    HydratedArtifact,
    MandatoryProviderError,
    MentionInference,
    MentionInferenceBackend,
    OfflineArtifactError,
    ProviderAgreement,
    ProviderOutputError,
    enforce_offline_model_environment,
    require_local_directory,
)
from kg_construction.unstructured.pipeline import MentionDetector

_TRANSFORMERS_TYPES = {
    "PER": "Person",
    "PERSON": "Person",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "LOC": "Location",
    "LOCATION": "Location",
    "GPE": "Location",
    "MISC": "Entity",
}
_SPACY_TYPES = {
    **_TRANSFORMERS_TYPES,
    "NORP": "Organization",
    "FAC": "Place",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "Artifact",
    "LAW": "Concept",
    "LANGUAGE": "Concept",
    "DATE": "Date",
    "TIME": "Concept",
    "PERCENT": "Concept",
    "MONEY": "Concept",
    "QUANTITY": "Concept",
    "ORDINAL": "Concept",
    "CARDINAL": "Concept",
}
_GLINER_ENTITY_REPO_ID = "gliner-community/gliner_small-v2.5"
_GLINER_ENCODER_MODEL_NAME = "microsoft/deberta-v3-small"
_GLINER_ENCODER_CONFIG_RESOURCE = "deberta_v3_small_encoder_config.json"
_GLINER_ENCODER_CONFIG_REVISION = "f439f2eae06a198f8fadc821afca725c8e4fc482"
_GLINER_ENCODER_CONFIG_SHA256 = (
    "1fc4f16e48c036926da4dda884bff5f7037dbf13f7a7846cd1fdee58d412eb16"
)
_GENERIC_ENTITY_TYPES = frozenset(
    {
        "entity",
        "generic",
        "misc",
        "miscellaneous",
        "other",
        "unknown",
    }
)
_MENTION_TYPE_SELECTION_POLICY = (
    "provider_vote>specific_over_generic>source_lexical_form>source_suffix>"
    "reject_unresolved_tie"
)
_LEXICAL_FORM_TYPE_HINTS = (
    ("date", re.compile(r"\d{4}-\d{2}-\d{2}")),
    ("emailaddress", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("version", re.compile(r"v\d+(?:\.\d+)+")),
    ("identifier", re.compile(r"[A-Z]{2,}-\d+")),
)
_LEXICAL_SUFFIX_TYPE_HINTS = {
    "api": "artifact",
    "bombe": "artifact",
    "company": "organization",
    "corp": "organization",
    "corporation": "organization",
    "engine": "artifact",
    "labs": "organization",
    "machine": "artifact",
    "park": "place",
}


class NestedMentionSuppression(BaseModel):
    """Audit record for one exact mention hidden by a unique maximal span."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fragment_item_key: str = Field(min_length=1)
    containing_item_key: str = Field(min_length=1)
    fragment_start: int = Field(ge=0)
    fragment_end: int = Field(gt=0)
    fragment_surface: str = Field(min_length=1)
    containing_start: int = Field(ge=0)
    containing_end: int = Field(gt=0)
    containing_surface: str = Field(min_length=1)
    reason: str = "strict_fragment_under_unique_maximal_mention"


class MentionProviderConfig(BaseModel):
    """Immutable provenance and threshold policy for one mention provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractor: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LocalGlinerBackend:
    """GLiNER client that loads only an already-hydrated snapshot directory."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        labels: Sequence[str],
        threshold: float,
        flat_ner: bool = True,
    ) -> None:
        if not labels:
            raise ValueError("GLiNER requires at least one entity label")
        self.artifact = artifact
        self.labels = tuple(dict.fromkeys(label.strip() for label in labels if label.strip()))
        if not self.labels:
            raise ValueError("GLiNER requires at least one non-empty entity label")
        self.threshold = threshold
        self.flat_ner = flat_ner
        self._model: Any | None = None

    def infer(self, text: str) -> tuple[MentionInference, ...]:
        model = self._load()
        raw_items = model.predict_entities(
            text,
            list(self.labels),
            threshold=self.threshold,
            flat_ner=self.flat_ner,
        )
        return tuple(
            _mapping_to_inference(
                raw,
                text=text,
                label_keys=("label", "entity"),
                text_keys=("text", "word"),
                type_map=_TRANSFORMERS_TYPES,
            )
            for raw in raw_items
        )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        require_local_directory(self.artifact.local_path, model_name=self.artifact.model_version)
        enforce_offline_model_environment()
        model = _load_gliner_snapshot(self.artifact)
        if hasattr(model, "to"):
            model = model.to(self.artifact.device)
        self._model = model
        return model


def _load_gliner_snapshot(artifact: HydratedArtifact) -> Any:
    """Load GLiNER without allowing its nested encoder config to hit the Hub.

    The locked ``gliner_small-v2.5`` snapshot contains its tokenizer and model
    weights but declares ``encoder_config: null``.  GLiNER 0.2.27 then asks
    Transformers for ``microsoft/deberta-v3-small`` by repository ID and does
    not propagate ``local_files_only`` to that nested call.  We inject the
    pinned 578-byte architecture config in memory and continue loading every
    tokenizer/weight byte from the exact locked GLiNER snapshot.
    """

    path = artifact.local_path.resolve()
    require_local_directory(path, model_name=artifact.model_version)
    config_path = path / "gliner_config.json"
    if not config_path.is_file():
        raise OfflineArtifactError(
            f"{artifact.model_version} is missing local gliner_config.json"
        )
    raw_config = _json_mapping(config_path)
    embedded_encoder = raw_config.get("encoder_config")
    if isinstance(embedded_encoder, dict):
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - mandatory dependency
            raise ProviderOutputError(
                "the mandatory gliner package is unavailable"
            ) from exc
        return GLiNER.from_pretrained(
            str(path),
            local_files_only=True,
        )
    if embedded_encoder is not None:
        raise OfflineArtifactError(
            f"{artifact.model_version} has an invalid embedded encoder_config"
        )
    if (
        artifact.repo_id != _GLINER_ENTITY_REPO_ID
        or raw_config.get("model_name") != _GLINER_ENCODER_MODEL_NAME
        or raw_config.get("span_mode") == "token_level"
        or any(
            raw_config.get(key) not in (None, "")
            for key in ("labels_encoder", "labels_decoder", "relations_layer")
        )
    ):
        raise OfflineArtifactError(
            "GLiNER snapshot requires an ungoverned nested model/config; "
            f"repo={artifact.repo_id!r}, "
            f"encoder={raw_config.get('model_name')!r}"
        )

    try:
        from gliner.config import UniEncoderSpanConfig
        from gliner.model import UniEncoderSpanGLiNER
    except ImportError as exc:  # pragma: no cover - mandatory dependency
        raise ProviderOutputError(
            "the mandatory gliner package does not expose its locked loader API"
        ) from exc

    encoder_config = _locked_gliner_encoder_config()
    gliner_vocab_size = raw_config.get("vocab_size")
    if isinstance(gliner_vocab_size, int) and gliner_vocab_size > 0:
        # This is the same resize GLiNER performs after AutoConfig resolution,
        # applied before model construction so the locked state dict is exact.
        encoder_config["vocab_size"] = gliner_vocab_size
    expected_config_path = config_path.resolve()

    class StrictLocalUniEncoderSpanGLiNER(UniEncoderSpanGLiNER):
        config_class = UniEncoderSpanConfig

        @classmethod
        def _load_config(
            cls,
            selected_config_path: Path,
            **overrides: object,
        ) -> Any:
            if Path(selected_config_path).resolve() != expected_config_path:
                raise OfflineArtifactError(
                    "GLiNER attempted to load configuration outside its "
                    f"locked snapshot: {selected_config_path}"
                )
            payload = dict(raw_config)
            payload.pop("model_type", None)
            payload["encoder_config"] = dict(encoder_config)
            for key, value in overrides.items():
                if value is not None:
                    payload[key] = value
            return cls.config_class(**payload)

    return StrictLocalUniEncoderSpanGLiNER.from_pretrained(
        str(path),
        local_files_only=True,
    )


def _locked_gliner_encoder_config() -> dict[str, object]:
    resource = files("kg_construction").joinpath(
        "resources",
        _GLINER_ENCODER_CONFIG_RESOURCE,
    )
    raw = resource.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _GLINER_ENCODER_CONFIG_SHA256:
        raise OfflineArtifactError(
            "pinned GLiNER encoder config failed its byte checksum: "
            f"source_revision={_GLINER_ENCODER_CONFIG_REVISION}, "
            f"expected={_GLINER_ENCODER_CONFIG_SHA256}, actual={digest}"
        )
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("model_type") != "deberta-v2"
        or payload.get("hidden_size") != 768
        or payload.get("num_hidden_layers") != 6
    ):
        raise OfflineArtifactError(
            "pinned GLiNER encoder config has an invalid architecture contract"
        )
    return dict(payload)


def _json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineArtifactError(
            f"invalid locked GLiNER configuration: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise OfflineArtifactError(
            f"locked GLiNER configuration must be an object: {path}"
        )
    return payload


class LocalTransformersNerBackend:
    """Hugging Face token-classification pipeline in strict local-files mode."""

    def __init__(self, artifact: HydratedArtifact) -> None:
        self.artifact = artifact
        self._pipeline: Any | None = None

    def infer(self, text: str) -> tuple[MentionInference, ...]:
        pipeline = self._load()
        raw_items = pipeline(text)
        if not isinstance(raw_items, list):
            raise ProviderOutputError("Transformers NER returned a non-list result")
        return tuple(
            _mapping_to_inference(
                raw,
                text=text,
                label_keys=("entity_group", "entity"),
                text_keys=("word",),
                type_map=_TRANSFORMERS_TYPES,
                trust_provider_text=False,
            )
            for raw in raw_items
        )

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        require_local_directory(self.artifact.local_path, model_name=self.artifact.model_version)
        enforce_offline_model_environment()
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:  # pragma: no cover - exercised by hard-pass environment
            raise ProviderOutputError("the mandatory transformers package is unavailable") from exc
        path = str(self.artifact.local_path)
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._pipeline = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=self.artifact.device,
        )
        return self._pipeline


class LocalSpacyBackend:
    """spaCy NER client loading a pipeline directly from its hydrated directory."""

    def __init__(self, artifact: HydratedArtifact) -> None:
        self.artifact = artifact
        self._pipeline: Any | None = None

    def infer(self, text: str) -> tuple[MentionInference, ...]:
        document = self._load()(text)
        return tuple(
            MentionInference(
                start=int(entity.start_char),
                end=int(entity.end_char),
                text=str(entity.text),
                entity_type=_canonical_entity_type(str(entity.label_), type_map=_SPACY_TYPES),
                # spaCy's standard EntityRecognizer does not expose calibrated
                # per-span probabilities.
                confidence=1.0,
            )
            for entity in document.ents
        )

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        require_local_directory(self.artifact.local_path, model_name=self.artifact.model_version)
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - exercised by hard-pass environment
            raise ProviderOutputError("the mandatory spacy package is unavailable") from exc
        self._pipeline = spacy.load(self.artifact.local_path)
        return self._pipeline


class ModelMentionDetector:
    """Convert backend spans into exact evidence-backed domain mentions."""

    def __init__(
        self,
        config: MentionProviderConfig,
        backend: MentionInferenceBackend,
    ) -> None:
        self.config = config
        self.backend = backend
        self.provider_id = config.provider

    def detect(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
    ) -> tuple[Mention, ...]:
        del segments
        if not document.matches_text(text):
            raise ProviderOutputError("mention input does not match its document snapshot")

        best: dict[tuple[int, int, str], MentionInference] = {}
        for raw in self.backend.infer(text):
            if not raw.text or not _span_is_exact(raw.start, raw.end, raw.text, text):
                raise ProviderOutputError(
                    f"{self.provider_id} returned an inexact mention span "
                    f"[{raw.start}, {raw.end}): {raw.text!r}"
                )
            if raw.confidence < self.config.minimum_confidence:
                continue
            key = (raw.start, raw.end, raw.entity_type.casefold())
            current = best.get(key)
            if current is None or raw.confidence > current.confidence:
                best[key] = raw

        mentions: list[Mention] = []
        for raw in sorted(best.values(), key=lambda item: (item.start, item.end, item.entity_type)):
            span = TextSpan(start=raw.start, end=raw.end, text=raw.text)
            mentions.append(
                Mention(
                    document_version=document.version,
                    evidence=Evidence.build(
                        document,
                        span=span,
                        kind=EvidenceKind.MENTION,
                    ),
                    surface=raw.text,
                    entity_type=raw.entity_type,
                    provenance=Provenance(
                        extractor=self.config.extractor,
                        provider=self.config.provider,
                        model_version=self.config.model_version,
                        configuration_id=self.config.configuration_id,
                        confidence=raw.confidence,
                    ),
                )
            )
        return tuple(mentions)


class GlinerMentionDetector(ModelMentionDetector):
    """Actual GLiNER zero-shot mention detector."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        labels: Sequence[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        backend: MentionInferenceBackend | None = None,
    ) -> None:
        label_tuple = tuple(dict.fromkeys(label.strip() for label in labels if label.strip()))
        configuration_id = stable_id("gliner-mention-config", threshold, flat_ner, *label_tuple)
        super().__init__(
            MentionProviderConfig(
                extractor="gliner_mention_detector",
                provider="gliner",
                model_version=artifact.model_version,
                configuration_id=configuration_id,
                minimum_confidence=threshold,
            ),
            backend
            or LocalGlinerBackend(
                artifact,
                labels=label_tuple,
                threshold=threshold,
                flat_ner=flat_ner,
            ),
        )


class TransformersMentionDetector(ModelMentionDetector):
    """Actual Transformers token-classification mention detector."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        minimum_confidence: float = 0.5,
        backend: MentionInferenceBackend | None = None,
    ) -> None:
        super().__init__(
            MentionProviderConfig(
                extractor="transformers_mention_detector",
                provider="transformers",
                model_version=artifact.model_version,
                configuration_id=stable_id(
                    "transformers-mention-config",
                    "aggregation=simple",
                    minimum_confidence,
                ),
                minimum_confidence=minimum_confidence,
            ),
            backend or LocalTransformersNerBackend(artifact),
        )


class SpacyMentionDetector(ModelMentionDetector):
    """Actual spaCy statistical NER mention detector."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        backend: MentionInferenceBackend | None = None,
    ) -> None:
        super().__init__(
            MentionProviderConfig(
                extractor="spacy_mention_detector",
                provider="spacy",
                model_version=artifact.model_version,
                configuration_id=stable_id(
                    "spacy-mention-config", artifact.repo_id, artifact.revision
                ),
                minimum_confidence=0.0,
            ),
            backend or LocalSpacyBackend(artifact),
        )


class MandatoryMentionEnsemble(MentionDetector):
    """Run every detector, union exact outputs, and retain provider agreement."""

    def __init__(self, providers: Sequence[MentionDetector]) -> None:
        if not providers:
            raise ValueError("mandatory mention ensemble requires at least one provider")
        self.providers = tuple(providers)
        self.provider_ids = tuple(_provider_id(item) for item in self.providers)
        if len(self.provider_ids) != len(set(self.provider_ids)):
            raise ValueError("mandatory mention provider IDs must be unique")
        self.last_agreements: tuple[ProviderAgreement, ...] = ()
        self.last_suppressions: tuple[NestedMentionSuppression, ...] = ()

    def detect(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
    ) -> tuple[Mention, ...]:
        self.last_agreements = ()
        self.last_suppressions = ()
        failures: dict[str, BaseException] = {}
        results: list[tuple[str, tuple[Mention, ...]]] = []
        for provider, provider_id in zip(self.providers, self.provider_ids, strict=True):
            try:
                output = tuple(provider.detect(document, text, segments))
                for mention in output:
                    _validate_provider_mention(mention, document=document, text=text)
                results.append((provider_id, output))
            except Exception as exc:  # noqa: BLE001 - run all providers, aggregate exact failures
                failures[provider_id] = exc
        if failures:
            raise MandatoryProviderError("mention detection", failures)

        grouped: dict[
            tuple[str, int, int, str],
            dict[str, list[Mention]],
        ] = {}
        for provider_id, mentions in results:
            for mention in mentions:
                key = (
                    mention.document_version,
                    mention.span.start,
                    mention.span.end,
                    mention.surface,
                )
                grouped.setdefault(key, {}).setdefault(provider_id, []).append(
                    mention
                )

        containing_by_fragment = _unique_maximal_containers(tuple(grouped))
        agreements: list[ProviderAgreement] = []
        merged_by_key: dict[tuple[str, int, int, str], Mention] = {}
        item_keys: dict[tuple[str, int, int, str], str] = {}
        provider_count = len(self.providers)
        for key, by_provider in sorted(
            grouped.items(),
            key=lambda row: (
                int(row[0][1]),
                int(row[0][2]),
                str(row[0][3]),
            ),
        ):
            provider_votes = {
                provider_id: _provider_type_vote(items)
                for provider_id, items in by_provider.items()
            }
            selected_type_key = _selected_type_key(
                provider_votes,
                surface=str(key[3]),
            )
            selected_votes = {
                provider_id: mention
                for provider_id, mention in provider_votes.items()
                if mention.entity_type.casefold() == selected_type_key
            }
            agreeing = tuple(sorted(selected_votes))
            participants = tuple(sorted(provider_votes))
            exemplar = min(
                selected_votes.values(),
                key=lambda item: (
                    -item.provenance.confidence,
                    item.provenance.provider,
                    item.entity_type,
                ),
            )
            selected_type = exemplar.entity_type
            item_key = stable_id(
                "mention-agreement",
                *key,
                selected_type_key,
                _MENTION_TYPE_SELECTION_POLICY,
            )
            item_keys[key] = item_key
            agreement = ProviderAgreement(
                item_key=item_key,
                agreeing_providers=agreeing,
                total_mandatory_providers=provider_count,
                ratio=len(agreeing) / provider_count,
                participating_providers=participants,
                selected_type=selected_type,
                provider_types={
                    provider_id: provider_votes[provider_id].entity_type
                    for provider_id in participants
                },
                provider_confidences={
                    provider_id: provider_votes[
                        provider_id
                    ].provenance.confidence
                    for provider_id in participants
                },
            )
            agreements.append(agreement)
            # This is a completeness-preserving occurrence union. Confidence
            # averages only selected-type voters; all losing type votes remain
            # in ProviderAgreement and cannot create a duplicate downstream.
            confidence = sum(
                item.provenance.confidence for item in selected_votes.values()
            ) / len(selected_votes)
            versions = tuple(
                sorted(
                    {
                        item.provenance.model_version
                        for item in selected_votes.values()
                    }
                )
            )
            merged_by_key[key] = exemplar.model_copy(
                update={
                    "provenance": Provenance(
                        extractor="mandatory_mention_ensemble",
                        provider="+".join(agreeing),
                        model_version="+".join(versions),
                        configuration_id=(
                            "mandatory-occurrence-type-union-v3;"
                            f"agreement={len(agreeing)}/{provider_count};"
                            f"participation={len(participants)}/{provider_count};"
                            f"type={selected_type};"
                            f"type_policy={_MENTION_TYPE_SELECTION_POLICY};"
                            f"item={item_key}"
                        ),
                        confidence=confidence,
                    )
                }
            )
        self.last_agreements = tuple(agreements)
        self.last_suppressions = tuple(
            NestedMentionSuppression(
                fragment_item_key=item_keys[fragment],
                containing_item_key=item_keys[container],
                fragment_start=fragment[1],
                fragment_end=fragment[2],
                fragment_surface=fragment[3],
                containing_start=container[1],
                containing_end=container[2],
                containing_surface=container[3],
            )
            for fragment, container in sorted(
                containing_by_fragment.items(),
                key=lambda row: (
                    row[0][1],
                    row[0][2],
                    row[0][3],
                ),
            )
        )
        return tuple(
            mention
            for key, mention in merged_by_key.items()
            if key not in containing_by_fragment
        )


def _provider_type_vote(mentions: list[Mention]) -> Mention:
    """Return one provider's deterministic type vote for an occurrence."""

    if not mentions:
        raise ValueError("provider type voting requires at least one mention")
    return min(
        mentions,
        key=lambda mention: (
            _is_generic_type(mention.entity_type),
            -mention.provenance.confidence,
            mention.entity_type.casefold(),
            mention.entity_type,
            mention.mention_id,
        ),
    )


def _selected_type_key(
    provider_votes: Mapping[str, Mention],
    *,
    surface: str,
) -> str:
    """Choose one type without using confidence or provider priority."""

    by_type: dict[str, list[Mention]] = {}
    for mention in provider_votes.values():
        by_type.setdefault(mention.entity_type.casefold(), []).append(mention)
    if not by_type:
        raise ValueError("mention type selection requires provider votes")

    maximum_votes = max(len(mentions) for mentions in by_type.values())
    tied = {
        entity_type
        for entity_type, mentions in by_type.items()
        if len(mentions) == maximum_votes
    }
    if len(tied) == 1:
        return next(iter(tied))

    specific = {
        entity_type
        for entity_type in tied
        if not _is_generic_type(entity_type)
    }
    if len(specific) == 1:
        return next(iter(specific))
    if specific:
        tied = specific

    lexical_form_hint = _lexical_form_type_hint(surface)
    if lexical_form_hint is not None and lexical_form_hint in tied:
        return lexical_form_hint

    suffix_hint = _lexical_suffix_type_hint(surface)
    if suffix_hint is not None and suffix_hint in tied:
        return suffix_hint
    raise ProviderOutputError(
        "unresolved exact-span type vote tie has no conservative source "
        f"lexical hint: surface={surface!r}, types={sorted(tied)}"
    )


def _lexical_form_type_hint(surface: str) -> str | None:
    """Return a type only for a reviewed whole-surface lexical form."""

    matches = tuple(
        entity_type
        for entity_type, pattern in _LEXICAL_FORM_TYPE_HINTS
        if pattern.fullmatch(surface) is not None
    )
    return matches[0] if len(matches) == 1 else None


def _lexical_suffix_type_hint(surface: str) -> str | None:
    """Return one reviewed whole-token suffix hint, never a guessed type."""

    normalized = surface.casefold().strip(
        " \t\n\r.,;:!?()[]{}'\""
    )
    if not normalized:
        return None
    suffix = normalized.rsplit(maxsplit=1)[-1]
    return _LEXICAL_SUFFIX_TYPE_HINTS.get(suffix)


def _unique_maximal_containers(
    occurrences: tuple[tuple[str, int, int, str], ...],
) -> dict[
    tuple[str, int, int, str],
    tuple[str, int, int, str],
]:
    """Map strict fragments only when one maximal containing span exists."""

    result: dict[
        tuple[str, int, int, str],
        tuple[str, int, int, str],
    ] = {}
    for fragment in occurrences:
        containers = [
            candidate
            for candidate in occurrences
            if _strictly_contains(candidate, fragment)
        ]
        maximal = [
            candidate
            for candidate in containers
            if not any(
                _strictly_contains(other, candidate)
                for other in containers
                if other != candidate
            )
        ]
        if len(maximal) == 1:
            result[fragment] = maximal[0]
    return result


def _strictly_contains(
    outer: tuple[str, int, int, str],
    inner: tuple[str, int, int, str],
) -> bool:
    """Return whether ``outer`` strictly contains ``inner`` in one document."""

    return (
        outer[0] == inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and (outer[1] < inner[1] or inner[2] < outer[2])
    )


def _is_generic_type(entity_type: str) -> bool:
    return entity_type.casefold().strip() in _GENERIC_ENTITY_TYPES


def _mapping_to_inference(
    raw: object,
    *,
    text: str,
    label_keys: Sequence[str],
    text_keys: Sequence[str],
    type_map: Mapping[str, str],
    trust_provider_text: bool = True,
) -> MentionInference:
    if not isinstance(raw, Mapping):
        raise ProviderOutputError(f"NER item must be a mapping, got {type(raw).__name__}")
    try:
        start = int(raw["start"])
        end = int(raw["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderOutputError(f"NER item has invalid offsets: {raw!r}") from exc
    if start < 0 or end <= start or end > len(text):
        raise ProviderOutputError(f"NER offsets are outside the source text: {raw!r}")

    label = next((str(raw[key]) for key in label_keys if raw.get(key)), "")
    if not label:
        raise ProviderOutputError(f"NER item has no entity label: {raw!r}")
    provided_text = next((str(raw[key]) for key in text_keys if raw.get(key)), "")
    exact_text = text[start:end]
    if trust_provider_text and provided_text and provided_text != exact_text:
        raise ProviderOutputError(
            f"NER text {provided_text!r} does not match source span {exact_text!r}"
        )
    try:
        confidence = float(raw.get("score", raw.get("confidence", 1.0)))
    except (TypeError, ValueError) as exc:
        raise ProviderOutputError(f"NER confidence is invalid: {raw!r}") from exc
    return MentionInference(
        start=start,
        end=end,
        text=exact_text,
        entity_type=_canonical_entity_type(label, type_map=type_map),
        confidence=confidence,
    )


def _canonical_entity_type(label: str, *, type_map: Mapping[str, str]) -> str:
    normalized = label.strip().upper()
    if normalized.startswith(("B-", "I-")):
        normalized = normalized[2:]
    mapped = type_map.get(normalized)
    if mapped:
        return mapped
    words = normalized.replace("-", " ").replace("_", " ").split()
    return "".join(word.title() for word in words) or "Entity"


def _span_is_exact(start: int, end: int, surface: str, text: str) -> bool:
    return (
        0 <= start < end <= len(text) and end - start == len(surface) and text[start:end] == surface
    )


def _provider_id(provider: object) -> str:
    provider_id = getattr(provider, "provider_id", provider.__class__.__name__)
    return str(provider_id)


def _validate_provider_mention(
    mention: Mention,
    *,
    document: DocumentRef,
    text: str,
) -> None:
    if mention.document_version != document.version:
        raise ProviderOutputError("mention references a different document version")
    if not mention.evidence.is_exact_for(document, text):
        raise ProviderOutputError("mention evidence is not exact for the document")
    if not mention.provenance.complete:
        raise ProviderOutputError("mention provider provenance is incomplete")


__all__ = [
    "GlinerMentionDetector",
    "LocalGlinerBackend",
    "LocalSpacyBackend",
    "LocalTransformersNerBackend",
    "MandatoryMentionEnsemble",
    "MentionProviderConfig",
    "ModelMentionDetector",
    "SpacyMentionDetector",
    "TransformersMentionDetector",
]
