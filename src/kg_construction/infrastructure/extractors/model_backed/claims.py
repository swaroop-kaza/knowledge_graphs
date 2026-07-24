"""Grounded Qwen3 claim extraction through Transformers, MLX, and Ollama."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.application.unstructured_pipeline import ClaimExtractor
from kg_construction.domain.claim import (
    Claim,
    ClaimKind,
    DocumentRef,
    EventRole,
    Evidence,
    EvidenceKind,
    Mention,
    Modality,
    Provenance,
    TextSpan,
)
from kg_construction.domain.models import canonical_json, stable_id
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    ExactSpanOutput,
    HydratedArtifact,
    HydratedServiceModel,
    MandatoryProviderError,
    ProviderAgreement,
    ProviderOutputError,
    StructuredGenerationBackend,
    enforce_offline_model_environment,
    require_local_directory,
)


class StructuredEventRoleOutput(BaseModel):
    """A grounded event role emitted by a structured generation provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    value_span: ExactSpanOutput


class StructuredClaimOutput(BaseModel):
    """One fully grounded model claim before conversion to the domain contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ClaimKind
    subject_span: ExactSpanOutput
    predicate: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    predicate_span: ExactSpanOutput
    object_span: ExactSpanOutput | None = None
    event_roles: tuple[StructuredEventRoleOutput, ...] = ()
    evidence_spans: tuple[ExactSpanOutput, ...] = Field(min_length=1)
    negated: bool = False
    negation_span: ExactSpanOutput | None = None
    modality: Modality = Modality.ASSERTED
    modality_span: ExactSpanOutput | None = None
    attribution_span: ExactSpanOutput | None = None
    temporal_span: ExactSpanOutput | None = None
    quantity_span: ExactSpanOutput | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _required_arguments_and_cues(self) -> StructuredClaimOutput:
        if (
            self.kind in {ClaimKind.RELATION, ClaimKind.FACT, ClaimKind.INTENT}
            and self.object_span is None
        ):
            raise ValueError(f"{self.kind.value} claims require object_span")
        if self.kind == ClaimKind.EVENT and not self.event_roles:
            raise ValueError("event claims require event_roles")
        if self.negated and self.negation_span is None:
            raise ValueError("negated claims require an exact negation_span")
        if self.modality != Modality.ASSERTED and self.modality_span is None:
            raise ValueError("non-asserted claims require an exact modality_span")
        return self


class StructuredClaimResponse(BaseModel):
    """Top-level response schema supplied to every structured backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: tuple[StructuredClaimOutput, ...]


class ClaimProviderConfig(BaseModel):
    """Immutable provider metadata for exact claim provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractor: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)


class LocalTransformersQwenBackend:
    """Qwen3 generation using only a local Transformers snapshot."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        max_new_tokens: int = 2048,
    ) -> None:
        self.artifact = artifact
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def generate_json(
        self,
        *,
        prompt: str,
        json_schema: Mapping[str, Any],
    ) -> object:
        model, tokenizer = self._load()
        messages = _structured_claim_messages(
            prompt=prompt,
            json_schema=json_schema,
        )
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        device_inputs = {key: value.to(self.artifact.device) for key, value in encoded.items()}
        input_length = int(device_inputs["input_ids"].shape[-1])
        generated = model.generate(
            **device_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = generated[0][input_length:]
        return tokenizer.decode(completion, skip_special_tokens=True).strip()

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        require_local_directory(self.artifact.local_path, model_name=self.artifact.model_version)
        enforce_offline_model_environment()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised by hard-pass environment
            raise ProviderOutputError("the mandatory transformers package is unavailable") from exc
        path = str(self.artifact.local_path)
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = model.to(self.artifact.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer


class LocalMlxQwenBackend:
    """Qwen3 generation through MLX-LM from an Apple-Silicon snapshot."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        max_new_tokens: int = 2048,
        max_retries: int = 3,
    ) -> None:
        self.artifact = artifact
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.last_attempt_count = 0
        self.last_quote_alignment_count = 0
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._generate: Any | None = None
        if max_retries < 0:
            raise ValueError("MLX max_retries must be non-negative")

    def generate_json(
        self,
        *,
        prompt: str,
        json_schema: Mapping[str, Any],
    ) -> object:
        model, tokenizer, generate = self._load()
        repair_context = ""
        last_error: BaseException | None = None
        last_output = ""
        for attempt in range(1, self.max_retries + 2):
            self.last_attempt_count = attempt
            messages = _structured_claim_messages(
                prompt=prompt,
                json_schema=json_schema,
                repair_context=repair_context,
            )
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            last_output = str(
                generate(
                    model,
                    tokenizer,
                    prompt=rendered,
                    max_tokens=self.max_new_tokens,
                    verbose=False,
                )
            ).strip()
            try:
                # MLX-LM does not currently expose JSON-schema constrained
                # decoding. Keep the model mandatory and fail closed, while
                # giving the same model bounded validator feedback so it can
                # correct its own output.
                return _validate_response(last_output)
            except Exception as exc:  # noqa: BLE001 - retry covers model output
                last_error = exc
                try:
                    aligned, alignment_count = _align_quoted_claim_response(
                        last_output,
                        prompt=prompt,
                    )
                except Exception:  # noqa: BLE001 - retain original validator feedback
                    pass
                else:
                    self.last_quote_alignment_count = alignment_count
                    return aligned
                repair_context = _structured_repair_context(
                    output=last_output,
                    error=exc,
                )
        raise ProviderOutputError(
            "MLX Qwen failed strict structured output after "
            f"{self.max_retries + 1} attempt(s): "
            f"{type(last_error).__name__}: {last_error}; "
            f"last_output={last_output[:500]!r}"
        ) from last_error

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._tokenizer is not None and self._generate is not None:
            return self._model, self._tokenizer, self._generate
        require_local_directory(self.artifact.local_path, model_name=self.artifact.model_version)
        enforce_offline_model_environment()
        try:
            from mlx_lm import generate, load
        except ImportError as exc:  # pragma: no cover - exercised by Darwin hard pass
            raise ProviderOutputError("the mandatory mlx-lm package is unavailable") from exc
        # `load` accepts a local directory. The tokenizer flag independently
        # forbids any missing tokenizer asset from being fetched.
        loaded = load(
            str(self.artifact.local_path),
            tokenizer_config={
                "local_files_only": True,
                "trust_remote_code": False,
            },
            return_config=False,
        )
        model, tokenizer = loaded[0], loaded[1]
        self._model = model
        self._tokenizer = tokenizer
        self._generate = generate
        return model, tokenizer, generate


class OllamaTransport(Protocol):
    """Injectable local Ollama transport."""

    def list_models(self, base_url: str) -> Mapping[str, str]: ...

    def chat(
        self,
        base_url: str,
        payload: Mapping[str, Any],
    ) -> object: ...


class UrlLibOllamaTransport:
    """Minimal Ollama HTTP transport; it deliberately has no pull operation."""

    def __init__(self, *, chat_timeout_seconds: int = 600) -> None:
        if chat_timeout_seconds <= 0:
            raise ValueError("Ollama chat timeout must be positive")
        self.chat_timeout_seconds = chat_timeout_seconds

    def list_models(self, base_url: str) -> Mapping[str, str]:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=10,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            str(row.get("name") or row.get("model")): _canonical_ollama_digest(
                row.get("digest")
            )
            for row in payload.get("models", [])
            if row.get("name") or row.get("model")
        }

    def chat(
        self,
        base_url: str,
        payload: Mapping[str, Any],
    ) -> object:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.chat_timeout_seconds,
        ) as response:
            return json.loads(response.read().decode("utf-8"))


class LocalOllamaQwenBackend:
    """Ollama structured output after exact local tag/digest verification."""

    def __init__(
        self,
        model: HydratedServiceModel,
        *,
        base_url: str = "http://localhost:11434",
        transport: OllamaTransport | None = None,
        retry_on_invalid_output: bool = True,
        max_retries: int = 3,
    ) -> None:
        if model.service != "ollama":
            raise ValueError(f"expected an Ollama service model, got {model.service!r}")
        self.model = model
        self.base_url = base_url
        self.transport = transport or UrlLibOllamaTransport()
        self.retry_on_invalid_output = retry_on_invalid_output
        self.max_retries = max_retries
        self.last_attempt_count = 0
        self.last_quote_alignment_count = 0
        if max_retries < 0:
            raise ValueError("Ollama max_retries must be non-negative")

    def generate_json(
        self,
        *,
        prompt: str,
        json_schema: Mapping[str, Any],
    ) -> object:
        installed = self.transport.list_models(self.base_url)
        actual_digest = _canonical_ollama_digest(installed.get(self.model.model))
        if actual_digest is None:
            raise ProviderOutputError(
                f"locked Ollama model {self.model.model!r} is not present locally"
            )
        if actual_digest != self.model.digest:
            raise ProviderOutputError(
                f"Ollama digest mismatch for {self.model.model}: "
                f"expected {self.model.digest}, got {actual_digest}"
            )
        attempts = self.max_retries + 1 if self.retry_on_invalid_output else 1
        repair_context = ""
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            self.last_attempt_count = attempt
            content = ""
            payload = {
                "model": self.model.model,
                "messages": _structured_claim_messages(
                    prompt=prompt,
                    json_schema=json_schema,
                    repair_context=repair_context,
                ),
                "stream": False,
                "format": dict(json_schema),
                # Qwen3 thinking can consume the complete prediction budget
                # before the schema-constrained answer. Ollama exposes this as
                # a first-class request field; disable it for extraction.
                "think": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "seed": 0,
                    "num_predict": 2048,
                },
            }
            try:
                content = _ollama_content(
                    self.transport.chat(self.base_url, payload)
                )
                # Validate here so malformed/schema-invalid output participates
                # in the configured retry policy instead of escaping to the
                # outer ensemble after only one attempt.
                try:
                    return _validate_response(content)
                except Exception:
                    aligned, alignment_count = _align_quoted_claim_response(
                        content,
                        prompt=prompt,
                    )
                    self.last_quote_alignment_count = alignment_count
                    return aligned
            except Exception as exc:  # noqa: BLE001 - retry contract covers transport/output
                last_error = exc
                repair_context = _structured_repair_context(
                    output=content,
                    error=exc,
                )
        raise ProviderOutputError(
            f"Ollama failed strict structured output after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


class ModelClaimExtractor:
    """Validate structured output and construct only exactly grounded claims."""

    def __init__(
        self,
        config: ClaimProviderConfig,
        backend: StructuredGenerationBackend,
    ) -> None:
        self.config = config
        self.backend = backend
        self.provider_id = config.provider

    def extract(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[Claim, ...]:
        if not document.matches_text(text):
            raise ProviderOutputError("claim input does not match its document snapshot")
        prompt = _claim_prompt(
            document=document,
            text=text,
            segments=segments,
            mentions=mentions,
        )
        raw_payload = self.backend.generate_json(
            prompt=prompt,
            json_schema=_strict_claim_json_schema(),
        )
        payload = _validate_response(raw_payload)
        claims = tuple(
            self._to_claim(
                output,
                document=document,
                text=text,
                mentions=mentions,
            )
            for output in payload.claims
        )
        # Exact duplicates from one provider are not distinct observations.
        unique: dict[tuple[object, ...], Claim] = {}
        for claim in claims:
            unique.setdefault(_claim_key(claim), claim)
        return tuple(unique.values())

    def _to_claim(
        self,
        output: StructuredClaimOutput,
        *,
        document: DocumentRef,
        text: str,
        mentions: tuple[Mention, ...],
    ) -> Claim:
        grounded_spans = [
            output.subject_span,
            output.predicate_span,
            *output.evidence_spans,
            *(role.value_span for role in output.event_roles),
        ]
        for optional_span in (
            output.object_span,
            output.negation_span,
            output.modality_span,
            output.attribution_span,
            output.temporal_span,
            output.quantity_span,
        ):
            if optional_span is not None:
                grounded_spans.append(optional_span)
        for span in grounded_spans:
            _require_exact_span(span, text=text, provider=self.provider_id)

        for semantic_span in (
            output.subject_span,
            output.predicate_span,
            output.object_span,
            output.negation_span,
            output.modality_span,
            output.attribution_span,
            output.temporal_span,
            output.quantity_span,
            *(role.value_span for role in output.event_roles),
        ):
            if semantic_span is not None and not any(
                _contains(evidence, semantic_span) for evidence in output.evidence_spans
            ):
                raise ProviderOutputError(
                    f"{self.provider_id} returned semantic span "
                    f"[{semantic_span.start}, {semantic_span.end}) outside claim evidence"
                )

        evidence_by_span: dict[tuple[int, int, str], Evidence] = {}
        for item in output.evidence_spans:
            key = (item.start, item.end, item.text)
            evidence_by_span.setdefault(
                key,
                Evidence.build(
                    document,
                    span=TextSpan(start=item.start, end=item.end, text=item.text),
                    kind=EvidenceKind.PASSAGE,
                ),
            )
        evidence = tuple(evidence_by_span.values())
        subject_mention = _matching_mention(output.subject_span, mentions)
        object_mention = (
            _matching_mention(output.object_span, mentions)
            if output.object_span is not None
            else None
        )
        roles = tuple(
            EventRole(
                role=role.role,
                value=role.value_span.text,
                mention_id=(
                    matched.mention_id
                    if (matched := _matching_mention(role.value_span, mentions)) is not None
                    else None
                ),
            )
            for role in output.event_roles
        )
        claim = Claim(
            document=document,
            kind=output.kind,
            subject=output.subject_span.text,
            predicate=_grounded_predicate(output, text=text),
            object=output.object_span.text if output.object_span is not None else None,
            subject_mention_id=(
                subject_mention.mention_id if subject_mention is not None else None
            ),
            object_mention_id=(object_mention.mention_id if object_mention is not None else None),
            event_roles=roles,
            evidence=evidence,
            negated=output.negated,
            modality=output.modality,
            attribution=(
                output.attribution_span.text if output.attribution_span is not None else None
            ),
            temporal=(output.temporal_span.text if output.temporal_span is not None else None),
            quantity=(output.quantity_span.text if output.quantity_span is not None else None),
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
                f"{self.provider_id} produced a claim without exact evidence/provenance"
            )
        return claim


class TransformersQwenClaimExtractor(ModelClaimExtractor):
    """Structured Qwen3 extraction through Hugging Face Transformers."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        max_new_tokens: int = 2048,
        backend: StructuredGenerationBackend | None = None,
    ) -> None:
        super().__init__(
            ClaimProviderConfig(
                extractor="transformers_qwen_claim_extractor",
                provider="transformers_qwen3",
                model_version=artifact.model_version,
                configuration_id=stable_id(
                    "transformers-qwen-claims",
                    "schema-v3",
                    max_new_tokens,
                    "thinking=false",
                ),
            ),
            backend
            or LocalTransformersQwenBackend(
                artifact,
                max_new_tokens=max_new_tokens,
            ),
        )


class MlxQwenClaimExtractor(ModelClaimExtractor):
    """Structured Qwen3 extraction through MLX-LM on Apple Silicon."""

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        max_new_tokens: int = 2048,
        max_retries: int = 3,
        backend: StructuredGenerationBackend | None = None,
    ) -> None:
        super().__init__(
            ClaimProviderConfig(
                extractor="mlx_qwen_claim_extractor",
                provider="mlx_qwen3",
                model_version=artifact.model_version,
                configuration_id=stable_id(
                    "mlx-qwen-claims",
                    "schema-v3",
                    max_new_tokens,
                    f"max_retries={max_retries}",
                    "thinking=false",
                ),
            ),
            backend
            or LocalMlxQwenBackend(
                artifact,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
            ),
        )


class OllamaQwenClaimExtractor(ModelClaimExtractor):
    """Structured Qwen3 extraction through a digest-locked local Ollama model."""

    def __init__(
        self,
        model: HydratedServiceModel,
        *,
        base_url: str = "http://localhost:11434",
        backend: StructuredGenerationBackend | None = None,
        transport: OllamaTransport | None = None,
        retry_on_invalid_output: bool = True,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            ClaimProviderConfig(
                extractor="ollama_qwen_claim_extractor",
                provider="ollama_qwen3",
                model_version=f"{model.model}@{model.digest}",
                configuration_id=stable_id(
                    "ollama-qwen-claims",
                    "schema-v3",
                    model.model,
                    model.digest,
                    "temperature=0",
                    f"retry_on_invalid_output={retry_on_invalid_output}",
                    f"max_retries={max_retries}",
                ),
            ),
            backend
            or LocalOllamaQwenBackend(
                model,
                base_url=base_url,
                transport=transport,
                retry_on_invalid_output=retry_on_invalid_output,
                max_retries=max_retries,
            ),
        )


class MandatoryClaimEnsemble(ClaimExtractor):
    """Run every generator, union exact claims, and retain provider agreement."""

    def __init__(self, providers: Sequence[ClaimExtractor]) -> None:
        if not providers:
            raise ValueError("mandatory claim ensemble requires at least one provider")
        self.providers = tuple(providers)
        self.provider_ids = tuple(_provider_id(item) for item in self.providers)
        if len(self.provider_ids) != len(set(self.provider_ids)):
            raise ValueError("mandatory claim provider IDs must be unique")
        self.last_agreements: tuple[ProviderAgreement, ...] = ()

    def extract(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[Claim, ...]:
        failures: dict[str, BaseException] = {}
        results: list[tuple[str, tuple[Claim, ...]]] = []
        for provider, provider_id in zip(self.providers, self.provider_ids, strict=True):
            try:
                output = tuple(provider.extract(document, text, segments, mentions))
                for claim in output:
                    _validate_provider_claim(claim, document=document, text=text)
                results.append((provider_id, output))
            except Exception as exc:  # noqa: BLE001 - run all providers, aggregate exact failures
                failures[provider_id] = exc
        if failures:
            self.last_agreements = ()
            raise MandatoryProviderError("claim extraction", failures)

        grouped: dict[tuple[object, ...], dict[str, Claim]] = {}
        for provider_id, claims in results:
            for claim in claims:
                key = _claim_key(claim)
                previous = grouped.setdefault(key, {}).get(provider_id)
                if previous is None or (
                    claim.provenance.confidence > previous.provenance.confidence
                ):
                    grouped[key][provider_id] = claim

        agreements: list[ProviderAgreement] = []
        merged: list[Claim] = []
        provider_count = len(self.providers)
        for key, by_provider in sorted(
            grouped.items(),
            key=lambda row: (
                min(evidence.span.start for evidence in next(iter(row[1].values())).evidence),
                str(row[0]),
            ),
        ):
            agreeing = tuple(sorted(by_provider))
            exemplar = min(
                by_provider.values(),
                key=lambda item: (
                    -item.provenance.confidence,
                    item.provenance.provider,
                ),
            )
            item_key = stable_id("claim-agreement", canonical_json(key))
            agreement = ProviderAgreement(
                item_key=item_key,
                agreeing_providers=agreeing,
                total_mandatory_providers=provider_count,
                ratio=len(agreeing) / provider_count,
            )
            agreements.append(agreement)
            # Preserve grounded facts/events/intents found by a specialist.
            # Provider agreement remains an explicit ratio rather than an
            # implicit confidence penalty that would discard union-only claims.
            confidence = sum(
                item.provenance.confidence for item in by_provider.values()
            ) / len(by_provider)
            versions = tuple(
                sorted({item.provenance.model_version for item in by_provider.values()})
            )
            merged.append(
                exemplar.model_copy(
                    update={
                        "provenance": Provenance(
                            extractor="mandatory_claim_ensemble",
                            provider="+".join(agreeing),
                            model_version="+".join(versions),
                            configuration_id=(
                                "mandatory-union-v1;"
                                f"agreement={len(agreeing)}/{provider_count};"
                                f"item={item_key}"
                            ),
                            confidence=confidence,
                        )
                    }
                )
            )
        self.last_agreements = tuple(agreements)
        return tuple(merged)


def _claim_prompt(
    *,
    document: DocumentRef,
    text: str,
    segments: tuple[Evidence, ...],
    mentions: tuple[Mention, ...],
) -> str:
    """Build a data-only prompt with absolute offset anchors."""

    payload = {
        "document": {
            "id": document.document_id,
            "version": document.version,
            "text_length": document.text_length,
            "text": text,
        },
        "segments": [
            {
                "start": item.span.start,
                "end": item.span.end,
                "text": item.span.text,
            }
            for item in segments
        ],
        "mentions": [
            {
                "mention_id": item.mention_id,
                "start": item.span.start,
                "end": item.span.end,
                "text": item.surface,
                "entity_type": item.entity_type,
            }
            for item in mentions
        ],
    }
    return (
        "Extract every relation, fact, event, and intent explicitly stated in the "
        "document. Preserve negation, modality, attribution, time, and quantity. "
        "All offsets are absolute, end-exclusive Unicode code-point indexes into "
        "document.text. Every semantic value and cue must have an exact span and "
        "must be contained by an evidence span. Use a lowercase snake_case "
        'predicate. Return {"claims": []} when there are no grounded claims.\n\n'
        f"INPUT={json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def _structured_claim_messages(
    *,
    prompt: str,
    json_schema: Mapping[str, Any],
    repair_context: str = "",
) -> list[dict[str, str]]:
    """Return the identical fail-closed structured contract for local Qwen backends."""

    rules = (
        "STRICT CONTRACT:\n"
        "1. The outer object has exactly one key, claims. claims is an array.\n"
        "2. Each claim uses subject_span, predicate_span, optional object_span, "
        "and evidence_spans; never use subject, object, relation, facts, entities, "
        "or evidence as replacement keys.\n"
        "3. Every span is exactly {\"start\": integer, \"end\": integer, "
        "\"text\": string}; start < end and text must equal "
        "document.text[start:end]. Never emit an empty or zero-width span.\n"
        "4. Every claim has kind, predicate, confidence, subject_span, "
        "predicate_span, and a non-empty evidence_spans array. relation, fact, "
        "and intent also require object_span. event requires a non-empty "
        "event_roles array whose values are exact value_span objects.\n"
        "5. confidence is a JSON number from 0 through 1. Use JSON null for an "
        "unknown optional span. Do not invent offsets or text. If no claim can "
        "satisfy every rule, return exactly {\"claims\":[]}.\n"
        "6. predicate is the normalized semantic relation. Include an explicit "
        "source preposition when it changes meaning: 'worked at' becomes "
        "worked_at and 'worked with' becomes worked_with. predicate_span quotes "
        "the complete source cue, including that preposition.\n"
        "7. Output the JSON object only: no Markdown, explanation, or thinking."
    )
    schema_text = json.dumps(json_schema, sort_keys=True, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are a grounded information-extraction engine. Return one "
                "strict JSON object and nothing else. Source evidence and exact "
                "offsets are mandatory."
            ),
        },
        {
            "role": "user",
            "content": (
                "FORMAT EXAMPLE ONLY. document.text is \"Ada founded Acme.\" "
                "Return its one grounded relation using the strict contract."
            ),
        },
        {
            "role": "assistant",
            "content": (
                '{"claims":[{"kind":"relation","subject_span":{"start":0,'
                '"end":3,"text":"Ada"},"predicate":"founded",'
                '"predicate_span":{"start":4,"end":11,"text":"founded"},'
                '"object_span":{"start":12,"end":16,"text":"Acme"},'
                '"evidence_spans":[{"start":0,"end":17,'
                '"text":"Ada founded Acme."}],"confidence":1.0}]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n{rules}\n\nThe required JSON Schema is:\n"
                f"{schema_text}{repair_context}"
            ),
        },
    ]


def _strict_claim_json_schema() -> dict[str, Any]:
    """Expose semantic model-validator rules to guided JSON decoders."""

    schema = StructuredClaimResponse.model_json_schema()
    claim_schema = schema["$defs"]["StructuredClaimOutput"]
    semantic_conditions = [
        {
            "if": {
                "properties": {
                    "kind": {"enum": ["relation", "fact", "intent"]},
                },
                "required": ["kind"],
            },
            "then": {
                "required": ["object_span"],
                "properties": {
                    "object_span": {"$ref": "#/$defs/ExactSpanOutput"},
                },
            },
        },
        {
            "if": {
                "properties": {"kind": {"const": "event"}},
                "required": ["kind"],
            },
            "then": {
                "required": ["event_roles"],
                "properties": {
                    "event_roles": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "$ref": "#/$defs/StructuredEventRoleOutput",
                        },
                    },
                },
            },
        },
        {
            "if": {
                "properties": {"negated": {"const": True}},
                "required": ["negated"],
            },
            "then": {
                "required": ["negation_span"],
                "properties": {
                    "negation_span": {"$ref": "#/$defs/ExactSpanOutput"},
                },
            },
        },
        {
            "if": {
                "properties": {
                    "modality": {
                        "enum": ["probable", "possible", "hypothetical"],
                    },
                },
                "required": ["modality"],
            },
            "then": {
                "required": ["modality_span"],
                "properties": {
                    "modality_span": {"$ref": "#/$defs/ExactSpanOutput"},
                },
            },
        },
    ]
    claim_schema["allOf"] = semantic_conditions

    # Ollama/llama.cpp guided decoding supports discriminated alternatives more
    # consistently than draft-2020 `if`/`then`. Put the kind-specific
    # requirements directly into four complete alternatives while retaining the
    # runtime conditions above for validators that implement them.
    variants: list[dict[str, Any]] = []
    common_required = list(claim_schema["required"])
    for kind in ("relation", "fact", "intent", "event"):
        variant = json.loads(json.dumps(claim_schema))
        variant.pop("allOf", None)
        variant["title"] = f"{kind.title()}ClaimOutput"
        variant["properties"]["kind"] = {"const": kind, "type": "string"}
        required = list(common_required)
        if kind == "event":
            required.append("event_roles")
            variant["properties"]["event_roles"] = {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/StructuredEventRoleOutput"},
            }
        else:
            required.append("object_span")
            variant["properties"]["object_span"] = {
                "$ref": "#/$defs/ExactSpanOutput",
            }
        variant["required"] = list(dict.fromkeys(required))
        variants.append(variant)
    schema["properties"]["claims"]["items"] = {"oneOf": variants}
    return schema


def _structured_repair_context(*, output: str, error: BaseException) -> str:
    """Bound validator feedback so the same mandatory model can repair its output."""

    rejected = output[:4000]
    validation_error = str(error)[:4000]
    return (
        "\n\nREPAIR REQUIRED. Your previous JSON was rejected by the strict "
        f"validator ({type(error).__name__}: {validation_error}). "
        "Correct every reported field and return a complete replacement object. "
        "Do not repeat the invalid shape. Previous rejected output:\n"
        f"{rejected}"
    )


_PREDICATE_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_SEMANTIC_PREPOSITIONS = {
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "to",
    "with",
}


def _align_quoted_claim_response(
    raw: object,
    *,
    prompt: str,
) -> tuple[StructuredClaimResponse, int]:
    """Align verbatim model quotes and return only a schema-valid response.

    Small local models can reliably copy a quote while being poor Unicode
    counters. This adapter never invents a quote: it accepts a quoted span only
    when the quote occurs uniquely in the immutable document, or when the
    supplied offsets already select one exact occurrence. Repeated ambiguous
    text remains an error.
    """

    if isinstance(raw, str):
        try:
            payload = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ProviderOutputError("quote alignment requires one JSON object") from exc
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise ProviderOutputError("quote alignment requires a JSON object")
    input_payload = _claim_input_payload(prompt)
    document = input_payload.get("document")
    if not isinstance(document, Mapping) or not isinstance(document.get("text"), str):
        raise ProviderOutputError("claim prompt has no immutable document text")
    text = str(document["text"])
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ProviderOutputError("quote alignment requires at least one model claim")

    aligned_claims: list[dict[str, object]] = []
    alignment_count = 0
    for raw_claim in claims:
        if not isinstance(raw_claim, Mapping):
            raise ProviderOutputError("model claim is not an object")
        claim = dict(raw_claim)
        for field_name in (
            "subject_span",
            "predicate_span",
            "object_span",
            "negation_span",
            "modality_span",
            "attribution_span",
            "temporal_span",
            "quantity_span",
        ):
            span = claim.get(field_name)
            if span is None:
                continue
            claim[field_name], changed = _align_one_quote(span, text=text)
            alignment_count += int(changed)

        evidence_spans = claim.get("evidence_spans")
        if not isinstance(evidence_spans, list) or not evidence_spans:
            raise ProviderOutputError("quote alignment requires quoted evidence_spans")
        aligned_evidence: list[dict[str, object]] = []
        for span in evidence_spans:
            aligned, changed = _align_one_quote(span, text=text)
            aligned_evidence.append(aligned)
            alignment_count += int(changed)
        claim["evidence_spans"] = aligned_evidence

        roles = claim.get("event_roles", [])
        if not isinstance(roles, list | tuple):
            raise ProviderOutputError("event_roles must be an array")
        aligned_roles: list[dict[str, object]] = []
        for raw_role in roles:
            if not isinstance(raw_role, Mapping):
                raise ProviderOutputError("event role is not an object")
            role = dict(raw_role)
            role["value_span"], changed = _align_one_quote(
                role.get("value_span"),
                text=text,
            )
            alignment_count += int(changed)
            aligned_roles.append(role)
        claim["event_roles"] = aligned_roles

        predicate = claim.get("predicate")
        predicate_span = claim.get("predicate_span")
        if not isinstance(predicate, str) or not isinstance(predicate_span, Mapping):
            raise ProviderOutputError("quote alignment requires predicate and predicate_span")
        claim["predicate"] = _normalize_aligned_predicate(
            predicate,
            cue=str(predicate_span["text"]),
        )
        # A missing model self-score is replaced by an explicit, conservative
        # quote-alignment calibration. It is not evidence confidence.
        if "confidence" not in claim:
            claim["confidence"] = 0.75
        aligned_claims.append(claim)

    response = _validate_response({"claims": aligned_claims})
    return response, alignment_count


def _claim_input_payload(prompt: str) -> Mapping[str, Any]:
    marker = "INPUT="
    if marker not in prompt:
        raise ProviderOutputError("claim prompt has no structured input payload")
    raw_payload = prompt.rsplit(marker, maxsplit=1)[1]
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ProviderOutputError("claim prompt input payload is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ProviderOutputError("claim prompt input payload is not an object")
    return payload


def _align_one_quote(
    raw_span: object,
    *,
    text: str,
) -> tuple[dict[str, object], bool]:
    if not isinstance(raw_span, Mapping):
        raise ProviderOutputError("quoted span is not an object")
    quote = raw_span.get("text")
    if not isinstance(quote, str) or not quote:
        raise ProviderOutputError("quoted span is missing verbatim text")
    starts: list[int] = []
    cursor = 0
    while (start := text.find(quote, cursor)) >= 0:
        starts.append(start)
        cursor = start + 1
    if not starts:
        raise ProviderOutputError(f"model quote is absent from source text: {quote!r}")
    proposed_start = raw_span.get("start")
    proposed_end = raw_span.get("end")
    if len(starts) == 1:
        start = starts[0]
    elif isinstance(proposed_start, int) and proposed_start in starts:
        start = proposed_start
    else:
        raise ProviderOutputError(
            f"model quote is ambiguous without an exact occurrence anchor: {quote!r}"
        )
    end = start + len(quote)
    aligned = {"start": start, "end": end, "text": quote}
    changed = proposed_start != start or proposed_end != end
    return aligned, changed


def _normalize_aligned_predicate(predicate: str, *, cue: str) -> str:
    normalized = _PREDICATE_TOKEN_RE.sub("_", predicate.casefold()).strip("_")
    cue_normalized = _PREDICATE_TOKEN_RE.sub("_", cue.casefold()).strip("_")
    if not normalized:
        raise ProviderOutputError("model predicate normalizes to an empty value")
    if cue_normalized in _SEMANTIC_PREPOSITIONS and not normalized.endswith(
        f"_{cue_normalized}"
    ):
        normalized = f"{normalized}_{cue_normalized}"
    return normalized


def _grounded_predicate(
    output: StructuredClaimOutput,
    *,
    text: str,
) -> str:
    """Normalize a relation label from its grounded cue and argument gap."""

    normalized = _PREDICATE_TOKEN_RE.sub(
        "_",
        output.predicate.casefold(),
    ).strip("_")
    cue = _PREDICATE_TOKEN_RE.sub(
        "_",
        output.predicate_span.text.casefold(),
    ).strip("_")
    cue_tail = cue.rsplit("_", maxsplit=1)[-1]
    if cue_tail in _SEMANTIC_PREPOSITIONS and not normalized.endswith(
        f"_{cue_tail}"
    ):
        normalized = f"{normalized}_{cue_tail}"
    if (
        output.object_span is not None
        and output.predicate_span.end <= output.object_span.start
    ):
        argument_gap = text[
            output.predicate_span.end : output.object_span.start
        ]
        gap = _PREDICATE_TOKEN_RE.sub("_", argument_gap.casefold()).strip("_")
        if gap in _SEMANTIC_PREPOSITIONS and not normalized.endswith(f"_{gap}"):
            normalized = f"{normalized}_{gap}"
    if not normalized:
        raise ProviderOutputError("grounded predicate normalizes to an empty value")
    return normalized


def _validate_response(raw: object) -> StructuredClaimResponse:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ProviderOutputError(
                "structured claim provider did not return one strict JSON object"
            ) from exc
    try:
        return StructuredClaimResponse.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise ProviderOutputError(f"structured claim response failed schema: {exc}") from exc


def _ollama_content(response: object) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, Mapping):
        raise ProviderOutputError("Ollama returned a non-object response")
    message = response.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return str(message["content"])
    if isinstance(response.get("response"), str):
        return str(response["response"])
    raise ProviderOutputError("Ollama response has no assistant content")


def _canonical_ollama_digest(value: object) -> str:
    digest = str(value or "").lower()
    if len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    ):
        return f"sha256:{digest}"
    return digest


def _require_exact_span(
    span: ExactSpanOutput,
    *,
    text: str,
    provider: str,
) -> None:
    if not span.is_exact_in(text):
        raise ProviderOutputError(
            f"{provider} returned an inexact source span [{span.start}, {span.end}): {span.text!r}"
        )


def _contains(container: ExactSpanOutput, item: ExactSpanOutput) -> bool:
    return container.start <= item.start and item.end <= container.end


def _matching_mention(
    span: ExactSpanOutput | None,
    mentions: tuple[Mention, ...],
) -> Mention | None:
    if span is None:
        return None
    matches = [
        item
        for item in mentions
        if item.span.start == span.start and item.span.end == span.end and item.surface == span.text
    ]
    if not matches:
        return None
    return min(matches, key=lambda item: (item.entity_type, item.mention_id))


def _claim_key(claim: Claim) -> tuple[object, ...]:
    return (
        claim.document.version,
        claim.kind.value,
        claim.subject,
        claim.predicate,
        claim.object,
        tuple(
            (
                role.role,
                role.value,
                role.mention_id,
                role.canonical_entity_id,
            )
            for role in claim.event_roles
        ),
        tuple(
            (
                evidence.span.start,
                evidence.span.end,
                evidence.span.text,
            )
            for evidence in claim.evidence
        ),
        claim.negated,
        claim.modality.value,
        claim.attribution,
        claim.temporal,
        claim.quantity,
    )


def _provider_id(provider: object) -> str:
    provider_id = getattr(provider, "provider_id", provider.__class__.__name__)
    return str(provider_id)


def _validate_provider_claim(
    claim: Claim,
    *,
    document: DocumentRef,
    text: str,
) -> None:
    if claim.document != document:
        raise ProviderOutputError("claim references a different document snapshot")
    if not claim.has_complete_provenance(text):
        raise ProviderOutputError("claim evidence or provider provenance is incomplete")


__all__ = [
    "ClaimProviderConfig",
    "LocalMlxQwenBackend",
    "LocalOllamaQwenBackend",
    "LocalTransformersQwenBackend",
    "MandatoryClaimEnsemble",
    "MlxQwenClaimExtractor",
    "ModelClaimExtractor",
    "OllamaQwenClaimExtractor",
    "OllamaTransport",
    "StructuredClaimOutput",
    "StructuredClaimResponse",
    "StructuredEventRoleOutput",
    "TransformersQwenClaimExtractor",
    "UrlLibOllamaTransport",
]
