"""Grounded Qwen3 claim extraction through Transformers, MLX, and Ollama."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import suppress
from functools import lru_cache
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from kg_construction.unstructured.pipeline import ClaimExtractor


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
        max_retries: int = 3,
    ) -> None:
        self.artifact = artifact
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.last_attempt_count = 0
        self.last_quote_alignment_count = 0
        self.last_rejected_claim_count = 0
        self.last_rejected_claim_diagnostics: tuple[str, ...] = ()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        if max_retries < 0:
            raise ValueError("Transformers max_retries must be non-negative")

    def generate_json(
        self,
        *,
        prompt: str,
        json_schema: Mapping[str, Any],
    ) -> object:
        self.last_rejected_claim_count = 0
        self.last_rejected_claim_diagnostics = ()
        repair_context = ""
        last_error: BaseException | None = None
        last_output = ""
        best_partial: tuple[
            StructuredClaimResponse,
            int,
            tuple[str, ...],
        ] | None = None
        for attempt in range(1, self.max_retries + 2):
            self.last_attempt_count = attempt
            messages = _structured_claim_messages(
                prompt=prompt,
                json_schema=json_schema,
                repair_context=repair_context,
            )
            last_output = self._complete(messages)
            try:
                response = _validate_response(last_output)
                return _validate_grounded_response(response, prompt=prompt)
            except Exception as exc:  # noqa: BLE001 - retry covers model output
                last_error = exc
                try:
                    aligned, alignment_count = _align_quoted_claim_response(
                        last_output,
                        prompt=prompt,
                    )
                    aligned = _validate_grounded_response(
                        aligned,
                        prompt=prompt,
                    )
                except Exception as alignment_error:  # noqa: BLE001
                    # The source-grounding error is more actionable than the
                    # raw output, but the schema diagnostic can identify other
                    # fields the model must repair in the same retry.
                    last_error = _combined_validation_error(
                        schema_error=exc,
                        alignment_error=alignment_error,
                    )
                    with suppress(Exception):
                        partial = _align_valid_quoted_claims(
                            last_output,
                            prompt=prompt,
                        )
                        if best_partial is None or len(
                            partial[0].claims
                        ) > len(best_partial[0].claims):
                            best_partial = partial
                else:
                    self.last_quote_alignment_count = alignment_count
                    return aligned
                repair_context = _structured_repair_context(
                    output=last_output,
                    error=last_error,
                )
        if best_partial is not None:
            response, alignment_count, diagnostics = best_partial
            self.last_quote_alignment_count = alignment_count
            self.last_rejected_claim_count = len(diagnostics)
            self.last_rejected_claim_diagnostics = diagnostics
            return response
        raise ProviderOutputError(
            "Transformers Qwen failed strict structured output after "
            f"{self.max_retries + 1} attempt(s): "
            f"{type(last_error).__name__}: {last_error}; "
            f"last_output={last_output[:500]!r}"
        ) from last_error

    def _complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        model, tokenizer = self._load()
        rendered = tokenizer.apply_chat_template(
            list(messages),
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
        self.last_rejected_claim_count = 0
        self.last_rejected_claim_diagnostics: tuple[str, ...] = ()
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
        self.last_rejected_claim_count = 0
        self.last_rejected_claim_diagnostics = ()
        repair_context = ""
        last_error: BaseException | None = None
        last_output = ""
        best_partial: tuple[
            StructuredClaimResponse,
            int,
            tuple[str, ...],
        ] | None = None
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
            try:
                last_output = str(
                    generate(
                        model,
                        tokenizer,
                        prompt=rendered,
                        max_tokens=self.max_new_tokens,
                        verbose=False,
                    )
                ).strip()
            finally:
                # MLX retains compiled graphs and temporary Metal allocations
                # unless its cache is explicitly cleared. The model weights
                # remain resident, so every mandatory replay still uses the
                # same locked model without accumulating per-request state.
                if str(getattr(generate, "__module__", "")).startswith("mlx_lm"):
                    try:
                        import mlx.core as mx
                    except ImportError:  # pragma: no cover - mandatory on Darwin
                        pass
                    else:
                        mx.clear_cache()
            try:
                # MLX-LM does not currently expose JSON-schema constrained
                # decoding. Keep the model mandatory and fail closed, while
                # giving the same model bounded validator feedback so it can
                # correct its own output.
                response = _validate_response(last_output)
                return _validate_grounded_response(response, prompt=prompt)
            except Exception as exc:  # noqa: BLE001 - retry covers model output
                last_error = exc
                try:
                    aligned, alignment_count = _align_quoted_claim_response(
                        last_output,
                        prompt=prompt,
                    )
                    aligned = _validate_grounded_response(
                        aligned,
                        prompt=prompt,
                    )
                except Exception as alignment_error:  # noqa: BLE001
                    last_error = _combined_validation_error(
                        schema_error=exc,
                        alignment_error=alignment_error,
                    )
                    with suppress(Exception):
                        partial = _align_valid_quoted_claims(
                            last_output,
                            prompt=prompt,
                        )
                        if best_partial is None or len(
                            partial[0].claims
                        ) > len(best_partial[0].claims):
                            best_partial = partial
                else:
                    self.last_quote_alignment_count = alignment_count
                    return aligned
                repair_context = _structured_repair_context(
                    output=last_output,
                    error=last_error,
                )
        if best_partial is not None:
            response, alignment_count, diagnostics = best_partial
            self.last_quote_alignment_count = alignment_count
            self.last_rejected_claim_count = len(diagnostics)
            self.last_rejected_claim_diagnostics = diagnostics
            return response
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
        self.last_rejected_claim_count = 0
        self.last_rejected_claim_diagnostics: tuple[str, ...] = ()
        if max_retries < 0:
            raise ValueError("Ollama max_retries must be non-negative")

    def generate_json(
        self,
        *,
        prompt: str,
        json_schema: Mapping[str, Any],
    ) -> object:
        self.last_rejected_claim_count = 0
        self.last_rejected_claim_diagnostics = ()
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
        best_partial: tuple[
            StructuredClaimResponse,
            int,
            tuple[str, ...],
        ] | None = None
        for attempt in range(1, attempts + 1):
            self.last_attempt_count = attempt
            content = ""
            payload = {
                "model": self.model.model,
                "messages": _structured_claim_messages(
                    prompt=prompt,
                    json_schema=json_schema,
                    repair_context=repair_context,
                    include_schema_text=False,
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
                    "num_ctx": 8192,
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
                    response = _validate_response(content)
                    return _validate_grounded_response(
                        response,
                        prompt=prompt,
                    )
                except Exception as schema_error:
                    try:
                        aligned, alignment_count = _align_quoted_claim_response(
                            content,
                            prompt=prompt,
                        )
                        self.last_quote_alignment_count = alignment_count
                        return _validate_grounded_response(
                            aligned,
                            prompt=prompt,
                        )
                    except Exception as alignment_error:
                        raise _combined_validation_error(
                            schema_error=schema_error,
                            alignment_error=alignment_error,
                        ) from alignment_error
            except Exception as exc:  # noqa: BLE001 - retry contract covers transport/output
                last_error = exc
                with suppress(Exception):
                    partial = _align_valid_quoted_claims(
                        content,
                        prompt=prompt,
                    )
                    if best_partial is None or len(partial[0].claims) > len(
                        best_partial[0].claims
                    ):
                        best_partial = partial
                repair_context = _structured_repair_context(
                    output=content,
                    error=exc,
                )
        if best_partial is not None:
            response, alignment_count, diagnostics = best_partial
            self.last_quote_alignment_count = alignment_count
            self.last_rejected_claim_count = len(diagnostics)
            self.last_rejected_claim_diagnostics = diagnostics
            return response
        raise ProviderOutputError(
            f"Ollama failed strict structured output after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}; "
            f"last_output={content[:500]!r}"
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
        _require_disjoint_argument_spans(output)

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
        max_retries: int = 3,
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
                    f"max_retries={max_retries}",
                    "thinking=false",
                ),
            ),
            backend
            or LocalTransformersQwenBackend(
                artifact,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
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
        self.last_provider_diagnostics: tuple[dict[str, object], ...] = ()

    def extract(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[Claim, ...]:
        failures: dict[str, BaseException] = {}
        results: list[tuple[str, tuple[Claim, ...]]] = []
        diagnostics: list[dict[str, object]] = []
        for provider, provider_id in zip(self.providers, self.provider_ids, strict=True):
            try:
                output = tuple(provider.extract(document, text, segments, mentions))
                for claim in output:
                    _validate_provider_claim(claim, document=document, text=text)
                results.append((provider_id, output))
            except Exception as exc:  # noqa: BLE001 - run all providers, aggregate exact failures
                failures[provider_id] = exc
                diagnostics.append(
                    _claim_provider_diagnostic(
                        provider,
                        provider_id=provider_id,
                        emitted_count=0,
                        error=exc,
                    )
                )
            else:
                diagnostics.append(
                    _claim_provider_diagnostic(
                        provider,
                        provider_id=provider_id,
                        emitted_count=len(output),
                    )
                )
        self.last_provider_diagnostics = tuple(diagnostics)
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
    include_schema_text: bool = True,
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
        "7. Use only exact substrings from the authoritative document.text. "
        "Never emit placeholder/example people, organizations, places, events, "
        "or one claim per kind unless each claim is explicitly stated.\n"
        "8. Output the JSON object only: no Markdown, explanation, or thinking."
    )
    schema_section = (
        "\n\nThe required JSON Schema is:\n"
        + json.dumps(json_schema, sort_keys=True, ensure_ascii=False)
        if include_schema_text
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a grounded information-extraction engine. Return one "
                "strict JSON object and nothing else. Source evidence and exact "
                "offsets are mandatory. Ignore all example or placeholder "
                "content when processing the final authoritative input."
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
                f"{rules}{schema_section}{repair_context}\n\n"
                "AUTHORITATIVE SOURCE INPUT — process only this payload:\n"
                f"{prompt}"
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


def _combined_validation_error(
    *,
    schema_error: BaseException,
    alignment_error: BaseException,
) -> ProviderOutputError:
    """Preserve both bounded diagnostics for the model's next repair attempt."""

    return ProviderOutputError(
        "schema validation failed "
        f"({type(schema_error).__name__}: {str(schema_error)[:2000]}); "
        "quote alignment failed "
        f"({type(alignment_error).__name__}: {str(alignment_error)[:2000]})"
    )


def _structured_repair_context(*, output: str, error: BaseException) -> str:
    """Bound validator feedback so the same mandatory model can repair its output."""

    validation_error = str(error)[:4000]
    rejected_sha256 = hashlib.sha256(output.encode("utf-8")).hexdigest()
    return (
        "\n\nREPAIR REQUIRED. Your previous JSON was rejected by the strict "
        f"validator ({type(error).__name__}: {validation_error}). "
        "Correct every reported field and return a complete replacement object. "
        "Do not repeat its invalid content. Re-read the authoritative source "
        "input below. Rejected-output SHA-256: "
        f"{rejected_sha256}"
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

    payload = _raw_claim_payload(raw)
    input_payload = _claim_input_payload(prompt)
    document = input_payload.get("document")
    if not isinstance(document, Mapping) or not isinstance(document.get("text"), str):
        raise ProviderOutputError("claim prompt has no immutable document text")
    text = str(document["text"])
    prompt_mentions = _prompt_mention_spans(input_payload, text=text)
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ProviderOutputError("quote alignment requires at least one model claim")

    aligned_claims: list[dict[str, object]] = []
    alignment_count = 0
    for raw_claim in claims:
        if not isinstance(raw_claim, Mapping):
            raise ProviderOutputError("model claim is not an object")
        claim = dict(raw_claim)
        repaired = _repair_exact_document_relation_field_shift(
            claim,
            input_payload=input_payload,
            text=text,
            prompt_mentions=prompt_mentions,
        )
        if repaired is not None:
            claim, repaired_count = repaired
            alignment_count += repaired_count

        evidence_spans = claim.get("evidence_spans")
        if not isinstance(evidence_spans, list) or not evidence_spans:
            raise ProviderOutputError(
                "quote alignment requires quoted evidence_spans"
            )
        aligned_evidence: list[dict[str, object]] = []
        for span in evidence_spans:
            aligned, changed = _align_one_quote(span, text=text)
            aligned_evidence.append(aligned)
            alignment_count += int(changed)
        claim["evidence_spans"] = aligned_evidence

        for field_name in (
            "subject_span",
            "modality_span",
            "attribution_span",
            "temporal_span",
            "quantity_span",
        ):
            span = claim.get(field_name)
            if span is None:
                continue
            claim[field_name], changed = _align_one_quote(
                span,
                text=text,
                containers=aligned_evidence,
            )
            alignment_count += int(changed)

        raw_predicate = claim.get("predicate")
        raw_predicate_span = claim.get("predicate_span")
        raw_object_span = claim.get("object_span")
        if not isinstance(raw_predicate, str):
            raise ProviderOutputError(
                "quote alignment requires a string predicate"
            )
        predicate, inferred_negation = _normalize_model_predicate(raw_predicate)
        claim["predicate"] = predicate
        displaced_predicate_span: dict[str, object] | None = None
        with suppress(ProviderOutputError):
            displaced_predicate_span, _ = _align_one_quote(
                raw_predicate_span,
                text=text,
                containers=aligned_evidence,
            )
        claim["predicate_span"], changed = _align_predicate_quote(
            raw_predicate_span,
            predicate=predicate,
            subject_span=claim.get("subject_span"),
            text=text,
            containers=aligned_evidence,
            kind=claim.get("kind"),
        )
        alignment_count += int(changed)

        kind = _claim_kind(claim.get("kind"))
        if kind in {ClaimKind.RELATION, ClaimKind.FACT, ClaimKind.INTENT}:
            claim["object_span"], changed = _align_object_argument_quote(
                raw_object_span,
                displaced_predicate_span=displaced_predicate_span,
                subject_span=claim.get("subject_span"),
                predicate_span=claim.get("predicate_span"),
                prompt_mentions=prompt_mentions,
                text=text,
                containers=aligned_evidence,
            )
            alignment_count += int(changed)
        elif raw_object_span is not None:
            claim["object_span"], changed = _align_one_quote(
                raw_object_span,
                text=text,
                containers=aligned_evidence,
            )
            alignment_count += int(changed)

        raw_negation_span = claim.get("negation_span")
        if inferred_negation is not None:
            claim["negated"] = True
            claim["negation_span"], changed = _align_negation_quote(
                raw_negation_span,
                inferred_negation=inferred_negation,
                text=text,
                containers=aligned_evidence,
            )
            alignment_count += int(changed)
        elif raw_negation_span is not None:
            claim["negation_span"], changed = _align_one_quote(
                raw_negation_span,
                text=text,
                containers=aligned_evidence,
            )
            alignment_count += int(changed)

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
                containers=aligned_evidence,
            )
            alignment_count += int(changed)
            aligned_roles.append(role)
        claim["event_roles"] = aligned_roles

        predicate_span = claim.get("predicate_span")
        if not isinstance(predicate, str) or not isinstance(predicate_span, Mapping):
            raise ProviderOutputError("quote alignment requires predicate and predicate_span")
        claim["predicate"] = _normalize_aligned_predicate(
            str(claim["predicate"]),
            cue=str(predicate_span["text"]),
        )
        # A missing model self-score is replaced by an explicit, conservative
        # quote-alignment calibration. It is not evidence confidence.
        if "confidence" not in claim:
            claim["confidence"] = 0.75
        aligned_claims.append(claim)

    response = _validate_response({"claims": aligned_claims})
    return response, alignment_count


def _raw_claim_payload(raw: object) -> dict[str, object]:
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ProviderOutputError(
                "quote alignment requires one JSON object"
            ) from exc
    else:
        decoded = raw
    if not isinstance(decoded, Mapping):
        raise ProviderOutputError("quote alignment requires a JSON object")
    payload = dict(decoded)
    if set(payload) != {"claims"}:
        raise ProviderOutputError(
            "quote alignment requires exactly one top-level claims key"
        )
    return payload


def _align_valid_quoted_claims(
    raw: object,
    *,
    prompt: str,
) -> tuple[StructuredClaimResponse, int, tuple[str, ...]]:
    """Quarantine invalid model claims after bounded whole-output retries.

    Each retained claim independently passes the same exact quote, schema,
    containment, and reviewed-predicate checks as a whole response. Invalid
    claims are never converted into empty success and their bounded diagnostics
    remain available on the mandatory backend.
    """

    payload = _raw_claim_payload(raw)
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ProviderOutputError(
            "claim-level quarantine requires at least one model claim"
        )
    retained: list[StructuredClaimOutput] = []
    diagnostics: list[str] = []
    alignment_count = 0
    for index, raw_claim in enumerate(raw_claims):
        try:
            response, claim_alignment_count = _align_quoted_claim_response(
                {"claims": [raw_claim]},
                prompt=prompt,
            )
            response = _validate_grounded_response(response, prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - preserve bounded model diagnostic
            diagnostics.append(
                f"claim[{index}] {type(exc).__name__}: {str(exc)[:500]}"
            )
            continue
        retained.extend(response.claims)
        alignment_count += claim_alignment_count
    if not retained:
        summary = "; ".join(diagnostics[:3])
        raise ProviderOutputError(
            "no model claim survived exact per-claim quarantine"
            + (f": {summary}" if summary else "")
        )
    return (
        StructuredClaimResponse(claims=tuple(retained)),
        alignment_count,
        tuple(diagnostics),
    )


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


def _claim_kind(raw: object) -> ClaimKind:
    try:
        return ClaimKind(str(raw))
    except ValueError as exc:
        raise ProviderOutputError(f"unsupported model claim kind: {raw!r}") from exc


def _prompt_mention_spans(
    input_payload: Mapping[str, Any],
    *,
    text: str,
) -> tuple[dict[str, object], ...]:
    """Return only immutable, source-exact mention anchors from the prompt."""

    raw_mentions = input_payload.get("mentions", [])
    if not isinstance(raw_mentions, list):
        raise ProviderOutputError("claim prompt mentions must be an array")
    mentions: dict[tuple[int, int, str], dict[str, object]] = {}
    for raw in raw_mentions:
        if not isinstance(raw, Mapping):
            raise ProviderOutputError("claim prompt mention is not an object")
        start = raw.get("start")
        end = raw.get("end")
        surface = raw.get("text")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(surface, str)
            or start < 0
            or start >= end
            or end > len(text)
            or text[start:end] != surface
        ):
            raise ProviderOutputError("claim prompt contains an inexact mention anchor")
        mentions.setdefault(
            (start, end, surface),
            {"start": start, "end": end, "text": surface},
        )
    return tuple(mentions.values())


def _prompt_segment_spans(
    input_payload: Mapping[str, Any],
    *,
    text: str,
) -> tuple[dict[str, object], ...]:
    """Return immutable, source-exact evidence anchors from the prompt."""

    raw_segments = input_payload.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ProviderOutputError("claim prompt segments must be an array")
    segments: dict[tuple[int, int, str], dict[str, object]] = {}
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise ProviderOutputError("claim prompt segment is not an object")
        start = raw.get("start")
        end = raw.get("end")
        surface = raw.get("text")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(surface, str)
            or start < 0
            or start >= end
            or end > len(text)
            or text[start:end] != surface
        ):
            raise ProviderOutputError("claim prompt contains an inexact segment anchor")
        segments.setdefault(
            (start, end, surface),
            {"start": start, "end": end, "text": surface},
        )
    return tuple(segments.values())


_FIELD_SHIFT_PADDING_RE = re.compile(r"^[\s,;:()\[\]{}]*$")


def _repair_exact_document_relation_field_shift(
    raw_claim: Mapping[str, object],
    *,
    input_payload: Mapping[str, Any],
    text: str,
    prompt_mentions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], int] | None:
    """Repair one uniquely grounded small-model field shift.

    A locked local decoder can occasionally put the exact evidence sentence in
    ``subject_span``, the exact subject mention's numeric window in
    ``predicate_span``, and the literal marker ``document`` in a full-document
    evidence span. This repair is deliberately atomic and narrow: every
    replacement must already be an immutable prompt segment/mention or a unique
    model quote inside that segment. Any ambiguity remains a hard failure.
    """

    raw_evidence = raw_claim.get("evidence_spans")
    if (
        not isinstance(raw_evidence, list)
        or len(raw_evidence) != 1
        or not isinstance(raw_evidence[0], Mapping)
        or raw_evidence[0].get("text") != "document"
    ):
        return None

    def fail(detail: str) -> ProviderOutputError:
        return ProviderOutputError(
            "full-document sentinel field shift is not uniquely grounded: "
            f"{detail}"
        )

    evidence_marker = raw_evidence[0]
    visible_document_end = len(text.rstrip("\r\n"))
    if (
        set(evidence_marker) != {"start", "end", "text"}
        or evidence_marker.get("start") != 0
        or evidence_marker.get("end")
        not in {len(text), visible_document_end}
    ):
        raise fail(
            "sentinel must select the immutable document, allowing only "
            "trailing line terminators to be excluded"
        )
    if _claim_kind(raw_claim.get("kind")) is not ClaimKind.RELATION:
        raise fail("repair is restricted to relation claims")

    raw_subject = raw_claim.get("subject_span")
    try:
        quoted_subject, _ = _align_one_quote(raw_subject, text=text)
    except ProviderOutputError as exc:
        raise fail("subject does not quote one exact source span") from exc
    matching_segments = [
        dict(segment)
        for segment in _prompt_segment_spans(input_payload, text=text)
        if _same_span(segment, quoted_subject)
    ]
    if len(matching_segments) != 1:
        raise fail("subject quote is not exactly one declared evidence segment")
    recovered_evidence = matching_segments[0]

    raw_predicate_span = raw_claim.get("predicate_span")
    if not isinstance(raw_predicate_span, Mapping):
        raise fail("predicate_span is not an object")
    window_start = raw_predicate_span.get("start")
    window_end = raw_predicate_span.get("end")
    if (
        not isinstance(window_start, int)
        or not isinstance(window_end, int)
        or window_start < 0
        or window_start >= window_end
        or window_end > len(text)
    ):
        raise fail("predicate_span numeric window is invalid")
    window_mentions: list[dict[str, object]] = []
    for mention in prompt_mentions:
        mention_start = mention.get("start")
        mention_end = mention.get("end")
        mention_text = mention.get("text")
        if not isinstance(mention_start, int) or not isinstance(mention_end, int):
            continue
        fully_contained = (
            window_start <= mention_start
            and mention_end <= window_end
            and _FIELD_SHIFT_PADDING_RE.fullmatch(
                text[window_start:mention_start]
            )
            is not None
            and _FIELD_SHIFT_PADDING_RE.fullmatch(
                text[mention_end:window_end]
            )
            is not None
        )
        prefix_overlap = (
            isinstance(mention_text, str)
            and window_start <= mention_start < window_end < mention_end
            and _FIELD_SHIFT_PADDING_RE.fullmatch(
                text[window_start:mention_start]
            )
            is not None
            and len(text[mention_start:window_end]) >= 4
            and len(text[mention_start:window_end]) * 2 >= len(mention_text)
            and mention_text.startswith(text[mention_start:window_end])
        )
        if fully_contained or prefix_overlap:
            window_mentions.append(dict(mention))
    subject_candidates = _maximal_non_nested_spans(window_mentions)
    if len(subject_candidates) != 1:
        raise fail("predicate_span window does not isolate one prompt mention")
    recovered_subject = subject_candidates[0]
    if not _span_contains(recovered_evidence, recovered_subject):
        raise fail("recovered subject lies outside the recovered evidence")

    raw_predicate = raw_claim.get("predicate")
    if not isinstance(raw_predicate, str):
        raise fail("predicate is not a string")
    predicate, inferred_negation = _normalize_model_predicate(raw_predicate)
    if inferred_negation is not None:
        raise fail("negated field shifts require explicit cue handling")
    try:
        recovered_predicate, _ = _align_predicate_quote(
            raw_predicate_span,
            predicate=predicate,
            subject_span=recovered_subject,
            text=text,
            containers=(recovered_evidence,),
            kind=raw_claim.get("kind"),
        )
        recovered_object, _ = _align_object_argument_quote(
            raw_claim.get("object_span"),
            displaced_predicate_span=None,
            subject_span=recovered_subject,
            predicate_span=recovered_predicate,
            prompt_mentions=prompt_mentions,
            text=text,
            containers=(recovered_evidence,),
        )
    except ProviderOutputError as exc:
        raise fail("predicate or object quote is not unique inside evidence") from exc
    if not any(
        _same_span(recovered_object, mention) for mention in prompt_mentions
    ):
        raise fail("recovered object is not an exact prompt mention")

    subject_end = recovered_subject.get("end")
    predicate_start = recovered_predicate.get("start")
    predicate_end = recovered_predicate.get("end")
    object_start = recovered_object.get("start")
    if not (
        isinstance(subject_end, int)
        and isinstance(predicate_start, int)
        and isinstance(predicate_end, int)
        and isinstance(object_start, int)
        and subject_end <= predicate_start
        and predicate_end <= object_start
    ):
        raise fail("recovered relation arguments overlap or have invalid order")
    if not all(
        _span_contains(recovered_evidence, span)
        for span in (
            recovered_subject,
            recovered_predicate,
            recovered_object,
        )
    ):
        raise fail("a recovered semantic span lies outside exact evidence")

    repaired = dict(raw_claim)
    repaired.update(
        {
            "evidence_spans": [recovered_evidence],
            "subject_span": recovered_subject,
            "predicate_span": recovered_predicate,
            "object_span": recovered_object,
        }
    )
    return repaired, 4


_NEGATED_PREDICATE_PREFIXES = (
    ("did", "not"),
    ("does", "not"),
    ("do", "not"),
    ("was", "not"),
    ("were", "not"),
    ("is", "not"),
    ("are", "not"),
    ("never",),
    ("not",),
)


def _normalize_model_predicate(predicate: str) -> tuple[str, str | None]:
    """Normalize syntax and separate an explicitly emitted negation cue."""

    tokens = _normalized_tokens(predicate)
    if not tokens:
        raise ProviderOutputError("model predicate normalizes to an empty value")
    inferred_negation: str | None = None
    for prefix in _NEGATED_PREDICATE_PREFIXES:
        if tuple(tokens[: len(prefix)]) != prefix:
            continue
        tokens = tokens[len(prefix) :]
        inferred_negation = prefix[-1]
        break
    if not tokens:
        raise ProviderOutputError("model predicate contains only a negation cue")
    return "_".join(tokens), inferred_negation


def _align_negation_quote(
    raw_span: object,
    *,
    inferred_negation: str,
    text: str,
    containers: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], bool]:
    """Anchor a model-emitted ``not``/``never`` cue inside exact evidence."""

    aligned: dict[str, object] | None = None
    changed = False
    with suppress(ProviderOutputError):
        aligned, changed = _align_one_quote(
            raw_span,
            text=text,
            containers=containers,
        )
    if aligned is not None and inferred_negation in _normalized_tokens(
        str(aligned["text"])
    ):
        return aligned, changed

    candidates: list[tuple[int, int]] = []
    pattern = re.compile(rf"\b{re.escape(inferred_negation)}\b", re.IGNORECASE)
    for container in containers:
        start = container.get("start")
        end = container.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        candidates.extend(
            (start + match.start(), start + match.end())
            for match in pattern.finditer(text[start:end])
        )
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ProviderOutputError(
            "model negation cue is absent or ambiguous inside exact evidence"
        )
    start, end = candidates[0]
    return {"start": start, "end": end, "text": text[start:end]}, True


def _align_object_argument_quote(
    raw_span: object,
    *,
    displaced_predicate_span: Mapping[str, object] | None,
    subject_span: object,
    predicate_span: object,
    prompt_mentions: Sequence[Mapping[str, object]],
    text: str,
    containers: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], bool]:
    """Recover a displaced object only from model quotes and mention anchors.

    Small local decoders sometimes shift a correct object quote into
    ``predicate_span`` while still quoting the complete exact evidence. The
    repair is deliberately bounded: prefer an explicitly quoted mention, then
    a single mention inside that displaced quote. It never chooses an argument
    merely because it is the closest or only remaining mention.
    """

    if not isinstance(subject_span, Mapping) or not isinstance(
        predicate_span,
        Mapping,
    ):
        raise ProviderOutputError(
            "object alignment requires grounded subject and predicate spans"
    )
    raw_aligned: dict[str, object] | None = None
    with suppress(ProviderOutputError):
        raw_aligned, _raw_changed = _align_one_quote(
            raw_span,
            text=text,
            containers=containers,
        )

    candidates = [
        dict(mention)
        for mention in prompt_mentions
        if not _same_span(mention, subject_span)
        and _span_is_in_any_container(mention, containers)
        and _span_is_on_object_side(
            mention,
            subject_span=subject_span,
            predicate_span=predicate_span,
        )
    ]
    if raw_aligned is not None and any(
        _same_span(raw_aligned, candidate) for candidate in candidates
    ):
        return raw_aligned, _span_changed(raw_span, raw_aligned)

    raw_contained_candidates = _maximal_non_nested_spans(
        [
            candidate
            for candidate in candidates
            if raw_aligned is not None and _span_contains(raw_aligned, candidate)
        ]
    )
    if len(raw_contained_candidates) == 1:
        selected = raw_contained_candidates[0]
        return selected, _span_changed(raw_span, selected)

    displaced_candidates = _maximal_non_nested_spans(
        [
            candidate
            for candidate in candidates
            if displaced_predicate_span is not None
            and _span_contains(displaced_predicate_span, candidate)
        ]
    )
    if len(displaced_candidates) == 1:
        selected = displaced_candidates[0]
        return selected, _span_changed(raw_span, selected)

    if (
        raw_aligned is not None
        and not _same_span(raw_aligned, subject_span)
        and _span_is_on_object_side(
            raw_aligned,
            subject_span=subject_span,
            predicate_span=predicate_span,
        )
        and not candidates
    ):
        return raw_aligned, _span_changed(raw_span, raw_aligned)

    raise ProviderOutputError(
        "model object quote is absent, displaced ambiguously, or conflicts "
        "with exact source mention anchors; "
        f"raw={raw_aligned!r}; displaced={displaced_predicate_span!r}; "
        f"eligible_mentions={candidates!r}"
    )


def _span_is_in_any_container(
    span: Mapping[str, object],
    containers: Sequence[Mapping[str, object]],
) -> bool:
    start = span.get("start")
    end = span.get("end")
    return isinstance(start, int) and isinstance(end, int) and any(
        _quote_is_contained(start, end, container) for container in containers
    )


def _span_is_on_object_side(
    span: Mapping[str, object],
    *,
    subject_span: Mapping[str, object],
    predicate_span: Mapping[str, object],
) -> bool:
    start = span.get("start")
    end = span.get("end")
    subject_start = subject_span.get("start")
    subject_end = subject_span.get("end")
    predicate_start = predicate_span.get("start")
    predicate_end = predicate_span.get("end")
    if not all(
        isinstance(value, int)
        for value in (
            start,
            end,
            subject_start,
            subject_end,
            predicate_start,
            predicate_end,
        )
    ):
        return False
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(subject_start, int)
    assert isinstance(subject_end, int)
    assert isinstance(predicate_start, int)
    assert isinstance(predicate_end, int)
    if subject_end <= predicate_start:
        return predicate_end <= start
    if predicate_end <= subject_start:
        return end <= predicate_start
    return False


def _span_contains(
    container: Mapping[str, object],
    item: Mapping[str, object],
) -> bool:
    container_start = container.get("start")
    container_end = container.get("end")
    item_start = item.get("start")
    item_end = item.get("end")
    return (
        isinstance(container_start, int)
        and isinstance(container_end, int)
        and isinstance(item_start, int)
        and isinstance(item_end, int)
        and container_start <= item_start
        and item_end <= container_end
    )


def _maximal_non_nested_spans(
    spans: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Discard only strict fragments contained by a more specific anchor."""

    return [
        span
        for span in spans
        if not any(
            span is not other
            and _span_contains(other, span)
            and not _same_span(other, span)
            for other in spans
        )
    ]


def _same_span(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    return (
        left.get("start"),
        left.get("end"),
        left.get("text"),
    ) == (
        right.get("start"),
        right.get("end"),
        right.get("text"),
    )


def _span_changed(raw: object, aligned: Mapping[str, object]) -> bool:
    return not isinstance(raw, Mapping) or not _same_span(raw, aligned)


def _align_one_quote(
    raw_span: object,
    *,
    text: str,
    containers: Sequence[Mapping[str, object]] = (),
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
    eligible_starts = [
        start
        for start in starts
        if not containers
        or any(
            _quote_is_contained(
                start,
                start + len(quote),
                container,
            )
            for container in containers
        )
    ]
    if not eligible_starts:
        raise ProviderOutputError(
            f"model quote is outside its exact evidence: {quote!r}"
        )
    proposed_start = raw_span.get("start")
    proposed_end = raw_span.get("end")
    if len(eligible_starts) == 1:
        start = eligible_starts[0]
    elif isinstance(proposed_start, int) and proposed_start in eligible_starts:
        start = proposed_start
    else:
        raise ProviderOutputError(
            "model quote is ambiguous without an exact occurrence or "
            f"unique evidence anchor: {quote!r}"
        )
    end = start + len(quote)
    aligned = {"start": start, "end": end, "text": quote}
    changed = proposed_start != start or proposed_end != end
    return aligned, changed


def _quote_is_contained(
    start: int,
    end: int,
    container: Mapping[str, object],
) -> bool:
    container_start = container.get("start")
    container_end = container.get("end")
    return (
        isinstance(container_start, int)
        and isinstance(container_end, int)
        and container_start <= start
        and end <= container_end
    )


def _align_predicate_quote(
    raw_span: object,
    *,
    predicate: str,
    subject_span: object,
    text: str,
    containers: Sequence[Mapping[str, object]],
    kind: object,
) -> tuple[dict[str, object], bool]:
    """Recover one predicate cue through exact or reviewed-alias grounding."""

    claim_kind = _claim_kind(kind)
    directly_aligned: dict[str, object] | None = None
    try:
        directly_aligned, directly_changed = _align_one_quote(
            raw_span,
            text=text,
            containers=containers,
        )
    except ProviderOutputError:
        pass
    else:
        if _predicate_quote_matches(
            str(directly_aligned["text"]),
            predicate=predicate,
            kind=claim_kind,
        ):
            return directly_aligned, directly_changed

    if not isinstance(subject_span, Mapping):
        raise ProviderOutputError(
            "predicate alignment requires a grounded subject span"
        )
    token_sets = _predicate_grounding_token_sets(predicate, kind=claim_kind)
    candidates: list[tuple[int, int]] = []
    # Prefer the longest reviewed surface forms. This avoids treating both
    # ``is a`` and its shorter alias ``is`` as two competing cues.
    for width in sorted({len(tokens) for tokens in token_sets}, reverse=True):
        width_candidates: list[tuple[int, int]] = []
        for token_set in (tokens for tokens in token_sets if len(tokens) == width):
            pattern = r"\b" + r"\s+".join(
                re.escape(token) for token in token_set
            ) + r"\b"
            for container in containers:
                container_start = container.get("start")
                container_end = container.get("end")
                if not isinstance(container_start, int) or not isinstance(
                    container_end,
                    int,
                ):
                    continue
                for match in re.finditer(
                    pattern,
                    text[container_start:container_end],
                    flags=re.IGNORECASE,
                ):
                    width_candidates.append(
                        (
                            container_start + match.start(),
                            container_start + match.end(),
                        )
                    )
        if width_candidates:
            candidates = width_candidates
            break
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ProviderOutputError(
            "model predicate cue is absent or ambiguous within exact evidence "
            "under the reviewed predicate vocabulary"
        )
    start, end = candidates[0]
    proposed_start = (
        raw_span.get("start") if isinstance(raw_span, Mapping) else None
    )
    proposed_end = (
        raw_span.get("end") if isinstance(raw_span, Mapping) else None
    )
    proposed_text = (
        raw_span.get("text") if isinstance(raw_span, Mapping) else None
    )
    aligned = {"start": start, "end": end, "text": text[start:end]}
    changed = (
        proposed_start != start
        or proposed_end != end
        or proposed_text != text[start:end]
    )
    return aligned, changed


@lru_cache(maxsize=1)
def _default_governed_predicate_schema() -> Any:
    """Load the immutable reviewed predicate vocabulary once per process."""

    from kg_construction.infrastructure.ontology.unstructured_governance import (
        GovernedUnstructuredSchema,
    )

    return GovernedUnstructuredSchema.from_package_resource()


def _predicate_grounding_token_sets(
    predicate: str,
    *,
    kind: ClaimKind,
) -> tuple[tuple[str, ...], ...]:
    """Return lexical plus reviewed source forms for one model predicate."""

    predicate_tokens = tuple(_normalized_tokens(predicate))
    token_sets: set[tuple[str, ...]] = {predicate_tokens}
    lexical_tokens = list(predicate_tokens)
    while (
        len(lexical_tokens) > 1
        and lexical_tokens[-1] in _SEMANTIC_PREPOSITIONS
    ):
        lexical_tokens.pop()
    if lexical_tokens:
        token_sets.add(tuple(lexical_tokens))

    schema = _default_governed_predicate_schema()
    canonical = schema.aligned_predicate(predicate, kind=kind)
    if canonical is not None:
        definition = next(
            item for item in schema.predicates if item.name == canonical
        )
        for term in (definition.name, *definition.aliases):
            tokens = tuple(_normalized_tokens(term))
            if tokens:
                token_sets.add(tokens)
    return tuple(
        sorted(
            token_sets,
            key=lambda tokens: (-len(tokens), tokens),
        )
    )


def _predicate_quote_matches(
    cue: str,
    *,
    predicate: str,
    kind: ClaimKind,
) -> bool:
    cue_tokens = _normalized_tokens(cue)
    return any(
        bool(_token_sequence_positions(cue_tokens, token_set))
        for token_set in _predicate_grounding_token_sets(
            predicate,
            kind=kind,
        )
    )


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
    """Normalize a relation label only from exact cue/argument text.

    Small constrained models occasionally quote an exact but over-wide
    ``predicate_span`` (for example ``"at Bletchley Park"``).  The immutable
    subject/object anchors still define an exact argument bridge
    (``"worked at"``).  We accept that bridge only when the model's lexical
    predicate occurs exactly once; otherwise the claim fails closed.
    """

    normalized = _PREDICATE_TOKEN_RE.sub(
        "_",
        output.predicate.casefold(),
    ).strip("_")
    if not normalized:
        raise ProviderOutputError("grounded predicate normalizes to an empty value")

    predicate_tokens = normalized.split("_")
    lexical_tokens = list(predicate_tokens)
    while (
        len(lexical_tokens) > 1
        and lexical_tokens[-1] in _SEMANTIC_PREPOSITIONS
    ):
        lexical_tokens.pop()
    grounding_token_sets = _predicate_grounding_token_sets(
        normalized,
        kind=output.kind,
    )
    cue_tokens = _normalized_tokens(output.predicate_span.text)
    cue_matches = _preferred_token_sequence_matches(
        cue_tokens,
        grounding_token_sets,
    )
    grounded = len(cue_matches) == 1
    if len(cue_matches) > 1:
        raise ProviderOutputError(
            "model predicate is ambiguous within its exact cue"
        )

    cue = _PREDICATE_TOKEN_RE.sub(
        "_",
        output.predicate_span.text.casefold(),
    ).strip("_")
    cue_tail = cue.rsplit("_", maxsplit=1)[-1]
    if cue_tail in _SEMANTIC_PREPOSITIONS and not normalized.endswith(
        f"_{cue_tail}"
    ):
        normalized = f"{normalized}_{cue_tail}"
    if output.object_span is not None:
        bridge_start = min(
            output.subject_span.end,
            output.object_span.end,
        )
        bridge_end = max(
            output.subject_span.start,
            output.object_span.start,
        )
        if bridge_start <= bridge_end:
            bridge_tokens = _normalized_tokens(text[bridge_start:bridge_end])
            bridge_matches = _preferred_token_sequence_matches(
                bridge_tokens,
                grounding_token_sets,
            )
            if len(bridge_matches) == 1:
                grounded = True
                trailing_index = bridge_matches[0][1]
                if (
                    trailing_index < len(bridge_tokens)
                    and bridge_tokens[trailing_index]
                    in _SEMANTIC_PREPOSITIONS
                    and not normalized.endswith(
                        f"_{bridge_tokens[trailing_index]}"
                    )
                ):
                    normalized = (
                        f"{normalized}_{bridge_tokens[trailing_index]}"
                    )
            elif not grounded and len(bridge_matches) > 1:
                raise ProviderOutputError(
                    "model predicate is ambiguous within the exact argument bridge"
                )
    if not grounded:
        raise ProviderOutputError(
            "model predicate is not lexically grounded by its exact cue or "
            "argument bridge"
        )
    return normalized


def _normalized_tokens(value: str) -> list[str]:
    normalized = _PREDICATE_TOKEN_RE.sub("_", value.casefold()).strip("_")
    return normalized.split("_") if normalized else []


def _token_sequence_positions(
    tokens: Sequence[str],
    sequence: Sequence[str],
) -> list[int]:
    if not sequence or len(sequence) > len(tokens):
        return []
    width = len(sequence)
    return [
        index
        for index in range(len(tokens) - width + 1)
        if list(tokens[index : index + width]) == list(sequence)
    ]


def _preferred_token_sequence_matches(
    tokens: Sequence[str],
    token_sets: Sequence[Sequence[str]],
) -> list[tuple[int, int]]:
    """Return matches from the longest exact/reviewed surface-form tier."""

    widths = sorted({len(sequence) for sequence in token_sets if sequence}, reverse=True)
    for width in widths:
        matches = {
            (start, start + width)
            for sequence in token_sets
            if len(sequence) == width
            for start in _token_sequence_positions(tokens, sequence)
        }
        if matches:
            return sorted(matches)
    return []


def _validate_grounded_response(
    response: StructuredClaimResponse,
    *,
    prompt: str,
) -> StructuredClaimResponse:
    """Validate source equality and semantic grounding inside retry scope."""

    if not response.claims:
        return response
    input_payload = _claim_input_payload(prompt)
    document = input_payload.get("document")
    if not isinstance(document, Mapping) or not isinstance(
        document.get("text"),
        str,
    ):
        raise ProviderOutputError(
            "claim prompt has no immutable document text"
        )
    text = str(document["text"])
    for output in response.claims:
        grounded_spans = [
            output.subject_span,
            output.predicate_span,
            *output.evidence_spans,
            *(role.value_span for role in output.event_roles),
        ]
        grounded_spans.extend(
            span
            for span in (
                output.object_span,
                output.negation_span,
                output.modality_span,
                output.attribution_span,
                output.temporal_span,
                output.quantity_span,
            )
            if span is not None
        )
        for span in grounded_spans:
            _require_exact_span(
                span,
                text=text,
                provider="structured_generation_backend",
            )
        semantic_spans = (
            output.subject_span,
            output.predicate_span,
            output.object_span,
            output.negation_span,
            output.modality_span,
            output.attribution_span,
            output.temporal_span,
            output.quantity_span,
            *(role.value_span for role in output.event_roles),
        )
        for semantic_span in semantic_spans:
            if semantic_span is not None and not any(
                _contains(evidence, semantic_span)
                for evidence in output.evidence_spans
            ):
                raise ProviderOutputError(
                    "structured model returned a semantic span outside claim "
                    "evidence"
                )
        _require_disjoint_argument_spans(output)
        _grounded_predicate(output, text=text)
    return response


def _require_disjoint_argument_spans(output: StructuredClaimOutput) -> None:
    """Reject a sentence-sized subject that overlaps its predicate or object.

    A predicate cue may intentionally be over-wide and overlap the object; the
    reviewed exact-argument bridge validates that case separately.
    """

    if (
        output.kind
        not in {ClaimKind.RELATION, ClaimKind.FACT, ClaimKind.INTENT}
        or output.object_span is None
    ):
        return
    for right_name, right in (
        ("predicate_span", output.predicate_span),
        ("object_span", output.object_span),
    ):
        if max(output.subject_span.start, right.start) < min(
            output.subject_span.end,
            right.end,
        ):
            raise ProviderOutputError(
                "structured claim argument spans overlap: "
                f"subject_span and {right_name}"
            )


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
    if len(matches) != 1:
        return None
    return matches[0]


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


def _claim_provider_diagnostic(
    provider: object,
    *,
    provider_id: str,
    emitted_count: int,
    error: BaseException | None = None,
) -> dict[str, object]:
    backend = getattr(provider, "backend", None)
    rejected_diagnostics = tuple(
        str(item)
        for item in getattr(
            backend,
            "last_rejected_claim_diagnostics",
            (),
        )
    )
    provider_rejections = tuple(
        dict(item)
        for item in getattr(provider, "last_rejections", ())
        if isinstance(item, Mapping)
    )
    diagnostic: dict[str, object] = {
        "provider_id": provider_id,
        "executed": True,
        "emitted_claim_count": emitted_count,
        "rejected_claim_count": len(rejected_diagnostics)
        + len(provider_rejections),
        "rejected_claim_diagnostics": list(rejected_diagnostics),
        "rejected_inferences": list(provider_rejections),
    }
    if error is not None:
        diagnostic["failed"] = True
        diagnostic["error_type"] = type(error).__name__
        diagnostic["error"] = str(error)[:1000]
    else:
        diagnostic["failed"] = False
    return diagnostic


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
