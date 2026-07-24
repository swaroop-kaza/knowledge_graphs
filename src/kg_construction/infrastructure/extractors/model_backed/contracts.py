"""Offline artifact and provider contracts for mandatory model-backed extraction.

The online provisioning step resolves model repositories to immutable revisions.
Runtime code receives only concrete snapshot directories and never accepts a Hub
repository identifier as a load target.  Ollama is treated similarly: the locked
tag and digest must already be present before inference is attempted.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProviderError(RuntimeError):
    """Base exception for mandatory model provider failures."""


class OfflineArtifactError(ModelProviderError):
    """Raised when a locked local artifact is absent or inconsistent."""


class ProviderOutputError(ModelProviderError):
    """Raised when a provider returns invalid or ungrounded output."""


class MandatoryProviderError(ModelProviderError):
    """Raised after every mandatory provider ran and at least one failed."""

    def __init__(self, stage: str, failures: Mapping[str, BaseException]) -> None:
        self.stage = stage
        self.failures = dict(failures)
        detail = "; ".join(
            f"{name}: {type(error).__name__}: {error}"
            for name, error in sorted(self.failures.items())
        )
        super().__init__(f"mandatory {stage} providers failed: {detail}")


class HydratedArtifact(BaseModel):
    """One immutable Hugging Face snapshot selected for a platform profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    task: str = Field(min_length=1)
    device: str = Field(min_length=1)
    local_path: Path

    @model_validator(mode="after")
    def _local_directory_exists(self) -> HydratedArtifact:
        if not self.local_path.is_absolute():
            raise ValueError(f"{self.name} local_path must be absolute")
        if not self.local_path.is_dir():
            raise ValueError(f"{self.name} is not hydrated at local directory {self.local_path}")
        return self

    @property
    def model_version(self) -> str:
        return f"{self.repo_id}@{self.revision}"


class HydratedServiceModel(BaseModel):
    """One locked, already-present model hosted by a local service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    service: str = Field(min_length=1)
    model: str = Field(min_length=1)
    digest: str = Field(min_length=1)


class HydratedModelSet(BaseModel):
    """All local model artifacts required by one exact platform profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str = Field(min_length=1)
    artifacts: tuple[HydratedArtifact, ...]
    service_models: tuple[HydratedServiceModel, ...]

    @model_validator(mode="after")
    def _unique_names(self) -> HydratedModelSet:
        artifact_names = [item.name for item in self.artifacts]
        service_names = [item.name for item in self.service_models]
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("hydrated artifact names must be unique")
        if len(service_names) != len(set(service_names)):
            raise ValueError("hydrated service model names must be unique")
        return self

    def artifact(self, name: str) -> HydratedArtifact:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise OfflineArtifactError(
            f"mandatory model artifact {name!r} is absent for {self.profile}"
        )

    def service_model(self, name: str) -> HydratedServiceModel:
        for model in self.service_models:
            if model.name == name:
                return model
        raise OfflineArtifactError(f"mandatory service model {name!r} is absent for {self.profile}")

    @classmethod
    def from_lock(
        cls,
        lock_path: Path,
        *,
        profile: str,
        cache_root: Path | None = None,
    ) -> HydratedModelSet:
        """Resolve a committed lock to local snapshot paths without network access."""

        if not lock_path.is_file():
            raise OfflineArtifactError(f"model lock does not exist: {lock_path}")
        payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise OfflineArtifactError(f"model lock must be a mapping: {lock_path}")

        resolved_cache = cache_root or _huggingface_cache_root()
        artifacts: list[HydratedArtifact] = []
        for raw in payload.get("models", []):
            if not isinstance(raw, dict) or profile not in raw.get("profiles", []):
                continue
            repo_id = str(raw.get("repo_id", ""))
            revision = str(raw.get("revision", ""))
            snapshot = _snapshot_path(resolved_cache, repo_id, revision)
            artifacts.append(
                HydratedArtifact(
                    name=str(raw.get("name", "")),
                    repo_id=repo_id,
                    revision=revision,
                    backend=str(raw.get("backend", "")),
                    task=str(raw.get("task", "")),
                    device=str(raw.get("device", "")),
                    local_path=snapshot,
                )
            )

        service_models: list[HydratedServiceModel] = []
        for raw in payload.get("service_models", []):
            if not isinstance(raw, dict) or profile not in raw.get("profiles", []):
                continue
            service_models.append(
                HydratedServiceModel(
                    name=str(raw.get("name", "")),
                    service=str(raw.get("service", "")),
                    model=str(raw.get("model", "")),
                    digest=str(raw.get("digest", "")),
                )
            )
        return cls(
            profile=profile,
            artifacts=tuple(artifacts),
            service_models=tuple(service_models),
        )


class ProviderAgreement(BaseModel):
    """Agreement evidence retained for one conservative ensemble item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_key: str = Field(min_length=1)
    agreeing_providers: tuple[str, ...] = Field(min_length=1)
    total_mandatory_providers: int = Field(gt=0)
    ratio: float = Field(ge=0.0, le=1.0)
    participating_providers: tuple[str, ...] = ()
    selected_type: str | None = None
    provider_types: dict[str, str] = Field(default_factory=dict)
    provider_confidences: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ratio_matches_counts(self) -> ProviderAgreement:
        expected = len(self.agreeing_providers) / self.total_mandatory_providers
        if abs(expected - self.ratio) > 1e-12:
            raise ValueError("agreement ratio must match provider counts")
        participants = self.participating_providers or self.agreeing_providers
        if len(participants) != len(set(participants)):
            raise ValueError("participating provider IDs must be unique")
        if not set(self.agreeing_providers) <= set(participants):
            raise ValueError("agreeing providers must have participated")
        if self.selected_type is None:
            if self.provider_types or self.provider_confidences:
                raise ValueError(
                    "provider type votes require a selected mention type"
                )
            return self
        if not self.selected_type.strip():
            raise ValueError("selected mention type must not be blank")
        if set(self.provider_types) != set(participants):
            raise ValueError("provider type votes must cover every participant")
        if set(self.provider_confidences) != set(participants):
            raise ValueError(
                "provider type confidences must cover every participant"
            )
        selected = self.selected_type.casefold()
        expected_agreeing = {
            provider_id
            for provider_id, entity_type in self.provider_types.items()
            if entity_type.casefold() == selected
        }
        if expected_agreeing != set(self.agreeing_providers):
            raise ValueError(
                "agreeing providers must be exactly the selected-type voters"
            )
        if any(
            confidence < 0.0 or confidence > 1.0
            for confidence in self.provider_confidences.values()
        ):
            raise ValueError("provider type confidences must be in [0, 1]")
        return self


class ExactSpanOutput(BaseModel):
    """An exact end-exclusive span required in structured model output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_width(self) -> ExactSpanOutput:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("span width must equal the text's Unicode code-point length")
        return self

    def is_exact_in(self, text: str) -> bool:
        return self.end <= len(text) and text[self.start : self.end] == self.text


class MentionInference(BaseModel):
    """Backend-neutral token-classification output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _valid_width(self) -> MentionInference:
        if self.end <= self.start:
            raise ValueError("mention end must be greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("mention offsets must have the same width as mention text")
        return self


class RelationInference(BaseModel):
    """Backend-neutral relation output with exact argument spans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_span: ExactSpanOutput
    predicate: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    object_span: ExactSpanOutput
    confidence: float = Field(ge=0.0, le=1.0)


class MentionInferenceBackend(Protocol):
    """Injectable client used by model-specific mention detectors."""

    def infer(self, text: str) -> tuple[MentionInference, ...]: ...


class RelationInferenceBackend(Protocol):
    """Injectable client used by model-specific relation extractors."""

    def infer(self, text: str) -> tuple[RelationInference, ...]: ...


class StructuredGenerationBackend(Protocol):
    """Injectable client used by structured claim extractors."""

    def generate_json(
        self,
        *,
        prompt: str,
        json_schema: Mapping[str, Any],
    ) -> object: ...


def require_local_directory(path: Path, *, model_name: str) -> None:
    """Hard-fail instead of allowing a model library to interpret a Hub ID."""

    if not path.is_absolute() or not path.is_dir():
        raise OfflineArtifactError(
            f"{model_name} must be an already-hydrated absolute local directory: {path}"
        )


def enforce_offline_model_environment() -> None:
    """Force supported model libraries into offline mode before importing them."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _huggingface_cache_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        explicit = os.environ.get("HF_HUB_CACHE")
        if explicit:
            return Path(explicit)
        return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_path(cache_root: Path, repo_id: str, revision: str) -> Path:
    repository = "models--" + repo_id.replace("/", "--")
    return (cache_root / repository / "snapshots" / revision).resolve()


__all__ = [
    "ExactSpanOutput",
    "HydratedArtifact",
    "HydratedModelSet",
    "HydratedServiceModel",
    "MandatoryProviderError",
    "MentionInference",
    "MentionInferenceBackend",
    "ModelProviderError",
    "OfflineArtifactError",
    "ProviderAgreement",
    "ProviderOutputError",
    "RelationInference",
    "RelationInferenceBackend",
    "StructuredGenerationBackend",
    "enforce_offline_model_environment",
    "require_local_directory",
]
