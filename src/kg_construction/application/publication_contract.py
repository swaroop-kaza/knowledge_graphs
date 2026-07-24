"""Trusted, content-addressed contracts for release-eligible graph construction.

An evaluation suite can seal its own contents, which is useful for research but
does not establish that independent reviewers approved the fixture or its
thresholds.  The public construction boundary therefore requires a second,
repository-owned contract.  That contract binds the curated suite, source
snapshot, semantic configuration, model lock, governed schema, held-out claim
set, graph-builder contract, and non-weakenable policy.
"""

from __future__ import annotations

import hashlib
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.application.config import DataToKGConfig
from kg_construction.application.source_catalog import SourceCatalog
from kg_construction.domain.models import SourceRecord, canonical_json
from kg_construction.infrastructure.evaluation.graph_bound_suite import (
    VersionedGraphEvaluationSuite,
)
from kg_construction.infrastructure.production.pii import PIIRedactionPolicy

PUBLICATION_CONTRACT_VERSION = "kg-publication-contract-registry-v1"
GRAPH_BUILDER_CONTRACT_VERSION: Literal[
    "property-rdf-evidence-builders-v1"
] = "property-rdf-evidence-builders-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicationMinimumPolicy(_ContractModel):
    """Repository-owned floors/ceilings that a fixture cannot relax."""

    policy_id: str = Field(min_length=1)
    policy_sha256: str
    quality_thresholds_version: Literal["kg-quality-v3"]
    quality_thresholds_sha256: str
    suite_minimums: dict[str, float] = Field(default_factory=dict)
    suite_maximums: dict[str, float] = Field(default_factory=dict)
    held_out_minimums: dict[str, float] = Field(default_factory=dict)
    held_out_maximums: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _digest_is_exact(self) -> PublicationMinimumPolicy:
        _require_sha256(self.policy_sha256, field="policy_sha256")
        _require_sha256(
            self.quality_thresholds_sha256,
            field="quality_thresholds_sha256",
        )
        actual_quality_thresholds = publication_quality_thresholds_sha256()
        if actual_quality_thresholds != self.quality_thresholds_sha256:
            raise ValueError(
                "quality_thresholds_sha256 does not match the executable "
                "VersionedQualityThresholds contract"
            )
        actual = publication_policy_sha256(self)
        if actual != self.policy_sha256:
            raise ValueError(
                "publication policy_sha256 mismatch: "
                f"declared={self.policy_sha256}, actual={actual}"
            )
        if not self.suite_minimums or not self.held_out_minimums:
            raise ValueError("publication policy requires suite and held-out minima")
        return self


class TrustedReleaseContract(_ContractModel):
    """One reviewed source/configuration/evaluation release tuple."""

    contract_id: str = Field(min_length=1)
    construction_profile: Literal["structured", "unstructured", "mixed"]
    fixture_id: str = Field(min_length=1)
    evaluation_fixture_sha256: str
    evaluation_file_sha256: str
    source_snapshot_sha256: tuple[str, ...] = Field(min_length=1)
    runtime_source_snapshot_sha256: tuple[str, ...] = Field(min_length=1)
    allowed_config_sha256: tuple[str, ...] = Field(min_length=1)
    allowed_source_catalog_sha256: tuple[str, ...] = Field(min_length=1)
    model_lock_sha256: str
    governed_schema_sha256: str | None = None
    held_out_gold_sha256: str | None = None
    required_gold_paths: tuple[str, ...] = Field(min_length=1)
    graph_builder_contract_version: Literal[
        "property-rdf-evidence-builders-v1"
    ] = GRAPH_BUILDER_CONTRACT_VERSION

    @model_validator(mode="after")
    def _hashes_are_exact(self) -> TrustedReleaseContract:
        for field_name in (
            "evaluation_fixture_sha256",
            "evaluation_file_sha256",
            "model_lock_sha256",
        ):
            _require_sha256(str(getattr(self, field_name)), field=field_name)
        for field_name, values in (
            ("source_snapshot_sha256", self.source_snapshot_sha256),
            (
                "runtime_source_snapshot_sha256",
                self.runtime_source_snapshot_sha256,
            ),
            ("allowed_config_sha256", self.allowed_config_sha256),
            (
                "allowed_source_catalog_sha256",
                self.allowed_source_catalog_sha256,
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} entries must be unique")
            for value in values:
                _require_sha256(value, field=field_name)
        for field_name in ("governed_schema_sha256", "held_out_gold_sha256"):
            value = getattr(self, field_name)
            if value is not None:
                _require_sha256(value, field=field_name)
        if self.construction_profile in {"unstructured", "mixed"} and (
            self.governed_schema_sha256 is None
            or self.held_out_gold_sha256 is None
        ):
            raise ValueError(
                "unstructured/mixed contracts require governed schema and "
                "held-out gold digests"
            )
        return self


class TrustedPublicationBinding(_ContractModel):
    """Auditable proof that one run matched a repository-owned release tuple."""

    registry_version: Literal["kg-publication-contract-registry-v1"]
    registry_sha256: str
    policy_id: str
    policy_sha256: str
    quality_thresholds_version: Literal["kg-quality-v3"]
    quality_thresholds_sha256: str
    contract_id: str
    construction_profile: Literal["structured", "unstructured", "mixed"]
    fixture_id: str
    evaluation_fixture_sha256: str
    evaluation_file_sha256: str
    source_snapshot_sha256: tuple[str, ...]
    runtime_source_snapshot_sha256: tuple[str, ...]
    config_sha256: str
    source_catalog_sha256: str
    model_lock_sha256: str
    governed_schema_sha256: str | None = None
    held_out_gold_sha256: str | None = None
    graph_builder_contract_version: Literal[
        "property-rdf-evidence-builders-v1"
    ]
    release_eligible: Literal[True] = True
    research_only: Literal[False] = False


class PublicationContractAssessment(_ContractModel):
    """Trust classification for a suite/run tuple."""

    release_eligible: bool
    research_only: bool
    reasons: tuple[str, ...]
    binding: TrustedPublicationBinding | None = None

    @model_validator(mode="after")
    def _classification_is_consistent(self) -> PublicationContractAssessment:
        if self.release_eligible == self.research_only:
            raise ValueError(
                "publication assessment must be exactly release-eligible or "
                "research-only"
            )
        if self.release_eligible != (self.binding is not None):
            raise ValueError("release eligibility requires a trusted binding")
        if self.release_eligible and self.reasons:
            raise ValueError("an eligible publication assessment has no rejection reasons")
        if self.research_only and not self.reasons:
            raise ValueError("a research-only assessment requires rejection reasons")
        return self


class UntrustedPublicationContractError(ValueError):
    """Raised when a self-sealed/custom fixture is used at the release boundary."""

    def __init__(self, assessment: PublicationContractAssessment) -> None:
        self.assessment = assessment
        detail = "; ".join(assessment.reasons)
        super().__init__(
            "evaluation suite/run tuple is research-only and cannot publish: "
            f"{detail}"
        )


class PublicationContractRegistry(_ContractModel):
    """Immutable, package-owned allowlist for public graph construction."""

    version: Literal["kg-publication-contract-registry-v1"]
    registry_sha256: str
    policy: PublicationMinimumPolicy
    contracts: tuple[TrustedReleaseContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _registry_is_exact(self) -> PublicationContractRegistry:
        _require_sha256(self.registry_sha256, field="registry_sha256")
        actual = publication_registry_sha256(self)
        if actual != self.registry_sha256:
            raise ValueError(
                "publication registry_sha256 mismatch: "
                f"declared={self.registry_sha256}, actual={actual}"
            )
        identifiers = [item.contract_id for item in self.contracts]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("publication contract IDs must be unique")
        tuples = [
            (
                item.fixture_id,
                item.evaluation_fixture_sha256,
                item.evaluation_file_sha256,
            )
            for item in self.contracts
        ]
        if len(tuples) != len(set(tuples)):
            raise ValueError("publication fixture registrations must be unique")
        return self

    @classmethod
    def from_package_resource(cls) -> PublicationContractRegistry:
        """Load the wheel-safe registry; callers cannot select another file."""

        raw = files("kg_construction").joinpath(
            "resources/publication_contracts_v1.yaml"
        ).read_bytes()
        payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise ValueError("publication contract registry must be a mapping")
        return cls.model_validate(payload)

    def assess(
        self,
        *,
        suite: VersionedGraphEvaluationSuite,
        suite_path: str | Path,
        sources: list[SourceRecord],
        config: DataToKGConfig,
        source_catalog: SourceCatalog | None = None,
    ) -> PublicationContractAssessment:
        """Classify a run without granting publication on self-assertion."""

        reasons: list[str] = []
        try:
            self._enforce_suite_policy(suite)
        except ValueError as exc:
            reasons.append(str(exc))

        suite_file = Path(suite_path)
        if not suite_file.is_file():
            reasons.append(f"evaluation suite file does not exist: {suite_file}")
            evaluation_file_hash = ""
        else:
            evaluation_file_hash = _sha256_bytes(suite_file.read_bytes())

        source_hashes = tuple(sorted(source_snapshot_sha256(item) for item in sources))
        pii_policy = PIIRedactionPolicy(
            **config.production.pii_redaction.model_dump(mode="python")
        )
        runtime_source_hashes = tuple(
            sorted(
                source_snapshot_sha256(pii_policy.redact_source_record(item))
                for item in sources
            )
        )
        config_hash = publication_config_sha256(config)
        source_catalog_hash = publication_source_catalog_sha256(
            config,
            source_catalog=source_catalog,
        )
        model_lock_hash = _artifact_sha256(
            Path(config.unstructured.models_lock_path).expanduser(),
            label="model lock",
            reasons=reasons,
        )
        governed_schema_hash: str | None = None
        held_out_hash: str | None = None
        if suite.construction_profile in {"unstructured", "mixed"}:
            governed_schema_hash = _configured_or_package_sha256(
                configured_path=config.unstructured.governed_schema_path,
                package_relative_path="resources/governed_unstructured_ontology.yaml",
                label="governed schema",
                reasons=reasons,
            )
            held_out_hash = _configured_or_package_sha256(
                configured_path=config.unstructured.held_out_gold_path,
                package_relative_path="resources/unstructured_claims_gold_v1.json",
                label="held-out claim gold",
                reasons=reasons,
            )
            if held_out_hash is not None:
                self._enforce_held_out_policy(
                    config=config,
                    reasons=reasons,
                )

        candidates = [
            item
            for item in self.contracts
            if item.fixture_id == suite.fixture_id
            and item.construction_profile == suite.construction_profile
        ]
        if not candidates:
            reasons.append(
                "fixture/profile is absent from the repository-owned release allowlist"
            )
            return _research_assessment(reasons)

        contract = candidates[0]
        comparisons: tuple[tuple[str, object, object], ...] = (
            (
                "evaluation fixture SHA-256",
                suite.fixture_sha256,
                contract.evaluation_fixture_sha256,
            ),
            (
                "evaluation file SHA-256",
                evaluation_file_hash,
                contract.evaluation_file_sha256,
            ),
            (
                "source snapshot SHA-256 tuple",
                source_hashes,
                contract.source_snapshot_sha256,
            ),
            (
                "runtime source snapshot SHA-256 tuple",
                runtime_source_hashes,
                contract.runtime_source_snapshot_sha256,
            ),
            (
                "model lock SHA-256",
                model_lock_hash,
                contract.model_lock_sha256,
            ),
            (
                "governed schema SHA-256",
                governed_schema_hash,
                contract.governed_schema_sha256,
            ),
            (
                "held-out gold SHA-256",
                held_out_hash,
                contract.held_out_gold_sha256,
            ),
            (
                "graph-builder contract version",
                GRAPH_BUILDER_CONTRACT_VERSION,
                contract.graph_builder_contract_version,
            ),
        )
        for label, actual, expected in comparisons:
            if actual != expected:
                reasons.append(f"{label} is not the reviewed release value")
        if config_hash not in contract.allowed_config_sha256:
            reasons.append("semantic configuration SHA-256 is not allowlisted")
        if source_catalog_hash not in contract.allowed_source_catalog_sha256:
            reasons.append("source-catalog SHA-256 is not allowlisted")
        for path in contract.required_gold_paths:
            try:
                value = _read_dotted(suite, path)
            except (AttributeError, KeyError, TypeError):
                reasons.append(f"required gold path is unavailable: {path}")
                continue
            if not value:
                reasons.append(f"required publication gold is empty: {path}")
        if reasons:
            return _research_assessment(reasons)

        binding = TrustedPublicationBinding(
            registry_version=self.version,
            registry_sha256=self.registry_sha256,
            policy_id=self.policy.policy_id,
            policy_sha256=self.policy.policy_sha256,
            quality_thresholds_version=self.policy.quality_thresholds_version,
            quality_thresholds_sha256=self.policy.quality_thresholds_sha256,
            contract_id=contract.contract_id,
            construction_profile=contract.construction_profile,
            fixture_id=contract.fixture_id,
            evaluation_fixture_sha256=suite.fixture_sha256,
            evaluation_file_sha256=evaluation_file_hash,
            source_snapshot_sha256=source_hashes,
            runtime_source_snapshot_sha256=runtime_source_hashes,
            config_sha256=config_hash,
            source_catalog_sha256=source_catalog_hash,
            model_lock_sha256=model_lock_hash,
            governed_schema_sha256=governed_schema_hash,
            held_out_gold_sha256=held_out_hash,
            graph_builder_contract_version=GRAPH_BUILDER_CONTRACT_VERSION,
        )
        return PublicationContractAssessment(
            release_eligible=True,
            research_only=False,
            reasons=(),
            binding=binding,
        )

    def authorize(
        self,
        *,
        suite: VersionedGraphEvaluationSuite,
        suite_path: str | Path,
        sources: list[SourceRecord],
        config: DataToKGConfig,
        source_catalog: SourceCatalog | None = None,
    ) -> TrustedPublicationBinding:
        """Return a binding or fail before construction/materialization."""

        assessment = self.assess(
            suite=suite,
            suite_path=suite_path,
            sources=sources,
            config=config,
            source_catalog=source_catalog,
        )
        if assessment.binding is None:
            raise UntrustedPublicationContractError(assessment)
        return assessment.binding

    def _enforce_suite_policy(
        self,
        suite: VersionedGraphEvaluationSuite,
    ) -> None:
        violations = _threshold_violations(
            suite,
            minimums=self.policy.suite_minimums,
            maximums=self.policy.suite_maximums,
            prefix="evaluation suite",
        )
        if violations:
            raise ValueError(
                "non-weakenable publication policy rejected thresholds: "
                + ", ".join(violations)
            )

    def _enforce_held_out_policy(
        self,
        *,
        config: DataToKGConfig,
        reasons: list[str],
    ) -> None:
        if config.unstructured.held_out_gold_path:
            path = Path(config.unstructured.held_out_gold_path).expanduser()
            if not path.is_file():
                return
            raw = path.read_bytes()
        else:
            raw = files("kg_construction").joinpath(
                "resources/unstructured_claims_gold_v1.json"
            ).read_bytes()
        try:
            payload = yaml.safe_load(raw)
            thresholds = payload["thresholds"]
            violations = _threshold_violations(
                thresholds,
                minimums=self.policy.held_out_minimums,
                maximums=self.policy.held_out_maximums,
                prefix="held-out claim gold",
            )
        except (KeyError, TypeError, yaml.YAMLError) as exc:
            reasons.append(f"held-out claim policy could not be evaluated: {exc}")
            return
        if violations:
            reasons.append(
                "non-weakenable publication policy rejected held-out thresholds: "
                + ", ".join(violations)
            )


def publication_config_sha256(config: DataToKGConfig) -> str:
    """Hash semantic/run policy while binding the suite separately."""

    payload = config.model_dump(mode="json")
    payload.pop("evaluation_suite_path", None)
    return _sha256_text(canonical_json(payload))


def source_snapshot_sha256(source: SourceRecord) -> str:
    """Bind source identity and immutable payload, excluding volatile timestamps."""

    payload = {
        "source_id": source.source_id,
        "source_type": source.source_type,
        "source_name": source.source_name,
        "content": source.content,
        "records": source.records,
    }
    return _sha256_text(canonical_json(payload))


def publication_source_catalog_sha256(
    config: DataToKGConfig,
    *,
    source_catalog: SourceCatalog | None,
) -> str:
    """Hash the modelling declarations supplied outside the main config."""

    if source_catalog is None:
        payload: object = {
            "mode": "config",
            "structured": config.structured.model_dump(mode="json"),
        }
    else:
        payload = {
            "mode": "explicit",
            "declarations": {
                name: declaration.model_dump(mode="json")
                for name, declaration in sorted(
                    source_catalog.declarations.items()
                )
            },
        }
    return _sha256_text(canonical_json(payload))


def publication_policy_sha256(policy: PublicationMinimumPolicy) -> str:
    payload = policy.model_dump(mode="json", exclude={"policy_sha256"})
    return _sha256_text(canonical_json(payload))


def publication_quality_thresholds_sha256() -> str:
    """Hash the complete executable KG quality-gate contract."""

    from kg_construction.infrastructure.evaluation.quality_gates import (
        VersionedQualityThresholds,
    )

    thresholds = VersionedQualityThresholds()
    return _sha256_text(canonical_json(thresholds.model_dump(mode="json")))


def publication_registry_sha256(registry: PublicationContractRegistry) -> str:
    payload = registry.model_dump(mode="json", exclude={"registry_sha256"})
    return _sha256_text(canonical_json(payload))


def _threshold_violations(
    root: object,
    *,
    minimums: dict[str, float],
    maximums: dict[str, float],
    prefix: str,
) -> list[str]:
    violations: list[str] = []
    for path, floor in minimums.items():
        try:
            actual = float(_read_dotted(root, path))
        except (AttributeError, KeyError, TypeError, ValueError):
            violations.append(f"{prefix}.{path} is missing")
            continue
        if actual < floor:
            violations.append(
                f"{prefix}.{path}={actual:g} is below floor {floor:g}"
            )
    for path, ceiling in maximums.items():
        try:
            actual = float(_read_dotted(root, path))
        except (AttributeError, KeyError, TypeError, ValueError):
            violations.append(f"{prefix}.{path} is missing")
            continue
        if actual > ceiling:
            violations.append(
                f"{prefix}.{path}={actual:g} exceeds ceiling {ceiling:g}"
            )
    return violations


def _read_dotted(root: object, path: str) -> Any:
    value = root
    for part in path.split("."):
        if isinstance(value, BaseModel):
            value = getattr(value, part)
        elif isinstance(value, dict):
            if part in value:
                value = value[part]
            elif part.isdigit() and int(part) in value:
                value = value[int(part)]
            else:
                raise KeyError(part)
        else:
            value = getattr(value, part)
    return value


def _configured_or_package_sha256(
    *,
    configured_path: str | None,
    package_relative_path: str,
    label: str,
    reasons: list[str],
) -> str | None:
    if configured_path:
        return _artifact_sha256(
            Path(configured_path).expanduser(),
            label=label,
            reasons=reasons,
        )
    raw = files("kg_construction").joinpath(package_relative_path).read_bytes()
    return _sha256_bytes(raw)


def _artifact_sha256(
    path: Path,
    *,
    label: str,
    reasons: list[str],
) -> str:
    if not path.is_file():
        reasons.append(f"{label} does not exist: {path}")
        return ""
    return _sha256_bytes(path.read_bytes())


def _research_assessment(reasons: list[str]) -> PublicationContractAssessment:
    return PublicationContractAssessment(
        release_eligible=False,
        research_only=True,
        reasons=tuple(dict.fromkeys(reasons)),
        binding=None,
    )


def _require_sha256(value: str, *, field: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


__all__ = [
    "GRAPH_BUILDER_CONTRACT_VERSION",
    "PUBLICATION_CONTRACT_VERSION",
    "PublicationContractAssessment",
    "PublicationContractRegistry",
    "PublicationMinimumPolicy",
    "TrustedPublicationBinding",
    "TrustedReleaseContract",
    "UntrustedPublicationContractError",
    "publication_config_sha256",
    "publication_policy_sha256",
    "publication_quality_thresholds_sha256",
    "publication_registry_sha256",
    "publication_source_catalog_sha256",
    "source_snapshot_sha256",
]
