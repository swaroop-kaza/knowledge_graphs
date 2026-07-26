"""Exact-platform factory for the mandatory model-backed extraction ensemble."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from kg_construction.infrastructure.extractors.model_backed.claims import (
    MandatoryClaimEnsemble,
    MlxQwenClaimExtractor,
    OllamaQwenClaimExtractor,
    OllamaTransport,
    TransformersQwenClaimExtractor,
)
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    HydratedArtifact,
    HydratedModelSet,
    MentionInferenceBackend,
    OfflineArtifactError,
    RelationInferenceBackend,
    StructuredGenerationBackend,
)
from kg_construction.infrastructure.extractors.model_backed.identity import (
    ExactClusterIdentityProvider,
    LockedCrossEncoderIdentityProvider,
    LockedEmbeddingIdentityProvider,
    MandatoryCanonicalIdentityEnsemble,
)
from kg_construction.infrastructure.extractors.model_backed.mentions import (
    GlinerMentionDetector,
    MandatoryMentionEnsemble,
    SpacyMentionDetector,
    TransformersMentionDetector,
)
from kg_construction.infrastructure.extractors.model_backed.relations import (
    GlinerRelExClaimExtractor,
)
from kg_construction.infrastructure.resolution.linking import (
    CrossEncoderModelProtocol,
    CrossEncoderResolver,
    CrossEncoderResolverConfig,
    DBpediaLinker,
    EmbeddingModelProtocol,
    EmbeddingResolver,
    EmbeddingResolverConfig,
    JsonTransport,
    OpenRefineReconciler,
    OpenRefineReconcilerConfig,
    WikidataLinker,
)
from kg_construction.infrastructure.resolution.mandatory_ensemble import (
    MandatoryScopedResolutionEnsemble,
    Resolver,
)
from kg_construction.unstructured.pipeline import (
    ClaimExtractor,
    PatternClaimExtractor,
    RuleMentionDetector,
)

DEFAULT_ENTITY_LABELS = (
    "Person",
    "Organization",
    "Location",
    "Event",
    "Product",
    "Date",
    "Concept",
)
DEFAULT_RELATION_LABELS = (
    "associated with",
    "part of",
    "has part",
    "located in",
    "worked at",
    "founded",
    "founded by",
    "created",
    "created by",
    "owns",
    "owned by",
    "member of",
    "reports to",
    "acquired",
    "supplies",
    "uses",
    "produces",
    "causes",
    "depends on",
    "precedes",
    "succeeds",
    "supports",
    "opposes",
    "intends",
    "requests",
    "requires",
    "has property",
)
_OPENREFINE_GOVERNED_TYPE_MAP = {
    # Reviewed Wikidata class binding: Q5 is "human".
    "Person": "Q5",
}


@dataclass(frozen=True)
class ModelBackendOverrides:
    """Injectable clients for tests or a controlled serving implementation."""

    gliner: MentionInferenceBackend | None = None
    gliner_relex: RelationInferenceBackend | None = None
    transformers_ner: MentionInferenceBackend | None = None
    spacy: MentionInferenceBackend | None = None
    transformers_claims: StructuredGenerationBackend | None = None
    mlx_claims: StructuredGenerationBackend | None = None
    ollama_claims: StructuredGenerationBackend | None = None
    ollama_transport: OllamaTransport | None = None
    embedding: EmbeddingModelProtocol | None = None
    cross_encoder: CrossEncoderModelProtocol | None = None
    external_link_transport: JsonTransport | None = None


@dataclass(frozen=True)
class MandatoryProviderBundle:
    """Every extraction and resolution component required for one platform."""

    profile: str
    mention_detector: MandatoryMentionEnsemble
    claim_extractor: MandatoryClaimEnsemble
    identity_resolver: MandatoryCanonicalIdentityEnsemble
    resolution_ensemble: MandatoryScopedResolutionEnsemble
    mandatory_provider_names: tuple[str, ...]
    mandatory_runtime_participation_ids: tuple[str, ...]
    locked_artifact_paths: tuple[Path, ...]


def detect_mandatory_profile(
    *,
    system: str | None = None,
    machine: str | None = None,
    cuda_available: bool | None = None,
) -> str:
    """Select the one declared profile for the exact host; never downgrade."""

    actual_system = system or platform.system()
    actual_machine = (machine or platform.machine()).lower()
    if actual_system == "Darwin" and actual_machine == "arm64":
        return "darwin-arm64"
    if actual_system == "Linux" and actual_machine in {"x86_64", "amd64"}:
        if cuda_available is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - mandatory dependency
                raise OfflineArtifactError(
                    "torch is required to select the Linux execution profile"
                ) from exc
            cuda_available = bool(torch.cuda.is_available())
        return "linux-cuda" if cuda_available else "linux-cpu"
    raise OfflineArtifactError(
        f"no mandatory model-backed profile exists for {actual_system}/{actual_machine}"
    )


def build_mandatory_model_providers_from_lock(
    lock_path: Path,
    *,
    profile: str | None = None,
    cache_root: Path | None = None,
    entity_labels: tuple[str, ...] = DEFAULT_ENTITY_LABELS,
    relation_labels: tuple[str, ...] = DEFAULT_RELATION_LABELS,
    overrides: ModelBackendOverrides | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_retry_on_invalid_output: bool = True,
    ollama_max_retries: int = 3,
    governed_schema_id: str = "governed-unstructured-core",
    governed_entity_types: tuple[str, ...] = DEFAULT_ENTITY_LABELS,
    system: str | None = None,
    machine: str | None = None,
    cuda_available: bool | None = None,
) -> MandatoryProviderBundle:
    """Load exact locked paths and construct all mandatory providers offline."""

    detected = detect_mandatory_profile(
        system=system,
        machine=machine,
        cuda_available=cuda_available,
    )
    if profile is not None and profile != detected:
        raise OfflineArtifactError(
            f"requested profile {profile!r} does not match detected profile {detected!r}"
        )
    models = HydratedModelSet.from_lock(
        lock_path,
        profile=detected,
        cache_root=cache_root,
    )
    return build_mandatory_model_providers(
        models,
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        overrides=overrides,
        ollama_base_url=ollama_base_url,
        ollama_retry_on_invalid_output=ollama_retry_on_invalid_output,
        ollama_max_retries=ollama_max_retries,
        governed_schema_id=governed_schema_id,
        governed_entity_types=governed_entity_types,
        system=system,
        machine=machine,
    )


def build_mandatory_model_providers(
    models: HydratedModelSet,
    *,
    entity_labels: tuple[str, ...] = DEFAULT_ENTITY_LABELS,
    relation_labels: tuple[str, ...] = DEFAULT_RELATION_LABELS,
    overrides: ModelBackendOverrides | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_retry_on_invalid_output: bool = True,
    ollama_max_retries: int = 3,
    governed_schema_id: str = "governed-unstructured-core",
    governed_entity_types: tuple[str, ...] = DEFAULT_ENTITY_LABELS,
    system: str | None = None,
    machine: str | None = None,
) -> MandatoryProviderBundle:
    """Construct the mandatory union ensemble for a validated model set."""

    actual_system = system or platform.system()
    actual_machine = (machine or platform.machine()).lower()
    _require_profile_matches_platform(
        models.profile,
        system=actual_system,
        machine=actual_machine,
    )
    clients = overrides or ModelBackendOverrides()

    gliner = _artifact(models, "gliner_entity_extractor", backend="gliner")
    gliner_relex = _artifact(
        models,
        "gliner_relex_extractor",
        backend="gliner",
    )
    transformer_ner = _artifact(models, "transformer_ner", backend="transformers")
    spacy = _artifact(models, "spacy_pipeline", backend="spacy")
    transformer_qwen = _artifact(
        models,
        "structured_transformer_extractor",
        backend="transformers",
    )
    embedding_artifact = _artifact(
        models,
        "embedding_resolver",
        backend="sentence-transformers",
    )
    cross_encoder_artifact = _artifact(
        models,
        "cross_encoder_resolver",
        backend="sentence-transformers",
    )
    ollama_qwen = models.service_model("ollama_structured_extractor")

    embedding_identity = LockedEmbeddingIdentityProvider(
        embedding_artifact,
        model=clients.embedding,
    )
    cross_encoder_identity = LockedCrossEncoderIdentityProvider(
        cross_encoder_artifact,
        model=clients.cross_encoder,
    )
    identity_resolver = MandatoryCanonicalIdentityEnsemble(
        (
            ExactClusterIdentityProvider(),
            embedding_identity,
            cross_encoder_identity,
        ),
        schema_id=governed_schema_id,
    )
    post_resolvers: tuple[Resolver, ...] = (
        EmbeddingResolver(
            EmbeddingResolverConfig(model_name=str(embedding_artifact.local_path)),
            model=embedding_identity,
        ),
        CrossEncoderResolver(
            CrossEncoderResolverConfig(
                model_name=str(cross_encoder_artifact.local_path)
            ),
            model=cross_encoder_identity,
        ),
        WikidataLinker(transport=clients.external_link_transport),
        DBpediaLinker(transport=clients.external_link_transport),
        OpenRefineReconciler(
            OpenRefineReconcilerConfig(
                merge_on_match=False,
                type_map={
                    entity_type: wikidata_type
                    for entity_type, wikidata_type
                    in _OPENREFINE_GOVERNED_TYPE_MAP.items()
                    if entity_type in governed_entity_types
                },
            ),
            transport=clients.external_link_transport,
        ),
    )
    resolution_ensemble = MandatoryScopedResolutionEnsemble(
        post_resolvers,
        governed_entity_types=governed_entity_types,
        schema_id=governed_schema_id,
    )

    mention_providers = (
        RuleMentionDetector(),
        GlinerMentionDetector(
            gliner,
            labels=entity_labels,
            backend=clients.gliner,
        ),
        TransformersMentionDetector(
            transformer_ner,
            backend=clients.transformers_ner,
        ),
        SpacyMentionDetector(
            spacy,
            backend=clients.spacy,
        ),
    )
    claim_providers: list[ClaimExtractor] = [
        PatternClaimExtractor(),
        GlinerRelExClaimExtractor(
            gliner_relex,
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            backend=clients.gliner_relex,
        ),
        TransformersQwenClaimExtractor(
            transformer_qwen,
            backend=clients.transformers_claims,
        ),
    ]
    provider_names = [
        "deterministic_rule_mentions",
        "gliner",
        "transformers_ner",
        "spacy",
        "deterministic_pattern_claims",
        "gliner_relex",
        "transformers_qwen3",
    ]

    if models.profile == "darwin-arm64":
        mlx_qwen = _artifact(
            models,
            "mlx_structured_extractor",
            backend="mlx",
        )
        claim_providers.append(
            MlxQwenClaimExtractor(
                mlx_qwen,
                backend=clients.mlx_claims,
            )
        )
        provider_names.append("mlx_qwen3")

    claim_providers.append(
        OllamaQwenClaimExtractor(
            ollama_qwen,
            base_url=ollama_base_url,
            backend=clients.ollama_claims,
            transport=clients.ollama_transport,
            retry_on_invalid_output=ollama_retry_on_invalid_output,
            max_retries=ollama_max_retries,
        )
    )
    provider_names.append("ollama_qwen3")
    mention_ensemble = MandatoryMentionEnsemble(mention_providers)
    claim_ensemble = MandatoryClaimEnsemble(claim_providers)
    return MandatoryProviderBundle(
        profile=models.profile,
        mention_detector=mention_ensemble,
        claim_extractor=claim_ensemble,
        identity_resolver=identity_resolver,
        resolution_ensemble=resolution_ensemble,
        mandatory_provider_names=tuple(provider_names),
        mandatory_runtime_participation_ids=(
            *mention_ensemble.provider_ids,
            *claim_ensemble.provider_ids,
            *identity_resolver.provider_ids,
            *resolution_ensemble.provider_ids,
        ),
        locked_artifact_paths=tuple(
            artifact.local_path for artifact in models.artifacts
        ),
    )


def _artifact(
    models: HydratedModelSet,
    name: str,
    *,
    backend: str,
) -> HydratedArtifact:
    artifact = models.artifact(name)
    if artifact.backend != backend:
        raise OfflineArtifactError(f"{name} must use backend {backend!r}, got {artifact.backend!r}")
    return artifact


def _require_profile_matches_platform(
    profile_name: str,
    *,
    system: str,
    machine: str,
) -> None:
    if system == "Darwin" and machine == "arm64":
        expected = {"darwin-arm64"}
    elif system == "Linux" and machine in {"x86_64", "amd64"}:
        expected = {"linux-cpu", "linux-cuda"}
    else:
        expected = set()
    if profile_name not in expected:
        raise OfflineArtifactError(f"profile {profile_name!r} cannot run on {system}/{machine}")


__all__ = [
    "DEFAULT_ENTITY_LABELS",
    "DEFAULT_RELATION_LABELS",
    "MandatoryProviderBundle",
    "ModelBackendOverrides",
    "build_mandatory_model_providers",
    "build_mandatory_model_providers_from_lock",
    "detect_mandatory_profile",
]
