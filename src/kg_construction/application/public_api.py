"""Stable public API facade for data scientists and application developers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias, TypeGuard

from kg_construction.application.benchmark_runner import (
    TechniqueEvaluationConfig,
    TechniqueEvaluationReport,
    TechniqueEvaluationRunner,
)
from kg_construction.application.config import DataToKGConfig
from kg_construction.application.dataset_profiler import DatasetProfile, DatasetProfiler
from kg_construction.application.mandatory_unstructured_runtime import (
    MandatoryUnstructuredRuntime,
)
from kg_construction.application.production import build_production_services
from kg_construction.application.publication_contract import (
    PublicationContractRegistry,
    TrustedPublicationBinding,
)
from kg_construction.application.registry import PluginRegistry, default_registry
from kg_construction.application.source_catalog import (
    CatalogTableMapper,
    DeclaredForeignKey,
    SourceCatalog,
    TableDeclaration,
)
from kg_construction.application.workflow import (
    ConstructionResult,
    ConstructionWorkflow,
    PublicationGateError,
)
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.evaluation import (
    GraphBoundIntendedUseEvaluator,
    GraphWorkloadRunner,
    Neo4jVectorWorkloadBackend,
    VersionedGraphEvaluationSuite,
)
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    HydratedModelSet,
)
from kg_construction.infrastructure.extractors.model_backed.factory import (
    detect_mandatory_profile,
)
from kg_construction.infrastructure.extractors.model_backed.identity import (
    LockedEmbeddingIdentityProvider,
)
from kg_construction.infrastructure.graph_stores import (
    Neo4jVectorStoreAdapter,
    Neo4jVectorStoreConfig,
)
from kg_construction.infrastructure.loaders import (
    APIResponseLoader,
    ChatTranscriptLoader,
    EmailLoader,
    EventPayloadLoader,
    ExcelLoader,
    HTMLLoader,
    JSONLDLoader,
    LogLoader,
    MarkdownLoader,
    PDFLoader,
    ReportLoader,
    ScientificPaperLoader,
    SQLDatabaseLoader,
    StructuredAPILoader,
    WebPageLoader,
    XMLLoader,
)
from kg_construction.infrastructure.loaders.csv_loader import CsvLoader
from kg_construction.infrastructure.loaders.json_loader import JsonLoader
from kg_construction.infrastructure.loaders.text_loader import TextLoader
from kg_construction.infrastructure.mapping import (
    DirectTableMapper,
    EventGraphMapper,
    HTMLMetadataMapper,
    JSONLDMapper,
    JsonPathMapper,
    LogEventMapper,
)
from kg_construction.infrastructure.ontology.validation import EvidenceRequiredValidator
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver

PathLike: TypeAlias = str | Path
SourceInput: TypeAlias = PathLike | SourceRecord | Sequence[SourceRecord]
ConfigInput: TypeAlias = DataToKGConfig | PathLike | None
EvaluationConfigInput: TypeAlias = TechniqueEvaluationConfig | PathLike | None

_DIRECTORY_SUFFIXES = {
    ".chat",
    ".csv",
    ".db",
    ".eml",
    ".htm",
    ".html",
    ".jats",
    ".json",
    ".jsonld",
    ".log",
    ".markdown",
    ".md",
    ".pdf",
    ".report",
    ".sqlite",
    ".tei",
    ".txt",
    ".xlsm",
    ".xlsx",
    ".xml",
}

_SOURCE_TYPE_ALIASES = {
    "api": "api_response",
    "api_response": "api_response",
    "chat": "chat_transcript",
    "chat_transcript": "chat_transcript",
    "csv": "csv",
    "db": "sql",
    "email": "email",
    "eml": "email",
    "event": "event_payload",
    "event_payload": "event_payload",
    "events": "event_payload",
    "excel": "excel",
    "htm": "html",
    "html": "html",
    "jats": "scientific_paper",
    "json": "json",
    "jsonld": "jsonld",
    "log": "logs",
    "logs": "logs",
    "markdown": "markdown",
    "md": "markdown",
    "pdf": "pdf",
    "report": "report",
    "scientific_paper": "scientific_paper",
    "sqlite": "sql",
    "sql": "sql",
    "structured_api": "structured_api",
    "tei": "scientific_paper",
    "text": "text",
    "transcript": "chat_transcript",
    "txt": "text",
    "web": "webpage",
    "webpage": "webpage",
    "xlsm": "excel",
    "xlsx": "excel",
    "xml": "xml",
}


def load_sources(
    input_data: SourceInput,
    *,
    source_type: str | None = None,
    source_name: str | None = None,
) -> list[SourceRecord]:
    """Load data into evidence-backed ``SourceRecord`` objects.

    ``input_data`` can be a file path, directory path, one ``SourceRecord``, a
    sequence of ``SourceRecord`` objects, or raw text when ``source_type="text"``
    or the string does not point to an existing file.
    """

    if isinstance(input_data, SourceRecord):
        return [input_data]
    if _is_source_record_sequence(input_data):
        return list(input_data)
    if not isinstance(input_data, str | Path):
        raise TypeError(
            "input_data must be a path, raw text string, SourceRecord, "
            "or sequence of SourceRecord objects."
        )

    path_or_text = Path(input_data)
    normalized_type = _normalize_source_type(source_type)
    if path_or_text.exists():
        if path_or_text.is_dir():
            return _load_directory(path_or_text)
        return _load_file(path_or_text, normalized_type)

    if normalized_type in {None, "text"} and isinstance(input_data, str):
        return [
            SourceRecord.from_text(
                input_data,
                source_type="text",
                source_name=source_name or "memory://kg-construction/text",
            )
        ]

    raise FileNotFoundError(f"Input path does not exist: {input_data}")


def profile_dataset(
    input_data: SourceInput,
    *,
    source_type: str | None = None,
) -> DatasetProfile:
    """Profile a dataset before choosing construction techniques."""

    if isinstance(input_data, str | Path) and Path(input_data).exists():
        return DatasetProfiler().profile_path(input_data, source_type=source_type)
    sources = load_sources(input_data, source_type=source_type)
    input_path = str(input_data) if isinstance(input_data, str | Path) else None
    return DatasetProfiler().profile_sources(sources, input_path=input_path)


def construct_kg(
    input_data: SourceInput,
    *,
    source_type: str | None = None,
    config: ConfigInput = None,
    source_name: str | None = None,
    source_catalog: SourceCatalog | None = None,
    evaluation_suite_path: PathLike | None = None,
) -> ConstructionResult:
    """Construct property, RDF, and evidence graphs from one input.

    Structured inputs use the semantic source-catalog path. Unstructured inputs
    use every model provider required by the detected platform profile and fail
    when the immutable model lock, a model artifact, or Ollama is unavailable.
    Every public run requires a sealed, versioned evaluation suite loaded from a
    file before construction. Self-sealing is not trust: the repository-owned
    publication registry must also bind the exact suite file, source snapshot,
    semantic config, model lock, governed schema, held-out gold, builder
    contract, and non-weakenable policy. The suite contains only gold judgments
    and declarative queries; observations are computed from the frozen
    constructed graph. Validation, quality, held-out extraction, all
    intended-use families, and the live workload backend form one conjunctive
    publication gate.

    Use ``ConstructionWorkflow(require_publication_gates=False).run(...)``
    directly when an explicitly non-publishable draft is needed for exploration
    or unit tests.
    """

    sources = load_sources(input_data, source_type=source_type, source_name=source_name)
    resolved_config = _data_config(config)
    suite_path = evaluation_suite_path or resolved_config.evaluation_suite_path
    if suite_path is None:
        raise ValueError(
            "public construct_kg requires evaluation_suite_path; "
            "unscored draft graphs cannot be published"
        )
    suite = VersionedGraphEvaluationSuite.from_path(suite_path)
    publication_contract = PublicationContractRegistry.from_package_resource().authorize(
        suite=suite,
        suite_path=suite_path,
        sources=sources,
        config=resolved_config,
        source_catalog=source_catalog,
    )
    primary_source_type = _primary_source_type(sources, source_type)
    production_kwargs = build_production_services(resolved_config).workflow_kwargs()
    workflow = default_workflow_for_source_type(
        primary_source_type,
        resolved_config,
        source_catalog=source_catalog,
        graph_evaluation_suite=suite,
        publication_contract=publication_contract,
        require_publication_gates=True,
        production_kwargs=production_kwargs,
    )
    result = workflow.run(sources, graph_evaluation_suite=suite)
    if not result.publication_gate.passed:
        raise PublicationGateError(result)
    if not result.release_eligible:
        raise RuntimeError(
            "public construction completed without trusted release-contract evidence"
        )
    return result


def evaluate_techniques(
    input_path: PathLike,
    *,
    source_type: str | None = None,
    config: EvaluationConfigInput = None,
    techniques: str = "applicable",
    output_dir: PathLike | None = None,
    registry: PluginRegistry | None = None,
) -> TechniqueEvaluationReport:
    """Profile, run, score, and rank KG construction techniques for a path.

    ``input_path`` must be a file or directory path because benchmark metadata
    includes a stable input hash. Use ``construct_kg`` for in-memory text or
    already-loaded ``SourceRecord`` objects.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark input path does not exist: {input_path}")
    return TechniqueEvaluationRunner(
        config=_evaluation_config(config),
        registry=registry,
    ).run(
        input_path=path,
        source_type=source_type,
        techniques=techniques,
        output_dir=output_dir,
    )


def _structured_mappers(
    config: DataToKGConfig,
    source_catalog: SourceCatalog | None = None,
) -> list[Any]:
    """Return the structured-table mapper: semantic by default, direct baseline on request."""

    if config.pipeline.structured_mapping == "direct":
        return [DirectTableMapper()]
    catalog = source_catalog or _catalog_from_config(config)
    if config.structured.require_declarations and not catalog.declarations:
        raise ValueError(
            "structured.require_declarations=true but no structured.tables were declared"
        )
    return [
        CatalogTableMapper(
            catalog,
            require_declarations=config.structured.require_declarations,
        )
    ]


def default_workflow_for_source_type(
    source_type: str | None,
    config: DataToKGConfig | None = None,
    *,
    source_catalog: SourceCatalog | None = None,
    graph_evaluation_suite: VersionedGraphEvaluationSuite | None = None,
    publication_contract: TrustedPublicationBinding | None = None,
    require_publication_gates: bool = False,
    production_kwargs: dict[str, Any] | None = None,
) -> ConstructionWorkflow:
    """Create the mandatory, fail-closed workflow used by ``construct_kg``."""

    resolved_config = config or DataToKGConfig()
    normalized_type = _normalize_source_type(source_type) or "text"
    validators: list[Any] = [EvidenceRequiredValidator(resolved_config.pipeline.validation_policy)]
    resolvers: list[Any] = [ExactKeyEntityResolver()]
    common_kwargs: dict[str, Any] = {
        "pipeline_name": resolved_config.pipeline.name,
        "config_payload": resolved_config.model_dump(mode="json"),
        "graph_evaluation_suite": graph_evaluation_suite,
        "publication_contract": publication_contract,
        "require_publication_gates": require_publication_gates,
        **(production_kwargs or {}),
    }
    if require_publication_gates:
        common_kwargs["graph_bound_evaluator"] = _live_graph_bound_evaluator(
            resolved_config
        )

    if normalized_type == "mixed":
        unstructured = _mandatory_unstructured_pipeline(resolved_config)
        return ConstructionWorkflow(
            mappers=[
                _DefaultSourceMapperRouter(
                    resolved_config,
                    source_catalog=source_catalog,
                )
            ],
            extractors=[_DefaultSourceExtractorRouter(unstructured)],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type in {"csv", "excel", "sql", "structured_api"}:
        return ConstructionWorkflow(
            mappers=_structured_mappers(resolved_config, source_catalog),
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "xml":
        return ConstructionWorkflow(
            mappers=[DirectTableMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type in {"json", "api_response"}:
        return ConstructionWorkflow(
            mappers=[JsonPathMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "jsonld":
        return ConstructionWorkflow(
            mappers=[JSONLDMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "event_payload":
        return ConstructionWorkflow(
            mappers=[EventGraphMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "logs":
        return ConstructionWorkflow(
            mappers=[LogEventMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type in {"html", "webpage"}:
        return ConstructionWorkflow(
            mappers=[HTMLMetadataMapper()],
            extractors=[_mandatory_unstructured_pipeline(resolved_config)],
            resolvers=[],
            validators=validators,
            **common_kwargs,
        )
    return ConstructionWorkflow(
        extractors=[_mandatory_unstructured_pipeline(resolved_config)],
        resolvers=[],
        validators=validators,
        **common_kwargs,
    )


def _load_directory(path: Path) -> list[SourceRecord]:
    sources: list[SourceRecord] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in _DIRECTORY_SUFFIXES:
            sources.extend(_load_file(child, None))
    return sources


def _load_file(path: Path, source_type: str | None) -> list[SourceRecord]:
    normalized_type = source_type or _infer_source_type(path)
    if normalized_type == "csv":
        return CsvLoader().load(path)
    if normalized_type == "excel":
        return ExcelLoader().load(path)
    if normalized_type == "sql":
        return SQLDatabaseLoader().load(path)
    if normalized_type == "json":
        return JsonLoader().load(path)
    if normalized_type == "jsonld":
        return JSONLDLoader().load(path)
    if normalized_type == "api_response":
        return APIResponseLoader().load(path)
    if normalized_type == "structured_api":
        return StructuredAPILoader().load(path)
    if normalized_type == "xml":
        return XMLLoader().load(path)
    if normalized_type == "markdown":
        return MarkdownLoader().load(path)
    if normalized_type == "pdf":
        return PDFLoader().load(path)
    if normalized_type == "report":
        return ReportLoader().load(path)
    if normalized_type == "email":
        return EmailLoader().load(path)
    if normalized_type == "chat_transcript":
        return ChatTranscriptLoader().load(path)
    if normalized_type == "html":
        return HTMLLoader().load(path)
    if normalized_type == "webpage":
        return WebPageLoader().load(path)
    if normalized_type == "logs":
        return LogLoader().load(path)
    if normalized_type == "event_payload":
        return EventPayloadLoader().load(path)
    if normalized_type == "scientific_paper":
        return ScientificPaperLoader().load(path)
    return TextLoader().load(path)


def _infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return _normalize_source_type(suffix) or "text"


def _normalize_source_type(source_type: str | None) -> str | None:
    if source_type is None:
        return None
    normalized = source_type.strip().lower().replace("-", "_")
    return _SOURCE_TYPE_ALIASES.get(normalized, normalized)


def _primary_source_type(sources: list[SourceRecord], source_type: str | None) -> str:
    explicit = _normalize_source_type(source_type)
    if explicit:
        return explicit
    if not sources:
        return "text"
    source_types = {_normalize_source_type(source.source_type) or source.source_type for source in sources}
    if len(source_types) == 1:
        return next(iter(source_types))
    return "mixed"


class _DefaultSourceMapperRouter:
    """Dispatch each source in a heterogeneous batch to its matching mapper."""

    name = "default_source_mapper_router"
    technique = "source_type_routed_mapping"

    def __init__(
        self,
        config: DataToKGConfig,
        *,
        source_catalog: SourceCatalog | None = None,
    ) -> None:
        direct = DirectTableMapper()
        semantic = _structured_mappers(config, source_catalog)[0]
        self._delegates: dict[str, Any] = {
            "api_response": JsonPathMapper(),
            "csv": semantic,
            "event_payload": EventGraphMapper(),
            "excel": semantic,
            "html": HTMLMetadataMapper(),
            "json": JsonPathMapper(),
            "jsonld": JSONLDMapper(),
            "logs": LogEventMapper(),
            "sql": semantic,
            "structured_api": semantic,
            "webpage": HTMLMetadataMapper(),
            "xml": direct,
        }

    def prepare_sources(self, sources: list[SourceRecord]) -> None:
        """Prepare each unique delegate once with only its applicable sources."""

        delegates = {id(delegate): delegate for delegate in self._delegates.values()}
        for delegate in delegates.values():
            prepare = getattr(delegate, "prepare_sources", None)
            if not callable(prepare):
                continue
            applicable = [
                source
                for source in sources
                if self._delegates.get(
                    _normalize_source_type(source.source_type) or source.source_type
                )
                is delegate
            ]
            prepare(applicable)

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        source_type = _normalize_source_type(source.source_type) or source.source_type
        delegate = self._delegates.get(source_type)
        if delegate is None:
            return ConstructionBatch(
                source_records=[source],
                metadata={"mapper_router": self.name, "mapper_route": "none"},
            )
        result = delegate.map_source(source)
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "mapper_router": self.name,
                    "mapper_route": delegate.name,
                }
            }
        )


class _DefaultSourceExtractorRouter:
    """Run text extraction only for source types with meaningful text content."""

    name = "default_source_extractor_router"
    technique = "source_type_routed_extraction"

    _TEXT_SOURCE_TYPES = {
        "chat_transcript",
        "email",
        "html",
        "markdown",
        "pdf",
        "report",
        "scientific_paper",
        "text",
        "webpage",
    }

    def __init__(self, delegate: MandatoryUnstructuredRuntime) -> None:
        self._delegate = delegate

    def prepare_sources(self, sources: list[SourceRecord]) -> None:
        applicable = [
            source
            for source in sources
            if (_normalize_source_type(source.source_type) or source.source_type)
            in self._TEXT_SOURCE_TYPES
        ]
        prepare = getattr(self._delegate, "prepare_sources", None)
        if applicable and callable(prepare):
            prepare(applicable)

    def reset_results(self) -> None:
        self._delegate.reset_results()

    @property
    def pipeline_results(self):
        return self._delegate.pipeline_results

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        source_type = _normalize_source_type(source.source_type) or source.source_type
        if source_type not in self._TEXT_SOURCE_TYPES:
            return ConstructionBatch(
                source_records=[source],
                metadata={"extractor_router": self.name, "extractor_route": "none"},
            )
        result = self._delegate.extract(source)
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "extractor_router": self.name,
                    "extractor_route": self._delegate.name,
                }
            }
        )


def _mandatory_unstructured_pipeline(
    config: DataToKGConfig,
) -> MandatoryUnstructuredRuntime:
    """Build the one governed, locked, conjunctive unstructured runtime."""

    settings = config.unstructured
    cache_root = Path(settings.cache_root).expanduser() if settings.cache_root else None
    return MandatoryUnstructuredRuntime.from_lock(
        Path(settings.models_lock_path).expanduser(),
        cache_root=cache_root,
        governed_schema_path=(
            Path(settings.governed_schema_path).expanduser()
            if settings.governed_schema_path
            else None
        ),
        held_out_gold_path=(
            Path(settings.held_out_gold_path).expanduser()
            if settings.held_out_gold_path
            else None
        ),
        entity_labels=settings.entity_labels,
        ollama_base_url=config.llm.base_url,
        ollama_retry_on_invalid_output=config.llm.retry_on_invalid_output,
        ollama_max_retries=config.llm.max_retries,
    )


def _data_config(config: ConfigInput) -> DataToKGConfig:
    if config is None:
        return DataToKGConfig()
    if isinstance(config, DataToKGConfig):
        return config
    return DataToKGConfig.from_yaml(config)


def _live_graph_bound_evaluator(
    config: DataToKGConfig,
) -> GraphBoundIntendedUseEvaluator:
    """Build the mandatory locked-embedding + live-Neo4j workload evaluator."""

    profile = detect_mandatory_profile()
    cache_root = (
        Path(config.unstructured.cache_root).expanduser()
        if config.unstructured.cache_root
        else None
    )
    models = HydratedModelSet.from_lock(
        Path(config.unstructured.models_lock_path).expanduser(),
        profile=profile,
        cache_root=cache_root,
    )
    artifact = models.artifact("embedding_resolver")
    embedding_model = LockedEmbeddingIdentityProvider(artifact)
    store = Neo4jVectorStoreAdapter(
        Neo4jVectorStoreConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="certification",
            index_name="kg_publication_vector_index",
            node_label="KGPublicationVector",
            dimensions=384,
        )
    )
    vector_backend = Neo4jVectorWorkloadBackend(
        store=store,
        embedding_model=embedding_model,
        model_id=artifact.model_version,
    )
    return GraphBoundIntendedUseEvaluator(
        runner=GraphWorkloadRunner(vector_backend=vector_backend)
    )


def _catalog_from_config(config: DataToKGConfig) -> SourceCatalog:
    catalog = SourceCatalog()
    for table in config.structured.tables:
        catalog.declare(
            TableDeclaration(
                table_name=table.table_name,
                record_type=table.record_type,
                primary_key=list(table.primary_key),
                alternate_keys=[
                    list(key) for key in table.alternate_keys
                ],
                foreign_keys=[
                    DeclaredForeignKey(
                        columns=list(foreign_key.columns),
                        target_table=foreign_key.target_table,
                        target_key=list(foreign_key.target_key),
                        relationship_name=foreign_key.relationship_name,
                    )
                    for foreign_key in table.foreign_keys
                ],
                datatypes=dict(table.datatypes),
                column_mappings=dict(table.column_mappings),
                identity_scope=table.identity_scope,
                controlled_vocabularies=list(table.controlled_vocabularies),
                table_kind=table.table_kind,
                association_relationship_name=table.association_relationship_name,
                origin="user",
            )
        )
    return catalog


def _evaluation_config(config: EvaluationConfigInput) -> TechniqueEvaluationConfig:
    if config is None:
        return TechniqueEvaluationConfig()
    if isinstance(config, TechniqueEvaluationConfig):
        return config
    return TechniqueEvaluationConfig.from_yaml(config)


def _is_source_record_sequence(value: object) -> TypeGuard[Sequence[SourceRecord]]:
    if isinstance(value, str | Path):
        return False
    return isinstance(value, Sequence) and all(isinstance(item, SourceRecord) for item in value)


__all__ = [
    "construct_kg",
    "default_registry",
    "default_workflow_for_source_type",
    "evaluate_techniques",
    "load_sources",
    "profile_dataset",
]
