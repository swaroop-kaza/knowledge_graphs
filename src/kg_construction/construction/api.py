"""Canonical construction API for loading data and building knowledge graphs.

This module owns construction only. Comparative technique benchmarking lives in
``kg_construction.research`` and release authorization lives in
``kg_construction.certification``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeAlias, TypeGuard

from kg_construction.certification.publication_contract import (
    TrustedPublicationBinding,
)
from kg_construction.construction.config import DataToKGConfig
from kg_construction.construction.profiling import DatasetProfile, DatasetProfiler
from kg_construction.construction.workflow import (
    ConstructionResult,
    ConstructionWorkflow,
    DraftConstructionWorkflow,
)
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.evaluation import (
    GraphBoundIntendedUseEvaluator,
    GraphWorkloadRunner,
    KnowledgeGraphQualityEvaluator,
    Neo4jGraphWorkloadBackend,
    Neo4jGraphWorkloadConfig,
    Neo4jVectorWorkloadBackend,
    VersionedGraphEvaluationSuite,
    VersionedQualityThresholds,
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
    DottedPathJsonMapper,
    EventGraphMapper,
    HTMLMetadataMapper,
    JSONLDMapper,
    JsonPathMapper,
    LogEventMapper,
    XMLPathMapper,
)
from kg_construction.infrastructure.ontology.validation import EvidenceRequiredValidator
from kg_construction.infrastructure.production import PIIRedactionPolicy
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver
from kg_construction.service_runtime import LockedServiceRuntime
from kg_construction.structured.source_catalog import (
    CatalogTableMapper,
    DeclaredForeignKey,
    SourceCatalog,
    TableDeclaration,
)
from kg_construction.unstructured.pipeline import UnstructuredPipeline
from kg_construction.unstructured.runtime import (
    MandatoryUnstructuredRuntime,
)

PathLike: TypeAlias = str | Path
SourceInput: TypeAlias = PathLike | SourceRecord | Sequence[SourceRecord]
ConfigInput: TypeAlias = DataToKGConfig | PathLike | None
EvaluationSuiteInput: TypeAlias = VersionedGraphEvaluationSuite | PathLike | None

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

_TECHNIQUES_BY_SOURCE_TYPE: dict[str, frozenset[str]] = {
    "api_response": frozenset({"dotted_path_json_mapper", "jsonpath_mapper"}),
    "csv": frozenset({"direct_table_mapper", "semantic_table_mapper"}),
    "event_payload": frozenset({"event_graph_mapper"}),
    "excel": frozenset({"direct_table_mapper", "semantic_table_mapper"}),
    "html": frozenset({"html_metadata_mapper"}),
    "json": frozenset({"dotted_path_json_mapper", "jsonpath_mapper"}),
    "jsonld": frozenset({"jsonld_mapper"}),
    "logs": frozenset({"log_event_mapper"}),
    "sql": frozenset({"direct_table_mapper", "semantic_table_mapper"}),
    "structured_api": frozenset({"direct_table_mapper", "semantic_table_mapper"}),
    "webpage": frozenset({"html_metadata_mapper"}),
    "xml": frozenset({"direct_table_mapper", "xmlpath_mapper"}),
}
_UNSTRUCTURED_TECHNIQUES = frozenset({"evidence_first_unstructured_pipeline"})
_TEXT_SOURCE_TYPES = frozenset(
    {
        "chat_transcript",
        "email",
        "markdown",
        "pdf",
        "report",
        "scientific_paper",
        "text",
    }
)


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
    normalized_type = _normalize_source_type(source_type, require_supported=True)
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


def convert(
    data: SourceInput,
    source_format: str | None = None,
    technique: str | None = None,
    *,
    config: ConfigInput = None,
    source_name: str | None = None,
    source_catalog: SourceCatalog | None = None,
    evaluation_suite: EvaluationSuiteInput = None,
) -> ConstructionResult:
    """Convert data into a knowledge graph plus every computed metric.

    Given ``data`` in a supported ``source_format`` and a valid ``technique``,
    this returns a :class:`~kg_construction.construction.workflow.ConstructionResult`
    holding the property, RDF, and evidence graphs together with all evaluation
    metrics (formation quality, schema quality, intended-use, validation, and
    extraction coverage).

    Thresholds are reported on ``result.quality_report`` and
    ``result.publication_gate`` but do not authorize publication. The mandatory,
    zero-skip certification workflow is the sole release-authorizing path.

    ``source_format`` is one of the loader formats (``csv``, ``excel``, ``sql``,
    ``json``, ``xml``, ``jsonld``, ``text``, ``logs``, ``event_payload``,
    ``html`` …). ``technique`` selects the converter — for example
    ``"semantic_table_mapper"``, ``"direct_table_mapper"``, ``"jsonpath_mapper"``,
    ``"dotted_path_json_mapper"``, or ``"evidence_first_unstructured_pipeline"``.
    When omitted, the governed default technique for the format is used. The
    deterministic and locked model-backed profiles are distinct explicit runtime
    profiles; an explicitly requested model-backed run never silently degrades to
    deterministic extraction.

    ``evaluation_suite`` may be a sealed suite object or its YAML/JSON path. It is
    executed against the constructed graph and contributes graph-bound formation,
    extraction, and intended-use metrics. A failed metric remains visible in this
    draft result; certification decides whether the graph may be published.
    """

    sources = load_sources(data, source_type=source_format, source_name=source_name)
    resolved_config = _data_config(config)
    primary_format = _primary_source_type(sources, source_format)
    suite = _evaluation_suite(
        evaluation_suite
        if evaluation_suite is not None
        else resolved_config.evaluation_suite_path
    )
    workflow = _draft_workflow_for_technique(
        primary_format,
        technique,
        resolved_config,
        source_catalog=source_catalog,
        graph_evaluation_suite=suite,
    )
    return workflow.run(sources)


def construct_kg(
    input_data: SourceInput,
    *,
    source_type: str | None = None,
    config: ConfigInput = None,
    source_name: str | None = None,
    source_catalog: SourceCatalog | None = None,
    technique: str | None = None,
    evaluation_suite_path: PathLike | None = None,
) -> ConstructionResult:
    """Backward-compatible alias for :func:`convert`.

    ``evaluation_suite_path`` is evaluated rather than treated as a publication
    bypass. The returned graph is a draft until the separate certification hard
    pass succeeds.
    """

    return convert(
        input_data,
        source_format=source_type,
        technique=technique,
        config=config,
        source_name=source_name,
        source_catalog=source_catalog,
        evaluation_suite=evaluation_suite_path,
    )


def _draft_workflow_for_technique(
    source_format: str | None,
    technique: str | None,
    config: DataToKGConfig,
    *,
    source_catalog: SourceCatalog | None = None,
    graph_evaluation_suite: VersionedGraphEvaluationSuite | None = None,
) -> DraftConstructionWorkflow:
    """Build the non-blocking workflow that always returns a graph plus metrics."""

    fmt = _normalize_source_type(source_format) or "text"
    if fmt == "mixed":
        if technique is not None:
            raise ValueError(
                "mixed-source conversion uses source-routed defaults and does not "
                "accept one technique for every source"
            )
        text_extractor = _text_extractor(config)
        mappers = [_DefaultSourceMapperRouter(config, source_catalog=source_catalog)]
        extractors = [_DefaultSourceExtractorRouter(text_extractor)]
    else:
        mappers, extractors = _technique_components(
            fmt,
            technique,
            config,
            source_catalog,
        )
    resolvers = (
        []
        if fmt in _TEXT_SOURCE_TYPES
        else [ExactKeyEntityResolver()]
    )
    return DraftConstructionWorkflow(
        mappers=mappers,
        extractors=extractors,
        resolvers=resolvers,
        validators=[EvidenceRequiredValidator(config.pipeline.validation_policy)],
        quality_evaluator=_quality_evaluator_for(config),
        graph_evaluation_suite=graph_evaluation_suite,
        pii_policy=PIIRedactionPolicy(
            **config.production.pii_redaction.model_dump()
        ),
        pipeline_name=config.pipeline.name,
        config_payload=config.model_dump(mode="json"),
    )


def _technique_components(
    source_format: str,
    technique: str | None,
    config: DataToKGConfig,
    source_catalog: SourceCatalog | None,
) -> tuple[list[Any], list[Any]]:
    """Map ``(format, technique)`` to the ``(mappers, extractors)`` that convert it."""

    tech = _validated_technique(source_format, technique)
    if source_format in {"csv", "excel", "sql", "structured_api"}:
        if tech == "direct_table_mapper":
            return [DirectTableMapper()], []
        return _structured_mappers(config, source_catalog), []
    if source_format == "xml":
        if tech == "xmlpath_mapper":
            return [XMLPathMapper()], []
        return [DirectTableMapper()], []
    if source_format in {"json", "api_response"}:
        if tech == "dotted_path_json_mapper":
            return [DottedPathJsonMapper()], []
        return [JsonPathMapper()], []
    if source_format == "jsonld":
        return [JSONLDMapper()], []
    if source_format == "event_payload":
        return [EventGraphMapper()], []
    if source_format == "logs":
        return [LogEventMapper()], []
    if source_format in {"html", "webpage"}:
        return [HTMLMetadataMapper()], [_text_extractor(config)]
    if source_format in _TEXT_SOURCE_TYPES:
        return [], [_text_extractor(config)]
    raise ValueError(f"unsupported source format for construction: {source_format!r}")


def _text_extractor(config: DataToKGConfig) -> Any:
    """Return the text extractor for the configured extraction runtime profile.

    ``deterministic`` runs the evidence-first rule pipeline. ``model_backed``
    builds the locked platform-applicable provider ensemble. Provider or artifact
    failures propagate; silently changing extraction semantics would make the
    resulting metrics and provenance misleading.
    """

    if config.extraction.runtime_profile == "model_backed":
        runtime = LockedServiceRuntime.from_environment()
        return _mandatory_unstructured_pipeline(config, runtime)
    return UnstructuredPipeline()


def _quality_evaluator_for(config: DataToKGConfig) -> KnowledgeGraphQualityEvaluator:
    """Build the quality evaluator honouring ``config.evaluation_thresholds``.

    Thresholds are reported, never enforced: they only change what the returned
    quality report calls pass vs. fail.
    """

    thresholds_config = config.evaluation_thresholds
    floors = thresholds_config.floor_overrides()
    overrides = {
        (item.family, item.metric): (item.operator, item.threshold)
        for item in thresholds_config.quality_overrides
    }
    default = VersionedQualityThresholds()
    known_keys = {(item.family, item.metric) for item in default.thresholds}
    unknown_keys = sorted(set(overrides) - known_keys)
    if unknown_keys:
        rendered = ", ".join(
            f"{family}.{metric}" for family, metric in unknown_keys
        )
        raise ValueError(f"unknown quality threshold override(s): {rendered}")
    adjusted = []
    for item in default.thresholds:
        key = (item.family, item.metric)
        if key in overrides:
            operator, threshold = overrides[key]
            adjusted.append(
                item.model_copy(update={"operator": operator, "threshold": threshold})
            )
        elif key in floors:
            adjusted.append(item.model_copy(update={"threshold": floors[key]}))
        else:
            adjusted.append(item)
    if adjusted == default.thresholds:
        return KnowledgeGraphQualityEvaluator(thresholds=default)
    return KnowledgeGraphQualityEvaluator(
        thresholds=VersionedQualityThresholds(
            version=f"{default.version}+user",
            thresholds=adjusted,
        )
    )


def _structured_mappers(
    config: DataToKGConfig,
    source_catalog: SourceCatalog | None = None,
) -> list[Any]:
    """Return the declaration-aware semantic mapper used by product construction."""
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
    service_runtime: LockedServiceRuntime | None = None,
    production_kwargs: dict[str, Any] | None = None,
) -> ConstructionWorkflow:
    """Create a mandatory workflow whose publication decision cannot be disabled."""

    return _workflow_for_source_type(
        source_type,
        config,
        source_catalog=source_catalog,
        graph_evaluation_suite=graph_evaluation_suite,
        publication_contract=publication_contract,
        service_runtime=service_runtime,
        production_kwargs=production_kwargs,
        workflow_class=ConstructionWorkflow,
    )


def _workflow_for_source_type(
    source_type: str | None,
    config: DataToKGConfig | None = None,
    *,
    source_catalog: SourceCatalog | None = None,
    graph_evaluation_suite: VersionedGraphEvaluationSuite | None = None,
    publication_contract: TrustedPublicationBinding | None = None,
    service_runtime: LockedServiceRuntime | None = None,
    production_kwargs: dict[str, Any] | None = None,
    workflow_class: type[ConstructionWorkflow],
) -> ConstructionWorkflow:
    """Build either the strict product workflow or the internal draft workflow."""

    if workflow_class not in {ConstructionWorkflow, DraftConstructionWorkflow}:
        raise TypeError("workflow_class must be a governed construction workflow")

    resolved_config = config or DataToKGConfig()
    runtime = service_runtime or LockedServiceRuntime.from_environment()
    _validate_service_runtime(resolved_config, runtime)
    normalized_type = _normalize_source_type(source_type) or "text"
    validators: list[Any] = [EvidenceRequiredValidator(resolved_config.pipeline.validation_policy)]
    resolvers: list[Any] = [ExactKeyEntityResolver()]
    common_kwargs: dict[str, Any] = {
        "pipeline_name": resolved_config.pipeline.name,
        "config_payload": resolved_config.model_dump(mode="json"),
        "graph_evaluation_suite": graph_evaluation_suite,
        "publication_contract": publication_contract,
        **(production_kwargs or {}),
    }
    if workflow_class is ConstructionWorkflow:
        common_kwargs["graph_bound_evaluator"] = _live_graph_bound_evaluator(
            resolved_config,
            runtime,
        )

    if normalized_type == "mixed":
        unstructured = _mandatory_unstructured_pipeline(resolved_config, runtime)
        return workflow_class(
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
        return workflow_class(
            mappers=_structured_mappers(resolved_config, source_catalog),
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "xml":
        return workflow_class(
            mappers=[DirectTableMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type in {"json", "api_response"}:
        return workflow_class(
            mappers=[JsonPathMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "jsonld":
        return workflow_class(
            mappers=[JSONLDMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "event_payload":
        return workflow_class(
            mappers=[EventGraphMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type == "logs":
        return workflow_class(
            mappers=[LogEventMapper()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type in {"html", "webpage"}:
        return workflow_class(
            mappers=[HTMLMetadataMapper()],
            extractors=[_mandatory_unstructured_pipeline(resolved_config, runtime)],
            resolvers=[],
            validators=validators,
            **common_kwargs,
        )
    return workflow_class(
        extractors=[_mandatory_unstructured_pipeline(resolved_config, runtime)],
        resolvers=[],
        validators=validators,
        **common_kwargs,
    )


def _load_directory(path: Path) -> list[SourceRecord]:
    """Load every inventoried file or fail with the unprocessed paths."""

    sources: list[SourceRecord] = []
    unprocessed: list[str] = []
    supported_files: list[Path] = []
    for child in sorted(path.rglob("*")):
        relative = str(child.relative_to(path))
        if child.is_symlink():
            unprocessed.append(f"{relative} (symbolic link)")
        elif child.is_dir():
            continue
        elif not child.is_file():
            unprocessed.append(f"{relative} (unsupported filesystem entry)")
        elif child.suffix.lower() not in _DIRECTORY_SUFFIXES:
            unprocessed.append(f"{relative} (unsupported suffix)")
        else:
            supported_files.append(child)
    if unprocessed:
        raise ValueError(
            "directory ingestion is incomplete; every input must be processed "
            "or removed under a reviewed exclusion policy. Unprocessed: "
            + ", ".join(unprocessed)
        )
    if not supported_files:
        raise ValueError(f"directory contains no supported input files: {path}")
    for child in supported_files:
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
    if normalized_type == "text":
        return TextLoader().load(path)
    raise ValueError(f"unsupported source type for file loading: {normalized_type!r}")


def _infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return _normalize_source_type(suffix) or "text"


def _normalize_source_type(
    source_type: str | None,
    *,
    require_supported: bool = False,
) -> str | None:
    if source_type is None:
        return None
    normalized = source_type.strip().lower().replace("-", "_")
    canonical = _SOURCE_TYPE_ALIASES.get(normalized)
    if canonical is not None:
        return canonical
    if require_supported:
        supported = ", ".join(sorted(_SOURCE_TYPE_ALIASES))
        raise ValueError(
            f"unsupported source type {source_type!r}; supported values: {supported}"
        )
    return normalized


def _primary_source_type(sources: list[SourceRecord], source_type: str | None) -> str:
    explicit = _normalize_source_type(source_type, require_supported=True)
    if explicit:
        return explicit
    if not sources:
        return "text"
    source_types = {
        _normalize_source_type(source.source_type, require_supported=True)
        or source.source_type
        for source in sources
    }
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

    _TEXT_SOURCE_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
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
    )

    def __init__(self, delegate: Any) -> None:
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
    runtime: LockedServiceRuntime,
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
        entity_labels=config.extraction.entity_labels,
        ollama_base_url=runtime.ollama_base_url,
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
    runtime: LockedServiceRuntime,
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
            uri=runtime.neo4j_uri,
            user=runtime.neo4j_user,
            password=runtime.neo4j_password,
            database=runtime.neo4j_database,
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
    graph_backend = Neo4jGraphWorkloadBackend(
        Neo4jGraphWorkloadConfig(
            uri=runtime.neo4j_uri,
            user=runtime.neo4j_user,
            password=runtime.neo4j_password,
            database=runtime.neo4j_database,
            snapshot_namespace="kg_publication",
        )
    )
    return GraphBoundIntendedUseEvaluator(
        runner=GraphWorkloadRunner(
            graph_backend=graph_backend,
            vector_backend=vector_backend,
        )
    )


def _validate_service_runtime(
    config: DataToKGConfig,
    runtime: LockedServiceRuntime,
) -> None:
    """Prevent config/environment endpoint drift in the mandatory product path."""

    conflicts: list[str] = []
    if config.llm.base_url.rstrip("/") != runtime.ollama_base_url:
        conflicts.append("llm.base_url/OLLAMA_BASE_URL")
    if config.llm.model != runtime.ollama_model:
        conflicts.append("llm.model/OLLAMA_MODEL")
    if conflicts:
        raise ValueError(
            "configuration conflicts with the locked service runtime: "
            + ", ".join(conflicts)
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


def _evaluation_suite(
    suite: EvaluationSuiteInput,
) -> VersionedGraphEvaluationSuite | None:
    """Resolve one sealed graph-bound suite without changing its trust status."""

    if suite is None:
        return None
    if isinstance(suite, VersionedGraphEvaluationSuite):
        return suite
    return VersionedGraphEvaluationSuite.from_path(suite)


def _validated_technique(source_format: str, technique: str | None) -> str | None:
    """Validate technique names instead of silently selecting another mapper."""

    normalized = (technique or "").strip().lower() or None
    allowed = _TECHNIQUES_BY_SOURCE_TYPE.get(
        source_format,
        _UNSTRUCTURED_TECHNIQUES if source_format in _TEXT_SOURCE_TYPES else frozenset(),
    )
    if normalized is not None and normalized not in allowed:
        raise ValueError(
            f"technique {technique!r} is not valid for {source_format!r}; "
            f"allowed values: {', '.join(sorted(allowed)) or '(none)'}"
        )
    return normalized


def _is_source_record_sequence(value: object) -> TypeGuard[Sequence[SourceRecord]]:
    if isinstance(value, str | Path):
        return False
    return isinstance(value, Sequence) and all(isinstance(item, SourceRecord) for item in value)


__all__ = [
    "construct_kg",
    "convert",
    "default_workflow_for_source_type",
    "load_sources",
    "profile_dataset",
]
