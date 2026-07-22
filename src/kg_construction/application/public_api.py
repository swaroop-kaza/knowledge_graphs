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
from kg_construction.application.production import build_production_services
from kg_construction.application.registry import PluginRegistry, default_registry
from kg_construction.application.workflow import ConstructionResult, ConstructionWorkflow
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
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
    workflow: ConstructionWorkflow | None = None,
    source_name: str | None = None,
) -> ConstructionResult:
    """Construct property, RDF, and evidence graphs from one input.

    The default workflow is intentionally local-first and deterministic. Provide
    a custom ``ConstructionWorkflow`` when a project needs specialized mappers,
    extractors, resolvers, validators, or graph builders.
    """

    sources = load_sources(input_data, source_type=source_type, source_name=source_name)
    if workflow is None:
        resolved_config = _data_config(config)
        primary_source_type = _primary_source_type(sources, source_type)
        production_kwargs = build_production_services(resolved_config).workflow_kwargs()
        workflow = default_workflow_for_source_type(
            primary_source_type,
            resolved_config,
            production_kwargs=production_kwargs,
        )
    return workflow.run(sources)


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


def default_workflow_for_source_type(
    source_type: str | None,
    config: DataToKGConfig | None = None,
    *,
    production_kwargs: dict[str, Any] | None = None,
) -> ConstructionWorkflow:
    """Create the conservative default workflow used by ``construct_kg``."""

    resolved_config = config or DataToKGConfig()
    normalized_type = _normalize_source_type(source_type) or "text"
    validators: list[Any] = [EvidenceRequiredValidator(resolved_config.pipeline.validation_policy)]
    resolvers: list[Any] = [ExactKeyEntityResolver()]
    common_kwargs: dict[str, Any] = {
        "pipeline_name": resolved_config.pipeline.name,
        "config_payload": resolved_config.model_dump(mode="json"),
        **(production_kwargs or {}),
    }

    if normalized_type == "mixed":
        return ConstructionWorkflow(
            mappers=[_DefaultSourceMapperRouter()],
            extractors=[_DefaultSourceExtractorRouter()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    if normalized_type in {"csv", "excel", "sql", "structured_api", "xml", "chat_transcript"}:
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
            extractors=[RegexEntityExtractor()],
            resolvers=resolvers,
            validators=validators,
            **common_kwargs,
        )
    return ConstructionWorkflow(
        extractors=[RegexEntityExtractor()],
        resolvers=resolvers,
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

    def __init__(self) -> None:
        direct = DirectTableMapper()
        self._delegates: dict[str, Any] = {
            "api_response": JsonPathMapper(),
            "chat_transcript": direct,
            "csv": direct,
            "event_payload": EventGraphMapper(),
            "excel": direct,
            "html": HTMLMetadataMapper(),
            "json": JsonPathMapper(),
            "jsonld": JSONLDMapper(),
            "logs": LogEventMapper(),
            "sql": direct,
            "structured_api": direct,
            "webpage": HTMLMetadataMapper(),
            "xml": direct,
        }

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
        "email",
        "html",
        "markdown",
        "pdf",
        "report",
        "scientific_paper",
        "text",
        "webpage",
    }

    def __init__(self) -> None:
        self._delegate = RegexEntityExtractor()

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


def _data_config(config: ConfigInput) -> DataToKGConfig:
    if config is None:
        return DataToKGConfig()
    if isinstance(config, DataToKGConfig):
        return config
    return DataToKGConfig.from_yaml(config)


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
