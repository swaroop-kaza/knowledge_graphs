"""One-cycle compatibility guarantees for former application import paths."""

from kg_construction.application.config import DataToKGConfig as LegacyConfig
from kg_construction.application.dataset_profiler import (
    DatasetProfiler as LegacyProfiler,
)
from kg_construction.application.publication_contract import (
    PublicationContractRegistry as LegacyPublicationRegistry,
)
from kg_construction.application.workflow import (
    ConstructionWorkflow as LegacyWorkflow,
)
from kg_construction.certification.publication_contract import (
    PublicationContractRegistry,
)
from kg_construction.construction.config import DataToKGConfig
from kg_construction.construction.profiling import DatasetProfiler
from kg_construction.construction.workflow import ConstructionWorkflow


def test_application_compatibility_exports_preserve_object_identity() -> None:
    assert LegacyConfig is DataToKGConfig
    assert LegacyProfiler is DatasetProfiler
    assert LegacyPublicationRegistry is PublicationContractRegistry
    assert LegacyWorkflow is ConstructionWorkflow
