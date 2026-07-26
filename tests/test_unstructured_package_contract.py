"""Package-ownership and compatibility checks for unstructured construction."""

from kg_construction.application.mandatory_unstructured_runtime import (
    MandatoryUnstructuredRuntime as ApplicationMandatoryUnstructuredRuntime,
)
from kg_construction.application.unstructured_pipeline import (
    UnstructuredPipeline as ApplicationUnstructuredPipeline,
)
from kg_construction.unstructured.pipeline import UnstructuredPipeline
from kg_construction.unstructured.runtime import MandatoryUnstructuredRuntime


def test_unstructured_implementations_are_owned_by_feature_package() -> None:
    assert UnstructuredPipeline.__module__ == "kg_construction.unstructured.pipeline"
    assert (
        MandatoryUnstructuredRuntime.__module__
        == "kg_construction.unstructured.runtime"
    )


def test_application_imports_are_identity_preserving_compatibility_shims() -> None:
    assert ApplicationUnstructuredPipeline is UnstructuredPipeline
    assert ApplicationMandatoryUnstructuredRuntime is MandatoryUnstructuredRuntime
