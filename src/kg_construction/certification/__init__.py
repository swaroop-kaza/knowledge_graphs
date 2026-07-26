"""Certification and release-authorization boundary.

The package facade is deliberately lazy: importing one publication-contract
type must not initialize every model, store, registry adapter, or capability
certifier. Every exported symbol still resolves to its canonical owning module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "CertCheck": ("kg_construction.certification.preflight", "CertCheck"),
    "ExecutionValidation": (
        "kg_construction.certification.execution",
        "ExecutionValidation",
    ),
    "PlatformProfile": (
        "kg_construction.certification.profiles",
        "PlatformProfile",
    ),
    "PluginRegistry": (
        "kg_construction.certification.registry",
        "PluginRegistry",
    ),
    "PublicationContractAssessment": (
        "kg_construction.certification.publication_contract",
        "PublicationContractAssessment",
    ),
    "PublicationContractRegistry": (
        "kg_construction.certification.publication_contract",
        "PublicationContractRegistry",
    ),
    "SUPPORT_LEVEL_DEFINITIONS": (
        "kg_construction.certification.registry",
        "SUPPORT_LEVEL_DEFINITIONS",
    ),
    "SUPPORT_LEVEL_LABELS": (
        "kg_construction.certification.registry",
        "SUPPORT_LEVEL_LABELS",
    ),
    "SupportLevel": (
        "kg_construction.certification.registry",
        "SupportLevel",
    ),
    "TechniqueDescriptor": (
        "kg_construction.certification.registry",
        "TechniqueDescriptor",
    ),
    "TrustedPublicationBinding": (
        "kg_construction.certification.publication_contract",
        "TrustedPublicationBinding",
    ),
    "UntrustedPublicationContractError": (
        "kg_construction.certification.publication_contract",
        "UntrustedPublicationContractError",
    ),
    "check_capability_manifest": (
        "kg_construction.certification.preflight",
        "check_capability_manifest",
    ),
    "check_execution_plan": (
        "kg_construction.certification.preflight",
        "check_execution_plan",
    ),
    "check_lockfiles_present": (
        "kg_construction.certification.preflight",
        "check_lockfiles_present",
    ),
    "check_makefile_certification_is_strict": (
        "kg_construction.certification.preflight",
        "check_makefile_certification_is_strict",
    ),
    "check_model_manifest": (
        "kg_construction.certification.preflight",
        "check_model_manifest",
    ),
    "check_no_cpu_fallback": (
        "kg_construction.certification.preflight",
        "check_no_cpu_fallback",
    ),
    "check_no_product_optional_extras": (
        "kg_construction.certification.preflight",
        "check_no_product_optional_extras",
    ),
    "check_offline_environment": (
        "kg_construction.certification.preflight",
        "check_offline_environment",
    ),
    "check_profile_contract": (
        "kg_construction.certification.preflight",
        "check_profile_contract",
    ),
    "check_service_lock": (
        "kg_construction.certification.preflight",
        "check_service_lock",
    ),
    "default_registry": (
        "kg_construction.certification.registry",
        "default_registry",
    ),
    "dispatch_capabilities": (
        "kg_construction.certification.runner",
        "dispatch_capabilities",
    ),
    "hardware_violations": (
        "kg_construction.certification.profiles",
        "hardware_violations",
    ),
    "research_registry": (
        "kg_construction.certification.registry",
        "research_registry",
    ),
    "run_preflight": (
        "kg_construction.certification.preflight",
        "run_preflight",
    ),
    "select_profile": (
        "kg_construction.certification.profiles",
        "select_profile",
    ),
    "validate_execution_report": (
        "kg_construction.certification.execution",
        "validate_execution_report",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve a certification symbol without eagerly loading unrelated stacks."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
