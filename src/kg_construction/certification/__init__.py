"""Certification: the reproducible locked-artifact hard pass.

The release is certified only when a clean machine can bootstrap entirely from
committed locks (``uv.lock``, ``models.lock.yaml``, ``certification/services.lock.yaml``)
and repeat without package, model, or container downloads. Mandatory remote
identity/reconciliation certifiers remain live network checks. This package
enforces that distinction with hard failures — no warnings, no
``--skip-services``, no silent CPU fallback, and no model-download-on-first-use.
"""

from kg_construction.certification.execution import (
    ExecutionValidation,
    validate_execution_report,
)
from kg_construction.certification.preflight import (
    CertCheck,
    check_capability_manifest,
    check_execution_plan,
    check_lockfiles_present,
    check_makefile_certification_is_strict,
    check_model_manifest,
    check_no_cpu_fallback,
    check_no_product_optional_extras,
    check_offline_environment,
    check_profile_contract,
    check_service_lock,
    run_preflight,
)
from kg_construction.certification.profiles import (
    PlatformProfile,
    hardware_violations,
    select_profile,
)
from kg_construction.certification.runner import dispatch_capabilities

__all__ = [
    "CertCheck",
    "ExecutionValidation",
    "PlatformProfile",
    "check_capability_manifest",
    "check_execution_plan",
    "check_lockfiles_present",
    "check_makefile_certification_is_strict",
    "check_model_manifest",
    "check_no_cpu_fallback",
    "check_no_product_optional_extras",
    "check_offline_environment",
    "check_profile_contract",
    "check_service_lock",
    "dispatch_capabilities",
    "hardware_violations",
    "run_preflight",
    "select_profile",
    "validate_execution_report",
]
