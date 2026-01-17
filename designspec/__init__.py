"""
DesignSpec - Declarative specification and runner for vascular network generation.

This package provides a unified, JSON-driven specification format for defining
vascular network generation pipelines with strict versioning, reproducibility,
and policy-driven execution.

Core Components:
    - DesignSpec: Main specification loader/validator/normalizer
    - DesignSpecRunner: Pipeline executor with staged execution
    - RunnerContext: Caching and artifact management
    - RunReport: Reproducibility metadata and execution reports

Usage:
    from designspec import DesignSpec, DesignSpecRunner

    # Load and validate a spec
    spec = DesignSpec.from_json("my_spec.json")

    # Run the pipeline
    runner = DesignSpecRunner(spec)
    result = runner.run()

    # Access outputs
    print(result.run_report.to_json())

Schema Version: aog_designspec 1.0.0

UNIT CONVENTIONS
----------------
All geometric values are normalized to METERS internally.
Input units are specified via meta.input_units and converted on load.
"""

from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    is_version_compatible,
    SchemaValidationError,
)

from .spec import (
    DesignSpec,
    DesignSpecError,
    DesignSpecValidationError,
)

from .context import (
    RunnerContext,
    ArtifactStore,
)

from .plan import (
    ExecutionPlan,
    Stage,
    STAGE_ORDER,
)

from .runner import (
    DesignSpecRunner,
    RunnerResult,
    StageReport,
    run_spec,
)

from .reports.run_report import (
    RunReport,
    MetaInfo,
    EnvInfo,
    HashInfo,
)

__version__ = "1.0.0"

__all__ = [
    # Schema
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SUPPORTED_VERSIONS",
    "is_version_compatible",
    "SchemaValidationError",
    # Spec
    "DesignSpec",
    "DesignSpecError",
    "DesignSpecValidationError",
    # Context
    "RunnerContext",
    "ArtifactStore",
    # Plan
    "ExecutionPlan",
    "Stage",
    "STAGE_ORDER",
    # Runner
    "DesignSpecRunner",
    "RunnerResult",
    "StageReport",
    "run_spec",
    # Reports
    "RunReport",
    "MetaInfo",
    "EnvInfo",
    "HashInfo",
]
