"""
Design specifications and evaluation results for LLM-driven vascular design.

DEPRECATION NOTICE
------------------
This module (generation.specs) is DEPRECATED and will be moved to legacy/ in a future version.

For new code, use:
    - JSON DesignSpec files loaded via `designspec.DesignSpec`
    - `designspec.DesignSpecRunner` for full pipeline orchestration
    - `aog_policies` for all policy configuration

This module is kept only for backward compatibility with existing automation/ and examples/ code.

UNIT CONVENTIONS
----------------
All spec classes use METERS internally. See generation/specs/compile.py for details on:
- Spec units (meters)
- Runtime units (meters)
- Output units (configurable, default mm)
- Coordinate frame conventions

Use compile_domain() to convert spec classes to runtime domain objects.
"""

import warnings

warnings.warn(
    "generation.specs is deprecated. For new code, use JSON DesignSpec files loaded via "
    "designspec.DesignSpec and designspec.DesignSpecRunner for pipeline orchestration. "
    "This module will be moved to legacy/ in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from .design_spec import (
    DomainSpec,
    EllipsoidSpec,
    BoxSpec,
    InletSpec,
    OutletSpec,
    ColonizationSpec,
    TreeSpec,
    DualTreeSpec,
    DesignSpec,
)

from .compile import (
    compile_domain,
    make_translation_transform,
    make_rotation_x_transform,
    make_rotation_y_transform,
    make_rotation_z_transform,
)

__all__ = [
    "DomainSpec",
    "EllipsoidSpec",
    "BoxSpec",
    "InletSpec",
    "OutletSpec",
    "ColonizationSpec",
    "TreeSpec",
    "DualTreeSpec",
    "DesignSpec",
    "compile_domain",
    "make_translation_transform",
    "make_rotation_x_transform",
    "make_rotation_y_transform",
    "make_rotation_z_transform",
]

from .eval_result import (
    CoverageMetrics,
    FlowMetrics,
    StructureMetrics,
    ValidityMetrics,
    EvalScores,
    EvalResult,
)

__all__.extend([
    "CoverageMetrics",
    "FlowMetrics",
    "StructureMetrics",
    "ValidityMetrics",
    "EvalScores",
    "EvalResult",
])
