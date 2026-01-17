"""
Reports package for DesignSpec.

This package provides structured reports for reproducibility and
metadata tracking during DesignSpec execution.
"""

from .run_report import (
    RunReport,
    EnvInfo,
    HashInfo,
    MetaInfo,
)

from .serializers import (
    to_json,
    from_json,
    make_json_safe,
    compute_content_hash,
)

__all__ = [
    # Run report
    "RunReport",
    "EnvInfo",
    "HashInfo",
    "MetaInfo",
    # Serializers
    "to_json",
    "from_json",
    "make_json_safe",
    "compute_content_hash",
]
