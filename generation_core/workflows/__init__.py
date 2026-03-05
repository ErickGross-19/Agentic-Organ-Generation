"""
Condensed workflow abstractions for generation_core.

This package provides cohesive, single-entry-point workflow modules
that consolidate previously fragmented implementations.

Modules
-------
space_colonization_workflow : Condensed space colonization workflow
    Consolidates the 4 previously scattered entry points into a single
    cohesive abstraction with internal variant selection.
"""

from .space_colonization_workflow import (
    SpaceColonizationWorkflow,
    WorkflowMode,
)

__all__ = [
    "SpaceColonizationWorkflow",
    "WorkflowMode",
]
