"""
Workflow implementations for automation layer.

This package provides workflow classes that integrate session management,
agent interaction, and GUI communication.
"""

from .designspec_workflow import (
    DesignSpecWorkflow,
    WorkflowEvent,
    WorkflowEventType,
)

__all__ = [
    "DesignSpecWorkflow",
    "WorkflowEvent",
    "WorkflowEventType",
]
