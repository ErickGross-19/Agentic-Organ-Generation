"""
Operation Report Module

B2 FIX: Re-export OperationReport at the import path expected by the runner contract.

Tests expect:
    from generation.core.report import OperationReport

This module re-exports OperationReport from aog_policies.base.
"""

from aog_policies.base import OperationReport

__all__ = ["OperationReport"]
