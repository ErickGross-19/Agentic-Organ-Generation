"""
V4 Single Agent Organ Generation (Legacy)

This is the V4 implementation kept for reference only.
V4 is no longer the default - use V5 instead.

V4 introduced:
- Two-layer architecture: Minimal Viable Spec (MVS) + Adaptive Modules
- Topology-first gating: PATH vs TREE vs BACKBONE as FIRST question
- Domain: explicit first, defaults only as fallback
- Ports: first-class and topology-dependent requirements
- Topology-specific "ready to generate" thresholds
"""

from .workflow_v4 import SingleAgentOrganGeneratorV4

__all__ = ["SingleAgentOrganGeneratorV4"]
