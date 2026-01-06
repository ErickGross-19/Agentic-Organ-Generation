"""
Post-Embedding Validity Checks

These checks validate the structure AFTER it is embedded into a domain.
They focus on manufacturability, connectivity, and physical constraints.

Modules:
    - connectivity_checks: Ports open, reachable fraction, trapped components
    - printability_checks: Min channel diameter, wall thickness, overhangs
    - domain_checks: Outlets not covered, channel continuity
"""

from .connectivity_checks import (
    check_port_accessibility,
    check_trapped_fluid,
    check_channel_continuity,
    run_all_connectivity_checks,
)
from .printability_checks import (
    check_min_channel_diameter,
    check_wall_thickness,
    check_unsupported_features,
    run_all_printability_checks,
)
from .domain_checks import (
    check_outlets_open,
    check_domain_coverage,
    run_all_domain_checks,
)

__all__ = [
    # Connectivity checks
    "check_port_accessibility",
    "check_trapped_fluid",
    "check_channel_continuity",
    "run_all_connectivity_checks",
    # Printability checks
    "check_min_channel_diameter",
    "check_wall_thickness",
    "check_unsupported_features",
    "run_all_printability_checks",
    # Domain checks
    "check_outlets_open",
    "check_domain_coverage",
    "run_all_domain_checks",
]
