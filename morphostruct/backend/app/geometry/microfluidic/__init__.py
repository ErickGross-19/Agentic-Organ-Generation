"""
Microfluidic scaffold generators for tissue engineering and lab-on-chip applications.

This module provides generators for:
- Organ-on-chip systems with microfluidic channels and tissue chambers
- Gradient scaffolds with continuous porosity/stiffness variation
- Perfusable vascular networks following Murray's law for thick tissue constructs
"""

from .organ_on_chip import (
    OrganOnChipParams,
    generate_organ_on_chip,
    generate_organ_on_chip_from_dict
)

from .gradient_scaffold import (
    GradientScaffoldParams,
    generate_gradient_scaffold,
    generate_gradient_scaffold_from_dict
)

from .perfusable_network import (
    PerfusableNetworkParams,
    generate_perfusable_network,
    generate_perfusable_network_from_dict
)

__all__ = [
    # Organ-on-chip
    'OrganOnChipParams',
    'generate_organ_on_chip',
    'generate_organ_on_chip_from_dict',

    # Gradient scaffold
    'GradientScaffoldParams',
    'generate_gradient_scaffold',
    'generate_gradient_scaffold_from_dict',

    # Perfusable network
    'PerfusableNetworkParams',
    'generate_perfusable_network',
    'generate_perfusable_network_from_dict',
]
