"""
Dental and craniofacial scaffold generators.

This module provides specialized scaffold generators for dental and craniofacial
tissue engineering applications:

- dentin_pulp: Tooth-like structures with outer dentin shell and inner pulp chamber
- ear_auricle: Complex curved cartilage frameworks for ear reconstruction
- nasal_septum: Curved cartilage sheets for nasal reconstruction

Each generator follows the standard pattern:
- Dataclass parameters with sensible defaults
- generate_X() function returning (manifold, stats)
- generate_X_from_dict() for API compatibility
"""

from .dentin_pulp import (
    DentinPulpParams,
    generate_dentin_pulp,
    generate_dentin_pulp_from_dict
)
from .ear_auricle import (
    EarAuricleParams,
    generate_ear_auricle,
    generate_ear_auricle_from_dict
)
from .nasal_septum import (
    NasalSeptumParams,
    generate_nasal_septum,
    generate_nasal_septum_from_dict
)

__all__ = [
    # Dentin-Pulp
    'DentinPulpParams',
    'generate_dentin_pulp',
    'generate_dentin_pulp_from_dict',
    # Ear Auricle
    'EarAuricleParams',
    'generate_ear_auricle',
    'generate_ear_auricle_from_dict',
    # Nasal Septum
    'NasalSeptumParams',
    'generate_nasal_septum',
    'generate_nasal_septum_from_dict',
]
