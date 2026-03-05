"""
Soft tissue scaffold generators.

Provides specialized generators for various soft tissue types:
- multilayer_skin: Epidermis/dermis/hypodermis with vascular channels
- skeletal_muscle: Aligned myofiber channels with pennate architectures
- cornea: Curved dome scaffold with lamellar structure
- adipose: Honeycomb structure for adipocyte housing
"""

from .multilayer_skin import (
    MultilayerSkinParams,
    generate_multilayer_skin,
    generate_multilayer_skin_from_dict
)
from .skeletal_muscle import (
    SkeletalMuscleParams,
    generate_skeletal_muscle,
    generate_skeletal_muscle_from_dict
)
from .cornea import (
    CorneaParams,
    generate_cornea,
    generate_cornea_from_dict
)
from .adipose import (
    AdiposeTissueParams,
    generate_adipose_tissue,
    generate_adipose_tissue_from_dict
)

__all__ = [
    # Multilayer skin
    'MultilayerSkinParams',
    'generate_multilayer_skin',
    'generate_multilayer_skin_from_dict',
    # Skeletal muscle
    'SkeletalMuscleParams',
    'generate_skeletal_muscle',
    'generate_skeletal_muscle_from_dict',
    # Cornea
    'CorneaParams',
    'generate_cornea',
    'generate_cornea_from_dict',
    # Adipose tissue
    'AdiposeTissueParams',
    'generate_adipose_tissue',
    'generate_adipose_tissue_from_dict',
]
