"""
Tubular organ scaffold generators.

Specialized generators for various tubular biological structures:
- Simple conduit (basic hollow tube)
- Blood vessels (tri-layer vascular walls)
- Nerve conduits (multi-channel guidance tubes)
- Spinal cord (gray matter core with white matter channels)
- Bladder (multi-layer dome structures)
- Trachea (C-shaped cartilage rings)
"""

from .simple_conduit import (
    TubularConduitParams,
    generate_tubular_conduit,
    generate_tubular_conduit_from_dict,
)
from .blood_vessel import (
    BloodVesselParams,
    generate_blood_vessel,
    generate_blood_vessel_from_dict
)
from .nerve_conduit import (
    NerveConduitParams,
    generate_nerve_conduit,
    generate_nerve_conduit_from_dict
)
from .spinal_cord import (
    SpinalCordParams,
    generate_spinal_cord,
    generate_spinal_cord_from_dict
)
from .bladder import (
    BladderParams,
    generate_bladder,
    generate_bladder_from_dict
)
from .trachea import (
    TracheaParams,
    generate_trachea,
    generate_trachea_from_dict
)

__all__ = [
    # Simple conduit
    'TubularConduitParams',
    'generate_tubular_conduit',
    'generate_tubular_conduit_from_dict',
    # Blood vessel
    'BloodVesselParams',
    'generate_blood_vessel',
    'generate_blood_vessel_from_dict',
    # Nerve conduit
    'NerveConduitParams',
    'generate_nerve_conduit',
    'generate_nerve_conduit_from_dict',
    # Spinal cord
    'SpinalCordParams',
    'generate_spinal_cord',
    'generate_spinal_cord_from_dict',
    # Bladder
    'BladderParams',
    'generate_bladder',
    'generate_bladder_from_dict',
    # Trachea
    'TracheaParams',
    'generate_trachea',
    'generate_trachea_from_dict',
]
