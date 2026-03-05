"""
Liver vascular network generator.

Generates realistic arterial and venous trees inside a liver-shaped volume,
following biophysical rules like Murray's law and space-filling constraints.
"""

from .config import LiverVascularConfig
from .growth import generate_liver_vasculature

__all__ = [
    "LiverVascularConfig",
    "generate_liver_vasculature",
]
