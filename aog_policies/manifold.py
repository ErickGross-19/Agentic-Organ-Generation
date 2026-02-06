"""
Policy for manifold-based scaffold generators.

Controls which MorphoStruct geometry generator to use and its parameters.
Used with build_type="manifold_generator" in DesignSpec components.

UNIT CONVENTION
---------------
MorphoStruct generators work in millimeters internally.
When convert_units is True (default), output is automatically converted
to meters for compatibility with the DesignSpec pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class ManifoldGeneratorPolicy:
    """
    Policy controlling manifold generator behavior.

    JSON Schema:
    {
        "generator_type": str,
        "generator_params": dict,
        "convert_units": bool
    }

    Attributes
    ----------
    generator_type : str
        Name of the geometry generator (e.g., "gyroid", "trabecular_bone").
        Must be a key in the ManifoldBackend's GENERATOR_REGISTRY.
    generator_params : dict
        Parameters passed to the generator's ``generate_*_from_dict`` function.
        Keys and values are generator-specific.
    convert_units : bool
        If True, convert output vertices from millimeters to meters.
        Should be True when used inside the DesignSpec pipeline.
    """
    generator_type: str = ""
    generator_params: Dict[str, Any] = field(default_factory=dict)
    convert_units: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ManifoldGeneratorPolicy":
        return ManifoldGeneratorPolicy(
            **{
                k: v
                for k, v in d.items()
                if k in ManifoldGeneratorPolicy.__dataclass_fields__
            }
        )
