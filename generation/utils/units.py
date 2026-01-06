"""
Unit conversion utilities for vascular library.

The library uses DIMENSIONLESS internal units where 1 internal unit = 1 output unit.
All internal calculations use dimensionless values (1 is 1), and scaling to user-specified
units (mm, cm, m, etc.) happens only at export/output time.

When a user specifies output_units="mm", then 1 internal unit = 1 mm in the output STL file.
This ensures consistent behavior regardless of the unit system chosen.
"""

from dataclasses import dataclass
from typing import Union, Literal
import numpy as np

# Internal units are dimensionless - 1 is 1
# The CANONICAL_UNIT specifies what 1 internal unit represents in the output
CANONICAL_UNIT = "mm"

# Supported unit types
UnitType = Literal["m", "mm", "cm", "um"]

_TO_METERS = {
    "m": 1.0,
    "mm": 0.001,
    "cm": 0.01,
    "um": 1e-6,
}

_FROM_METERS = {
    "m": 1.0,
    "mm": 1000.0,
    "cm": 100.0,
    "um": 1e6,
}


def to_si_length(value: Union[float, np.ndarray], from_unit: str = CANONICAL_UNIT) -> Union[float, np.ndarray]:
    """
    Convert length from specified unit to SI (meters).
    
    Parameters
    ----------
    value : float or ndarray
        Length value(s) to convert
    from_unit : str
        Source unit ('mm', 'cm', 'm', 'um'). Default: 'mm'
        
    Returns
    -------
    float or ndarray
        Length in meters (SI)
        
    Examples
    --------
    >>> to_si_length(5.0, 'mm')  # 5mm to meters
    0.005
    >>> to_si_length(100.0, 'mm')  # 100mm to meters
    0.1
    """
    if from_unit not in _TO_METERS:
        raise ValueError(f"Unknown unit '{from_unit}'. Supported: {list(_TO_METERS.keys())}")
    
    return value * _TO_METERS[from_unit]


def from_si_length(value: Union[float, np.ndarray], to_unit: str = CANONICAL_UNIT) -> Union[float, np.ndarray]:
    """
    Convert length from SI (meters) to specified unit.
    
    Parameters
    ----------
    value : float or ndarray
        Length value(s) in meters
    to_unit : str
        Target unit ('mm', 'cm', 'm', 'um'). Default: 'mm'
        
    Returns
    -------
    float or ndarray
        Length in target unit
        
    Examples
    --------
    >>> from_si_length(0.005, 'mm')  # 0.005m to mm
    5.0
    >>> from_si_length(0.1, 'mm')  # 0.1m to mm
    100.0
    """
    if to_unit not in _FROM_METERS:
        raise ValueError(f"Unknown unit '{to_unit}'. Supported: {list(_FROM_METERS.keys())}")
    
    return value * _FROM_METERS[to_unit]


def convert_length(value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Convert length between any supported units.
    
    Parameters
    ----------
    value : float or ndarray
        Length value(s) to convert
    from_unit : str
        Source unit
    to_unit : str
        Target unit
        
    Returns
    -------
    float or ndarray
        Length in target unit
        
    Examples
    --------
    >>> convert_length(100.0, 'mm', 'cm')
    10.0
    >>> convert_length(5.0, 'cm', 'mm')
    50.0
    """
    if from_unit == to_unit:
        return value
    
    meters = to_si_length(value, from_unit)
    return from_si_length(meters, to_unit)


def detect_unit(value: float, context: str = "length") -> str:
    """
    Auto-detect likely unit based on magnitude.
    
    This is a heuristic for backward compatibility with meter-based code.
    
    Parameters
    ----------
    value : float
        A typical length value from the data
    context : str
        Context hint ('length', 'radius', 'domain_size')
        
    Returns
    -------
    str
        Likely unit ('m' or 'mm')
        
    Examples
    --------
    >>> detect_unit(0.005)  # Likely 5mm in meters
    'm'
    >>> detect_unit(5.0)  # Likely 5mm in mm
    'mm'
    >>> detect_unit(120.0)  # Likely 120mm in mm
    'mm'
    """
    abs_value = abs(value)
    
    if context in ["radius", "length"]:
        if abs_value < 0.1:
            return "m"  # Likely meters (e.g., 0.005 = 5mm)
        else:
            return "mm"  # Likely mm (e.g., 5.0 = 5mm)
    
    elif context == "domain_size":
        if abs_value < 1.0:
            return "m"  # Likely meters (e.g., 0.12 = 120mm)
        else:
            return "mm"  # Likely mm (e.g., 120 = 120mm)
    
    if abs_value < 1.0:
        return "m"
    else:
        return "mm"


def warn_if_legacy_units(value: float, context: str = "length", param_name: str = "value") -> None:
    """
    Warn if value appears to be in legacy meter units.
    
    Parameters
    ----------
    value : float
        Value to check
    context : str
        Context hint
    param_name : str
        Parameter name for warning message
    """
    import warnings
    
    detected = detect_unit(value, context)
    if detected == "m":
        warnings.warn(
            f"Parameter '{param_name}' value {value} appears to be in meters (legacy). "
            f"The library now uses millimeters as the default unit. "
            f"If this is intentional, multiply by 1000 to convert to mm. "
            f"To silence this warning, explicitly set units='m' in your spec.",
            UserWarning,
            stacklevel=3
        )


@dataclass
class UnitContext:
    """
    Context for unit handling in the library.
    
    The library uses DIMENSIONLESS internal units where 1 internal unit = 1 output unit.
    All internal calculations use dimensionless values (1 is 1), and scaling to 
    user-specified units happens only at export/output time.
    
    When output_units="mm", then 1 internal unit = 1 mm in the output STL file.
    
    Attributes
    ----------
    output_units : str
        Units for exported files (STL, JSON, etc.). Default: "mm"
        Supported: "m", "mm", "cm", "um"
    
    Examples
    --------
    >>> ctx = UnitContext(output_units="mm")
    >>> # Internal value of 50 (dimensionless) becomes 50mm in output
    >>> ctx.to_output(50.0)
    50.0
    >>> 
    >>> ctx = UnitContext(output_units="m")
    >>> # Internal value of 50 (dimensionless) becomes 0.05m in output
    >>> ctx.to_output(50.0)
    0.05
    """
    
    output_units: str = "mm"
    
    def __post_init__(self):
        """Validate output_units."""
        if self.output_units not in _TO_METERS:
            raise ValueError(
                f"Unknown output_units '{self.output_units}'. "
                f"Supported: {list(_TO_METERS.keys())}"
            )
    
    @property
    def scale_factor(self) -> float:
        """
        Get scale factor for converting internal units to output units.
        
        Internal units are dimensionless where 1 = 1mm equivalent.
        This factor converts to the requested output units.
        
        Returns
        -------
        float
            Scale factor to multiply internal values by for output.
            - output_units="mm": 1.0 (no scaling)
            - output_units="m": 0.001 (divide by 1000)
            - output_units="cm": 0.1 (divide by 10)
            - output_units="um": 1000.0 (multiply by 1000)
        """
        # Internal units are mm-equivalent, so convert from mm to output
        return convert_length(1.0, "mm", self.output_units)
    
    def to_output(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert internal dimensionless value to output units.
        
        Parameters
        ----------
        value : float or ndarray
            Internal dimensionless value(s)
            
        Returns
        -------
        float or ndarray
            Value(s) in output units
        """
        return value * self.scale_factor
    
    def from_output(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert value from output units to internal dimensionless units.
        
        Parameters
        ----------
        value : float or ndarray
            Value(s) in output units
            
        Returns
        -------
        float or ndarray
            Internal dimensionless value(s)
        """
        return value / self.scale_factor
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "output_units": self.output_units,
            "scale_factor": self.scale_factor,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "UnitContext":
        """Create from dictionary."""
        return cls(output_units=d.get("output_units", "mm"))
    
    def scale_mesh(self, mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
        """
        Scale a mesh from internal units to output units.
        
        Creates a copy of the mesh with scaled vertices.
        Does NOT modify the original mesh.
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh in internal units
            
        Returns
        -------
        trimesh.Trimesh
            New mesh with vertices scaled to output units
        """
        import trimesh
        
        # Create a copy to avoid modifying the original
        scaled_mesh = mesh.copy()
        scaled_mesh.vertices = scaled_mesh.vertices * self.scale_factor
        return scaled_mesh
    
    def get_metadata(self) -> dict:
        """
        Get metadata dict describing the unit context.
        
        Useful for including in output files (JSON sidecar, etc.)
        to document what units were used.
        
        Returns
        -------
        dict
            Metadata including units and scale factor
        """
        return {
            "units": self.output_units,
            "scale_factor_applied": self.scale_factor,
            "internal_units": "dimensionless (1 = 1mm equivalent)",
        }


# Default unit context (mm output)
DEFAULT_UNIT_CONTEXT = UnitContext(output_units="mm")
