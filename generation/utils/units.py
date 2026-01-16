"""
Unit conversion utilities for vascular library.

INTERNAL UNITS: The library uses METER-SCALE values internally.
All coordinates, lengths, and radii are stored in meters throughout the codebase.

For example:
- EllipsoidSpec defaults to semi_axes=(0.05, 0.045, 0.035) which is 50mm, 45mm, 35mm
- A vessel radius of 0.002 represents 2mm
- A segment length of 0.01 represents 10mm

OUTPUT UNITS: At export time (STL, JSON, etc.), internal meter values are converted
to the user-specified output_units via UnitContext:
- output_units="mm" (default): internal 0.05 becomes 50.0 in output
- output_units="m": internal 0.05 stays 0.05 in output
- output_units="cm": internal 0.05 becomes 5.0 in output

This design ensures consistent internal calculations while allowing flexible output formats.
"""

from dataclasses import dataclass
from typing import Union, Literal
import numpy as np

# Internal units are meter-scale (legacy convention from the codebase)
# The INTERNAL_UNIT specifies what internal coordinates represent
INTERNAL_UNIT = "m"

# Default output unit for STL exports and user-facing values
DEFAULT_OUTPUT_UNIT = "mm"

# CANONICAL_UNIT is kept for backward compatibility but INTERNAL_UNIT is the truth
# DEPRECATED: Use INTERNAL_UNIT for internal coordinates, DEFAULT_OUTPUT_UNIT for exports
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


def to_meters(value: Union[float, np.ndarray], units: str) -> Union[float, np.ndarray]:
    """
    Convert value from specified units to meters.
    
    This is the primary function for converting user inputs (typically in mm or µm)
    to the internal meter representation used throughout the library.
    
    Parameters
    ----------
    value : float or ndarray
        Value(s) to convert
    units : str
        Source unit ('m', 'mm', 'cm', 'um')
        
    Returns
    -------
    float or ndarray
        Value in meters
        
    Examples
    --------
    >>> to_meters(20, 'um')  # 20µm to meters
    2e-05
    >>> to_meters(5.0, 'mm')  # 5mm to meters
    0.005
    >>> to_meters(np.array([10, 20, 30]), 'um')  # Array conversion
    array([1.e-05, 2.e-05, 3.e-05])
    """
    if units not in _TO_METERS:
        raise ValueError(f"Unknown unit '{units}'. Supported: {list(_TO_METERS.keys())}")
    
    return value * _TO_METERS[units]


def from_meters(value: Union[float, np.ndarray], units: str) -> Union[float, np.ndarray]:
    """
    Convert value from meters to specified units.
    
    This is the primary function for converting internal meter values
    to user-facing units (typically mm or µm).
    
    Parameters
    ----------
    value : float or ndarray
        Value(s) in meters
    units : str
        Target unit ('m', 'mm', 'cm', 'um')
        
    Returns
    -------
    float or ndarray
        Value in target units
        
    Examples
    --------
    >>> from_meters(2e-5, 'um')  # 2e-5m to µm
    20.0
    >>> from_meters(0.005, 'mm')  # 0.005m to mm
    5.0
    >>> from_meters(np.array([1e-5, 2e-5]), 'um')  # Array conversion
    array([10., 20.])
    """
    if units not in _FROM_METERS:
        raise ValueError(f"Unknown unit '{units}'. Supported: {list(_FROM_METERS.keys())}")
    
    return value * _FROM_METERS[units]


def to_si_length(value: Union[float, np.ndarray], from_unit: str = CANONICAL_UNIT) -> Union[float, np.ndarray]:
    """
    Convert length from specified unit to SI (meters).
    
    DEPRECATED: Use to_meters() instead for clarity.
    
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
    return to_meters(value, from_unit)


def from_si_length(value: Union[float, np.ndarray], to_unit: str = CANONICAL_UNIT) -> Union[float, np.ndarray]:
    """
    Convert length from SI (meters) to specified unit.
    
    DEPRECATED: Use from_meters() instead for clarity.
    
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
    return from_meters(value, to_unit)


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
    Warn if value appears to be in millimeters when meters are expected.
    
    The library uses METERS internally for all geometry. This function warns
    when a value looks like it might be in millimeters (e.g., 50.0 meaning 50mm)
    being passed where meters are expected (should be 0.05).
    
    Parameters
    ----------
    value : float
        Value to check
    context : str
        Context hint ('length', 'radius', 'domain_size')
    param_name : str
        Parameter name for warning message
    """
    import warnings
    
    detected = detect_unit(value, context)
    if detected == "mm":
        warnings.warn(
            f"Parameter '{param_name}' value {value} appears to be in millimeters. "
            f"The library uses METERS internally for all geometry. "
            f"If you intended {value}mm, divide by 1000 to convert to meters (e.g., {value / 1000}). "
            f"To silence this warning, ensure your values are in meters.",
            UserWarning,
            stacklevel=3
        )


@dataclass
class UnitContext:
    """
    Context for unit handling in the library.
    
    The library uses meter-scale values internally (legacy convention).
    For example, EllipsoidSpec defaults to semi_axes=(0.05, 0.045, 0.035) 
    which represents 50mm, 45mm, 35mm in meters.
    
    At export time, internal values are converted to the user-specified output_units.
    When output_units="mm", internal value 0.05 becomes 50mm in the output STL file.
    
    The input_units field specifies what units user inputs are in, enabling
    automatic conversion to internal meters at API boundaries.
    
    Attributes
    ----------
    output_units : str
        Units for exported files (STL, JSON, etc.). Default: "mm"
        Supported: "m", "mm", "cm", "um"
    input_units : str
        Units for user inputs. Default: "mm"
        Supported: "m", "mm", "cm", "um"
    
    Examples
    --------
    >>> ctx = UnitContext(output_units="mm", input_units="um")
    >>> # User input of 20 (µm) becomes 2e-5 meters internally
    >>> ctx.to_internal(20)
    2e-05
    >>> # Internal value of 2e-5 (meters) becomes 20µm in output
    >>> ctx.to_output(2e-5)
    0.02  # 20µm in mm
    >>> 
    >>> ctx = UnitContext(input_units="um", output_units="um")
    >>> ctx.to_internal(20)  # 20µm to meters
    2e-05
    >>> ctx.to_output(2e-5)  # 2e-5m to µm
    20.0
    """
    
    output_units: str = "mm"
    input_units: str = "mm"
    
    def __post_init__(self):
        """Validate output_units and input_units."""
        if self.output_units not in _TO_METERS:
            raise ValueError(
                f"Unknown output_units '{self.output_units}'. "
                f"Supported: {list(_TO_METERS.keys())}"
            )
        if self.input_units not in _TO_METERS:
            raise ValueError(
                f"Unknown input_units '{self.input_units}'. "
                f"Supported: {list(_TO_METERS.keys())}"
            )
    
    @property
    def input_scale_factor(self) -> float:
        """
        Get scale factor for converting input units to internal meters.
        
        Returns
        -------
        float
            Scale factor to multiply input values by for internal use.
            - input_units="um": 1e-6 (multiply by 1e-6)
            - input_units="mm": 0.001 (multiply by 0.001)
            - input_units="m": 1.0 (no scaling)
        """
        return _TO_METERS[self.input_units]
    
    @property
    def scale_factor(self) -> float:
        """
        Get scale factor for converting internal units to output units.
        
        Internal units are meter-scale (legacy convention from the codebase).
        This factor converts from meters to the requested output units.
        
        Returns
        -------
        float
            Scale factor to multiply internal values by for output.
            - output_units="mm": 1000.0 (multiply by 1000)
            - output_units="m": 1.0 (no scaling)
            - output_units="cm": 100.0 (multiply by 100)
            - output_units="um": 1000000.0 (multiply by 1e6)
        """
        # Internal units are meters, so convert from m to output
        return convert_length(1.0, INTERNAL_UNIT, self.output_units)
    
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
    
    def to_internal(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert value from input units to internal meters.
        
        This is the primary method for converting user inputs to internal representation.
        
        Parameters
        ----------
        value : float or ndarray
            Value(s) in input_units
            
        Returns
        -------
        float or ndarray
            Value(s) in meters (internal units)
            
        Examples
        --------
        >>> ctx = UnitContext(input_units="um")
        >>> ctx.to_internal(20)  # 20µm to meters
        2e-05
        """
        return value * self.input_scale_factor
    
    def from_internal(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert value from internal meters to input units.
        
        Parameters
        ----------
        value : float or ndarray
            Value(s) in meters (internal units)
            
        Returns
        -------
        float or ndarray
            Value(s) in input_units
            
        Examples
        --------
        >>> ctx = UnitContext(input_units="um")
        >>> ctx.from_internal(2e-5)  # 2e-5m to µm
        20.0
        """
        return value / self.input_scale_factor
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "output_units": self.output_units,
            "input_units": self.input_units,
            "scale_factor": self.scale_factor,
            "input_scale_factor": self.input_scale_factor,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "UnitContext":
        """Create from dictionary."""
        return cls(
            output_units=d.get("output_units", "mm"),
            input_units=d.get("input_units", "mm"),
        )
    
    def scale_mesh(self, mesh):
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
            "internal_units": f"meters (converted to {self.output_units} on export)",
        }


# Default unit context (mm output)
DEFAULT_UNIT_CONTEXT = UnitContext(output_units="mm")
