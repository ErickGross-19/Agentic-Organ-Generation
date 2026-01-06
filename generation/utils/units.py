"""
Unit conversion utilities for vascular library.

The library uses millimeters (mm) as the default unit for all spatial measurements.
Physics calculations internally convert to SI units (meters) as needed.
"""

from typing import Union
import numpy as np

CANONICAL_UNIT = "mm"

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
