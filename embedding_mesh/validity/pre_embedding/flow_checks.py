"""
Pre-Embedding Flow Checks

Validates hemodynamic flow properties before embedding.
Checks include flow plausibility, Reynolds number, and pressure monotonicity.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from generation.core import VascularNetwork


@dataclass
class FlowCheckResult:
    """Result of a flow check."""
    passed: bool
    check_name: str
    message: str
    details: Dict[str, Any]
    warnings: List[str]


def check_flow_plausibility(
    network: "VascularNetwork",
    max_flow_balance_error: float = 0.05,
) -> FlowCheckResult:
    """
    Check if flow solution is plausible (conservation at junctions).
    
    At each junction node, inflow should equal outflow (mass conservation).
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network with solved flow
    max_flow_balance_error : float
        Maximum acceptable relative flow balance error at junctions
        
    Returns
    -------
    FlowCheckResult
        Result with pass/fail status and details
    """
    # Check if network has flow solution
    has_flow = any('flow' in seg.attributes for seg in network.segments.values())
    
    if not has_flow:
        return FlowCheckResult(
            passed=True,
            check_name="flow_plausibility",
            message="No flow solution found - skipping check",
            details={"has_flow_solution": False},
            warnings=["Network does not have flow solution"],
        )
    
    balance_errors = []
    violations = []
    
    for node_id, node in network.nodes.items():
        if node.node_type == "junction":
            inflow = 0.0
            outflow = 0.0
            
            for seg in network.segments.values():
                flow = seg.attributes.get('flow', 0.0)
                
                if seg.end_node_id == node_id:
                    inflow += flow
                elif seg.start_node_id == node_id:
                    outflow += flow
            
            if inflow > 1e-10:
                balance = abs(inflow - outflow) / inflow
                balance_errors.append(balance)
                
                if balance > max_flow_balance_error:
                    violations.append({
                        "node_id": node_id,
                        "inflow": inflow,
                        "outflow": outflow,
                        "balance_error": balance,
                    })
    
    mean_error = float(np.mean(balance_errors)) if balance_errors else 0.0
    max_error = float(np.max(balance_errors)) if balance_errors else 0.0
    passed = max_error <= max_flow_balance_error
    
    details = {
        "has_flow_solution": True,
        "junctions_checked": len(balance_errors),
        "mean_balance_error": mean_error,
        "max_balance_error": max_error,
        "violations_count": len(violations),
        "max_allowed_error": max_flow_balance_error,
    }
    
    warnings = []
    if violations:
        warnings.append(f"{len(violations)} junctions have flow balance error > {max_flow_balance_error:.0%}")
    
    return FlowCheckResult(
        passed=passed,
        check_name="flow_plausibility",
        message=f"Flow balance: max error {max_error:.1%}" if balance_errors else "No junctions found",
        details=details,
        warnings=warnings,
    )


def check_reynolds_number(
    network: "VascularNetwork",
    max_reynolds: float = 2300.0,
    blood_density: float = 1060.0,
    blood_viscosity: float = 1.0e-3,
) -> FlowCheckResult:
    """
    Check Reynolds number to ensure laminar flow assumption is valid.
    
    Reynolds number Re = rho * v * D / mu
    For blood flow, Re < 2300 indicates laminar flow.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network with solved flow
    max_reynolds : float
        Maximum acceptable Reynolds number (default 2300 for laminar flow)
    blood_density : float
        Blood density in kg/m^3 (default 1060)
    blood_viscosity : float
        Blood dynamic viscosity in Pa*s (default 1e-3)
        
    Returns
    -------
    FlowCheckResult
        Result with pass/fail status and details
    """
    reynolds_numbers = []
    turbulent_segments = []
    
    for seg_id, seg in network.segments.items():
        velocity = seg.attributes.get('velocity', 0.0)
        
        if velocity > 0:
            # Get radius (assume mm, convert to m)
            if hasattr(seg, 'geometry') and seg.geometry is not None:
                radius_mm = (seg.geometry.radius_start + seg.geometry.radius_end) / 2
            else:
                radius_mm = seg.attributes.get("radius", 1.0)
            
            radius_m = radius_mm / 1000.0
            diameter_m = 2 * radius_m
            
            re = blood_density * velocity * diameter_m / blood_viscosity
            reynolds_numbers.append(re)
            
            if re > max_reynolds:
                turbulent_segments.append({
                    "segment_id": seg_id,
                    "reynolds": re,
                    "velocity": velocity,
                    "diameter_m": diameter_m,
                })
    
    if not reynolds_numbers:
        return FlowCheckResult(
            passed=True,
            check_name="reynolds_number",
            message="No velocity data found - skipping check",
            details={"has_velocity_data": False},
            warnings=["Network does not have velocity data"],
        )
    
    max_re = float(np.max(reynolds_numbers))
    mean_re = float(np.mean(reynolds_numbers))
    passed = max_re <= max_reynolds
    
    details = {
        "has_velocity_data": True,
        "segments_checked": len(reynolds_numbers),
        "max_reynolds": max_re,
        "mean_reynolds": mean_re,
        "turbulent_segments": len(turbulent_segments),
        "is_laminar": max_re < max_reynolds,
        "max_allowed_reynolds": max_reynolds,
    }
    
    warnings = []
    if turbulent_segments:
        warnings.append(f"{len(turbulent_segments)} segments have turbulent flow (Re > {max_reynolds})")
    
    return FlowCheckResult(
        passed=passed,
        check_name="reynolds_number",
        message=f"Reynolds: max={max_re:.0f}, {'laminar' if passed else 'TURBULENT'}",
        details=details,
        warnings=warnings,
    )


def check_pressure_monotonicity(
    network: "VascularNetwork",
) -> FlowCheckResult:
    """
    Check that pressure decreases along flow direction.
    
    In a properly solved flow, pressure should decrease from inlet to outlet.
    Pressure increasing along flow direction indicates a solver issue.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network with solved flow
        
    Returns
    -------
    FlowCheckResult
        Result with pass/fail status and details
    """
    has_pressure = any('pressure' in node.attributes for node in network.nodes.values())
    has_flow = any('flow' in seg.attributes for seg in network.segments.values())
    
    if not has_pressure or not has_flow:
        return FlowCheckResult(
            passed=True,
            check_name="pressure_monotonicity",
            message="No pressure/flow data found - skipping check",
            details={"has_pressure_data": has_pressure, "has_flow_data": has_flow},
            warnings=["Network does not have complete flow solution"],
        )
    
    violations = []
    segments_checked = 0
    
    for seg_id, seg in network.segments.items():
        p_start = network.nodes[seg.start_node_id].attributes.get('pressure', 0)
        p_end = network.nodes[seg.end_node_id].attributes.get('pressure', 0)
        flow = seg.attributes.get('flow', 0)
        
        segments_checked += 1
        
        # If flow is positive (start -> end), pressure should decrease
        if flow > 0 and p_end > p_start:
            violations.append({
                "segment_id": seg_id,
                "p_start": p_start,
                "p_end": p_end,
                "flow": flow,
                "pressure_increase": p_end - p_start,
            })
    
    passed = len(violations) == 0
    
    details = {
        "has_pressure_data": True,
        "has_flow_data": True,
        "segments_checked": segments_checked,
        "violations_count": len(violations),
    }
    
    warnings = []
    if violations:
        warnings.append(f"{len(violations)} segments have pressure increasing along flow direction")
    
    return FlowCheckResult(
        passed=passed,
        check_name="pressure_monotonicity",
        message="Pressure monotonic" if passed else f"{len(violations)} pressure violations",
        details=details,
        warnings=warnings,
    )


@dataclass
class FlowCheckReport:
    """Aggregated report of all flow checks."""
    passed: bool
    status: str  # "ok", "warnings", "fail"
    checks: List[FlowCheckResult]
    summary: Dict[str, Any]


def run_all_flow_checks(
    network: "VascularNetwork",
    max_flow_balance_error: float = 0.05,
    max_reynolds: float = 2300.0,
    blood_density: float = 1060.0,
    blood_viscosity: float = 1.0e-3,
) -> FlowCheckReport:
    """
    Run all pre-embedding flow checks.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network with solved flow
    max_flow_balance_error : float
        Maximum acceptable flow balance error at junctions
    max_reynolds : float
        Maximum acceptable Reynolds number
    blood_density : float
        Blood density in kg/m^3
    blood_viscosity : float
        Blood dynamic viscosity in Pa*s
        
    Returns
    -------
    FlowCheckReport
        Aggregated report with all check results
    """
    checks = [
        check_flow_plausibility(network, max_flow_balance_error),
        check_reynolds_number(network, max_reynolds, blood_density, blood_viscosity),
        check_pressure_monotonicity(network),
    ]
    
    all_passed = all(c.passed for c in checks)
    has_warnings = any(len(c.warnings) > 0 for c in checks)
    
    if all_passed and not has_warnings:
        status = "ok"
    elif all_passed:
        status = "warnings"
    else:
        status = "fail"
    
    summary = {
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c.passed),
        "failed_checks": sum(1 for c in checks if not c.passed),
        "total_warnings": sum(len(c.warnings) for c in checks),
    }
    
    return FlowCheckReport(
        passed=all_passed,
        status=status,
        checks=checks,
        summary=summary,
    )
