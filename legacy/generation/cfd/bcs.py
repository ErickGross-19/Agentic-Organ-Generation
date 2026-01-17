"""
Boundary condition definitions for CFD simulations.

Supports inlet conditions (steady/pulsatile flow, pressure) and
outlet conditions (zero-pressure, resistance, Windkessel RC/RCR).

Note: The library uses METERS internally for all geometry.
Pressures are in Pascals (Pa), flows in m^3/s.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Callable, Any
from enum import Enum
import numpy as np


class InletType(Enum):
    """Type of inlet boundary condition."""
    STEADY_FLOW = "steady_flow"
    PULSATILE_FLOW = "pulsatile_flow"
    STEADY_PRESSURE = "steady_pressure"


class OutletType(Enum):
    """Type of outlet boundary condition."""
    ZERO_PRESSURE = "zero_pressure"
    RESISTANCE = "R"
    RC = "RC"
    RCR = "RCR"
    IMPEDANCE = "impedance"


@dataclass
class InletBC:
    """
    Inlet boundary condition specification.
    
    Supports steady flow, pulsatile flow waveform, or steady pressure.
    """
    
    node_id: int
    bc_type: InletType = InletType.STEADY_FLOW
    
    flow_rate: float = 1e-6
    pressure: float = 13332.0
    
    waveform_times: Optional[np.ndarray] = None
    waveform_values: Optional[np.ndarray] = None
    
    def get_flow_at_time(self, t: float) -> float:
        """Get flow rate at time t (for pulsatile flow)."""
        if self.bc_type == InletType.STEADY_FLOW:
            return self.flow_rate
        elif self.bc_type == InletType.PULSATILE_FLOW:
            if self.waveform_times is None or self.waveform_values is None:
                return self.flow_rate
            return float(np.interp(t, self.waveform_times, self.waveform_values))
        else:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "node_id": self.node_id,
            "bc_type": self.bc_type.value,
            "flow_rate": self.flow_rate,
            "pressure": self.pressure,
        }
        if self.waveform_times is not None:
            result["waveform_times"] = self.waveform_times.tolist()
        if self.waveform_values is not None:
            result["waveform_values"] = self.waveform_values.tolist()
        return result
    
    @classmethod
    def steady_flow(cls, node_id: int, flow_rate: float) -> "InletBC":
        """Create steady flow inlet BC."""
        return cls(
            node_id=node_id,
            bc_type=InletType.STEADY_FLOW,
            flow_rate=flow_rate,
        )
    
    @classmethod
    def pulsatile_flow(
        cls,
        node_id: int,
        times: np.ndarray,
        flows: np.ndarray,
    ) -> "InletBC":
        """Create pulsatile flow inlet BC from waveform."""
        return cls(
            node_id=node_id,
            bc_type=InletType.PULSATILE_FLOW,
            flow_rate=float(np.mean(flows)),
            waveform_times=times,
            waveform_values=flows,
        )
    
    @classmethod
    def steady_pressure(cls, node_id: int, pressure: float) -> "InletBC":
        """Create steady pressure inlet BC."""
        return cls(
            node_id=node_id,
            bc_type=InletType.STEADY_PRESSURE,
            pressure=pressure,
        )


@dataclass
class OutletBC:
    """
    Outlet boundary condition specification.
    
    Supports zero-pressure, resistance (R), RC, and RCR Windkessel models.
    """
    
    node_id: int
    bc_type: OutletType = OutletType.ZERO_PRESSURE
    
    resistance: float = 1e9
    capacitance: float = 1e-9
    resistance_proximal: float = 1e8
    resistance_distal: float = 1e9
    
    reference_pressure: float = 0.0
    
    def compute_pressure(self, flow: float, capacitor_pressure: float = 0.0) -> float:
        """
        Compute outlet pressure given flow.
        
        Parameters
        ----------
        flow : float
            Flow rate at outlet (m^3/s)
        capacitor_pressure : float
            Current capacitor pressure for RC/RCR models (Pa)
            
        Returns
        -------
        float
            Outlet pressure (Pa)
        """
        if self.bc_type == OutletType.ZERO_PRESSURE:
            return self.reference_pressure
        elif self.bc_type == OutletType.RESISTANCE:
            return self.reference_pressure + self.resistance * flow
        elif self.bc_type == OutletType.RC:
            return capacitor_pressure
        elif self.bc_type == OutletType.RCR:
            return self.reference_pressure + self.resistance_proximal * flow + capacitor_pressure
        else:
            return self.reference_pressure
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "bc_type": self.bc_type.value,
            "resistance": self.resistance,
            "capacitance": self.capacitance,
            "resistance_proximal": self.resistance_proximal,
            "resistance_distal": self.resistance_distal,
            "reference_pressure": self.reference_pressure,
        }
    
    @classmethod
    def zero_pressure(cls, node_id: int, reference: float = 0.0) -> "OutletBC":
        """Create zero-pressure outlet BC."""
        return cls(
            node_id=node_id,
            bc_type=OutletType.ZERO_PRESSURE,
            reference_pressure=reference,
        )
    
    @classmethod
    def resistance_bc(cls, node_id: int, resistance: float, reference: float = 0.0) -> "OutletBC":
        """Create resistance outlet BC."""
        return cls(
            node_id=node_id,
            bc_type=OutletType.RESISTANCE,
            resistance=resistance,
            reference_pressure=reference,
        )
    
    @classmethod
    def rc_windkessel(
        cls,
        node_id: int,
        resistance: float,
        capacitance: float,
        reference: float = 0.0,
    ) -> "OutletBC":
        """Create RC Windkessel outlet BC."""
        return cls(
            node_id=node_id,
            bc_type=OutletType.RC,
            resistance=resistance,
            capacitance=capacitance,
            reference_pressure=reference,
        )
    
    @classmethod
    def rcr_windkessel(
        cls,
        node_id: int,
        r_proximal: float,
        capacitance: float,
        r_distal: float,
        reference: float = 0.0,
    ) -> "OutletBC":
        """Create RCR Windkessel outlet BC."""
        return cls(
            node_id=node_id,
            bc_type=OutletType.RCR,
            resistance_proximal=r_proximal,
            capacitance=capacitance,
            resistance_distal=r_distal,
            reference_pressure=reference,
        )


@dataclass
class FluidProperties:
    """Blood/fluid properties for CFD simulation."""
    
    density: float = 1060.0
    viscosity: float = 0.0035
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "density": self.density,
            "viscosity": self.viscosity,
        }


@dataclass
class BoundaryConditions:
    """
    Complete boundary condition specification for CFD simulation.
    
    Contains inlet BCs, outlet BCs, and fluid properties.
    """
    
    inlets: List[InletBC] = field(default_factory=list)
    outlets: List[OutletBC] = field(default_factory=list)
    fluid: FluidProperties = field(default_factory=FluidProperties)
    wall_type: Literal["rigid", "fsi"] = "rigid"
    
    def add_inlet(self, inlet: InletBC) -> None:
        """Add an inlet boundary condition."""
        self.inlets.append(inlet)
    
    def add_outlet(self, outlet: OutletBC) -> None:
        """Add an outlet boundary condition."""
        self.outlets.append(outlet)
    
    def get_inlet_node_ids(self) -> List[int]:
        """Get list of inlet node IDs."""
        return [inlet.node_id for inlet in self.inlets]
    
    def get_outlet_node_ids(self) -> List[int]:
        """Get list of outlet node IDs."""
        return [outlet.node_id for outlet in self.outlets]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "inlets": [inlet.to_dict() for inlet in self.inlets],
            "outlets": [outlet.to_dict() for outlet in self.outlets],
            "fluid": self.fluid.to_dict(),
            "wall_type": self.wall_type,
        }


def create_default_bcs_from_network(
    network: "VascularNetwork",
    inlet_flow: float = 1e-6,
    outlet_type: OutletType = OutletType.ZERO_PRESSURE,
    outlet_resistance: Optional[float] = None,
) -> BoundaryConditions:
    """
    Create default boundary conditions from network topology.
    
    Automatically identifies inlet and outlet nodes and assigns BCs.
    
    Parameters
    ----------
    network : VascularNetwork
        Vascular network
    inlet_flow : float
        Total inlet flow rate (m^3/s)
    outlet_type : OutletType
        Type of outlet BC to use
    outlet_resistance : float, optional
        Resistance for outlet BCs (if using resistance-based outlets)
        
    Returns
    -------
    BoundaryConditions
        Configured boundary conditions
    """
    bcs = BoundaryConditions()
    
    inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
    outlet_nodes = [n for n in network.nodes.values() if n.node_type in ["outlet", "terminal"]]
    
    for inlet_node in inlet_nodes:
        bcs.add_inlet(InletBC.steady_flow(inlet_node.id, inlet_flow / len(inlet_nodes)))
    
    if outlet_resistance is None and outlet_type != OutletType.ZERO_PRESSURE:
        total_resistance = 1e10
        outlet_resistance = total_resistance * len(outlet_nodes)
    
    for outlet_node in outlet_nodes:
        if outlet_type == OutletType.ZERO_PRESSURE:
            bcs.add_outlet(OutletBC.zero_pressure(outlet_node.id))
        elif outlet_type == OutletType.RESISTANCE:
            bcs.add_outlet(OutletBC.resistance_bc(outlet_node.id, outlet_resistance))
        else:
            bcs.add_outlet(OutletBC.zero_pressure(outlet_node.id))
    
    return bcs


def calibrate_outlet_resistances(
    network: "VascularNetwork",
    bcs: BoundaryConditions,
    target_pressure_drop: float,
    target_flow_splits: Optional[Dict[int, float]] = None,
) -> BoundaryConditions:
    """
    Calibrate outlet resistances to achieve target pressure drop and flow splits.
    
    Parameters
    ----------
    network : VascularNetwork
        Vascular network
    bcs : BoundaryConditions
        Initial boundary conditions
    target_pressure_drop : float
        Target pressure drop from inlet to outlets (Pa)
    target_flow_splits : dict, optional
        Target flow fraction at each outlet (should sum to 1.0)
        
    Returns
    -------
    BoundaryConditions
        Calibrated boundary conditions
    """
    if not bcs.inlets:
        return bcs
    
    total_inlet_flow = sum(inlet.flow_rate for inlet in bcs.inlets)
    
    if target_flow_splits is None:
        n_outlets = len(bcs.outlets)
        target_flow_splits = {outlet.node_id: 1.0 / n_outlets for outlet in bcs.outlets}
    
    for outlet in bcs.outlets:
        flow_fraction = target_flow_splits.get(outlet.node_id, 1.0 / len(bcs.outlets))
        outlet_flow = total_inlet_flow * flow_fraction
        
        if outlet_flow > 0:
            outlet.resistance = target_pressure_drop / outlet_flow
        else:
            outlet.resistance = 1e12
        
        outlet.bc_type = OutletType.RESISTANCE
    
    return bcs
