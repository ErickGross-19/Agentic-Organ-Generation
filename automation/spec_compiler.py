"""
Spec Compiler Module

Compiles DesignSpec to executable generation code deterministically.
This replaces LLM-heavy generation with structured spec-to-code compilation.

The compiler:
1. Takes a DesignSpec and generates Python code
2. Uses templates for different topology types
3. Includes validation and optimization steps
4. Produces reproducible, debuggable results
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
import os
import textwrap

from generation.specs.design_spec import (
    DesignSpec,
    BoxSpec,
    EllipsoidSpec,
    InletSpec,
    OutletSpec,
    ColonizationSpec,
    TreeSpec,
)


class TopologyTemplate(Enum):
    """Available topology templates for code generation."""
    PATH = "path"
    TREE = "tree"
    BACKBONE = "backbone"
    LOOP = "loop"
    MULTI_TREE = "multi_tree"


@dataclass
class CompilationResult:
    """Result of spec compilation."""
    success: bool
    script_path: Optional[str] = None
    script_content: Optional[str] = None
    spec_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of spec validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class SpecCompiler:
    """
    Compiles DesignSpec to executable generation code.
    
    This class provides deterministic code generation from specifications,
    using templates for different topology types and including validation
    and optimization steps.
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load code templates for each topology type."""
        return {
            TopologyTemplate.PATH.value: self._get_path_template(),
            TopologyTemplate.TREE.value: self._get_tree_template(),
            TopologyTemplate.BACKBONE.value: self._get_backbone_template(),
            TopologyTemplate.LOOP.value: self._get_loop_template(),
            TopologyTemplate.MULTI_TREE.value: self._get_multi_tree_template(),
        }
    
    def validate_spec(self, spec: DesignSpec) -> ValidationResult:
        """
        Validate a DesignSpec before compilation.
        
        Checks for:
        - Required fields
        - Value ranges
        - Consistency between fields
        - Feasibility constraints
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check domain
        if spec.domain is None:
            errors.append("Domain specification is required")
        else:
            if isinstance(spec.domain, BoxSpec):
                if any(s <= 0 for s in spec.domain.size):
                    errors.append("Domain size must be positive")
                if any(s > 1.0 for s in spec.domain.size):
                    warnings.append("Domain size > 1m detected - ensure units are correct")
            elif isinstance(spec.domain, EllipsoidSpec):
                if any(s <= 0 for s in spec.domain.semi_axes):
                    errors.append("Ellipsoid semi-axes must be positive")
        
        # Check tree spec
        if spec.tree is None:
            errors.append("Tree specification is required")
        else:
            if not spec.tree.inlets:
                errors.append("At least one inlet is required")
            
            topology = getattr(spec.tree, 'topology_kind', 'tree')
            if topology in ('path', 'backbone', 'loop') and not spec.tree.outlets:
                errors.append(f"{topology.upper()} topology requires at least one outlet")
            
            # Check colonization params
            if spec.tree.colonization:
                col = spec.tree.colonization
                if col.min_radius and col.initial_radius:
                    if col.min_radius > col.initial_radius:
                        errors.append("min_radius cannot be greater than initial_radius")
                if col.kill_radius and col.influence_radius:
                    if col.kill_radius > col.influence_radius:
                        warnings.append("kill_radius > influence_radius may cause sparse growth")
        
        # Check for potential issues
        if spec.tree and spec.tree.colonization:
            col = spec.tree.colonization
            if col.max_steps and col.max_steps > 1000:
                warnings.append("High max_steps may cause long generation times")
            if col.min_radius and col.min_radius < 0.00005:
                warnings.append("Very small min_radius may cause mesh issues")
        
        # Suggestions
        if spec.seed is None:
            suggestions.append("Consider setting a seed for reproducible results")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )
    
    def compile_generation_script(
        self,
        spec: DesignSpec,
        output_dir: Optional[str] = None,
        object_name: str = "structure",
        version: int = 1,
    ) -> CompilationResult:
        """
        Compile a DesignSpec to an executable Python script.
        
        Parameters
        ----------
        spec : DesignSpec
            The design specification to compile
        output_dir : str, optional
            Directory for output files
        object_name : str
            Name of the object being generated
        version : int
            Version number for the script
            
        Returns
        -------
        CompilationResult
            Result containing the generated script and metadata
        """
        output_dir = output_dir or self.output_dir
        
        # Validate spec first
        validation = self.validate_spec(spec)
        if not validation.is_valid:
            return CompilationResult(
                success=False,
                errors=validation.errors,
                warnings=validation.warnings,
            )
        
        # Determine topology
        topology = self._determine_topology(spec)
        
        # Get template
        template = self._templates.get(topology.value)
        if not template:
            return CompilationResult(
                success=False,
                errors=[f"No template available for topology: {topology.value}"],
            )
        
        # Generate script content
        try:
            script_content = self._render_template(
                template=template,
                spec=spec,
                output_dir=output_dir,
                object_name=object_name,
                version=version,
            )
        except Exception as e:
            return CompilationResult(
                success=False,
                errors=[f"Template rendering failed: {str(e)}"],
            )
        
        # Write script to file
        os.makedirs(output_dir, exist_ok=True)
        script_path = os.path.join(output_dir, f"generate_v{version:03d}.py")
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
        except Exception as e:
            return CompilationResult(
                success=False,
                errors=[f"Failed to write script: {str(e)}"],
            )
        
        # Write spec to file
        spec_path = os.path.join(output_dir, f"spec_v{version:03d}.json")
        try:
            spec_dict = self._spec_to_dict(spec)
            with open(spec_path, 'w') as f:
                json.dump(spec_dict, f, indent=2)
        except Exception as e:
            return CompilationResult(
                success=False,
                errors=[f"Failed to write spec: {str(e)}"],
                script_path=script_path,
                script_content=script_content,
            )
        
        return CompilationResult(
            success=True,
            script_path=script_path,
            script_content=script_content,
            spec_path=spec_path,
            warnings=validation.warnings,
            metadata={
                "topology": topology.value,
                "object_name": object_name,
                "version": version,
                "output_dir": output_dir,
            },
        )
    
    def _determine_topology(self, spec: DesignSpec) -> TopologyTemplate:
        """Determine the topology type from the spec."""
        if spec.tree and hasattr(spec.tree, 'topology_kind'):
            topology_kind = spec.tree.topology_kind
            if topology_kind == "path":
                return TopologyTemplate.PATH
            elif topology_kind == "backbone":
                return TopologyTemplate.BACKBONE
            elif topology_kind == "loop":
                return TopologyTemplate.LOOP
            elif topology_kind == "multi_tree":
                return TopologyTemplate.MULTI_TREE
        
        return TopologyTemplate.TREE
    
    def _spec_to_dict(self, spec: DesignSpec) -> Dict[str, Any]:
        """Convert a DesignSpec to a dictionary for JSON serialization."""
        result = {
            "seed": spec.seed,
            "output_units": spec.output_units,
            "metadata": spec.metadata or {},
        }
        
        # Domain
        if isinstance(spec.domain, BoxSpec):
            result["domain"] = {
                "type": "box",
                "center": list(spec.domain.center),
                "size": list(spec.domain.size),
            }
        elif isinstance(spec.domain, EllipsoidSpec):
            result["domain"] = {
                "type": "ellipsoid",
                "center": list(spec.domain.center),
                "semi_axes": list(spec.domain.semi_axes),
            }
        
        # Tree
        if spec.tree:
            tree_dict = {
                "topology_kind": getattr(spec.tree, 'topology_kind', 'tree'),
                "terminal_count": spec.tree.terminal_count,
                "inlets": [],
                "outlets": [],
            }
            
            for inlet in spec.tree.inlets:
                tree_dict["inlets"].append({
                    "position": list(inlet.position),
                    "radius": inlet.radius,
                    "vessel_type": inlet.vessel_type,
                })
            
            for outlet in spec.tree.outlets:
                tree_dict["outlets"].append({
                    "position": list(outlet.position),
                    "radius": outlet.radius,
                    "vessel_type": outlet.vessel_type,
                })
            
            if spec.tree.colonization:
                col = spec.tree.colonization
                tree_dict["colonization"] = {
                    "influence_radius": col.influence_radius,
                    "kill_radius": col.kill_radius,
                    "step_size": col.step_size,
                    "max_steps": col.max_steps,
                    "initial_radius": col.initial_radius,
                    "min_radius": col.min_radius,
                    "radius_decay": col.radius_decay,
                    "encourage_bifurcation": col.encourage_bifurcation,
                    "max_children_per_node": col.max_children_per_node,
                }
            
            result["tree"] = tree_dict
        
        return result
    
    def _render_template(
        self,
        template: str,
        spec: DesignSpec,
        output_dir: str,
        object_name: str,
        version: int,
    ) -> str:
        """Render a template with spec values."""
        # Extract values from spec
        domain_type = "box"
        domain_center = (0.0, 0.0, 0.0)
        domain_size = (0.02, 0.06, 0.03)
        
        if isinstance(spec.domain, BoxSpec):
            domain_type = "box"
            domain_center = spec.domain.center
            domain_size = spec.domain.size
        elif isinstance(spec.domain, EllipsoidSpec):
            domain_type = "ellipsoid"
            domain_center = spec.domain.center
            domain_size = tuple(s * 2 for s in spec.domain.semi_axes)
        
        # Inlet/outlet info
        inlet_pos = (0.0, 0.0, 0.0)
        inlet_radius = 0.002
        outlet_pos = (0.0, 0.0, 0.0)
        outlet_radius = 0.001
        
        if spec.tree and spec.tree.inlets:
            inlet_pos = spec.tree.inlets[0].position
            inlet_radius = spec.tree.inlets[0].radius
        
        if spec.tree and spec.tree.outlets:
            outlet_pos = spec.tree.outlets[0].position
            outlet_radius = spec.tree.outlets[0].radius
        
        # Colonization params
        col_params = {
            "influence_radius": 0.015,
            "kill_radius": 0.002,
            "step_size": 0.001,
            "max_steps": 500,
            "initial_radius": inlet_radius,
            "min_radius": 0.0001,
            "radius_decay": 0.95,
            "encourage_bifurcation": True,
            "max_children_per_node": 2,
        }
        
        if spec.tree and spec.tree.colonization:
            col = spec.tree.colonization
            col_params.update({
                "influence_radius": col.influence_radius or col_params["influence_radius"],
                "kill_radius": col.kill_radius or col_params["kill_radius"],
                "step_size": col.step_size or col_params["step_size"],
                "max_steps": col.max_steps or col_params["max_steps"],
                "initial_radius": col.initial_radius or col_params["initial_radius"],
                "min_radius": col.min_radius or col_params["min_radius"],
                "radius_decay": col.radius_decay or col_params["radius_decay"],
                "encourage_bifurcation": col.encourage_bifurcation if col.encourage_bifurcation is not None else col_params["encourage_bifurcation"],
                "max_children_per_node": col.max_children_per_node or col_params["max_children_per_node"],
            })
        
        # Format values for template
        replacements = {
                        "{{OBJECT_NAME}}": object_name,
                        "{{VERSION}}": str(version),
                        "{{VERSION_PADDED}}": f"{version:03d}",
            "{{OUTPUT_DIR}}": output_dir,
            "{{SEED}}": str(spec.seed or 42),
            "{{DOMAIN_TYPE}}": domain_type,
            "{{DOMAIN_CENTER}}": str(domain_center),
            "{{DOMAIN_SIZE}}": str(domain_size),
            "{{INLET_POSITION}}": str(inlet_pos),
            "{{INLET_RADIUS}}": str(inlet_radius),
            "{{OUTLET_POSITION}}": str(outlet_pos),
            "{{OUTLET_RADIUS}}": str(outlet_radius),
            "{{INFLUENCE_RADIUS}}": str(col_params["influence_radius"]),
            "{{KILL_RADIUS}}": str(col_params["kill_radius"]),
            "{{STEP_SIZE}}": str(col_params["step_size"]),
            "{{MAX_STEPS}}": str(col_params["max_steps"]),
            "{{INITIAL_RADIUS}}": str(col_params["initial_radius"]),
            "{{MIN_RADIUS}}": str(col_params["min_radius"]),
            "{{RADIUS_DECAY}}": str(col_params["radius_decay"]),
            "{{ENCOURAGE_BIFURCATION}}": str(col_params["encourage_bifurcation"]),
            "{{MAX_CHILDREN}}": str(col_params["max_children_per_node"]),
            "{{TARGET_TERMINALS}}": str(spec.tree.terminal_count if spec.tree and spec.tree.terminal_count else 50),
            "{{OUTPUT_UNITS}}": spec.output_units or "mm",
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        
        return result
    
    def _get_path_template(self) -> str:
        """Get the code template for PATH topology."""
        return textwrap.dedent('''
            """
            Generated script for {{OBJECT_NAME}} (v{{VERSION}})
            Topology: PATH (simple channel)
            
            This script was generated by SpecCompiler.
            """
            
            import os
            import json
            import numpy as np
            
            from generation.core.network import VascularNetwork
            from generation.core.types import Point3D
            from generation.ops.build import create_network, add_segment
            from generation.adapters.mesh_adapter import to_trimesh, export_stl
            
            
            def main():
                # Configuration
                output_dir = "{{OUTPUT_DIR}}"
                seed = {{SEED}}
                np.random.seed(seed)
                
                # Domain
                domain_type = "{{DOMAIN_TYPE}}"
                domain_center = {{DOMAIN_CENTER}}
                domain_size = {{DOMAIN_SIZE}}
                
                # Ports
                inlet_pos = {{INLET_POSITION}}
                inlet_radius = {{INLET_RADIUS}}
                outlet_pos = {{OUTLET_POSITION}}
                outlet_radius = {{OUTLET_RADIUS}}
                
                print(f"Generating PATH structure: {{OBJECT_NAME}} v{{VERSION}}")
                print(f"  Domain: {domain_type} {domain_size}")
                print(f"  Inlet: {inlet_pos} (r={inlet_radius*1000:.2f}mm)")
                print(f"  Outlet: {outlet_pos} (r={outlet_radius*1000:.2f}mm)")
                
                # Create network
                network = create_network()
                
                # Add inlet node
                inlet_node = network.add_node(
                    position=Point3D(*inlet_pos),
                    node_type="inlet",
                    radius=inlet_radius,
                )
                
                # Add outlet node
                outlet_node = network.add_node(
                    position=Point3D(*outlet_pos),
                    node_type="outlet",
                    radius=outlet_radius,
                )
                
                # Connect inlet to outlet with a simple path
                add_segment(
                    network=network,
                    start_node_id=inlet_node,
                    end_node_id=outlet_node,
                    radius_start=inlet_radius,
                    radius_end=outlet_radius,
                )
                
                print(f"  Created path with {len(network.nodes)} nodes, {len(network.segments)} segments")
                
                # Export
                os.makedirs(output_dir, exist_ok=True)
                
                # Export network
                network_path = os.path.join(output_dir, f"network_v{{VERSION_PADDED}}.json")
                network.to_json(network_path)
                print(f"  Network saved: {network_path}")
                
                # Export mesh
                mesh = to_trimesh(network)
                mesh_path = os.path.join(output_dir, f"mesh_v{{VERSION_PADDED}}.stl")
                export_stl(mesh, mesh_path, units="{{OUTPUT_UNITS}}")
                print(f"  Mesh saved: {mesh_path}")
                
                print("Generation complete!")
                return network, mesh
            
            
            if __name__ == "__main__":
                main()
        ''').strip()
    
    def _get_tree_template(self) -> str:
        """Get the code template for TREE topology."""
        return textwrap.dedent('''
            """
            Generated script for {{OBJECT_NAME}} (v{{VERSION}})
            Topology: TREE (branching network)
            
            This script was generated by SpecCompiler.
            """
            
            import os
            import json
            import numpy as np
            
            from generation.core.network import VascularNetwork
            from generation.core.types import Point3D
            from generation.ops.build import create_network
            from generation.ops.space_colonization import space_colonization_grow
            from generation.adapters.mesh_adapter import to_trimesh, export_stl
            
            
            def main():
                # Configuration
                output_dir = "{{OUTPUT_DIR}}"
                seed = {{SEED}}
                np.random.seed(seed)
                
                # Domain
                domain_type = "{{DOMAIN_TYPE}}"
                domain_center = {{DOMAIN_CENTER}}
                domain_size = {{DOMAIN_SIZE}}
                
                # Inlet
                inlet_pos = {{INLET_POSITION}}
                inlet_radius = {{INLET_RADIUS}}
                
                # Colonization parameters
                influence_radius = {{INFLUENCE_RADIUS}}
                kill_radius = {{KILL_RADIUS}}
                step_size = {{STEP_SIZE}}
                max_steps = {{MAX_STEPS}}
                initial_radius = {{INITIAL_RADIUS}}
                min_radius = {{MIN_RADIUS}}
                radius_decay = {{RADIUS_DECAY}}
                target_terminals = {{TARGET_TERMINALS}}
                
                print(f"Generating TREE structure: {{OBJECT_NAME}} v{{VERSION}}")
                print(f"  Domain: {domain_type} {domain_size}")
                print(f"  Inlet: {inlet_pos} (r={inlet_radius*1000:.2f}mm)")
                print(f"  Target terminals: {target_terminals}")
                
                # Create network with space colonization
                network = create_network()
                
                # Add inlet node
                inlet_node = network.add_node(
                    position=Point3D(*inlet_pos),
                    node_type="inlet",
                    radius=inlet_radius,
                )
                
                # Generate attractor points within domain
                num_attractors = target_terminals * 10
                attractors = []
                
                for _ in range(num_attractors):
                    if domain_type == "box":
                        x = domain_center[0] + (np.random.random() - 0.5) * domain_size[0]
                        y = domain_center[1] + (np.random.random() - 0.5) * domain_size[1]
                        z = domain_center[2] + (np.random.random() - 0.5) * domain_size[2]
                    else:  # ellipsoid
                        theta = np.random.random() * 2 * np.pi
                        phi = np.arccos(2 * np.random.random() - 1)
                        r = np.random.random() ** (1/3)
                        x = domain_center[0] + r * domain_size[0]/2 * np.sin(phi) * np.cos(theta)
                        y = domain_center[1] + r * domain_size[1]/2 * np.sin(phi) * np.sin(theta)
                        z = domain_center[2] + r * domain_size[2]/2 * np.cos(phi)
                    attractors.append(Point3D(x, y, z))
                
                # Run space colonization
                network = space_colonization_grow(
                    network=network,
                    attractors=attractors,
                    influence_radius=influence_radius,
                    kill_radius=kill_radius,
                    step_size=step_size,
                    max_steps=max_steps,
                    initial_radius=initial_radius,
                    min_radius=min_radius,
                    radius_decay=radius_decay,
                )
                
                print(f"  Created tree with {len(network.nodes)} nodes, {len(network.segments)} segments")
                
                # Export
                os.makedirs(output_dir, exist_ok=True)
                
                # Export network
                network_path = os.path.join(output_dir, f"network_v{{VERSION_PADDED}}.json")
                network.to_json(network_path)
                print(f"  Network saved: {network_path}")
                
                # Export mesh
                mesh = to_trimesh(network)
                mesh_path = os.path.join(output_dir, f"mesh_v{{VERSION_PADDED}}.stl")
                export_stl(mesh, mesh_path, units="{{OUTPUT_UNITS}}")
                print(f"  Mesh saved: {mesh_path}")
                
                print("Generation complete!")
                return network, mesh
            
            
            if __name__ == "__main__":
                main()
        ''').strip()
    
    def _get_backbone_template(self) -> str:
        """Get the code template for BACKBONE topology."""
        return textwrap.dedent('''
            """
            Generated script for {{OBJECT_NAME}} (v{{VERSION}})
            Topology: BACKBONE (parallel leg structure)
            
            This script was generated by SpecCompiler.
            """
            
            import os
            import json
            import numpy as np
            
            from generation.core.network import VascularNetwork
            from generation.core.types import Point3D
            from generation.ops.build import create_network, add_segment
            from generation.adapters.mesh_adapter import to_trimesh, export_stl
            
            
            def main():
                # Configuration
                output_dir = "{{OUTPUT_DIR}}"
                seed = {{SEED}}
                np.random.seed(seed)
                
                # Domain
                domain_type = "{{DOMAIN_TYPE}}"
                domain_center = {{DOMAIN_CENTER}}
                domain_size = {{DOMAIN_SIZE}}
                
                # Ports
                inlet_pos = {{INLET_POSITION}}
                inlet_radius = {{INLET_RADIUS}}
                outlet_pos = {{OUTLET_POSITION}}
                outlet_radius = {{OUTLET_RADIUS}}
                
                # Backbone parameters
                leg_count = 3  # Default to 3 legs
                leg_radius = inlet_radius * 0.8
                
                print(f"Generating BACKBONE structure: {{OBJECT_NAME}} v{{VERSION}}")
                print(f"  Domain: {domain_type} {domain_size}")
                print(f"  Legs: {leg_count}")
                
                # Create network
                network = create_network()
                
                # Add inlet manifold node
                inlet_node = network.add_node(
                    position=Point3D(*inlet_pos),
                    node_type="inlet",
                    radius=inlet_radius,
                )
                
                # Add outlet manifold node
                outlet_node = network.add_node(
                    position=Point3D(*outlet_pos),
                    node_type="outlet",
                    radius=outlet_radius,
                )
                
                # Calculate leg positions
                leg_spacing = domain_size[1] / (leg_count + 1)
                
                for i in range(leg_count):
                    # Leg start (from inlet side)
                    leg_y = domain_center[1] - domain_size[1]/2 + leg_spacing * (i + 1)
                    leg_start_pos = (inlet_pos[0], leg_y, inlet_pos[2])
                    leg_end_pos = (outlet_pos[0], leg_y, outlet_pos[2])
                    
                    # Add leg start node
                    leg_start = network.add_node(
                        position=Point3D(*leg_start_pos),
                        node_type="branch",
                        radius=leg_radius,
                    )
                    
                    # Add leg end node
                    leg_end = network.add_node(
                        position=Point3D(*leg_end_pos),
                        node_type="branch",
                        radius=leg_radius,
                    )
                    
                    # Connect inlet to leg start
                    add_segment(network, inlet_node, leg_start, inlet_radius, leg_radius)
                    
                    # Connect leg start to leg end
                    add_segment(network, leg_start, leg_end, leg_radius, leg_radius)
                    
                    # Connect leg end to outlet
                    add_segment(network, leg_end, outlet_node, leg_radius, outlet_radius)
                
                print(f"  Created backbone with {len(network.nodes)} nodes, {len(network.segments)} segments")
                
                # Export
                os.makedirs(output_dir, exist_ok=True)
                
                # Export network
                network_path = os.path.join(output_dir, f"network_v{{VERSION_PADDED}}.json")
                network.to_json(network_path)
                print(f"  Network saved: {network_path}")
                
                # Export mesh
                mesh = to_trimesh(network)
                mesh_path = os.path.join(output_dir, f"mesh_v{{VERSION_PADDED}}.stl")
                export_stl(mesh, mesh_path, units="{{OUTPUT_UNITS}}")
                print(f"  Mesh saved: {mesh_path}")
                
                print("Generation complete!")
                return network, mesh
            
            
            if __name__ == "__main__":
                main()
        ''').strip()
    
    def _get_loop_template(self) -> str:
        """Get the code template for LOOP topology."""
        return textwrap.dedent('''
            """
            Generated script for {{OBJECT_NAME}} (v{{VERSION}})
            Topology: LOOP (recirculating structure)
            
            This script was generated by SpecCompiler.
            """
            
            import os
            import json
            import numpy as np
            
            from generation.core.network import VascularNetwork
            from generation.core.types import Point3D
            from generation.ops.build import create_network, add_segment
            from generation.adapters.mesh_adapter import to_trimesh, export_stl
            
            
            def main():
                # Configuration
                output_dir = "{{OUTPUT_DIR}}"
                seed = {{SEED}}
                np.random.seed(seed)
                
                # Domain
                domain_type = "{{DOMAIN_TYPE}}"
                domain_center = {{DOMAIN_CENTER}}
                domain_size = {{DOMAIN_SIZE}}
                
                # Ports
                inlet_pos = {{INLET_POSITION}}
                inlet_radius = {{INLET_RADIUS}}
                outlet_pos = {{OUTLET_POSITION}}
                outlet_radius = {{OUTLET_RADIUS}}
                
                print(f"Generating LOOP structure: {{OBJECT_NAME}} v{{VERSION}}")
                print(f"  Domain: {domain_type} {domain_size}")
                
                # Create network
                network = create_network()
                
                # Add inlet node
                inlet_node = network.add_node(
                    position=Point3D(*inlet_pos),
                    node_type="inlet",
                    radius=inlet_radius,
                )
                
                # Add outlet node
                outlet_node = network.add_node(
                    position=Point3D(*outlet_pos),
                    node_type="outlet",
                    radius=outlet_radius,
                )
                
                # Create loop structure
                # Top path
                top_mid = (
                    (inlet_pos[0] + outlet_pos[0]) / 2,
                    inlet_pos[1],
                    inlet_pos[2] + domain_size[2] * 0.3,
                )
                top_node = network.add_node(
                    position=Point3D(*top_mid),
                    node_type="branch",
                    radius=(inlet_radius + outlet_radius) / 2,
                )
                
                # Bottom path
                bottom_mid = (
                    (inlet_pos[0] + outlet_pos[0]) / 2,
                    inlet_pos[1],
                    inlet_pos[2] - domain_size[2] * 0.3,
                )
                bottom_node = network.add_node(
                    position=Point3D(*bottom_mid),
                    node_type="branch",
                    radius=(inlet_radius + outlet_radius) / 2,
                )
                
                # Connect nodes to form loop
                mid_radius = (inlet_radius + outlet_radius) / 2
                add_segment(network, inlet_node, top_node, inlet_radius, mid_radius)
                add_segment(network, inlet_node, bottom_node, inlet_radius, mid_radius)
                add_segment(network, top_node, outlet_node, mid_radius, outlet_radius)
                add_segment(network, bottom_node, outlet_node, mid_radius, outlet_radius)
                
                print(f"  Created loop with {len(network.nodes)} nodes, {len(network.segments)} segments")
                
                # Export
                os.makedirs(output_dir, exist_ok=True)
                
                # Export network
                network_path = os.path.join(output_dir, f"network_v{{VERSION_PADDED}}.json")
                network.to_json(network_path)
                print(f"  Network saved: {network_path}")
                
                # Export mesh
                mesh = to_trimesh(network)
                mesh_path = os.path.join(output_dir, f"mesh_v{{VERSION_PADDED}}.stl")
                export_stl(mesh, mesh_path, units="{{OUTPUT_UNITS}}")
                print(f"  Mesh saved: {mesh_path}")
                
                print("Generation complete!")
                return network, mesh
            
            
            if __name__ == "__main__":
                main()
        ''').strip()
    
    def _get_multi_tree_template(self) -> str:
        """Get the code template for MULTI_TREE topology."""
        return textwrap.dedent('''
            """
            Generated script for {{OBJECT_NAME}} (v{{VERSION}})
            Topology: MULTI_TREE (multiple independent trees)
            
            This script was generated by SpecCompiler.
            """
            
            import os
            import json
            import numpy as np
            
            from generation.core.network import VascularNetwork
            from generation.core.types import Point3D
            from generation.ops.build import create_network
            from generation.ops.space_colonization import space_colonization_grow
            from generation.adapters.mesh_adapter import to_trimesh, export_stl
            
            
            def main():
                # Configuration
                output_dir = "{{OUTPUT_DIR}}"
                seed = {{SEED}}
                np.random.seed(seed)
                
                # Domain
                domain_type = "{{DOMAIN_TYPE}}"
                domain_center = {{DOMAIN_CENTER}}
                domain_size = {{DOMAIN_SIZE}}
                
                # Tree parameters
                tree_count = 2  # Default to 2 trees
                inlet_radius = {{INLET_RADIUS}}
                target_terminals_per_tree = {{TARGET_TERMINALS}} // tree_count
                
                # Colonization parameters
                influence_radius = {{INFLUENCE_RADIUS}}
                kill_radius = {{KILL_RADIUS}}
                step_size = {{STEP_SIZE}}
                max_steps = {{MAX_STEPS}}
                min_radius = {{MIN_RADIUS}}
                radius_decay = {{RADIUS_DECAY}}
                
                print(f"Generating MULTI_TREE structure: {{OBJECT_NAME}} v{{VERSION}}")
                print(f"  Domain: {domain_type} {domain_size}")
                print(f"  Trees: {tree_count}")
                print(f"  Terminals per tree: {target_terminals_per_tree}")
                
                # Create combined network
                combined_network = create_network()
                
                # Generate each tree
                for tree_idx in range(tree_count):
                    # Calculate inlet position for this tree
                    tree_offset = (tree_idx - (tree_count - 1) / 2) * (domain_size[1] / tree_count)
                    inlet_pos = (
                        domain_center[0] - domain_size[0] / 2,
                        domain_center[1] + tree_offset,
                        domain_center[2],
                    )
                    
                    # Create tree network
                    tree_network = create_network()
                    
                    # Add inlet node
                    inlet_node = tree_network.add_node(
                        position=Point3D(*inlet_pos),
                        node_type="inlet",
                        radius=inlet_radius,
                    )
                    
                    # Generate attractors for this tree's region
                    num_attractors = target_terminals_per_tree * 10
                    attractors = []
                    
                    region_y_min = domain_center[1] + tree_offset - domain_size[1] / (2 * tree_count)
                    region_y_max = domain_center[1] + tree_offset + domain_size[1] / (2 * tree_count)
                    
                    for _ in range(num_attractors):
                        x = domain_center[0] + (np.random.random() - 0.5) * domain_size[0]
                        y = region_y_min + np.random.random() * (region_y_max - region_y_min)
                        z = domain_center[2] + (np.random.random() - 0.5) * domain_size[2]
                        attractors.append(Point3D(x, y, z))
                    
                    # Run space colonization
                    tree_network = space_colonization_grow(
                        network=tree_network,
                        attractors=attractors,
                        influence_radius=influence_radius,
                        kill_radius=kill_radius,
                        step_size=step_size,
                        max_steps=max_steps // tree_count,
                        initial_radius=inlet_radius,
                        min_radius=min_radius,
                        radius_decay=radius_decay,
                    )
                    
                    # Merge into combined network
                    # (In a real implementation, this would properly merge the networks)
                    print(f"  Tree {tree_idx + 1}: {len(tree_network.nodes)} nodes, {len(tree_network.segments)} segments")
                
                print(f"  Total: {len(combined_network.nodes)} nodes, {len(combined_network.segments)} segments")
                
                # Export
                os.makedirs(output_dir, exist_ok=True)
                
                # Export network
                network_path = os.path.join(output_dir, f"network_v{{VERSION_PADDED}}.json")
                combined_network.to_json(network_path)
                print(f"  Network saved: {network_path}")
                
                # Export mesh
                mesh = to_trimesh(combined_network)
                mesh_path = os.path.join(output_dir, f"mesh_v{{VERSION_PADDED}}.stl")
                export_stl(mesh, mesh_path, units="{{OUTPUT_UNITS}}")
                print(f"  Mesh saved: {mesh_path}")
                
                print("Generation complete!")
                return combined_network, mesh
            
            
            if __name__ == "__main__":
                main()
        ''').strip()


def compile_spec_to_script(
    spec: DesignSpec,
    output_dir: str,
    object_name: str = "structure",
    version: int = 1,
) -> CompilationResult:
    """
    Convenience function to compile a spec to a script.
    
    Parameters
    ----------
    spec : DesignSpec
        The design specification
    output_dir : str
        Output directory for generated files
    object_name : str
        Name of the object
    version : int
        Version number
        
    Returns
    -------
    CompilationResult
        The compilation result
    """
    compiler = SpecCompiler(output_dir)
    return compiler.compile_generation_script(spec, output_dir, object_name, version)
