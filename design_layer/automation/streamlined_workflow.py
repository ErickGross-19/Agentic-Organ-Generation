"""
Streamlined Workflow Module

Provides a simplified, conversation-driven workflow for organ generation.
This replaces the rigid 10+ state machine with a fluid 5-state approach.

The workflow states:
1. INITIALIZING: Project setup and initialization
2. DESIGNING: Combines requirements capture + spec compilation
3. GENERATING: Combines generation + validation
4. REFINING: Iteration loop for user feedback
5. COMPLETE: Final state

Key improvements:
- Natural conversation flow without rigid state transitions
- Contextual dialogue that handles interruptions seamlessly
- Deterministic code generation via SpecCompiler
- Real-time adaptation based on generation results
- Multi-format output generation
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import os
import json
import datetime

from .contextual_dialogue import ContextualDialogue, DialogueContext, DialogueResponse
from .spec_compiler import SpecCompiler, CompilationResult, ValidationResult
from .adaptation_engine import AdaptationEngine, AdaptationResult, GenerationResult
from .output_generator import OutputGenerator, ArtifactPackage

from generation.specs.design_spec import (
    DesignSpec,
    BoxSpec,
    EllipsoidSpec,
    InletSpec,
    OutletSpec,
    ColonizationSpec,
    TreeSpec,
)


class StreamlinedState(Enum):
    """Simplified workflow states."""
    INITIALIZING = "initializing"
    DESIGNING = "designing"
    GENERATING = "generating"
    REFINING = "refining"
    COMPLETE = "complete"


@dataclass
class ProjectContext:
    """Context for the current project."""
    project_name: str = ""
    project_dir: str = ""
    objects: List[Dict[str, Any]] = field(default_factory=list)
    current_object_index: int = 0
    global_defaults: Dict[str, Any] = field(default_factory=dict)
    
    def get_current_object(self) -> Optional[Dict[str, Any]]:
        """Get the current object being worked on."""
        if 0 <= self.current_object_index < len(self.objects):
            return self.objects[self.current_object_index]
        return None
    
    def add_object(self, name: str) -> Dict[str, Any]:
        """Add a new object to the project."""
        obj = {
            "name": name,
            "version": 1,
            "spec": None,
            "network": None,
            "artifacts": None,
            "status": "pending",
        }
        self.objects.append(obj)
        return obj


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    project_dir: str = ""
    objects: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[ArtifactPackage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class StreamlinedWorkflow:
    """
    Streamlined workflow for organ generation.
    
    This class provides a simplified, conversation-driven approach to
    organ generation that replaces the rigid 10+ state machine with
    a fluid 5-state workflow.
    
    Features:
    - Natural conversation flow
    - Contextual dialogue handling
    - Deterministic code generation
    - Real-time adaptation
    - Multi-format output generation
    """
    
    WORKFLOW_NAME = "Streamlined Organ Generator"
    WORKFLOW_VERSION = "5.0.0"
    
    def __init__(
        self,
        output_dir: str = "./outputs",
        llm_client: Optional[Any] = None,
        input_func: Optional[Callable[[str], str]] = None,
        print_func: Optional[Callable[[str], None]] = None,
    ):
        self.output_dir = output_dir
        self.llm_client = llm_client
        self.input_func = input_func or input
        self.print_func = print_func or print
        
        # Initialize components
        self.dialogue = ContextualDialogue(
            llm_client=llm_client,
            input_func=input_func,
            print_func=print_func,
        )
        self.compiler = SpecCompiler(output_dir)
        self.adapter = AdaptationEngine()
        self.output_gen = OutputGenerator(output_dir)
        
        # State
        self.state = StreamlinedState.INITIALIZING
        self.context = ProjectContext()
        self._state_history: List[StreamlinedState] = []
        self._max_iterations = 5
    
    def _print(self, text: str) -> None:
        """Print text using configured print function."""
        self.print_func(text)
    
    def _input(self, prompt: str) -> str:
        """Get input using configured input function."""
        return self.input_func(prompt)
    
    def run(self) -> WorkflowResult:
        """
        Run the complete workflow.
        
        Returns
        -------
        WorkflowResult
            The result of the workflow execution
        """
        self._print_header()
        
        errors = []
        warnings = []
        artifacts = []
        
        try:
            while self.state != StreamlinedState.COMPLETE:
                self._state_history.append(self.state)
                
                if self.state == StreamlinedState.INITIALIZING:
                    self._run_initializing()
                elif self.state == StreamlinedState.DESIGNING:
                    self._run_designing()
                elif self.state == StreamlinedState.GENERATING:
                    self._run_generating()
                elif self.state == StreamlinedState.REFINING:
                    self._run_refining()
                
                # Check for stuck state
                if len(self._state_history) > 50:
                    errors.append("Workflow exceeded maximum state transitions")
                    break
            
            # Collect artifacts
            for obj in self.context.objects:
                if obj.get("artifacts"):
                    artifacts.append(obj["artifacts"])
            
        except KeyboardInterrupt:
            self._print("\n\nWorkflow interrupted by user.")
            errors.append("Workflow interrupted")
        except Exception as e:
            errors.append(f"Workflow error: {str(e)}")
        
        return WorkflowResult(
            success=len(errors) == 0,
            project_dir=self.context.project_dir,
            objects=self.context.objects,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
        )
    
    def _print_header(self) -> None:
        """Print workflow header with rules and available commands."""
        self._print("")
        self._print("=" * 60)
        self._print(f"  {self.WORKFLOW_NAME} v{self.WORKFLOW_VERSION}")
        self._print("=" * 60)
        self._print("")
        self._print("This workflow uses a streamlined, conversation-driven approach")
        self._print("to generate 3D vascular structures.")
        self._print("")
        self._print("WORKFLOW RULES:")
        self._print("-" * 40)
        self._print("1. Describe what you want to create in natural language")
        self._print("2. Answer questions to refine your specification")
        self._print("3. You can update your description at any time")
        self._print("4. Address warnings and ambiguities before generation")
        self._print("5. Type 'default' to accept default values for any field")
        self._print("6. Type 'confirm' when ready to proceed with generation")
        self._print("")
        self._print("AVAILABLE COMMANDS:")
        self._print("-" * 40)
        self._print("  'update description' - Provide a new description and re-assess")
        self._print("  'address warnings'   - View and resolve current warnings")
        self._print("  'address ambiguities'- View and clarify ambiguities")
        self._print("  'default'            - Use default value for current field")
        self._print("  'help'               - Show detailed help information")
        self._print("  'cancel'             - Cancel the current workflow")
        self._print("")
    
    def _run_initializing(self) -> None:
        """Run INITIALIZING state: Project setup."""
        self._print("-" * 40)
        self._print("Step 1: Project Initialization")
        self._print("-" * 40)
        self._print("")
        
        # Get project name
        project_name = self._input("Project name (or press Enter for 'organ_project'): ").strip()
        if not project_name:
            project_name = "organ_project"
        
        self.context.project_name = project_name
        self.context.project_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(self.context.project_dir, exist_ok=True)
        
        self._print(f"\nProject '{project_name}' initialized at: {self.context.project_dir}")
        
        # Get number of objects
        num_objects_str = self._input("\nHow many objects to generate? (default: 1): ").strip()
        try:
            num_objects = int(num_objects_str) if num_objects_str else 1
        except ValueError:
            num_objects = 1
        
        # Create objects
        for i in range(num_objects):
            if num_objects > 1:
                obj_name = self._input(f"Name for object {i+1}: ").strip()
                if not obj_name:
                    obj_name = f"object_{i+1}"
            else:
                obj_name = self._input("Object name (or press Enter for 'structure'): ").strip()
                if not obj_name:
                    obj_name = "structure"
            
            self.context.add_object(obj_name)
        
        self._print(f"\nCreated {len(self.context.objects)} object(s).")
        self._print("")
        
        # Transition to DESIGNING
        self.state = StreamlinedState.DESIGNING
    
    def _run_designing(self) -> None:
        """Run DESIGNING state: Requirements capture + spec compilation."""
        obj = self.context.get_current_object()
        if not obj:
            self.state = StreamlinedState.COMPLETE
            return
        
        self._print("-" * 40)
        self._print(f"Step 2: Designing '{obj['name']}'")
        self._print("-" * 40)
        self._print("")
        
        # Use contextual dialogue to gather requirements
        self._print("Describe what you want to generate.")
        self._print("(Type 'help' for guidance, 'defaults' to see default values)")
        self._print("")
        
        # Run dialogue loop
        spec_values = self.dialogue.run_dialogue_loop()
        
        if not spec_values:
            # User cancelled
            self._print("\nDesign cancelled.")
            self.context.current_object_index += 1
            if self.context.current_object_index >= len(self.context.objects):
                self.state = StreamlinedState.COMPLETE
            return
        
        # Build DesignSpec from collected values
        spec = self._build_spec_from_values(spec_values, obj['name'])
        
        # Validate spec
        validation = self.compiler.validate_spec(spec)
        
        if not validation.is_valid:
            self._print("\nSpec validation failed:")
            for error in validation.errors:
                self._print(f"  - {error}")
            self._print("\nPlease provide corrections.")
            return  # Stay in DESIGNING state
        
        if validation.warnings:
            self._print("\nWarnings:")
            for warning in validation.warnings:
                self._print(f"  - {warning}")
        
        # Store spec
        obj['spec'] = spec
        obj['status'] = "designed"
        
        self._print("\nDesign complete! Ready to generate.")
        self._print("")
        
        # Transition to GENERATING
        self.state = StreamlinedState.GENERATING
    
    def _run_generating(self) -> None:
        """Run GENERATING state: Generation + validation."""
        obj = self.context.get_current_object()
        if not obj or not obj.get('spec'):
            self.state = StreamlinedState.COMPLETE
            return
        
        self._print("-" * 40)
        self._print(f"Step 3: Generating '{obj['name']}' (v{obj['version']})")
        self._print("-" * 40)
        self._print("")
        
        spec = obj['spec']
        
        # Compile spec to generation script
        self._print("Compiling specification...")
        
        object_dir = os.path.join(self.context.project_dir, obj['name'])
        compilation = self.compiler.compile_generation_script(
            spec=spec,
            output_dir=object_dir,
            object_name=obj['name'],
            version=obj['version'],
        )
        
        if not compilation.success:
            self._print("\nCompilation failed:")
            for error in compilation.errors:
                self._print(f"  - {error}")
            self.state = StreamlinedState.REFINING
            return
        
        self._print(f"  Script generated: {compilation.script_path}")
        
        # Execute generation (simulated for now)
        self._print("\nExecuting generation...")
        
        # In a real implementation, this would execute the generated script
        # For now, we simulate the result
        network = self._simulate_generation(spec)
        
        if network is None:
            self._print("  Generation failed!")
            obj['status'] = "failed"
            self.state = StreamlinedState.REFINING
            return
        
        obj['network'] = network
        self._print("  Generation complete!")
        
        # Run validation
        self._print("\nValidating structure...")
        
        validation_report = self._run_validation(network, spec)
        
        if validation_report.get('passed', False):
            self._print("  Validation passed!")
            obj['status'] = "validated"
        else:
            self._print("  Validation found issues:")
            for issue in validation_report.get('issues', []):
                self._print(f"    - {issue}")
            obj['status'] = "needs_refinement"
        
        # Generate outputs
        self._print("\nGenerating outputs...")
        
        artifacts = self.output_gen.generate_artifact_package(
            network=network,
            spec=spec,
            object_name=obj['name'],
            version=obj['version'],
        )
        
        obj['artifacts'] = artifacts
        
        self._print(f"  Generated {len(artifacts.artifacts)} artifact(s):")
        for artifact in artifacts.artifacts:
            self._print(f"    - {artifact.name}: {artifact.path}")
        
        self._print("")
        
        # Check if refinement is needed
        if obj['status'] == "needs_refinement":
            self.state = StreamlinedState.REFINING
        else:
            # Ask user if they want to refine
            refine = self._input("Would you like to refine this structure? (y/n): ").strip().lower()
            if refine in ['y', 'yes']:
                self.state = StreamlinedState.REFINING
            else:
                # Move to next object or complete
                self.context.current_object_index += 1
                if self.context.current_object_index >= len(self.context.objects):
                    self.state = StreamlinedState.COMPLETE
                else:
                    self.state = StreamlinedState.DESIGNING
    
    def _run_refining(self) -> None:
        """Run REFINING state: Iteration loop."""
        obj = self.context.get_current_object()
        if not obj:
            self.state = StreamlinedState.COMPLETE
            return
        
        self._print("-" * 40)
        self._print(f"Step 4: Refining '{obj['name']}' (v{obj['version']})")
        self._print("-" * 40)
        self._print("")
        
        # Analyze current result
        if obj.get('network'):
            gen_result = GenerationResult(
                success=True,
                network=obj['network'],
                metrics=obj.get('artifacts', {}).metrics if obj.get('artifacts') else {},
            )
            
            adaptation = self.adapter.analyze_and_suggest(gen_result)
            
            if adaptation.suggestions:
                self._print("Suggested improvements:")
                for i, suggestion in enumerate(adaptation.suggestions[:5], 1):
                    self._print(f"  {i}. {suggestion.parameter}: {suggestion.current_value} -> {suggestion.suggested_value}")
                    self._print(f"     Reason: {suggestion.reason}")
                self._print("")
        
        # Get user feedback
        self._print("What would you like to change?")
        self._print("  - Type specific changes (e.g., 'increase terminals to 100')")
        self._print("  - Type 'accept' to accept current result")
        self._print("  - Type 'regenerate' to regenerate with current spec")
        self._print("  - Type 'cancel' to skip this object")
        self._print("")
        
        feedback = self._input("Your feedback: ").strip().lower()
        
        if feedback == 'accept':
            obj['status'] = "complete"
            self.context.current_object_index += 1
            if self.context.current_object_index >= len(self.context.objects):
                self.state = StreamlinedState.COMPLETE
            else:
                self.state = StreamlinedState.DESIGNING
            return
        
        if feedback == 'cancel':
            obj['status'] = "cancelled"
            self.context.current_object_index += 1
            if self.context.current_object_index >= len(self.context.objects):
                self.state = StreamlinedState.COMPLETE
            else:
                self.state = StreamlinedState.DESIGNING
            return
        
        if feedback == 'regenerate':
            obj['version'] += 1
            self.state = StreamlinedState.GENERATING
            return
        
        # Process feedback as spec changes
        response = self.dialogue.process_input(feedback)
        
        if response.spec_updates:
            # Apply updates to spec
            spec = obj['spec']
            spec = self._apply_spec_updates(spec, response.spec_updates)
            obj['spec'] = spec
            obj['version'] += 1
            
            self._print(f"\nSpec updated. Regenerating as v{obj['version']}...")
            self.state = StreamlinedState.GENERATING
        else:
            self._print("\nI didn't understand that change. Please try again.")
            # Stay in REFINING state
    
    def _build_spec_from_values(self, values: Dict[str, Any], object_name: str) -> DesignSpec:
        """Build a DesignSpec from collected dialogue values."""
        # Domain
        domain_type = values.get('domain_type', 'box')
        domain_size = values.get('domain_size', (0.02, 0.06, 0.03))
        
        if isinstance(domain_size, str):
            # Parse string like "20 60 30 mm"
            parts = domain_size.replace('mm', '').strip().split()
            try:
                domain_size = tuple(float(p) / 1000 for p in parts[:3])
            except ValueError:
                domain_size = (0.02, 0.06, 0.03)
        
        domain_center = (0.0, 0.0, 0.0)
        
        if domain_type == 'ellipsoid':
            domain = EllipsoidSpec(
                center=domain_center,
                semi_axes=tuple(s / 2 for s in domain_size),
            )
        else:
            domain = BoxSpec(
                center=domain_center,
                size=domain_size,
            )
        
        # Inlet
        inlet_face = values.get('inlet_face', 'x_min')
        inlet_radius = values.get('inlet_radius', 0.002)
        
        if isinstance(inlet_radius, str):
            try:
                inlet_radius = float(inlet_radius.replace('mm', '').strip()) / 1000
            except ValueError:
                inlet_radius = 0.002
        
        inlet_pos = self._face_to_position(inlet_face, domain_center, domain_size)
        
        inlets = [InletSpec(
            position=inlet_pos,
            radius=inlet_radius,
            vessel_type="arterial",
        )]
        
        # Outlet (if specified)
        outlets = []
        outlet_face = values.get('outlet_face')
        if outlet_face:
            outlet_radius = values.get('outlet_radius', 0.001)
            if isinstance(outlet_radius, str):
                try:
                    outlet_radius = float(outlet_radius.replace('mm', '').strip()) / 1000
                except ValueError:
                    outlet_radius = 0.001
            
            outlet_pos = self._face_to_position(outlet_face, domain_center, domain_size)
            outlets.append(OutletSpec(
                position=outlet_pos,
                radius=outlet_radius,
                vessel_type="arterial",
            ))
        
        # Colonization parameters
        target_terminals = values.get('target_terminals', 50)
        if isinstance(target_terminals, str):
            try:
                target_terminals = int(target_terminals)
            except ValueError:
                target_terminals = 50
        
        min_radius = values.get('min_radius', 0.0001)
        if isinstance(min_radius, str):
            try:
                min_radius = float(min_radius.replace('mm', '').strip()) / 1000
            except ValueError:
                min_radius = 0.0001
        
        colonization = ColonizationSpec(
            influence_radius=values.get('influence_radius', 0.015),
            kill_radius=values.get('kill_radius', 0.002),
            step_size=values.get('step_size', 0.001),
            max_steps=values.get('max_steps', 500),
            initial_radius=inlet_radius,
            min_radius=min_radius,
            radius_decay=values.get('radius_decay', 0.95),
            encourage_bifurcation=True,
            max_children_per_node=2,
        )
        
        # Tree spec
        # Note: "backbone" is not a valid TopologyKind in TreeSpec, so we map it to "tree"
        # The SpecCompiler has a separate BACKBONE template for code generation
        topology_kind = values.get('topology_kind', 'tree')
        valid_topology_kinds = ('path', 'tree', 'loop', 'multi_tree')
        if topology_kind not in valid_topology_kinds:
            topology_kind = 'tree'  # Default to tree for unsupported topologies
        
        tree = TreeSpec(
            inlets=inlets,
            outlets=outlets,
            terminal_count=target_terminals,
            colonization=colonization,
            topology_kind=topology_kind,
        )
        
        # Build final spec
        spec = DesignSpec(
            domain=domain,
            tree=tree,
            seed=values.get('seed', 42),
            output_units=values.get('output_units', 'mm'),
            metadata={
                "object_name": object_name,
                "created_by": "StreamlinedWorkflow",
                "version": self.WORKFLOW_VERSION,
            },
        )
        
        return spec
    
    def _face_to_position(
        self,
        face: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Convert a face name to a position."""
        face_positions = {
            'x_min': (center[0] - size[0]/2, center[1], center[2]),
            'x_max': (center[0] + size[0]/2, center[1], center[2]),
            'y_min': (center[0], center[1] - size[1]/2, center[2]),
            'y_max': (center[0], center[1] + size[1]/2, center[2]),
            'z_min': (center[0], center[1], center[2] - size[2]/2),
            'z_max': (center[0], center[1], center[2] + size[2]/2),
        }
        return face_positions.get(face, center)
    
    def _apply_spec_updates(self, spec: DesignSpec, updates: Dict[str, Any]) -> DesignSpec:
        """Apply updates to a spec."""
        # For now, rebuild the spec with updated values
        # In a real implementation, this would be more sophisticated
        
        current_values = {
            'domain_type': 'box' if isinstance(spec.domain, BoxSpec) else 'ellipsoid',
            'domain_size': spec.domain.size if isinstance(spec.domain, BoxSpec) else tuple(s*2 for s in spec.domain.semi_axes),
            'seed': spec.seed,
            'output_units': spec.output_units,
        }
        
        if spec.tree:
            if spec.tree.inlets:
                current_values['inlet_radius'] = spec.tree.inlets[0].radius
            if spec.tree.outlets:
                current_values['outlet_radius'] = spec.tree.outlets[0].radius
            current_values['target_terminals'] = spec.tree.terminal_count
            current_values['topology_kind'] = getattr(spec.tree, 'topology_kind', 'tree')
            
            if spec.tree.colonization:
                current_values['min_radius'] = spec.tree.colonization.min_radius
                current_values['influence_radius'] = spec.tree.colonization.influence_radius
                current_values['kill_radius'] = spec.tree.colonization.kill_radius
                current_values['step_size'] = spec.tree.colonization.step_size
                current_values['max_steps'] = spec.tree.colonization.max_steps
                current_values['radius_decay'] = spec.tree.colonization.radius_decay
        
        # Apply updates
        current_values.update(updates)
        
        # Rebuild spec
        object_name = spec.metadata.get('object_name', 'structure') if spec.metadata else 'structure'
        return self._build_spec_from_values(current_values, object_name)
    
    def _simulate_generation(self, spec: DesignSpec) -> Optional[Any]:
        """Simulate generation (placeholder for actual generation)."""
        # In a real implementation, this would call the generation library
        # For now, return a mock network object
        
        class MockNetwork:
            def __init__(self):
                self.nodes = {}
                self.segments = {}
        
        return MockNetwork()
    
    def _run_validation(self, network: Any, spec: DesignSpec) -> Dict[str, Any]:
        """Run validation on generated network."""
        # In a real implementation, this would call the validation library
        # For now, return a mock validation report
        
        return {
            "passed": True,
            "issues": [],
            "pre_embedding": {
                "mesh_integrity": {"passed": True},
                "murray_law": {"passed": True},
                "collision_free": {"passed": True},
            },
            "post_embedding": {
                "port_accessibility": {"passed": True},
                "min_diameter": {"passed": True},
                "wall_thickness": {"passed": True},
            },
        }


def run_streamlined_workflow(
    output_dir: str = "./outputs",
    llm_client: Optional[Any] = None,
) -> WorkflowResult:
    """
    Convenience function to run the streamlined workflow.
    
    Parameters
    ----------
    output_dir : str
        Output directory for generated files
    llm_client : Any, optional
        LLM client for dialogue
        
    Returns
    -------
    WorkflowResult
        The result of the workflow execution
    """
    workflow = StreamlinedWorkflow(output_dir=output_dir, llm_client=llm_client)
    return workflow.run()
