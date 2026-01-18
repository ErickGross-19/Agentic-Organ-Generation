"""
DesignSpecRunner - Core pipeline for executing DesignSpec specifications.

This module provides the main runner that orchestrates the full generation
pipeline from spec to validated outputs.

PIPELINE STAGES
---------------
1. compile_policies - Compile policy dicts to aog_policies objects
2. compile_domains - Compile domain dicts to runtime Domain objects
3. component_ports:<id> - Resolve port positions for each component
4. component_build:<id> - Generate network/mesh for each component
5. component_mesh:<id> - Convert network to void mesh for each component
6. union_voids - Union all component void meshes
7. mesh_domain - Generate domain mesh
8. embed - Embed unified void into domain
9. port_recarve - Recarve ports if enabled
10. validity - Run validity checks
11. export - Export outputs to files

KEY INVARIANTS
--------------
1. Multi-component: Union all void contributions, then embed once
2. All behavior controlled through aog_policies objects
3. Every stage emits structured report
4. Hierarchical pathfinding mandatory when used

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path
import logging
import time
import json

from .spec import DesignSpec
from .plan import ExecutionPlan, Stage, STAGE_ORDER
from .context import RunnerContext, ArtifactStore

if TYPE_CHECKING:
    import trimesh
    from generation.core.domain import DomainSpec as RuntimeDomain
    from generation.core.network import VascularNetwork

logger = logging.getLogger(__name__)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries with override semantics.
    
    Merge rules:
    - dict values are merged recursively
    - lists are replaced (no concatenation)
    - scalar values override
    
    Parameters
    ----------
    base : dict
        Base dictionary (typically global policies)
    override : dict
        Override dictionary (typically component-level overrides)
        
    Returns
    -------
    dict
        Merged dictionary with override values taking precedence
    """
    if not override:
        return dict(base) if base else {}
    if not base:
        return dict(override)
    
    result = dict(base)
    for key, override_value in override.items():
        if key in result:
            base_value = result[key]
            # Both are dicts: merge recursively
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                result[key] = deep_merge(base_value, override_value)
            else:
                # Lists and scalars: override completely
                result[key] = override_value
        else:
            # Key only in override
            result[key] = override_value
    
    return result


@dataclass
class StageReport:
    """Report from a single pipeline stage."""
    stage: str
    success: bool
    duration_s: float = 0.0
    requested_policy: Optional[Dict[str, Any]] = None
    effective_policy: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "success": self.success,
            "duration_s": self.duration_s,
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class RunnerResult:
    """Result from a DesignSpecRunner execution."""
    success: bool
    spec_hash: str
    stages_completed: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    stage_reports: List[StageReport] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    total_duration_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "spec_hash": self.spec_hash,
            "stages_completed": self.stages_completed,
            "stages_skipped": self.stages_skipped,
            "stage_reports": [r.to_dict() for r in self.stage_reports],
            "warnings": self.warnings,
            "errors": self.errors,
            "artifacts": self.artifacts,
            "total_duration_s": self.total_duration_s,
        }


class DesignSpecRunner:
    """
    Runner for executing DesignSpec specifications.
    
    Orchestrates the full pipeline from spec to validated outputs,
    enforcing key invariants:
    - Multi-component: Union all voids, then embed once
    - All behavior controlled through policy objects
    - Every stage emits structured report
    
    Parameters
    ----------
    spec : DesignSpec
        The loaded and normalized specification
    plan : ExecutionPlan, optional
        Execution plan for partial execution control
    output_dir : str or Path, optional
        Output directory for artifacts
        
    Attributes
    ----------
    spec : DesignSpec
        The specification being executed
    plan : ExecutionPlan
        The execution plan
    context : RunnerContext
        Caching context for intermediate results
    artifacts : ArtifactStore
        Store for named artifacts
    """
    
    def __init__(
        self,
        spec: DesignSpec,
        plan: Optional[ExecutionPlan] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.spec = spec
        self.plan = plan or ExecutionPlan()
        
        if output_dir is None:
            output_policy = spec.policies.get("output", {})
            output_dir = output_policy.get("output_dir", "./out")
        
        self.output_dir = Path(output_dir)
        self.context = RunnerContext(spec_hash=spec.spec_hash, seed=spec.seed)
        self.artifacts = ArtifactStore(self.output_dir)
        
        self._compiled_policies: Dict[str, Any] = {}
        self._compiled_domains: Dict[str, Any] = {}
        self._component_networks: Dict[str, Any] = {}
        self._component_voids: Dict[str, "trimesh.Trimesh"] = {}
        self._resolved_ports: Dict[str, Dict[str, Any]] = {}  # component_id -> resolved ports
        self._union_void: Optional["trimesh.Trimesh"] = None
        self._domain_mesh: Optional["trimesh.Trimesh"] = None
        self._embedded_solid: Optional["trimesh.Trimesh"] = None
        self._embedded_shell: Optional["trimesh.Trimesh"] = None
        self._validity_report: Optional[Any] = None
        
        self._stage_reports: List[StageReport] = []
        self._warnings: List[str] = []
        self._errors: List[str] = []
        
        component_ids = [c["id"] for c in spec.components]
        self.plan.set_component_ids(component_ids)
        
        self._register_requested_artifacts()
    
    def _register_requested_artifacts(self) -> None:
        """Register artifacts requested in the spec."""
        for component in self.spec.components:
            save_artifacts = component.get("save_artifacts", {})
            for name, path in save_artifacts.items():
                artifact_name = f"{component['id']}_{name}"
                self.artifacts.request_artifact(artifact_name, path)
        
        composition = self.spec.normalized.get("composition", {})
        if "save_union" in composition:
            self.artifacts.request_artifact("union_void", composition["save_union"])
        
        embedding = self.spec.normalized.get("embedding", {})
        outputs = embedding.get("outputs", {})
        for name, path in outputs.items():
            self.artifacts.request_artifact(name, path)
        
        spec_outputs = self.spec.normalized.get("outputs", {})
        named = spec_outputs.get("named", {})
        for name, path in named.items():
            self.artifacts.request_artifact(name, path)
    
    def run(self) -> RunnerResult:
        """
        Execute the pipeline according to the execution plan.
        
        Returns
        -------
        RunnerResult
            Result containing success status, stage reports, and artifacts
        """
        start_time = time.time()
        
        stages_to_run = self.plan.compute_stages()
        stages_completed = []
        stages_skipped = []
        
        try:
            for stage in stages_to_run:
                if not self.plan.should_run(stage):
                    stages_skipped.append(stage)
                    continue
                
                stage_start = time.time()
                
                try:
                    report = self._execute_stage(stage)
                    report.duration_s = time.time() - stage_start
                    self._stage_reports.append(report)
                    
                    if report.success:
                        stages_completed.append(stage)
                    else:
                        self._errors.extend(report.errors)
                        break
                    
                    self._warnings.extend(report.warnings)
                    
                except Exception as e:
                    logger.exception(f"Stage {stage} failed with exception")
                    self._errors.append(f"Stage {stage} failed: {e}")
                    self._stage_reports.append(StageReport(
                        stage=stage,
                        success=False,
                        duration_s=time.time() - stage_start,
                        errors=[str(e)],
                    ))
                    break
            
            success = len(self._errors) == 0
            
        except Exception as e:
            logger.exception("Runner failed with exception")
            self._errors.append(f"Runner failed: {e}")
            success = False
            stages_completed = []
        
        total_duration = time.time() - start_time
        
        return RunnerResult(
            success=success,
            spec_hash=self.spec.spec_hash,
            stages_completed=stages_completed,
            stages_skipped=stages_skipped,
            stage_reports=self._stage_reports,
            warnings=self._warnings,
            errors=self._errors,
            artifacts=self.artifacts.build_manifest(),
            total_duration_s=total_duration,
        )
    
    def _execute_stage(self, stage: str) -> StageReport:
        """Execute a single pipeline stage."""
        base_stage = stage.split(":")[0]
        component_id = stage.split(":")[1] if ":" in stage else None
        
        if base_stage == Stage.COMPILE_POLICIES.value:
            return self._stage_compile_policies()
        elif base_stage == Stage.COMPILE_DOMAINS.value:
            return self._stage_compile_domains()
        elif base_stage == Stage.COMPONENT_PORTS.value:
            return self._stage_component_ports(component_id)
        elif base_stage == Stage.COMPONENT_BUILD.value:
            return self._stage_component_build(component_id)
        elif base_stage == Stage.COMPONENT_MESH.value:
            return self._stage_component_mesh(component_id)
        elif base_stage == Stage.UNION_VOIDS.value:
            return self._stage_union_voids()
        elif base_stage == Stage.MESH_DOMAIN.value:
            return self._stage_mesh_domain()
        elif base_stage == Stage.EMBED.value:
            return self._stage_embed()
        elif base_stage == Stage.PORT_RECARVE.value:
            return self._stage_port_recarve()
        elif base_stage == Stage.VALIDITY.value:
            return self._stage_validity()
        elif base_stage == Stage.EXPORT.value:
            return self._stage_export()
        else:
            return StageReport(
                stage=stage,
                success=False,
                errors=[f"Unknown stage: {stage}"],
            )
    
    def _stage_compile_policies(self) -> StageReport:
        """Compile policy dicts to aog_policies objects."""
        from aog_policies import (
            ResolutionPolicy,
            GrowthPolicy,
            CollisionPolicy,
            EmbeddingPolicy,
            TissueSamplingPolicy,
            ValidationPolicy,
            OpenPortPolicy,
            RepairPolicy,
            ComposePolicy,
            PathfindingPolicy,
            PortPlacementPolicy,
            ChannelPolicy,
            OutputPolicy,
            DomainMeshingPolicy,
        )
        
        warnings = []
        metadata = {}
        
        policy_classes = {
            "resolution": ResolutionPolicy,
            "growth": GrowthPolicy,
            "collision": CollisionPolicy,
            "embedding": EmbeddingPolicy,
            "tissue_sampling": TissueSamplingPolicy,
            "validation": ValidationPolicy,
            "validity": ValidationPolicy,
            "open_port": OpenPortPolicy,
            "repair": RepairPolicy,
            "composition": ComposePolicy,
            "pathfinding": PathfindingPolicy,
            "ports": PortPlacementPolicy,
            "channels": ChannelPolicy,
            "output": OutputPolicy,
            "domain_meshing": DomainMeshingPolicy,
        }
        
        for policy_name, policy_dict in self.spec.policies.items():
            policy_class = policy_classes.get(policy_name)
            
            if policy_class is None:
                warnings.append(f"Unknown policy type: {policy_name}")
                self._compiled_policies[policy_name] = policy_dict
                continue
            
            try:
                if hasattr(policy_class, "from_dict"):
                    policy_obj = policy_class.from_dict(policy_dict)
                else:
                    policy_obj = policy_class(**policy_dict)
                
                self._compiled_policies[policy_name] = policy_obj
                metadata[policy_name] = "compiled"
                
            except Exception as e:
                warnings.append(f"Failed to compile {policy_name}: {e}")
                self._compiled_policies[policy_name] = policy_dict
                metadata[policy_name] = f"fallback: {e}"
        
        return StageReport(
            stage=Stage.COMPILE_POLICIES.value,
            success=True,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _stage_compile_domains(self) -> StageReport:
        """
        Compile domain dicts to runtime Domain objects.
        
        By default, fails loudly if domain compilation fails.
        Set meta.allow_domain_compile_fallback = true to allow fallback to dict.
        """
        from generation.core.domain import domain_from_dict
        
        # Check if fallback is allowed via meta flag
        meta = self.spec.normalized.get("meta", {})
        allow_fallback = meta.get("allow_domain_compile_fallback", False)
        
        warnings = []
        errors = []
        metadata = {}
        
        for domain_name, domain_dict in self.spec.domains.items():
            try:
                domain_obj = domain_from_dict(domain_dict, input_units="m")
                self._compiled_domains[domain_name] = domain_obj
                metadata[domain_name] = "compiled"
                
            except Exception as e:
                if allow_fallback:
                    # Fallback allowed - store dict and warn
                    warnings.append(
                        f"Failed to compile domain {domain_name}: {e}. "
                        "Using dict fallback (allow_domain_compile_fallback=true)."
                    )
                    self._compiled_domains[domain_name] = domain_dict
                    metadata[domain_name] = f"fallback: {e}"
                else:
                    # Fallback not allowed - fail loudly
                    errors.append(f"Failed to compile domain {domain_name}: {e}")
                    metadata[domain_name] = f"error: {e}"
        
        # If any errors and fallback not allowed, fail the stage
        if errors:
            return StageReport(
                stage=Stage.COMPILE_DOMAINS.value,
                success=False,
                warnings=warnings,
                errors=errors,
                metadata=metadata,
            )
        
        return StageReport(
            stage=Stage.COMPILE_DOMAINS.value,
            success=True,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _stage_component_ports(self, component_id: str) -> StageReport:
        """
        Resolve port positions for a component.
        
        This stage resolves ports domain-aware using PortPlacementPolicy + ridge
        constraints + face rules. It outputs resolved ports to _resolved_ports
        so later stages use resolved positions instead of raw ports.
        
        Supports:
        - layout on reference plane
        - clamp to face region
        - optional project to boundary
        - ridge effective radius convention
        - port validation: min separation, clamp/project, autofix warnings
        """
        from generation.utils.port_placement import place_ports_on_domain
        from aog_policies import PortPlacementPolicy, RidgePolicy
        
        component = self._get_component(component_id)
        if component is None:
            return StageReport(
                stage=f"{Stage.COMPONENT_PORTS.value}:{component_id}",
                success=False,
                errors=[f"Component not found: {component_id}"],
            )
        
        # Get domain for this component
        domain_ref = component.get("domain_ref", "main_domain")
        domain = self._compiled_domains.get(domain_ref)
        
        if domain is None:
            return StageReport(
                stage=f"{Stage.COMPONENT_PORTS.value}:{component_id}",
                success=False,
                errors=[f"Domain not found for port resolution: {domain_ref}"],
            )
        
        # Check if domain is a dict (fallback) - cannot resolve ports on dict
        if isinstance(domain, dict):
            return StageReport(
                stage=f"{Stage.COMPONENT_PORTS.value}:{component_id}",
                success=False,
                errors=[f"Domain {domain_ref} is not compiled - cannot resolve ports"],
            )
        
        ports = component.get("ports", {})
        inlets = ports.get("inlets", [])
        outlets = ports.get("outlets", [])
        
        # Get effective port placement policy for this component
        effective_ports_dict = self._get_effective_policy_dict(component, "ports")
        port_policy = PortPlacementPolicy.from_dict(effective_ports_dict)
        
        # Get effective ridge policy if present
        effective_ridge_dict = self._get_effective_policy_dict(component, "ridge")
        ridge_policy = None
        if effective_ridge_dict:
            ridge_policy = RidgePolicy.from_dict(effective_ridge_dict)
            # Apply ridge constraints to port policy if ridge is defined
            if ridge_policy.thickness > 0:
                port_policy.ridge_width = ridge_policy.thickness
                port_policy.ridge_constraint_enabled = True
        
        warnings = []
        resolved_inlets = []
        resolved_outlets = []
        
        # Resolve inlets
        for i, inlet in enumerate(inlets):
            resolved_inlet = dict(inlet)
            
            # If position is already specified, use it directly
            if "position" in inlet and inlet["position"] is not None:
                resolved_inlet["resolved"] = True
                resolved_inlet["resolution_method"] = "explicit"
            else:
                # Need to place this port using the policy
                port_radius = inlet.get("radius", 0.0005)  # Default 0.5mm
                try:
                    result, report = place_ports_on_domain(
                        domain=domain,
                        num_ports=1,
                        port_radius=port_radius,
                        policy=port_policy,
                    )
                    if result.positions:
                        resolved_inlet["position"] = list(result.positions[0])
                        resolved_inlet["direction"] = list(result.directions[0])
                        resolved_inlet["resolved"] = True
                        resolved_inlet["resolution_method"] = "policy_placement"
                        resolved_inlet["effective_radius"] = result.effective_radius
                        if result.warnings:
                            warnings.extend([f"Inlet {i}: {w}" for w in result.warnings])
                    else:
                        warnings.append(f"Inlet {i}: No position resolved")
                        resolved_inlet["resolved"] = False
                except Exception as e:
                    warnings.append(f"Inlet {i}: Port placement failed: {e}")
                    resolved_inlet["resolved"] = False
            
            resolved_inlets.append(resolved_inlet)
        
        # Resolve outlets
        for i, outlet in enumerate(outlets):
            resolved_outlet = dict(outlet)
            
            # If position is already specified, use it directly
            if "position" in outlet and outlet["position"] is not None:
                resolved_outlet["resolved"] = True
                resolved_outlet["resolution_method"] = "explicit"
            else:
                # Need to place this port using the policy
                port_radius = outlet.get("radius", 0.0005)  # Default 0.5mm
                try:
                    result, report = place_ports_on_domain(
                        domain=domain,
                        num_ports=1,
                        port_radius=port_radius,
                        policy=port_policy,
                    )
                    if result.positions:
                        resolved_outlet["position"] = list(result.positions[0])
                        resolved_outlet["direction"] = list(result.directions[0])
                        resolved_outlet["resolved"] = True
                        resolved_outlet["resolution_method"] = "policy_placement"
                        resolved_outlet["effective_radius"] = result.effective_radius
                        if result.warnings:
                            warnings.extend([f"Outlet {i}: {w}" for w in result.warnings])
                    else:
                        warnings.append(f"Outlet {i}: No position resolved")
                        resolved_outlet["resolved"] = False
                except Exception as e:
                    warnings.append(f"Outlet {i}: Port placement failed: {e}")
                    resolved_outlet["resolved"] = False
            
            resolved_outlets.append(resolved_outlet)
        
        # Store resolved ports for later stages
        self._resolved_ports[component_id] = {
            "inlets": resolved_inlets,
            "outlets": resolved_outlets,
        }
        
        metadata = {
            "inlet_count": len(inlets),
            "outlet_count": len(outlets),
            "resolved_inlet_count": sum(1 for p in resolved_inlets if p.get("resolved")),
            "resolved_outlet_count": sum(1 for p in resolved_outlets if p.get("resolved")),
            "domain_ref": domain_ref,
        }
        
        return StageReport(
            stage=f"{Stage.COMPONENT_PORTS.value}:{component_id}",
            success=True,
            requested_policy=self.spec.policies.get("ports", {}),
            effective_policy=effective_ports_dict,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _stage_component_build(self, component_id: str) -> StageReport:
        """
        Generate network/mesh for a component.
        
        Uses resolved ports from _resolved_ports if available, otherwise falls
        back to raw ports. Applies component-level policy overrides via deep_merge.
        """
        component = self._get_component(component_id)
        if component is None:
            return StageReport(
                stage=f"{Stage.COMPONENT_BUILD.value}:{component_id}",
                success=False,
                errors=[f"Component not found: {component_id}"],
            )
        
        build = component.get("build", {})
        build_type = build.get("type", "backend_network")
        
        domain_ref = component.get("domain_ref", "main_domain")
        domain = self._compiled_domains.get(domain_ref)
        
        if domain is None:
            return StageReport(
                stage=f"{Stage.COMPONENT_BUILD.value}:{component_id}",
                success=False,
                errors=[f"Domain not found: {domain_ref}"],
            )
        
        # Check if domain is a dict (fallback) - cannot build on dict
        if isinstance(domain, dict):
            return StageReport(
                stage=f"{Stage.COMPONENT_BUILD.value}:{component_id}",
                success=False,
                errors=[f"Domain {domain_ref} is not compiled - cannot build component"],
            )
        
        # Use resolved ports if available, otherwise fall back to raw ports
        if component_id in self._resolved_ports:
            ports = self._resolved_ports[component_id]
        else:
            ports = component.get("ports", {})
        
        warnings = []
        metadata = {"build_type": build_type, "domain_ref": domain_ref}
        
        # Get effective policy dicts for this component (with overrides applied)
        effective_growth_dict = self._get_effective_policy_dict(component, "growth")
        effective_collision_dict = self._get_effective_policy_dict(component, "collision")
        
        try:
            if build_type == "backend_network":
                network, report = self._build_backend_network(
                    domain, ports, build, component_id, component
                )
                self._component_networks[component_id] = network
                metadata["node_count"] = len(network.nodes)
                metadata["segment_count"] = len(network.segments)
                
            elif build_type == "primitive_channels":
                void_mesh, report = self._build_primitive_channels(
                    domain, ports, build, component_id, component
                )
                self._component_voids[component_id] = void_mesh
                metadata["vertex_count"] = len(void_mesh.vertices)
                metadata["face_count"] = len(void_mesh.faces)
                
            elif build_type == "import_void_mesh":
                void_mesh = self._import_void_mesh(build)
                self._component_voids[component_id] = void_mesh
                metadata["vertex_count"] = len(void_mesh.vertices)
                metadata["face_count"] = len(void_mesh.faces)
                
            else:
                return StageReport(
                    stage=f"{Stage.COMPONENT_BUILD.value}:{component_id}",
                    success=False,
                    errors=[f"Unknown build type: {build_type}"],
                )
            
            return StageReport(
                stage=f"{Stage.COMPONENT_BUILD.value}:{component_id}",
                success=True,
                requested_policy={
                    "growth": self.spec.policies.get("growth", {}),
                    "collision": self.spec.policies.get("collision", {}),
                },
                effective_policy={
                    "growth": effective_growth_dict,
                    "collision": effective_collision_dict,
                    "backend_params": build.get("backend_params", {}),
                },
                warnings=warnings,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.exception(f"Component build failed: {component_id}")
            return StageReport(
                stage=f"{Stage.COMPONENT_BUILD.value}:{component_id}",
                success=False,
                errors=[str(e)],
                metadata=metadata,
            )
    
    def _stage_component_mesh(self, component_id: str) -> StageReport:
        """
        Convert network to void mesh for a component.
        
        Uses aog_policies MeshSynthesisPolicy with component-level overrides.
        """
        if component_id in self._component_voids:
            return StageReport(
                stage=f"{Stage.COMPONENT_MESH.value}:{component_id}",
                success=True,
                metadata={"skipped": "already has void mesh"},
            )
        
        network = self._component_networks.get(component_id)
        if network is None:
            return StageReport(
                stage=f"{Stage.COMPONENT_MESH.value}:{component_id}",
                success=False,
                errors=[f"No network found for component: {component_id}"],
            )
        
        component = self._get_component(component_id)
        
        try:
            from generation.ops.mesh.synthesis import synthesize_mesh
            from aog_policies import MeshSynthesisPolicy
            
            # Get effective mesh synthesis policy with component overrides
            if component is not None:
                effective_mesh_dict = self._get_effective_policy_dict(component, "mesh_synthesis")
                policy = MeshSynthesisPolicy.from_dict(effective_mesh_dict)
            else:
                # Fallback to global compiled policy
                policy = self._compiled_policies.get("mesh_synthesis")
                if policy is None:
                    mesh_policy_dict = self.spec.policies.get("mesh_synthesis", {})
                    policy = MeshSynthesisPolicy.from_dict(mesh_policy_dict)
            
            void_mesh, synth_report = synthesize_mesh(network, policy)
            
            # Check for mesh synthesis failure or empty/degenerate mesh
            is_empty_mesh = (
                void_mesh is None or 
                len(void_mesh.vertices) == 0 or 
                len(void_mesh.faces) == 0
            )
            synth_failed = synth_report is not None and not synth_report.success
            
            if is_empty_mesh or synth_failed:
                error_msg = "Mesh synthesis produced empty mesh"
                if synth_report is not None and synth_report.errors:
                    error_msg = f"Mesh synthesis failed: {'; '.join(synth_report.errors)}"
                return StageReport(
                    stage=f"{Stage.COMPONENT_MESH.value}:{component_id}",
                    success=False,
                    errors=[error_msg],
                    metadata={
                        "vertex_count": len(void_mesh.vertices) if void_mesh is not None else 0,
                        "face_count": len(void_mesh.faces) if void_mesh is not None else 0,
                        "synth_report_success": synth_report.success if synth_report else None,
                    },
                )
            
            self._component_voids[component_id] = void_mesh
            
            artifact_name = f"{component_id}_void_mesh"
            self.artifacts.register(
                artifact_name,
                f"{Stage.COMPONENT_MESH.value}:{component_id}",
                void_mesh,
            )
            self.artifacts.save_mesh(artifact_name, void_mesh)
            
            # Get effective policy dict for reporting
            effective_mesh_dict = self._get_effective_policy_dict(component, "mesh_synthesis") if component else {}
            
            return StageReport(
                stage=f"{Stage.COMPONENT_MESH.value}:{component_id}",
                success=True,
                requested_policy=self.spec.policies.get("mesh_synthesis", {}),
                effective_policy=effective_mesh_dict,
                metadata={
                    "vertex_count": len(void_mesh.vertices),
                    "face_count": len(void_mesh.faces),
                },
            )
            
        except Exception as e:
            logger.exception(f"Mesh synthesis failed: {component_id}")
            return StageReport(
                stage=f"{Stage.COMPONENT_MESH.value}:{component_id}",
                success=False,
                errors=[str(e)],
            )
    
    def _stage_union_voids(self) -> StageReport:
        """Union all component void meshes."""
        if not self._component_voids:
            return StageReport(
                stage=Stage.UNION_VOIDS.value,
                success=False,
                errors=["No component voids to union"],
            )
        
        try:
            from generation.ops.compose import compose_components, ComponentSpec
            from aog_policies import ComposePolicy
            
            compose_policy_dict = self.spec.policies.get("composition", {})
            compose_policy = self._compiled_policies.get("composition")
            if compose_policy is None:
                compose_policy = ComposePolicy.from_dict(compose_policy_dict)
            
            components = []
            for comp_id, void_mesh in self._component_voids.items():
                components.append(ComponentSpec.from_mesh(void_mesh, name=comp_id))
            
            union_mesh, compose_report = compose_components(components, compose_policy)
            
            self._union_void = union_mesh
            
            self.artifacts.register("union_void", Stage.UNION_VOIDS.value, union_mesh)
            self.artifacts.save_mesh("union_void", union_mesh)
            
            return StageReport(
                stage=Stage.UNION_VOIDS.value,
                success=compose_report.success,
                warnings=compose_report.warnings,
                errors=compose_report.errors,
                metadata={
                    "components_merged": compose_report.meshes_merged,
                    "vertex_count": compose_report.vertex_count,
                    "face_count": compose_report.face_count,
                    "is_watertight": compose_report.is_watertight,
                },
            )
            
        except Exception as e:
            logger.exception("Union voids failed")
            return StageReport(
                stage=Stage.UNION_VOIDS.value,
                success=False,
                errors=[str(e)],
            )
    
    def _stage_mesh_domain(self) -> StageReport:
        """
        Generate domain mesh.
        
        Uses explicit domain selection rule:
        1. If spec.embedding.domain_ref exists, use it
        2. Else if single domain, use it
        3. Else error with clear message
        """
        domain_name, domain, error = self._select_domain_for_embedding()
        
        if error is not None:
            return StageReport(
                stage=Stage.MESH_DOMAIN.value,
                success=False,
                errors=[error],
            )
        
        # Check if domain is a dict (fallback) - cannot mesh a dict
        if isinstance(domain, dict):
            return StageReport(
                stage=Stage.MESH_DOMAIN.value,
                success=False,
                errors=[f"Domain '{domain_name}' is not compiled - cannot generate mesh"],
            )
        
        try:
            domain_mesh = domain.to_mesh()
            self._domain_mesh = domain_mesh
            
            self.artifacts.register("domain_mesh", Stage.MESH_DOMAIN.value, domain_mesh)
            
            return StageReport(
                stage=Stage.MESH_DOMAIN.value,
                success=True,
                metadata={
                    "vertex_count": len(domain_mesh.vertices),
                    "face_count": len(domain_mesh.faces),
                    "domain_name": domain_name,
                },
            )
            
        except Exception as e:
            logger.exception("Domain meshing failed")
            return StageReport(
                stage=Stage.MESH_DOMAIN.value,
                success=False,
                errors=[str(e)],
                metadata={"domain_name": domain_name},
            )
    
    def _stage_embed(self) -> StageReport:
        """
        Embed unified void into domain.
        
        Uses explicit domain selection rule:
        1. If spec.embedding.domain_ref exists, use it
        2. Else if single domain, use it
        3. Else error with clear message
        """
        if self._union_void is None:
            return StageReport(
                stage=Stage.EMBED.value,
                success=False,
                errors=["No union void to embed"],
            )
        
        domain_name, domain, error = self._select_domain_for_embedding()
        
        if error is not None:
            return StageReport(
                stage=Stage.EMBED.value,
                success=False,
                errors=[error],
            )
        
        # Check if domain is a dict (fallback) - cannot embed into dict
        if isinstance(domain, dict):
            return StageReport(
                stage=Stage.EMBED.value,
                success=False,
                errors=[f"Domain '{domain_name}' is not compiled - cannot embed void"],
            )
        
        try:
            from generation.api.embed import embed_void
            from aog_policies import EmbeddingPolicy, ResolutionPolicy
            
            embedding_policy = self._compiled_policies.get("embedding")
            if embedding_policy is None:
                embedding_policy_dict = self.spec.policies.get("embedding", {})
                embedding_policy = EmbeddingPolicy.from_dict(embedding_policy_dict)
            
            resolution_policy = self._compiled_policies.get("resolution")
            if resolution_policy is None:
                resolution_policy_dict = self.spec.policies.get("resolution", {})
                resolution_policy = ResolutionPolicy.from_dict(resolution_policy_dict)
            
            all_ports = self._collect_all_ports()
            
            solid, void_out, shell, embed_report = embed_void(
                domain=domain,
                void_mesh=self._union_void,
                embedding_policy=embedding_policy,
                ports=all_ports,
                resolution_policy=resolution_policy,
            )
            
            self._embedded_solid = solid
            self._embedded_shell = shell
            
            self.artifacts.register("domain_with_void", Stage.EMBED.value, solid)
            self.artifacts.save_mesh("domain_with_void", solid)
            
            self.artifacts.register("void_mesh", Stage.EMBED.value, void_out)
            self.artifacts.save_mesh("void_mesh", void_out)
            
            if shell is not None and len(shell.vertices) > 0:
                self.artifacts.register("shell", Stage.EMBED.value, shell)
                self.artifacts.save_mesh("shell", shell)
            
            # Add domain_name to metadata
            metadata = dict(embed_report.metadata) if embed_report.metadata else {}
            metadata["domain_name"] = domain_name
            
            return StageReport(
                stage=Stage.EMBED.value,
                success=embed_report.success,
                requested_policy=embed_report.requested_policy,
                effective_policy=embed_report.effective_policy,
                warnings=embed_report.warnings,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.exception("Embedding failed")
            return StageReport(
                stage=Stage.EMBED.value,
                success=False,
                errors=[str(e)],
            )
    
    def _stage_port_recarve(self) -> StageReport:
        """Recarve ports if enabled."""
        embedding_policy = self._compiled_policies.get("embedding")
        
        if embedding_policy is None or not getattr(embedding_policy, "preserve_ports_enabled", False):
            return StageReport(
                stage=Stage.PORT_RECARVE.value,
                success=True,
                metadata={"skipped": "port preservation not enabled"},
            )
        
        return StageReport(
            stage=Stage.PORT_RECARVE.value,
            success=True,
            metadata={"note": "port recarve handled in embed stage"},
        )
    
    def _stage_validity(self) -> StageReport:
        """Run validity checks."""
        try:
            from validity.runner import run_validity_checks
            from aog_policies import ValidationPolicy, OpenPortPolicy, RepairPolicy, ResolutionPolicy
            
            validation_policy = self._compiled_policies.get("validation")
            if validation_policy is None:
                validation_policy = self._compiled_policies.get("validity")
            if validation_policy is None:
                validation_policy_dict = self.spec.policies.get("validity", {})
                validation_policy = ValidationPolicy.from_dict(validation_policy_dict)
            
            open_port_policy = self._compiled_policies.get("open_port")
            if open_port_policy is None:
                open_port_policy_dict = self.spec.policies.get("open_port", {})
                open_port_policy = OpenPortPolicy.from_dict(open_port_policy_dict)
            
            resolution_policy = self._compiled_policies.get("resolution")
            if resolution_policy is None:
                resolution_policy_dict = self.spec.policies.get("resolution", {})
                resolution_policy = ResolutionPolicy.from_dict(resolution_policy_dict)
            
            repair_policy = self._compiled_policies.get("repair")
            if repair_policy is None:
                repair_policy_dict = self.spec.policies.get("repair", {})
                repair_policy = RepairPolicy.from_dict(repair_policy_dict)
            
            first_domain_name = list(self._compiled_domains.keys())[0]
            domain = self._compiled_domains.get(first_domain_name)
            
            all_ports = self._collect_all_ports()
            
            validity_report = run_validity_checks(
                domain=domain,
                domain_mesh=self._domain_mesh,
                domain_with_void=self._embedded_solid,
                void_mesh=self._union_void,
                ports=all_ports,
                validation_policy=validation_policy,
                open_port_policy=open_port_policy,
                resolution_policy=resolution_policy,
                repair_policy=repair_policy,
            )
            
            self._validity_report = validity_report
            
            validity_spec = self.spec.normalized.get("validity", {})
            if validity_spec.get("save_report"):
                report_path = self.output_dir / validity_spec["save_report"]
                report_path.parent.mkdir(parents=True, exist_ok=True)
                validity_report.save(report_path)
            
            return StageReport(
                stage=Stage.VALIDITY.value,
                success=validity_report.success,
                warnings=validity_report.warnings,
                errors=validity_report.errors,
                metadata={
                    "status": validity_report.status,
                    "total_checks": validity_report.metadata.get("total_checks", 0),
                    "passed_checks": validity_report.metadata.get("passed_checks", 0),
                    "failed_checks": validity_report.metadata.get("failed_checks", 0),
                },
            )
            
        except Exception as e:
            logger.exception("Validity checks failed")
            return StageReport(
                stage=Stage.VALIDITY.value,
                success=False,
                errors=[str(e)],
            )
    
    def _stage_export(self) -> StageReport:
        """Export outputs to files."""
        warnings = []
        metadata = {"exported": []}
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            output_policy = self._compiled_policies.get("output")
            if output_policy is None:
                output_policy_dict = self.spec.policies.get("output", {})
            
            if self._embedded_solid is not None:
                solid_path = self.output_dir / "domain_with_void.stl"
                self._embedded_solid.export(str(solid_path))
                metadata["exported"].append("domain_with_void.stl")
            
            if self._union_void is not None:
                void_path = self.output_dir / "void_mesh.stl"
                self._union_void.export(str(void_path))
                metadata["exported"].append("void_mesh.stl")
            
            if self._embedded_shell is not None and len(self._embedded_shell.vertices) > 0:
                shell_path = self.output_dir / "shell.stl"
                self._embedded_shell.export(str(shell_path))
                metadata["exported"].append("shell.stl")
            
            self.artifacts.save_manifest()
            metadata["exported"].append("artifact_manifest.json")
            
            return StageReport(
                stage=Stage.EXPORT.value,
                success=True,
                warnings=warnings,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.exception("Export failed")
            return StageReport(
                stage=Stage.EXPORT.value,
                success=False,
                errors=[str(e)],
                metadata=metadata,
            )
    
    def _get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a component by ID."""
        for component in self.spec.components:
            if component.get("id") == component_id:
                return component
        return None
    
    def _select_domain_for_embedding(self) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
        """
        Select domain for embedding/validity using explicit rules.
        
        Selection rule:
        1. If spec.embedding.domain_ref exists, use it
        2. Else if single domain, use it
        3. Else error with clear message
        
        Returns
        -------
        tuple
            (domain_name, domain_object, error_message)
            If successful: (name, domain, None)
            If error: (None, None, error_message)
        """
        embedding = self.spec.normalized.get("embedding", {})
        domain_ref = embedding.get("domain_ref")
        
        if domain_ref is not None:
            # Rule 1: Use explicit domain_ref from embedding
            domain = self._compiled_domains.get(domain_ref)
            if domain is None:
                return (None, None, f"Embedding domain_ref '{domain_ref}' not found in compiled domains")
            return (domain_ref, domain, None)
        
        # Rule 2/3: Check domain count
        domain_names = list(self._compiled_domains.keys())
        
        if len(domain_names) == 0:
            return (None, None, "No compiled domains available")
        
        if len(domain_names) == 1:
            # Rule 2: Single domain - use it
            domain_name = domain_names[0]
            return (domain_name, self._compiled_domains[domain_name], None)
        
        # Rule 3: Multiple domains without explicit domain_ref - error
        return (
            None,
            None,
            f"Multiple domains ({', '.join(domain_names)}) but no embedding.domain_ref specified. "
            "Please specify which domain to use for embedding via embedding.domain_ref."
        )
    
    def _get_effective_policy_dict(
        self,
        component: Dict[str, Any],
        policy_name: str,
    ) -> Dict[str, Any]:
        """
        Get effective policy dict for a component by merging global with overrides.
        
        Parameters
        ----------
        component : dict
            Component specification dict
        policy_name : str
            Name of the policy (e.g., "growth", "ports", "channels")
            
        Returns
        -------
        dict
            Effective policy dict with component overrides applied
        """
        global_policy = self.spec.policies.get(policy_name, {})
        component_overrides = component.get("policy_overrides", {}).get(policy_name, {})
        return deep_merge(global_policy, component_overrides)
    
    def _compile_effective_policy(
        self,
        component: Dict[str, Any],
        policy_name: str,
        policy_class: type,
    ) -> Any:
        """
        Compile effective policy object for a component.
        
        Parameters
        ----------
        component : dict
            Component specification dict
        policy_name : str
            Name of the policy (e.g., "growth", "ports", "channels")
        policy_class : type
            Policy class with from_dict method
            
        Returns
        -------
        Policy object
            Compiled policy object with component overrides applied
        """
        effective_dict = self._get_effective_policy_dict(component, policy_name)
        return policy_class.from_dict(effective_dict)
    
    def _collect_all_ports(self) -> List[Dict[str, Any]]:
        """Collect all ports from all components."""
        all_ports = []
        
        for component in self.spec.components:
            ports = component.get("ports", {})
            
            for inlet in ports.get("inlets", []):
                port_dict = dict(inlet)
                port_dict["component_id"] = component["id"]
                port_dict["port_type"] = "inlet"
                all_ports.append(port_dict)
            
            for outlet in ports.get("outlets", []):
                port_dict = dict(outlet)
                port_dict["component_id"] = component["id"]
                port_dict["port_type"] = "outlet"
                all_ports.append(port_dict)
        
        return all_ports
    
    def _build_backend_network(
        self,
        domain: "RuntimeDomain",
        ports: Dict[str, Any],
        build: Dict[str, Any],
        component_id: str,
        component: Optional[Dict[str, Any]] = None,
    ) -> Tuple["VascularNetwork", Any]:
        """
        Build a network using a generation backend.
        
        Uses component-level policy overrides if component is provided.
        Wires backend_params into GrowthPolicy.backend_params.
        """
        from generation.api.generate import generate_network
        from aog_policies import GrowthPolicy, CollisionPolicy, TissueSamplingPolicy
        
        backend = build.get("backend", "space_colonization")
        backend_params = build.get("backend_params", {})
        
        # Get effective policies with component overrides applied
        if component is not None:
            effective_growth_dict = self._get_effective_policy_dict(component, "growth")
            effective_collision_dict = self._get_effective_policy_dict(component, "collision")
            growth_policy = GrowthPolicy.from_dict(effective_growth_dict)
            collision_policy = CollisionPolicy.from_dict(effective_collision_dict)
        else:
            # Fallback to global compiled policies
            growth_policy = self._compiled_policies.get("growth")
            if growth_policy is None:
                growth_policy_dict = self.spec.policies.get("growth", {})
                growth_policy = GrowthPolicy.from_dict(growth_policy_dict)
            
            collision_policy = self._compiled_policies.get("collision")
            if collision_policy is None:
                collision_policy_dict = self.spec.policies.get("collision", {})
                collision_policy = CollisionPolicy.from_dict(collision_policy_dict)
        
        # Wire backend_params into GrowthPolicy
        # This ensures backend_params from component.build affects behavior
        if backend_params:
            # Merge backend_params into growth_policy.backend_params
            existing_backend_params = getattr(growth_policy, 'backend_params', {}) or {}
            merged_backend_params = deep_merge(existing_backend_params, backend_params)
            growth_policy.backend_params = merged_backend_params
        
        tissue_sampling_policy = self._compiled_policies.get("tissue_sampling")
        
        network, report = generate_network(
            generator_kind=backend,
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
            collision_policy=collision_policy,
            seed=self.spec.seed,
        )
        
        self.artifacts.register(
            f"{component_id}_network",
            f"{Stage.COMPONENT_BUILD.value}:{component_id}",
            network,
        )
        
        return network, report
    
    def _build_primitive_channels(
        self,
        domain: "RuntimeDomain",
        ports: Dict[str, Any],
        build: Dict[str, Any],
        component_id: str,
        component: Optional[Dict[str, Any]] = None,
    ) -> Tuple["trimesh.Trimesh", Any]:
        """
        Build primitive channels (e.g., fang hooks).
        
        Uses component-level policy overrides if component is provided.
        """
        from generation.api.generate import generate_void_mesh
        from aog_policies import ChannelPolicy
        
        # Get effective channel policy with component overrides applied
        if component is not None:
            effective_channel_dict = self._get_effective_policy_dict(component, "channels")
            channel_policy = ChannelPolicy.from_dict(effective_channel_dict)
        else:
            # Fallback to global compiled policies
            channel_policy = self._compiled_policies.get("channels")
            if channel_policy is None:
                channel_policy_dict = self.spec.policies.get("channels", {})
                channel_policy = ChannelPolicy.from_dict(channel_policy_dict)
        
        void_mesh, report = generate_void_mesh(
            kind="primitive_channels",
            domain=domain,
            ports=ports,
            channel_policy=channel_policy,
        )
        
        self.artifacts.register(
            f"{component_id}_void_mesh",
            f"{Stage.COMPONENT_BUILD.value}:{component_id}",
            void_mesh,
        )
        
        return void_mesh, report
    
    def _import_void_mesh(self, build: Dict[str, Any]) -> "trimesh.Trimesh":
        """Import a void mesh from file."""
        import trimesh
        
        path = build.get("path")
        if path is None:
            raise ValueError("import_void_mesh requires 'path' in build config")
        
        mesh = trimesh.load(path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh, got {type(mesh)}")
        
        return mesh


def run_spec(
    spec: Union[DesignSpec, Dict[str, Any], str, Path],
    plan: Optional[ExecutionPlan] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> RunnerResult:
    """
    Convenience function to run a DesignSpec.
    
    Parameters
    ----------
    spec : DesignSpec, dict, str, or Path
        The specification to run (DesignSpec, dict, or path to JSON file)
    plan : ExecutionPlan, optional
        Execution plan for partial execution
    output_dir : str or Path, optional
        Output directory
        
    Returns
    -------
    RunnerResult
        Result of the run
    """
    if isinstance(spec, (str, Path)):
        spec = DesignSpec.from_json(spec)
    elif isinstance(spec, dict):
        spec = DesignSpec.from_dict(spec)
    
    runner = DesignSpecRunner(spec, plan, output_dir)
    return runner.run()


__all__ = [
    "DesignSpecRunner",
    "RunnerResult",
    "StageReport",
    "run_spec",
]
