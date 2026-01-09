"""
MOGS Runner

Main workflow runner for the MultiAgentOrgan Generation System.
Orchestrates the three-agent workflow with approval gates.
"""

import os
import sys
import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
from enum import Enum

from .models import (
    ObjectManifest,
    SpecVersion,
    RunEntry,
    RunIndex,
    VersionStatus,
    ApprovalStatus,
    GateType,
    get_timestamp,
    generate_object_uuid,
)
from .folder_manager import FolderManager, create_new_object, load_object
from .object_registry import ObjectRegistry
from .agents import ConceptSpecAgent, CodingBuildAgent, ValidationQAAgent
from .gates import GateManager, GateContext, GateResult, ApprovalChoice
from .retention import RetentionManager
from .safety import SafetyManager, SafetyConfig


class WorkflowState(Enum):
    """State of the MOGS workflow."""
    IDLE = "idle"
    SPEC_CREATION = "spec_creation"
    SPEC_APPROVAL = "spec_approval"
    CODE_GENERATION = "code_generation"
    CODE_APPROVAL = "code_approval"
    EXECUTION = "execution"
    VALIDATION = "validation"
    RESULTS_APPROVAL = "results_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REFINEMENT = "needs_refinement"


@dataclass
class WorkflowContext:
    """
    Context for a MOGS workflow run.
    """
    object_uuid: str
    spec_version: int
    state: WorkflowState = WorkflowState.IDLE
    started_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.started_at:
            self.started_at = get_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_uuid": self.object_uuid,
            "spec_version": self.spec_version,
            "state": self.state.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "artifacts": self.artifacts,
        }


@dataclass
class WorkflowResult:
    """
    Result of a MOGS workflow run.
    """
    success: bool
    context: WorkflowContext
    final_outputs: Dict[str, str] = field(default_factory=dict)
    validation_passed: bool = False
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "context": self.context.to_dict(),
            "final_outputs": self.final_outputs,
            "validation_passed": self.validation_passed,
            "message": self.message,
        }


class MOGSRunner:
    """
    Main runner for the MOGS workflow.
    
    Orchestrates:
    1. CSA: Spec creation from requirements
    2. Gate A: Spec approval
    3. CBA: Script generation from spec
    4. Gate B: Code + plan approval
    5. Script execution (with safety constraints)
    6. VQA: Output validation
    7. Gate C: Results approval
    """
    
    def __init__(
        self,
        objects_base_dir: str,
        llm_client: Optional[Any] = None,
        approval_callback: Optional[Callable[[GateContext], GateResult]] = None,
        auto_approve: bool = False,
    ):
        """
        Initialize the MOGS runner.
        
        Parameters
        ----------
        objects_base_dir : str
            Base directory for all objects
        llm_client : Any, optional
            LLM client for agent interactions
        approval_callback : Callable, optional
            Callback for approval gates
        auto_approve : bool
            If True, automatically approve all gates (for testing)
        """
        self.objects_base_dir = os.path.abspath(objects_base_dir)
        self.llm_client = llm_client
        self.approval_callback = approval_callback
        self.auto_approve = auto_approve
        
        # Create registry
        self.registry = ObjectRegistry(objects_base_dir)
        
        # Current context
        self._current_context: Optional[WorkflowContext] = None
        self._folder_manager: Optional[FolderManager] = None
    
    def create_object(self, object_name: str) -> FolderManager:
        """
        Create a new MOGS object.
        
        Parameters
        ----------
        object_name : str
            Human-readable name for the object
            
        Returns
        -------
        FolderManager
            Folder manager for the new object
        """
        return self.registry.create_object(object_name)
    
    def load_object(self, object_uuid: str) -> Optional[FolderManager]:
        """
        Load an existing MOGS object.
        
        Parameters
        ----------
        object_uuid : str
            UUID of the object
            
        Returns
        -------
        FolderManager or None
            Folder manager for the object
        """
        return self.registry.get_object(object_uuid)
    
    def run_workflow(
        self,
        object_uuid: str,
        requirements: Dict[str, Any],
        user_description: str = "",
        parent_version: Optional[int] = None,
    ) -> WorkflowResult:
        """
        Run the complete MOGS workflow.
        
        Parameters
        ----------
        object_uuid : str
            UUID of the object
        requirements : Dict[str, Any]
            Requirements for the spec
        user_description : str
            User's description of intent
        parent_version : int, optional
            Parent version for refinements
            
        Returns
        -------
        WorkflowResult
            Result of the workflow
        """
        # Load object
        self._folder_manager = self.load_object(object_uuid)
        if self._folder_manager is None:
            return WorkflowResult(
                success=False,
                context=WorkflowContext(
                    object_uuid=object_uuid,
                    spec_version=0,
                    state=WorkflowState.FAILED,
                    error=f"Object not found: {object_uuid}",
                ),
                message=f"Object not found: {object_uuid}",
            )
        
        # Initialize agents
        csa = ConceptSpecAgent(self._folder_manager, self.llm_client)
        cba = CodingBuildAgent(self._folder_manager, self.llm_client)
        vqa = ValidationQAAgent(self._folder_manager, self.llm_client)
        
        # Initialize gate manager
        gate_callback = self._get_approval_callback()
        gate_manager = GateManager(self._folder_manager, gate_callback)
        
        # Initialize safety manager
        safety_manager = SafetyManager(self._folder_manager)
        
        # Initialize retention manager
        retention_manager = RetentionManager(self._folder_manager)
        
        # Create workflow context
        next_version = csa.get_next_version()
        self._current_context = WorkflowContext(
            object_uuid=object_uuid,
            spec_version=next_version,
        )
        
        try:
            # Step 1: CSA - Create spec
            self._current_context.state = WorkflowState.SPEC_CREATION
            self._folder_manager.log_event(f"Starting spec creation for version {next_version}")
            
            csa_output = csa.create_spec_from_requirements(
                requirements=requirements,
                user_description=user_description,
                parent_version=parent_version,
            )
            csa_paths = csa.save_output(csa_output)
            self._current_context.artifacts.update(csa_paths)
            
            # Step 2: Gate A - Spec approval
            self._current_context.state = WorkflowState.SPEC_APPROVAL
            spec_result = gate_manager.run_spec_approval(next_version)
            
            if spec_result.choice == ApprovalChoice.REFINE:
                self._current_context.state = WorkflowState.NEEDS_REFINEMENT
                return WorkflowResult(
                    success=False,
                    context=self._current_context,
                    message=f"Spec needs refinement: {spec_result.refinement_notes}",
                )
            elif spec_result.choice == ApprovalChoice.REJECT:
                self._current_context.state = WorkflowState.FAILED
                return WorkflowResult(
                    success=False,
                    context=self._current_context,
                    message=f"Spec rejected: {spec_result.comments}",
                )
            
            # Mark spec as approved
            csa.approve_spec(next_version)
            
            # Step 3: CBA - Generate scripts
            self._current_context.state = WorkflowState.CODE_GENERATION
            self._folder_manager.log_event(f"Starting code generation for version {next_version}")
            
            spec_version = csa.load_spec_version(next_version)
            cba_output = cba.generate_scripts(spec_version)
            cba_paths = cba.save_output(cba_output)
            self._current_context.artifacts.update(cba_paths)
            
            # Step 4: Gate B - Code approval
            self._current_context.state = WorkflowState.CODE_APPROVAL
            code_result = gate_manager.run_code_approval(next_version)
            
            if code_result.choice == ApprovalChoice.REFINE:
                self._current_context.state = WorkflowState.NEEDS_REFINEMENT
                return WorkflowResult(
                    success=False,
                    context=self._current_context,
                    message=f"Code needs refinement (loops back to CSA): {code_result.refinement_notes}",
                )
            elif code_result.choice == ApprovalChoice.REJECT:
                self._current_context.state = WorkflowState.FAILED
                return WorkflowResult(
                    success=False,
                    context=self._current_context,
                    message=f"Code rejected: {code_result.comments}",
                )
            
            # Step 5: Execute scripts
            self._current_context.state = WorkflowState.EXECUTION
            self._folder_manager.log_event(f"Starting script execution for version {next_version}")
            
            # Create version directories
            version_dirs = self._folder_manager.create_version_directories(next_version)
            outputs_dir = version_dirs["outputs"]
            
            # Execute scripts
            scripts_dir = self._folder_manager.get_scripts_version_dir(next_version)
            run_dir = self._folder_manager.get_run_dir(
                datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                next_version,
            )
            os.makedirs(run_dir, exist_ok=True)
            
            execution_success = self._execute_scripts(
                scripts_dir=scripts_dir,
                outputs_dir=outputs_dir,
                run_dir=run_dir,
                spec_version=next_version,
                safety_manager=safety_manager,
            )
            
            if not execution_success:
                self._current_context.state = WorkflowState.FAILED
                return WorkflowResult(
                    success=False,
                    context=self._current_context,
                    message="Script execution failed",
                )
            
            # Step 6: VQA - Validate outputs
            self._current_context.state = WorkflowState.VALIDATION
            self._folder_manager.log_event(f"Starting validation for version {next_version}")
            
            # Validate scripts
            script_checks = vqa.validate_scripts(next_version, scripts_dir)
            
            # Validate outputs
            output_checks, metrics = vqa.validate_outputs(
                next_version,
                outputs_dir,
                cba_output.expected_artifacts,
            )
            
            # Generate report
            vqa_output = vqa.generate_report(
                next_version,
                script_checks,
                output_checks,
                metrics,
            )
            vqa_paths = vqa.save_output(vqa_output)
            self._current_context.artifacts.update(vqa_paths)
            
            # Step 7: Gate C - Results approval
            self._current_context.state = WorkflowState.RESULTS_APPROVAL
            results_result = gate_manager.run_results_approval(next_version)
            
            if results_result.choice == ApprovalChoice.REFINE:
                self._current_context.state = WorkflowState.NEEDS_REFINEMENT
                
                # Propose refinements to CSA
                if vqa_output.suggested_refinements:
                    vqa.propose_refinement_to_csa(next_version, vqa_output.suggested_refinements)
                
                return WorkflowResult(
                    success=False,
                    context=self._current_context,
                    validation_passed=vqa_output.validation_report.overall_passed,
                    message=f"Results need refinement (loops back to CSA): {results_result.refinement_notes}",
                )
            elif results_result.choice == ApprovalChoice.REJECT:
                self._current_context.state = WorkflowState.FAILED
                return WorkflowResult(
                    success=False,
                    context=self._current_context,
                    validation_passed=vqa_output.validation_report.overall_passed,
                    message=f"Results rejected: {results_result.comments}",
                )
            
            # Mark version as accepted
            retention_manager.mark_version_accepted(next_version)
            
            # Update run index
            run_index = self._folder_manager.load_run_index()
            run_entry = RunEntry(
                run_id=f"run_{next_version:03d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                spec_version=next_version,
                timestamp=get_timestamp(),
                scripts_path=scripts_dir,
                outputs_path=outputs_dir,
                validation_path=version_dirs["validation"],
                status="completed",
                accepted=True,
            )
            run_index.add_run(run_entry)
            self._folder_manager.save_run_index(run_index)
            
            # Run retention cleanup
            cleanup_report = retention_manager.run_cleanup()
            if cleanup_report.versions_deleted:
                self._folder_manager.log_event(
                    f"Retention cleanup: deleted versions {cleanup_report.versions_deleted}"
                )
            
            # Workflow completed
            self._current_context.state = WorkflowState.COMPLETED
            self._current_context.completed_at = get_timestamp()
            
            # Get final outputs
            final_outputs = {
                "void_stl": os.path.join(outputs_dir, "final", "void.stl"),
                "scaffold_stl": os.path.join(outputs_dir, "final", "scaffold.stl"),
                "final_manifest": os.path.join(outputs_dir, "final", "final_manifest.json"),
            }
            
            return WorkflowResult(
                success=True,
                context=self._current_context,
                final_outputs=final_outputs,
                validation_passed=vqa_output.validation_report.overall_passed,
                message=f"Workflow completed successfully for version {next_version}",
            )
            
        except Exception as e:
            self._current_context.state = WorkflowState.FAILED
            self._current_context.error = str(e)
            self._folder_manager.log_event(f"Workflow failed: {e}")
            
            return WorkflowResult(
                success=False,
                context=self._current_context,
                message=f"Workflow failed: {e}",
            )
    
    def _get_approval_callback(self) -> Optional[Callable[[GateContext], GateResult]]:
        """Get the approval callback to use."""
        if self.auto_approve:
            def auto_approve_callback(context: GateContext) -> GateResult:
                return GateResult(
                    gate_type=context.gate_type,
                    spec_version=context.spec_version,
                    choice=ApprovalChoice.APPROVE,
                    comments="Auto-approved",
                )
            return auto_approve_callback
        
        return self.approval_callback
    
    def _execute_scripts(
        self,
        scripts_dir: str,
        outputs_dir: str,
        run_dir: str,
        spec_version: int,
        safety_manager: SafetyManager,
    ) -> bool:
        """
        Execute the three scripts in order.
        
        Parameters
        ----------
        scripts_dir : str
            Directory containing scripts
        outputs_dir : str
            Output directory
        run_dir : str
            Run directory for logs
        spec_version : int
            Spec version
        safety_manager : SafetyManager
            Safety manager
            
        Returns
        -------
        bool
            True if all scripts executed successfully
        """
        scripts = ["01_generate.py", "02_analyze.py", "03_finalize.py"]
        
        # Build safe environment
        env = safety_manager.build_safe_environment(spec_version, outputs_dir)
        
        # Save execution metadata
        metadata = safety_manager.create_execution_metadata(
            spec_version=spec_version,
            script_name="all",
            output_dir=outputs_dir,
        )
        safety_manager.save_execution_metadata(run_dir, metadata)
        safety_manager.save_sanitized_environment(run_dir, env)
        
        for script_name in scripts:
            script_path = os.path.join(scripts_dir, script_name)
            
            if not os.path.exists(script_path):
                self._folder_manager.log_event(f"Script not found: {script_path}")
                return False
            
            # Validate script safety
            warnings = safety_manager.validate_script_content(script_path)
            if warnings:
                self._folder_manager.log_warning(f"Script {script_name} warnings: {warnings}")
            
            # Execute script
            self._folder_manager.log_event(f"Executing {script_name}")
            
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    env=env,
                    cwd=scripts_dir,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    preexec_fn=safety_manager.get_preexec_fn(),
                )
                
                # Save logs
                stdout_path = os.path.join(run_dir, f"{script_name}_stdout.log")
                stderr_path = os.path.join(run_dir, f"{script_name}_stderr.log")
                
                with open(stdout_path, 'w') as f:
                    f.write(result.stdout)
                with open(stderr_path, 'w') as f:
                    f.write(result.stderr)
                
                if result.returncode != 0:
                    self._folder_manager.log_event(
                        f"Script {script_name} failed with return code {result.returncode}"
                    )
                    return False
                
                self._folder_manager.log_event(f"Script {script_name} completed successfully")
                
            except subprocess.TimeoutExpired:
                self._folder_manager.log_event(f"Script {script_name} timed out")
                return False
            except Exception as e:
                self._folder_manager.log_event(f"Script {script_name} error: {e}")
                return False
        
        return True
    
    def run_refinement(
        self,
        object_uuid: str,
        refinement_notes: str,
        parent_version: int,
    ) -> WorkflowResult:
        """
        Run a refinement workflow.
        
        This creates a new spec version based on refinement notes
        and runs the full workflow.
        
        Parameters
        ----------
        object_uuid : str
            UUID of the object
        refinement_notes : str
            Notes for refinement
        parent_version : int
            Version to refine
            
        Returns
        -------
        WorkflowResult
            Result of the workflow
        """
        # Load the parent spec
        folder_manager = self.load_object(object_uuid)
        if folder_manager is None:
            return WorkflowResult(
                success=False,
                context=WorkflowContext(
                    object_uuid=object_uuid,
                    spec_version=0,
                    state=WorkflowState.FAILED,
                ),
                message=f"Object not found: {object_uuid}",
            )
        
        csa = ConceptSpecAgent(folder_manager, self.llm_client)
        parent_spec = csa.load_spec_version(parent_version)
        
        if parent_spec is None:
            return WorkflowResult(
                success=False,
                context=WorkflowContext(
                    object_uuid=object_uuid,
                    spec_version=0,
                    state=WorkflowState.FAILED,
                ),
                message=f"Parent version not found: {parent_version}",
            )
        
        # Create refined requirements
        requirements = parent_spec.spec_data.copy()
        requirements["_refinement_notes"] = refinement_notes
        requirements["_parent_version"] = parent_version
        
        return self.run_workflow(
            object_uuid=object_uuid,
            requirements=requirements,
            user_description=f"Refinement of version {parent_version}: {refinement_notes}",
            parent_version=parent_version,
        )
    
    def get_workflow_status(self) -> Optional[Dict[str, Any]]:
        """Get the current workflow status."""
        if self._current_context is None:
            return None
        return self._current_context.to_dict()
    
    def list_objects(self) -> List[Dict[str, Any]]:
        """List all objects in the registry."""
        entries = self.registry.list_objects()
        return [e.to_dict() for e in entries]


def create_mogs_runner(
    objects_base_dir: str = "./objects",
    llm_client: Optional[Any] = None,
    auto_approve: bool = False,
) -> MOGSRunner:
    """
    Create a MOGS runner with default settings.
    
    Parameters
    ----------
    objects_base_dir : str
        Base directory for objects
    llm_client : Any, optional
        LLM client
    auto_approve : bool
        Auto-approve all gates
        
    Returns
    -------
    MOGSRunner
        Configured MOGS runner
    """
    return MOGSRunner(
        objects_base_dir=objects_base_dir,
        llm_client=llm_client,
        auto_approve=auto_approve,
    )
