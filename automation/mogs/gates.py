"""
MOGS Approval Gates

Implements the three approval gates in the MOGS workflow:
- Gate A: Spec Approval (CSA -> user)
- Gate B: Code + Plan Approval (CBA -> user)
- Gate C: Results Approval (VQA -> user)
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum

from .models import (
    ApprovalStatus,
    ApprovalRecord,
    GateType,
    VersionStatus,
    get_timestamp,
)
from .folder_manager import FolderManager


class ApprovalChoice(Enum):
    """User choices at approval gates."""
    APPROVE = "approve"
    REFINE = "refine"
    REJECT = "reject"


@dataclass
class GateContext:
    """
    Context for an approval gate.
    
    Contains all the information needed for the user to make a decision.
    """
    gate_type: GateType
    spec_version: int
    summary: str
    files_to_review: List[str]
    risk_flags: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_type": self.gate_type.value,
            "spec_version": self.spec_version,
            "summary": self.summary,
            "files_to_review": self.files_to_review,
            "risk_flags": self.risk_flags,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass
class GateResult:
    """
    Result of an approval gate.
    """
    gate_type: GateType
    spec_version: int
    choice: ApprovalChoice
    comments: str = ""
    refinement_notes: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = get_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_type": self.gate_type.value,
            "spec_version": self.spec_version,
            "choice": self.choice.value,
            "comments": self.comments,
            "refinement_notes": self.refinement_notes,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GateResult":
        return cls(
            gate_type=GateType(d["gate_type"]),
            spec_version=d["spec_version"],
            choice=ApprovalChoice(d["choice"]),
            comments=d.get("comments", ""),
            refinement_notes=d.get("refinement_notes", ""),
            timestamp=d.get("timestamp", ""),
        )


class ApprovalGate:
    """
    Base class for approval gates.
    
    Provides common functionality for all gates.
    """
    
    def __init__(
        self,
        folder_manager: FolderManager,
        gate_type: GateType,
        approval_callback: Optional[Callable[[GateContext], GateResult]] = None,
    ):
        """
        Initialize the approval gate.
        
        Parameters
        ----------
        folder_manager : FolderManager
            Folder manager for the object
        gate_type : GateType
            Type of this gate
        approval_callback : Callable, optional
            Callback function for getting user approval.
            If None, uses interactive mode.
        """
        self.folder_manager = folder_manager
        self.gate_type = gate_type
        self.approval_callback = approval_callback
        self._approval_history: List[ApprovalRecord] = []
    
    def prepare_context(self, spec_version: int) -> GateContext:
        """
        Prepare the context for this gate.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def request_approval(self, context: GateContext) -> GateResult:
        """
        Request approval from the user.
        
        Parameters
        ----------
        context : GateContext
            Context for the approval decision
            
        Returns
        -------
        GateResult
            The user's decision
        """
        if self.approval_callback:
            return self.approval_callback(context)
        else:
            return self._interactive_approval(context)
    
    def _interactive_approval(self, context: GateContext) -> GateResult:
        """
        Interactive approval mode (for CLI use).
        
        This is a placeholder that returns auto-approval.
        In a real implementation, this would prompt the user.
        """
        return GateResult(
            gate_type=context.gate_type,
            spec_version=context.spec_version,
            choice=ApprovalChoice.APPROVE,
            comments="Auto-approved (no callback provided)",
        )
    
    def record_approval(self, result: GateResult) -> ApprovalRecord:
        """
        Record an approval decision.
        
        Parameters
        ----------
        result : GateResult
            The approval result
            
        Returns
        -------
        ApprovalRecord
            The recorded approval
        """
        status_map = {
            ApprovalChoice.APPROVE: ApprovalStatus.APPROVED,
            ApprovalChoice.REFINE: ApprovalStatus.NEEDS_REVISION,
            ApprovalChoice.REJECT: ApprovalStatus.REJECTED,
        }
        
        record = ApprovalRecord(
            gate_type=self.gate_type,
            spec_version=result.spec_version,
            status=status_map[result.choice],
            timestamp=result.timestamp,
            comments=result.comments,
        )
        
        self._approval_history.append(record)
        self._save_approval_record(record)
        
        self.folder_manager.log_event(
            f"Gate {self.gate_type.value} for version {result.spec_version}: {result.choice.value}"
        )
        
        return record
    
    def _save_approval_record(self, record: ApprovalRecord) -> None:
        """Save an approval record to disk."""
        approvals_dir = os.path.join(self.folder_manager.admin_dir, "approvals")
        os.makedirs(approvals_dir, exist_ok=True)
        
        filename = f"{record.gate_type.value}_v{record.spec_version:03d}_{record.timestamp.replace(':', '-')}.json"
        filepath = os.path.join(approvals_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
    
    def get_approval_history(self, spec_version: Optional[int] = None) -> List[ApprovalRecord]:
        """
        Get approval history.
        
        Parameters
        ----------
        spec_version : int, optional
            Filter by spec version
            
        Returns
        -------
        List[ApprovalRecord]
            Approval history
        """
        if spec_version is None:
            return self._approval_history
        return [r for r in self._approval_history if r.spec_version == spec_version]


class SpecApprovalGate(ApprovalGate):
    """
    Gate A: Spec Approval (CSA -> user)
    
    User reviews:
    - spec_summary.md
    - risk flags + assumptions
    
    Choices:
    - Approve -> proceeds to CBA
    - Refine -> CSA increments version and repeats
    """
    
    def __init__(
        self,
        folder_manager: FolderManager,
        approval_callback: Optional[Callable[[GateContext], GateResult]] = None,
    ):
        super().__init__(folder_manager, GateType.SPEC_APPROVAL, approval_callback)
    
    def prepare_context(self, spec_version: int) -> GateContext:
        """Prepare context for spec approval."""
        files_to_review = []
        risk_flags = []
        summary = ""
        
        # Load spec summary
        summary_path = self.folder_manager.get_spec_summary_path(spec_version)
        if os.path.exists(summary_path):
            files_to_review.append(summary_path)
            with open(summary_path, 'r') as f:
                summary = f.read()
        
        # Load risk flags
        risk_flags_path = self.folder_manager.get_spec_risk_flags_path(spec_version)
        if os.path.exists(risk_flags_path):
            files_to_review.append(risk_flags_path)
            with open(risk_flags_path, 'r') as f:
                risk_flags = json.load(f)
        
        # Load spec for additional context
        spec_path = self.folder_manager.get_spec_path(spec_version)
        if os.path.exists(spec_path):
            files_to_review.append(spec_path)
        
        return GateContext(
            gate_type=GateType.SPEC_APPROVAL,
            spec_version=spec_version,
            summary=summary,
            files_to_review=files_to_review,
            risk_flags=risk_flags,
        )


class CodeApprovalGate(ApprovalGate):
    """
    Gate B: Code + Plan Approval (CBA -> user)
    
    User reviews:
    - CBA_buildplan.md (plain English)
    - all three scripts (full code)
    - expected_artifacts.json
    
    Choices:
    - Approve -> execution begins
    - Refine -> loops back to CSA (creates new spec version)
    
    Key rule: refinements always modify spec first; code is regenerated from spec.
    """
    
    def __init__(
        self,
        folder_manager: FolderManager,
        approval_callback: Optional[Callable[[GateContext], GateResult]] = None,
    ):
        super().__init__(folder_manager, GateType.CODE_APPROVAL, approval_callback)
    
    def prepare_context(self, spec_version: int) -> GateContext:
        """Prepare context for code approval."""
        files_to_review = []
        warnings = []
        summary = ""
        
        scripts_dir = self.folder_manager.get_scripts_version_dir(spec_version)
        cba_docs_dir = self.folder_manager.get_agent_docs_dir("CBA")
        
        # Load build plan
        buildplan_path = os.path.join(cba_docs_dir, f"CBA_buildplan_v{spec_version:03d}.md")
        if os.path.exists(buildplan_path):
            files_to_review.append(buildplan_path)
            with open(buildplan_path, 'r') as f:
                summary = f.read()
        
        # Add scripts
        for script_name in ["01_generate.py", "02_analyze.py", "03_finalize.py"]:
            script_path = os.path.join(scripts_dir, script_name)
            if os.path.exists(script_path):
                files_to_review.append(script_path)
            else:
                warnings.append(f"Script not found: {script_name}")
        
        # Add expected artifacts
        artifacts_path = os.path.join(scripts_dir, "expected_artifacts.json")
        if os.path.exists(artifacts_path):
            files_to_review.append(artifacts_path)
        else:
            warnings.append("expected_artifacts.json not found")
        
        return GateContext(
            gate_type=GateType.CODE_APPROVAL,
            spec_version=spec_version,
            summary=summary,
            files_to_review=files_to_review,
            warnings=warnings,
        )


class ResultsApprovalGate(ApprovalGate):
    """
    Gate C: Results Approval (VQA -> user)
    
    User reviews:
    - summary of checks and metrics
    - confirms both outputs exist:
      - final/void.stl
      - final/scaffold.stl
    
    Choices:
    - Approve outputs -> mark version "accepted"
    - Refine -> loops back to CSA with structured deltas
    """
    
    def __init__(
        self,
        folder_manager: FolderManager,
        approval_callback: Optional[Callable[[GateContext], GateResult]] = None,
    ):
        super().__init__(folder_manager, GateType.RESULTS_APPROVAL, approval_callback)
    
    def prepare_context(self, spec_version: int) -> GateContext:
        """Prepare context for results approval."""
        files_to_review = []
        warnings = []
        metrics = {}
        summary = ""
        
        outputs_dir = self.folder_manager.get_outputs_version_dir(spec_version)
        validation_dir = self.folder_manager.get_validation_version_dir(spec_version)
        vqa_docs_dir = self.folder_manager.get_agent_docs_dir("VQA")
        
        # Load VQA report
        report_path = os.path.join(vqa_docs_dir, f"VQA_report_v{spec_version:03d}.md")
        if os.path.exists(report_path):
            files_to_review.append(report_path)
            with open(report_path, 'r') as f:
                summary = f.read()
        
        # Load validation.json
        validation_path = os.path.join(validation_dir, "validation.json")
        if os.path.exists(validation_path):
            files_to_review.append(validation_path)
            with open(validation_path, 'r') as f:
                validation_data = json.load(f)
                metrics["validation"] = validation_data
        
        # Load metrics
        metrics_path = os.path.join(vqa_docs_dir, f"VQA_metrics_v{spec_version:03d}.json")
        if os.path.exists(metrics_path):
            files_to_review.append(metrics_path)
            with open(metrics_path, 'r') as f:
                metrics["vqa_metrics"] = json.load(f)
        
        # Check for final outputs
        void_path = os.path.join(outputs_dir, "final", "void.stl")
        scaffold_path = os.path.join(outputs_dir, "final", "scaffold.stl")
        
        if os.path.exists(void_path):
            files_to_review.append(void_path)
        else:
            warnings.append("void.stl not found")
        
        if os.path.exists(scaffold_path):
            files_to_review.append(scaffold_path)
        else:
            warnings.append("scaffold.stl not found")
        
        return GateContext(
            gate_type=GateType.RESULTS_APPROVAL,
            spec_version=spec_version,
            summary=summary,
            files_to_review=files_to_review,
            warnings=warnings,
            metrics=metrics,
        )


class GateManager:
    """
    Manages all approval gates for an object.
    """
    
    def __init__(
        self,
        folder_manager: FolderManager,
        approval_callback: Optional[Callable[[GateContext], GateResult]] = None,
    ):
        """
        Initialize the gate manager.
        
        Parameters
        ----------
        folder_manager : FolderManager
            Folder manager for the object
        approval_callback : Callable, optional
            Callback function for getting user approval
        """
        self.folder_manager = folder_manager
        self.approval_callback = approval_callback
        
        self.spec_gate = SpecApprovalGate(folder_manager, approval_callback)
        self.code_gate = CodeApprovalGate(folder_manager, approval_callback)
        self.results_gate = ResultsApprovalGate(folder_manager, approval_callback)
    
    def get_gate(self, gate_type: GateType) -> ApprovalGate:
        """Get a gate by type."""
        gates = {
            GateType.SPEC_APPROVAL: self.spec_gate,
            GateType.CODE_APPROVAL: self.code_gate,
            GateType.RESULTS_APPROVAL: self.results_gate,
        }
        return gates[gate_type]
    
    def run_gate(self, gate_type: GateType, spec_version: int) -> GateResult:
        """
        Run an approval gate.
        
        Parameters
        ----------
        gate_type : GateType
            Type of gate to run
        spec_version : int
            Version to approve
            
        Returns
        -------
        GateResult
            The approval result
        """
        gate = self.get_gate(gate_type)
        context = gate.prepare_context(spec_version)
        result = gate.request_approval(context)
        gate.record_approval(result)
        return result
    
    def run_spec_approval(self, spec_version: int) -> GateResult:
        """Run Gate A: Spec Approval."""
        return self.run_gate(GateType.SPEC_APPROVAL, spec_version)
    
    def run_code_approval(self, spec_version: int) -> GateResult:
        """Run Gate B: Code + Plan Approval."""
        return self.run_gate(GateType.CODE_APPROVAL, spec_version)
    
    def run_results_approval(self, spec_version: int) -> GateResult:
        """Run Gate C: Results Approval."""
        return self.run_gate(GateType.RESULTS_APPROVAL, spec_version)
    
    def get_all_approvals(self, spec_version: Optional[int] = None) -> Dict[str, List[ApprovalRecord]]:
        """
        Get all approval records.
        
        Parameters
        ----------
        spec_version : int, optional
            Filter by spec version
            
        Returns
        -------
        Dict[str, List[ApprovalRecord]]
            Approval records by gate type
        """
        return {
            "spec_approval": self.spec_gate.get_approval_history(spec_version),
            "code_approval": self.code_gate.get_approval_history(spec_version),
            "results_approval": self.results_gate.get_approval_history(spec_version),
        }
