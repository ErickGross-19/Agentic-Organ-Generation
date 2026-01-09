"""
Validation & QA Agent (VQA)

The VQA is responsible for:
- Validating scripts before execution
- Validating outputs after execution
- Generating validation reports
- Proposing refinements back to CSA
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from ..models import (
    ValidationReport,
    ValidationCheck,
    ExpectedArtifact,
    AgentType,
    get_timestamp,
)
from ..folder_manager import FolderManager


@dataclass
class VQAMetrics:
    """
    Metrics collected by VQA during validation.
    """
    mesh_metrics: Dict[str, Any] = field(default_factory=dict)
    network_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mesh_metrics": self.mesh_metrics,
            "network_metrics": self.network_metrics,
            "performance_metrics": self.performance_metrics,
            "quality_scores": self.quality_scores,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VQAMetrics":
        return cls(
            mesh_metrics=d.get("mesh_metrics", {}),
            network_metrics=d.get("network_metrics", {}),
            performance_metrics=d.get("performance_metrics", {}),
            quality_scores=d.get("quality_scores", {}),
        )


@dataclass
class SuggestedRefinement:
    """
    A refinement suggested by VQA to be sent back to CSA.
    """
    id: str
    category: str  # "spec", "geometry", "topology", "manufacturing"
    severity: str  # "suggestion", "warning", "required"
    description: str
    affected_field: str
    current_value: Any
    suggested_value: Any
    rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "affected_field": self.affected_field,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "rationale": self.rationale,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SuggestedRefinement":
        return cls(**d)


@dataclass
class VQAOutput:
    """
    Output from a VQA session.
    
    Contains all the files that VQA must produce.
    """
    spec_version: int
    validation_report: ValidationReport
    metrics: VQAMetrics
    suggested_refinements: List[SuggestedRefinement]
    report_md: str


class ValidationQAAgent:
    """
    Validation & QA Agent (VQA)
    
    Responsible for:
    - Validating scripts before execution
    - Validating outputs after execution
    - Generating validation reports
    - Proposing refinements back to CSA
    
    Required outputs per session:
    - VQA_report_v###.md
    - VQA_metrics_v###.json
    - validation.json (checklist pass/fail)
    - VQA_suggested_refinements_v###.json (structured deltas)
    """
    
    def __init__(self, folder_manager: FolderManager, llm_client: Optional[Any] = None):
        """
        Initialize the VQA.
        
        Parameters
        ----------
        folder_manager : FolderManager
            Folder manager for the object
        llm_client : Any, optional
            LLM client for generating reports (if None, uses templates)
        """
        self.folder_manager = folder_manager
        self.llm_client = llm_client
    
    def validate_scripts(
        self,
        spec_version: int,
        scripts_dir: str,
    ) -> List[ValidationCheck]:
        """
        Validate scripts before execution.
        
        Parameters
        ----------
        spec_version : int
            Version of the spec
        scripts_dir : str
            Directory containing the scripts
            
        Returns
        -------
        List[ValidationCheck]
            List of validation checks
        """
        checks = []
        
        # Check that all required scripts exist
        required_scripts = ["01_generate.py", "02_analyze.py", "03_finalize.py"]
        for script_name in required_scripts:
            script_path = os.path.join(scripts_dir, script_name)
            exists = os.path.exists(script_path)
            checks.append(ValidationCheck(
                check_id=f"script_exists_{script_name}",
                name=f"Script exists: {script_name}",
                passed=exists,
                message="" if exists else f"Script not found: {script_path}",
            ))
        
        # Check for expected_artifacts.json
        artifacts_path = os.path.join(scripts_dir, "expected_artifacts.json")
        exists = os.path.exists(artifacts_path)
        checks.append(ValidationCheck(
            check_id="expected_artifacts_exists",
            name="Expected artifacts file exists",
            passed=exists,
            message="" if exists else "expected_artifacts.json not found",
        ))
        
        # Check for run_manifest.json
        manifest_path = os.path.join(scripts_dir, "run_manifest.json")
        exists = os.path.exists(manifest_path)
        checks.append(ValidationCheck(
            check_id="run_manifest_exists",
            name="Run manifest file exists",
            passed=exists,
            message="" if exists else "run_manifest.json not found",
        ))
        
        # Validate script syntax
        for script_name in required_scripts:
            script_path = os.path.join(scripts_dir, script_name)
            if os.path.exists(script_path):
                syntax_valid, error = self._check_script_syntax(script_path)
                checks.append(ValidationCheck(
                    check_id=f"script_syntax_{script_name}",
                    name=f"Script syntax valid: {script_name}",
                    passed=syntax_valid,
                    message="" if syntax_valid else f"Syntax error: {error}",
                ))
        
        # Check for suspicious patterns
        for script_name in required_scripts:
            script_path = os.path.join(scripts_dir, script_name)
            if os.path.exists(script_path):
                safe, warnings = self._check_script_safety(script_path)
                checks.append(ValidationCheck(
                    check_id=f"script_safety_{script_name}",
                    name=f"Script safety check: {script_name}",
                    passed=safe,
                    message="" if safe else f"Safety warnings: {', '.join(warnings)}",
                    details={"warnings": warnings} if warnings else {},
                ))
        
        return checks
    
    def validate_outputs(
        self,
        spec_version: int,
        outputs_dir: str,
        expected_artifacts: List[ExpectedArtifact],
    ) -> Tuple[List[ValidationCheck], VQAMetrics]:
        """
        Validate outputs after execution.
        
        Parameters
        ----------
        spec_version : int
            Version of the spec
        outputs_dir : str
            Directory containing the outputs
        expected_artifacts : List[ExpectedArtifact]
            List of expected artifacts
            
        Returns
        -------
        Tuple[List[ValidationCheck], VQAMetrics]
            Validation checks and collected metrics
        """
        checks = []
        metrics = VQAMetrics()
        
        # Check each expected artifact
        for artifact in expected_artifacts:
            artifact_path = os.path.join(outputs_dir, artifact.filename)
            
            # Check existence
            exists = os.path.exists(artifact_path)
            if artifact.required:
                checks.append(ValidationCheck(
                    check_id=f"artifact_exists_{artifact.filename}",
                    name=f"Required artifact exists: {artifact.filename}",
                    passed=exists,
                    message="" if exists else f"Required artifact not found: {artifact_path}",
                ))
            else:
                checks.append(ValidationCheck(
                    check_id=f"artifact_exists_{artifact.filename}",
                    name=f"Optional artifact exists: {artifact.filename}",
                    passed=True,  # Optional artifacts don't fail
                    message="" if exists else f"Optional artifact not found: {artifact_path}",
                ))
            
            if not exists:
                continue
            
            # Validate based on type
            if artifact.validation_type == "json":
                valid, error, data = self._validate_json_file(artifact_path)
                checks.append(ValidationCheck(
                    check_id=f"artifact_valid_{artifact.filename}",
                    name=f"Artifact valid JSON: {artifact.filename}",
                    passed=valid,
                    message="" if valid else f"Invalid JSON: {error}",
                ))
                
                # Collect metrics from JSON files
                if valid and data:
                    if "metrics" in artifact.filename.lower():
                        metrics.network_metrics.update(data.get("network", {}))
                        metrics.mesh_metrics.update(data.get("mesh", {}))
            
            elif artifact.validation_type == "stl":
                valid, error, mesh_stats = self._validate_stl_file(artifact_path)
                checks.append(ValidationCheck(
                    check_id=f"artifact_valid_{artifact.filename}",
                    name=f"Artifact valid STL: {artifact.filename}",
                    passed=valid,
                    message="" if valid else f"Invalid STL: {error}",
                    details=mesh_stats if mesh_stats else {},
                ))
                
                # Collect mesh metrics
                if valid and mesh_stats:
                    key = artifact.filename.replace("/", "_").replace(".stl", "")
                    metrics.mesh_metrics[key] = mesh_stats
        
        # Check for both final outputs
        void_path = os.path.join(outputs_dir, "final", "void.stl")
        scaffold_path = os.path.join(outputs_dir, "final", "scaffold.stl")
        
        both_exist = os.path.exists(void_path) and os.path.exists(scaffold_path)
        checks.append(ValidationCheck(
            check_id="both_final_outputs",
            name="Both final outputs exist (void.stl and scaffold.stl)",
            passed=both_exist,
            message="" if both_exist else "Missing one or both final outputs",
        ))
        
        # Compute quality scores
        metrics.quality_scores = self._compute_quality_scores(checks, metrics)
        
        return checks, metrics
    
    def generate_report(
        self,
        spec_version: int,
        script_checks: List[ValidationCheck],
        output_checks: List[ValidationCheck],
        metrics: VQAMetrics,
    ) -> VQAOutput:
        """
        Generate a complete validation report.
        
        Parameters
        ----------
        spec_version : int
            Version of the spec
        script_checks : List[ValidationCheck]
            Checks from script validation
        output_checks : List[ValidationCheck]
            Checks from output validation
        metrics : VQAMetrics
            Collected metrics
            
        Returns
        -------
        VQAOutput
            Complete VQA output
        """
        all_checks = script_checks + output_checks
        overall_passed = all(c.passed for c in all_checks if "required" in c.check_id.lower() or c.check_id.startswith("both_"))
        
        # Generate suggested refinements
        refinements = self._generate_refinements(all_checks, metrics)
        
        # Create validation report
        report = ValidationReport(
            spec_version=spec_version,
            timestamp=get_timestamp(),
            overall_passed=overall_passed,
            checks=all_checks,
            metrics=metrics.to_dict(),
            suggested_refinements=[r.to_dict() for r in refinements],
        )
        
        # Generate report markdown
        report_md = self._format_report_md(report, metrics, refinements)
        
        return VQAOutput(
            spec_version=spec_version,
            validation_report=report,
            metrics=metrics,
            suggested_refinements=refinements,
            report_md=report_md,
        )
    
    def save_output(self, output: VQAOutput) -> Dict[str, str]:
        """
        Save all VQA output files.
        
        Parameters
        ----------
        output : VQAOutput
            The VQA output to save
            
        Returns
        -------
        Dict[str, str]
            Dictionary of file types to paths
        """
        version = output.spec_version
        paths = {}
        
        # Create validation directory
        validation_dir = self.folder_manager.get_validation_version_dir(version)
        os.makedirs(validation_dir, exist_ok=True)
        
        # Save validation.json
        validation_path = os.path.join(validation_dir, "validation.json")
        output.validation_report.save(validation_path)
        paths["validation"] = validation_path
        
        # Save repair_log.md (placeholder)
        repair_log_path = os.path.join(validation_dir, "repair_log.md")
        with open(repair_log_path, 'w') as f:
            f.write(f"# Repair Log - Version {version:03d}\n\n")
            f.write("No repairs performed.\n")
        paths["repair_log"] = repair_log_path
        
        # Save VQA docs
        vqa_docs_dir = self.folder_manager.get_agent_docs_dir("VQA")
        os.makedirs(vqa_docs_dir, exist_ok=True)
        
        # Save report MD
        report_path = os.path.join(vqa_docs_dir, f"VQA_report_v{version:03d}.md")
        with open(report_path, 'w') as f:
            f.write(output.report_md)
        paths["report"] = report_path
        
        # Save metrics JSON
        metrics_path = os.path.join(vqa_docs_dir, f"VQA_metrics_v{version:03d}.json")
        with open(metrics_path, 'w') as f:
            json.dump(output.metrics.to_dict(), f, indent=2)
        paths["metrics"] = metrics_path
        
        # Save suggested refinements JSON
        refinements_path = os.path.join(vqa_docs_dir, f"VQA_suggested_refinements_v{version:03d}.json")
        with open(refinements_path, 'w') as f:
            json.dump([r.to_dict() for r in output.suggested_refinements], f, indent=2)
        paths["refinements"] = refinements_path
        
        self.folder_manager.log_event(f"VQA output saved for version {version}")
        
        return paths
    
    def _check_script_syntax(self, script_path: str) -> Tuple[bool, str]:
        """Check if a Python script has valid syntax."""
        try:
            with open(script_path, 'r') as f:
                source = f.read()
            compile(source, script_path, 'exec')
            return True, ""
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    def _check_script_safety(self, script_path: str) -> Tuple[bool, List[str]]:
        """Check a script for suspicious patterns."""
        import re
        
        warnings = []
        
        suspicious_patterns = [
            (r'\bsubprocess\b', "Uses subprocess module"),
            (r'\bos\.system\s*\(', "Uses os.system()"),
            (r'\bos\.popen\s*\(', "Uses os.popen()"),
            (r'\beval\s*\(', "Uses eval()"),
            (r'\bexec\s*\(', "Uses exec()"),
            (r'\bshutil\.rmtree\s*\(', "Uses shutil.rmtree()"),
            (r'\bos\.remove\s*\(', "Uses os.remove()"),
            (r'__import__\s*\(', "Uses __import__()"),
        ]
        
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            for pattern, message in suspicious_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    warnings.append(message)
        except Exception as e:
            warnings.append(f"Could not read script: {e}")
        
        return len(warnings) == 0, warnings
    
    def _validate_json_file(self, file_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """Validate a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return True, "", data
        except json.JSONDecodeError as e:
            return False, str(e), None
        except Exception as e:
            return False, str(e), None
    
    def _validate_stl_file(self, file_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """Validate an STL file."""
        try:
            import trimesh
            mesh = trimesh.load(file_path)
            
            stats = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": mesh.is_watertight,
                "is_volume": mesh.is_volume,
            }
            
            if len(mesh.vertices) == 0:
                return False, "Mesh has no vertices", stats
            
            return True, "", stats
        except ImportError:
            return True, "", {"note": "trimesh not available"}
        except Exception as e:
            return False, str(e), None
    
    def _compute_quality_scores(
        self,
        checks: List[ValidationCheck],
        metrics: VQAMetrics,
    ) -> Dict[str, float]:
        """Compute quality scores from checks and metrics."""
        scores = {}
        
        # Completeness score (% of checks passed)
        if checks:
            passed = sum(1 for c in checks if c.passed)
            scores["completeness"] = passed / len(checks)
        else:
            scores["completeness"] = 0.0
        
        # Mesh quality score
        mesh_scores = []
        for key, stats in metrics.mesh_metrics.items():
            if isinstance(stats, dict):
                if stats.get("is_watertight", False):
                    mesh_scores.append(1.0)
                elif stats.get("is_volume", False):
                    mesh_scores.append(0.7)
                else:
                    mesh_scores.append(0.3)
        
        if mesh_scores:
            scores["mesh_quality"] = sum(mesh_scores) / len(mesh_scores)
        else:
            scores["mesh_quality"] = 0.0
        
        # Overall score
        scores["overall"] = (scores["completeness"] + scores["mesh_quality"]) / 2
        
        return scores
    
    def _generate_refinements(
        self,
        checks: List[ValidationCheck],
        metrics: VQAMetrics,
    ) -> List[SuggestedRefinement]:
        """Generate suggested refinements based on validation results."""
        refinements = []
        refinement_id = 0
        
        # Check for failed checks
        for check in checks:
            if not check.passed and "required" in check.check_id.lower():
                refinement_id += 1
                refinements.append(SuggestedRefinement(
                    id=f"ref_{refinement_id:03d}",
                    category="spec",
                    severity="required",
                    description=f"Fix failed check: {check.name}",
                    affected_field="unknown",
                    current_value=None,
                    suggested_value=None,
                    rationale=check.message,
                ))
        
        # Check mesh quality
        for key, stats in metrics.mesh_metrics.items():
            if isinstance(stats, dict) and not stats.get("is_watertight", True):
                refinement_id += 1
                refinements.append(SuggestedRefinement(
                    id=f"ref_{refinement_id:03d}",
                    category="geometry",
                    severity="warning",
                    description=f"Mesh {key} is not watertight",
                    affected_field="geometry.mesh_repair",
                    current_value=False,
                    suggested_value=True,
                    rationale="Non-watertight meshes may cause issues in 3D printing",
                ))
        
        return refinements
    
    def _format_report_md(
        self,
        report: ValidationReport,
        metrics: VQAMetrics,
        refinements: List[SuggestedRefinement],
    ) -> str:
        """Format the validation report as markdown."""
        md = f"""# VQA Validation Report - Version {report.spec_version:03d}

## Summary

- **Timestamp**: {report.timestamp}
- **Overall Status**: {'PASSED' if report.overall_passed else 'FAILED'}
- **Total Checks**: {len(report.checks)}
- **Passed**: {sum(1 for c in report.checks if c.passed)}
- **Failed**: {sum(1 for c in report.checks if not c.passed)}

## Quality Scores

"""
        for score_name, score_value in metrics.quality_scores.items():
            md += f"- **{score_name.replace('_', ' ').title()}**: {score_value:.1%}\n"
        
        md += "\n## Validation Checks\n\n"
        md += "| Check | Status | Message |\n"
        md += "|-------|--------|--------|\n"
        
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            message = check.message[:50] + "..." if len(check.message) > 50 else check.message
            md += f"| {check.name} | {status} | {message} |\n"
        
        md += "\n## Metrics\n\n"
        
        if metrics.network_metrics:
            md += "### Network Metrics\n\n"
            for key, value in metrics.network_metrics.items():
                md += f"- **{key}**: {value}\n"
        
        if metrics.mesh_metrics:
            md += "\n### Mesh Metrics\n\n"
            for mesh_name, stats in metrics.mesh_metrics.items():
                md += f"#### {mesh_name}\n\n"
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        md += f"- **{key}**: {value}\n"
                md += "\n"
        
        md += "\n## Suggested Refinements\n\n"
        
        if refinements:
            for ref in refinements:
                md += f"### {ref.id} ({ref.severity.upper()})\n\n"
                md += f"**Category**: {ref.category}\n\n"
                md += f"**Description**: {ref.description}\n\n"
                md += f"**Rationale**: {ref.rationale}\n\n"
        else:
            md += "No refinements suggested.\n"
        
        return md
    
    def propose_refinement_to_csa(
        self,
        spec_version: int,
        refinements: List[SuggestedRefinement],
    ) -> Dict[str, Any]:
        """
        Propose refinements to CSA for spec modification.
        
        Parameters
        ----------
        spec_version : int
            Version of the spec
        refinements : List[SuggestedRefinement]
            Refinements to propose
            
        Returns
        -------
        Dict[str, Any]
            Structured refinement proposal
        """
        proposal = {
            "from_agent": "VQA",
            "to_agent": "CSA",
            "spec_version": spec_version,
            "timestamp": get_timestamp(),
            "refinements": [r.to_dict() for r in refinements],
            "status": "pending",
        }
        
        # Save proposal to VQA docs
        vqa_docs_dir = self.folder_manager.get_agent_docs_dir("VQA")
        os.makedirs(vqa_docs_dir, exist_ok=True)
        
        proposal_path = os.path.join(
            vqa_docs_dir,
            f"VQA_refinement_proposal_v{spec_version:03d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(proposal_path, 'w') as f:
            json.dump(proposal, f, indent=2)
        
        self.folder_manager.log_event(f"VQA proposed refinements for version {spec_version}")
        
        return proposal
