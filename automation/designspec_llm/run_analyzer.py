"""
Run Analyzer for DesignSpec LLM Agent

This module analyzes run results and proposes corrections automatically.
When a run fails, the analyzer identifies the root cause and suggests
targeted fixes without requiring user to say "fix it".
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .error_taxonomy import ErrorParser, StructuredError
from .patch_generator import PatchGenerator

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """
    Result of analyzing a failed run.

    Contains the root cause, affected components, and suggested patches.
    """
    root_cause: str                     # "Pitch too large for domain scale"
    affected_policies: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    suggested_patches: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0             # 0.0 to 1.0
    reasoning: str = ""                 # Explanation of the analysis
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "affected_policies": self.affected_policies,
            "affected_components": self.affected_components,
            "suggested_patches": self.suggested_patches,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "error_messages": self.error_messages,
        }


class RunAnalyzer:
    """
    Analyzes run results and proposes corrections.

    Examines failed runs to identify root causes and generate
    fix proposals automatically.
    """

    def __init__(self):
        """Initialize the run analyzer."""
        self.error_parser = ErrorParser()
        self.patch_generator = PatchGenerator()

    def analyze_run_failure(
        self,
        run_result: Dict[str, Any],
        spec: Dict[str, Any],
    ) -> Optional[AnalysisResult]:
        """
        Analyze why a run failed and suggest fixes.

        Parameters
        ----------
        run_result : dict
            The run result dictionary from the pipeline
        spec : dict
            The current spec that was run

        Returns
        -------
        AnalysisResult or None
            Analysis result with suggested fixes, or None if unable to analyze
        """
        if run_result.get("success", False):
            # Run succeeded, no analysis needed
            return None

        errors = run_result.get("errors", [])
        if not errors:
            # No errors to analyze
            return None

        # Parse errors into structured format
        error_messages = [str(e) for e in errors]
        structured_errors = self.error_parser.parse_multiple(error_messages, spec)

        # Find the highest confidence error
        best_error = max(structured_errors, key=lambda e: e.confidence)

        if best_error.confidence < 0.5:
            # Low confidence - fall back to basic analysis
            return self._fallback_analysis(errors, spec)

        # Generate patches for the errors
        patches = self.patch_generator.generate_fix_patches(structured_errors, spec)

        # Create analysis result
        reasoning = f"{best_error.suggested_fix}"
        if best_error.affected_policy:
            reasoning += f" (affects policy: {best_error.affected_policy})"

        return AnalysisResult(
            root_cause=best_error.suggested_fix,
            affected_policies=[best_error.affected_policy] if best_error.affected_policy else [],
            affected_components=[best_error.affected_component] if best_error.affected_component else [],
            suggested_patches=patches,
            confidence=best_error.confidence,
            reasoning=reasoning,
            error_messages=error_messages[:3],
        )

    def _fallback_analysis(
        self,
        errors: List[Any],
        spec: Dict[str, Any],
    ) -> AnalysisResult:
        """
        Fallback analysis when structured parsing fails.

        Uses simple pattern matching for basic error detection.
        """
        for error in errors:
            error_str = str(error)

            # Check for common error patterns with simple string matching
            analysis = self._analyze_pitch_error(error_str, spec)
            if analysis:
                return analysis

            analysis = self._analyze_units_error(error_str, spec)
            if analysis:
                return analysis

            analysis = self._analyze_component_outside_domain(error_str, spec)
            if analysis:
                return analysis

            analysis = self._analyze_port_direction_error(error_str, spec)
            if analysis:
                return analysis

        # Could not identify specific error pattern
        return AnalysisResult(
            root_cause="Unknown error",
            error_messages=[str(e) for e in errors[:3]],
            confidence=0.2,
            reasoning="Unable to identify specific error pattern",
        )

    def _analyze_pitch_error(
        self,
        error: str,
        spec: Dict[str, Any],
    ) -> Optional[AnalysisResult]:
        """
        Analyze pitch-related errors.

        Common pattern: "domain scale is ~X but voxel_pitch is Y"
        """
        if "voxel_pitch" not in error.lower() and "pitch" not in error.lower():
            return None

        if "domain" not in error.lower() and "scale" not in error.lower():
            return None

        # Extract domain scale if possible
        import re
        domain_match = re.search(r"domain scale is ~?([\d.]+)", error, re.IGNORECASE)
        pitch_match = re.search(r"voxel_pitch is ([\d.]+)", error, re.IGNORECASE)

        domain_scale = float(domain_match.group(1)) if domain_match else None
        current_pitch = float(pitch_match.group(1)) if pitch_match else None

        # Calculate recommended pitch
        recommended_pitch = None
        if domain_scale:
            recommended_pitch = domain_scale / 100.0  # Common heuristic

        # Generate patch
        patches = []
        if recommended_pitch:
            patches.append({
                "op": "replace",
                "path": "/policies/mesh_merge/voxel_pitch",
                "value": recommended_pitch,
            })

        reasoning = (
            f"The voxel_pitch ({current_pitch}m) is too large for the domain scale "
            f"({domain_scale}m). A good rule of thumb is voxel_pitch = domain_scale / 100. "
            f"Recommended: {recommended_pitch}m"
        )

        return AnalysisResult(
            root_cause="Voxel pitch too large for domain scale",
            affected_policies=["mesh_merge"],
            suggested_patches=patches,
            confidence=0.9,
            reasoning=reasoning,
            error_messages=[error],
        )

    def _analyze_units_error(
        self,
        error: str,
        spec: Dict[str, Any],
    ) -> Optional[AnalysisResult]:
        """
        Analyze units mismatch errors.

        Common pattern: values in wrong units
        """
        if "unit" not in error.lower() and "mm" not in error.lower() and "meter" not in error.lower():
            return None

        return AnalysisResult(
            root_cause="Units mismatch",
            confidence=0.7,
            reasoning=(
                "Possible units mismatch detected. Ensure all geometric values are in "
                "the correct units (meters internally, but input_units can be mm)."
            ),
            error_messages=[error],
        )

    def _analyze_component_outside_domain(
        self,
        error: str,
        spec: Dict[str, Any],
    ) -> Optional[AnalysisResult]:
        """
        Analyze component outside domain errors.

        Common pattern: component position or size extends beyond domain bounds
        """
        if "outside" not in error.lower() and "beyond" not in error.lower():
            return None

        if "domain" not in error.lower():
            return None

        return AnalysisResult(
            root_cause="Component extends outside domain bounds",
            confidence=0.8,
            reasoning=(
                "One or more components are positioned or sized such that they "
                "extend beyond the domain boundaries. Consider adjusting component "
                "positions or increasing domain size."
            ),
            error_messages=[error],
        )

    def _analyze_port_direction_error(
        self,
        error: str,
        spec: Dict[str, Any],
    ) -> Optional[AnalysisResult]:
        """
        Analyze port direction errors.

        Common pattern: inlet/outlet direction not aligned with domain faces
        """
        if "port" not in error.lower() and "inlet" not in error.lower() and "outlet" not in error.lower():
            return None

        if "direction" not in error.lower():
            return None

        return AnalysisResult(
            root_cause="Port direction misalignment",
            confidence=0.7,
            reasoning=(
                "Port (inlet/outlet) direction may not be properly aligned with domain "
                "faces. Ensure port directions are specified as valid face normals "
                "(e.g., [1,0,0], [-1,0,0], [0,1,0], etc.)."
            ),
            error_messages=[error],
        )

    def should_auto_analyze(self, run_result: Dict[str, Any]) -> bool:
        """
        Determine if auto-analysis should be triggered.

        Parameters
        ----------
        run_result : dict
            The run result

        Returns
        -------
        bool
            True if auto-analysis should run
        """
        # Auto-analyze if run failed with errors
        if not run_result.get("success", False) and run_result.get("errors"):
            return True

        return False
