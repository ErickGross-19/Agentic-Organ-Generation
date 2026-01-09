"""
Adaptation Engine Module

Dynamically adjusts specs based on generation results.
This provides real-time spec adjustment during the generation process.

The engine:
1. Analyzes validation failures and generation results
2. Suggests parameter adjustments
3. Predicts impact of changes
4. Auto-corrects common issues
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json


class IssueCategory(Enum):
    """Categories of issues that can be detected."""
    COLLISION = "collision"
    COVERAGE = "coverage"
    FLOW = "flow"
    MESH_QUALITY = "mesh_quality"
    MANUFACTURABILITY = "manufacturability"
    TOPOLOGY = "topology"
    PERFORMANCE = "performance"


class SuggestionPriority(Enum):
    """Priority levels for suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Issue:
    """An issue detected during generation or validation."""
    category: IssueCategory
    description: str
    severity: str  # "error", "warning", "info"
    location: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Suggestion:
    """A suggested parameter adjustment."""
    parameter: str
    current_value: Any
    suggested_value: Any
    reason: str
    priority: SuggestionPriority
    expected_impact: str
    confidence: float = 0.8


@dataclass
class AdaptationResult:
    """Result of adaptation analysis."""
    issues: List[Issue] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    auto_corrections: Dict[str, Any] = field(default_factory=dict)
    analysis_summary: str = ""
    can_auto_fix: bool = False


@dataclass
class GenerationResult:
    """Result from a generation attempt."""
    success: bool
    network: Optional[Any] = None
    mesh: Optional[Any] = None
    validation_report: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AdaptationEngine:
    """
    Dynamically adjusts specs based on generation results.
    
    This class analyzes validation failures and generation results,
    suggests parameter adjustments, and can auto-correct common issues.
    """
    
    def __init__(self):
        self._issue_handlers = {
            IssueCategory.COLLISION: self._handle_collision_issue,
            IssueCategory.COVERAGE: self._handle_coverage_issue,
            IssueCategory.FLOW: self._handle_flow_issue,
            IssueCategory.MESH_QUALITY: self._handle_mesh_quality_issue,
            IssueCategory.MANUFACTURABILITY: self._handle_manufacturability_issue,
            IssueCategory.TOPOLOGY: self._handle_topology_issue,
            IssueCategory.PERFORMANCE: self._handle_performance_issue,
        }
        
        self._adaptation_history: List[AdaptationResult] = []
    
    def analyze_and_suggest(self, result: GenerationResult) -> AdaptationResult:
        """
        Analyze generation results and suggest parameter adjustments.
        
        Parameters
        ----------
        result : GenerationResult
            The result from a generation attempt
            
        Returns
        -------
        AdaptationResult
            Analysis with issues, suggestions, and auto-corrections
        """
        issues = self._detect_issues(result)
        suggestions = []
        auto_corrections = {}
        
        for issue in issues:
            handler = self._issue_handlers.get(issue.category)
            if handler:
                issue_suggestions, issue_corrections = handler(issue, result)
                suggestions.extend(issue_suggestions)
                auto_corrections.update(issue_corrections)
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda s: list(SuggestionPriority).index(s.priority))
        
        # Determine if we can auto-fix
        can_auto_fix = bool(auto_corrections) and all(
            s.priority != SuggestionPriority.CRITICAL for s in suggestions
        )
        
        # Generate summary
        summary = self._generate_summary(issues, suggestions)
        
        adaptation_result = AdaptationResult(
            issues=issues,
            suggestions=suggestions,
            auto_corrections=auto_corrections,
            analysis_summary=summary,
            can_auto_fix=can_auto_fix,
        )
        
        self._adaptation_history.append(adaptation_result)
        
        return adaptation_result
    
    def _detect_issues(self, result: GenerationResult) -> List[Issue]:
        """Detect issues from generation result."""
        issues = []
        
        # Check for errors
        for error in result.errors:
            issue = self._classify_error(error)
            if issue:
                issues.append(issue)
        
        # Check for warnings
        for warning in result.warnings:
            issue = self._classify_warning(warning)
            if issue:
                issues.append(issue)
        
        # Check validation report
        if result.validation_report:
            issues.extend(self._analyze_validation_report(result.validation_report))
        
        # Check metrics
        if result.metrics:
            issues.extend(self._analyze_metrics(result.metrics))
        
        return issues
    
    def _classify_error(self, error: str) -> Optional[Issue]:
        """Classify an error string into an Issue."""
        error_lower = error.lower()
        
        if "collision" in error_lower or "intersect" in error_lower:
            return Issue(
                category=IssueCategory.COLLISION,
                description=error,
                severity="error",
                details={"raw_error": error},
            )
        
        if "coverage" in error_lower or "uncovered" in error_lower:
            return Issue(
                category=IssueCategory.COVERAGE,
                description=error,
                severity="error",
                details={"raw_error": error},
            )
        
        if "flow" in error_lower or "pressure" in error_lower or "reynolds" in error_lower:
            return Issue(
                category=IssueCategory.FLOW,
                description=error,
                severity="error",
                details={"raw_error": error},
            )
        
        if "mesh" in error_lower or "manifold" in error_lower or "watertight" in error_lower:
            return Issue(
                category=IssueCategory.MESH_QUALITY,
                description=error,
                severity="error",
                details={"raw_error": error},
            )
        
        if "diameter" in error_lower or "thickness" in error_lower or "overhang" in error_lower:
            return Issue(
                category=IssueCategory.MANUFACTURABILITY,
                description=error,
                severity="error",
                details={"raw_error": error},
            )
        
        if "topology" in error_lower or "branch" in error_lower or "terminal" in error_lower:
            return Issue(
                category=IssueCategory.TOPOLOGY,
                description=error,
                severity="error",
                details={"raw_error": error},
            )
        
        return Issue(
            category=IssueCategory.PERFORMANCE,
            description=error,
            severity="error",
            details={"raw_error": error},
        )
    
    def _classify_warning(self, warning: str) -> Optional[Issue]:
        """Classify a warning string into an Issue."""
        issue = self._classify_error(warning)
        if issue:
            issue.severity = "warning"
        return issue
    
    def _analyze_validation_report(self, report: Dict[str, Any]) -> List[Issue]:
        """Analyze a validation report for issues."""
        issues = []
        
        # Check pre-embedding validation
        pre_embedding = report.get("pre_embedding", {})
        if not pre_embedding.get("mesh_integrity", {}).get("passed", True):
            issues.append(Issue(
                category=IssueCategory.MESH_QUALITY,
                description="Mesh integrity check failed",
                severity="error",
                details=pre_embedding.get("mesh_integrity", {}),
            ))
        
        if not pre_embedding.get("murray_law", {}).get("passed", True):
            issues.append(Issue(
                category=IssueCategory.TOPOLOGY,
                description="Murray's law validation failed",
                severity="warning",
                details=pre_embedding.get("murray_law", {}),
            ))
        
        if not pre_embedding.get("collision_free", {}).get("passed", True):
            issues.append(Issue(
                category=IssueCategory.COLLISION,
                description="Collision detected in network",
                severity="error",
                details=pre_embedding.get("collision_free", {}),
            ))
        
        # Check post-embedding validation
        post_embedding = report.get("post_embedding", {})
        if not post_embedding.get("port_accessibility", {}).get("passed", True):
            issues.append(Issue(
                category=IssueCategory.MANUFACTURABILITY,
                description="Port accessibility check failed",
                severity="error",
                details=post_embedding.get("port_accessibility", {}),
            ))
        
        if not post_embedding.get("min_diameter", {}).get("passed", True):
            issues.append(Issue(
                category=IssueCategory.MANUFACTURABILITY,
                description="Minimum diameter constraint violated",
                severity="error",
                details=post_embedding.get("min_diameter", {}),
            ))
        
        if not post_embedding.get("wall_thickness", {}).get("passed", True):
            issues.append(Issue(
                category=IssueCategory.MANUFACTURABILITY,
                description="Wall thickness constraint violated",
                severity="warning",
                details=post_embedding.get("wall_thickness", {}),
            ))
        
        return issues
    
    def _analyze_metrics(self, metrics: Dict[str, Any]) -> List[Issue]:
        """Analyze generation metrics for issues."""
        issues = []
        
        # Check coverage
        coverage = metrics.get("coverage", 1.0)
        if coverage < 0.8:
            issues.append(Issue(
                category=IssueCategory.COVERAGE,
                description=f"Low tissue coverage: {coverage*100:.1f}%",
                severity="warning" if coverage > 0.6 else "error",
                details={"coverage": coverage},
            ))
        
        # Check terminal count
        target_terminals = metrics.get("target_terminals")
        actual_terminals = metrics.get("actual_terminals")
        if target_terminals and actual_terminals:
            ratio = actual_terminals / target_terminals
            if ratio < 0.7:
                issues.append(Issue(
                    category=IssueCategory.TOPOLOGY,
                    description=f"Terminal count below target: {actual_terminals}/{target_terminals}",
                    severity="warning",
                    details={"target": target_terminals, "actual": actual_terminals},
                ))
        
        # Check generation time
        generation_time = metrics.get("generation_time_seconds", 0)
        if generation_time > 300:  # 5 minutes
            issues.append(Issue(
                category=IssueCategory.PERFORMANCE,
                description=f"Long generation time: {generation_time:.1f}s",
                severity="info",
                details={"time_seconds": generation_time},
            ))
        
        return issues
    
    def _handle_collision_issue(
        self,
        issue: Issue,
        result: GenerationResult
    ) -> Tuple[List[Suggestion], Dict[str, Any]]:
        """Handle collision issues."""
        suggestions = []
        auto_corrections = {}
        
        # Suggest increasing clearance
        suggestions.append(Suggestion(
            parameter="min_clearance",
            current_value=result.metrics.get("min_clearance", 0.0005),
            suggested_value=result.metrics.get("min_clearance", 0.0005) * 1.5,
            reason="Increase minimum clearance to prevent collisions",
            priority=SuggestionPriority.HIGH,
            expected_impact="Reduces collision probability but may reduce coverage",
            confidence=0.85,
        ))
        
        # Suggest reducing step size
        suggestions.append(Suggestion(
            parameter="step_size",
            current_value=result.metrics.get("step_size", 0.001),
            suggested_value=result.metrics.get("step_size", 0.001) * 0.8,
            reason="Smaller steps allow finer collision avoidance",
            priority=SuggestionPriority.MEDIUM,
            expected_impact="Slower generation but better collision avoidance",
            confidence=0.75,
        ))
        
        # Auto-correct: increase clearance by 20%
        if issue.severity == "warning":
            current_clearance = result.metrics.get("min_clearance", 0.0005)
            auto_corrections["min_clearance"] = current_clearance * 1.2
        
        return suggestions, auto_corrections
    
    def _handle_coverage_issue(
        self,
        issue: Issue,
        result: GenerationResult
    ) -> Tuple[List[Suggestion], Dict[str, Any]]:
        """Handle coverage issues."""
        suggestions = []
        auto_corrections = {}
        
        coverage = issue.details.get("coverage", 0.5)
        
        # Suggest increasing terminal count
        current_terminals = result.metrics.get("target_terminals", 50)
        suggested_terminals = int(current_terminals * (1.0 / coverage) * 0.9)
        
        suggestions.append(Suggestion(
            parameter="target_terminals",
            current_value=current_terminals,
            suggested_value=suggested_terminals,
            reason="More terminals improve tissue coverage",
            priority=SuggestionPriority.HIGH,
            expected_impact=f"Expected coverage improvement to ~{min(coverage * 1.3, 0.95)*100:.0f}%",
            confidence=0.8,
        ))
        
        # Suggest increasing influence radius
        suggestions.append(Suggestion(
            parameter="influence_radius",
            current_value=result.metrics.get("influence_radius", 0.015),
            suggested_value=result.metrics.get("influence_radius", 0.015) * 1.2,
            reason="Larger influence radius allows growth toward more attractors",
            priority=SuggestionPriority.MEDIUM,
            expected_impact="Better coverage but potentially less uniform distribution",
            confidence=0.7,
        ))
        
        # Auto-correct: increase terminals by 30% if coverage is moderate
        if coverage > 0.5:
            auto_corrections["target_terminals"] = int(current_terminals * 1.3)
        
        return suggestions, auto_corrections
    
    def _handle_flow_issue(
        self,
        issue: Issue,
        result: GenerationResult
    ) -> Tuple[List[Suggestion], Dict[str, Any]]:
        """Handle flow-related issues."""
        suggestions = []
        auto_corrections = {}
        
        # Suggest adjusting inlet radius
        suggestions.append(Suggestion(
            parameter="inlet_radius",
            current_value=result.metrics.get("inlet_radius", 0.002),
            suggested_value=result.metrics.get("inlet_radius", 0.002) * 1.2,
            reason="Larger inlet improves flow characteristics",
            priority=SuggestionPriority.MEDIUM,
            expected_impact="Better flow but may require domain adjustment",
            confidence=0.7,
        ))
        
        # Suggest adjusting radius decay
        suggestions.append(Suggestion(
            parameter="radius_decay",
            current_value=result.metrics.get("radius_decay", 0.95),
            suggested_value=0.92,
            reason="Slower radius decay maintains flow capacity",
            priority=SuggestionPriority.MEDIUM,
            expected_impact="Better flow but larger terminal vessels",
            confidence=0.75,
        ))
        
        return suggestions, auto_corrections
    
    def _handle_mesh_quality_issue(
        self,
        issue: Issue,
        result: GenerationResult
    ) -> Tuple[List[Suggestion], Dict[str, Any]]:
        """Handle mesh quality issues."""
        suggestions = []
        auto_corrections = {}
        
        # Suggest increasing minimum radius
        suggestions.append(Suggestion(
            parameter="min_radius",
            current_value=result.metrics.get("min_radius", 0.0001),
            suggested_value=result.metrics.get("min_radius", 0.0001) * 1.5,
            reason="Larger minimum radius produces cleaner meshes",
            priority=SuggestionPriority.HIGH,
            expected_impact="Better mesh quality but fewer fine branches",
            confidence=0.85,
        ))
        
        # Suggest enabling mesh repair
        suggestions.append(Suggestion(
            parameter="enable_mesh_repair",
            current_value=False,
            suggested_value=True,
            reason="Mesh repair can fix non-manifold edges and holes",
            priority=SuggestionPriority.HIGH,
            expected_impact="Cleaner mesh but slightly longer processing",
            confidence=0.9,
        ))
        
        # Auto-correct: enable mesh repair
        auto_corrections["enable_mesh_repair"] = True
        
        return suggestions, auto_corrections
    
    def _handle_manufacturability_issue(
        self,
        issue: Issue,
        result: GenerationResult
    ) -> Tuple[List[Suggestion], Dict[str, Any]]:
        """Handle manufacturability issues."""
        suggestions = []
        auto_corrections = {}
        
        if "diameter" in issue.description.lower():
            # Suggest increasing minimum radius
            current_min = result.metrics.get("min_radius", 0.0001)
            manufacturing_min = 0.00025  # 0.5mm diameter
            
            suggestions.append(Suggestion(
                parameter="min_radius",
                current_value=current_min,
                suggested_value=max(current_min, manufacturing_min),
                reason="Minimum diameter must meet manufacturing constraints",
                priority=SuggestionPriority.CRITICAL,
                expected_impact="Printable structure but fewer fine branches",
                confidence=0.95,
            ))
            
            auto_corrections["min_radius"] = max(current_min, manufacturing_min)
        
        if "thickness" in issue.description.lower():
            suggestions.append(Suggestion(
                parameter="wall_thickness",
                current_value=result.metrics.get("wall_thickness", 0.0003),
                suggested_value=0.0005,  # 0.5mm
                reason="Wall thickness must meet manufacturing constraints",
                priority=SuggestionPriority.CRITICAL,
                expected_impact="Stronger walls but larger overall structure",
                confidence=0.95,
            ))
        
        if "overhang" in issue.description.lower():
            suggestions.append(Suggestion(
                parameter="max_overhang_angle",
                current_value=result.metrics.get("max_overhang_angle", 45),
                suggested_value=40,
                reason="Reduce overhang angle for better printability",
                priority=SuggestionPriority.HIGH,
                expected_impact="Better printability but may affect topology",
                confidence=0.8,
            ))
        
        return suggestions, auto_corrections
    
    def _handle_topology_issue(
        self,
        issue: Issue,
        result: GenerationResult
    ) -> Tuple[List[Suggestion], Dict[str, Any]]:
        """Handle topology issues."""
        suggestions = []
        auto_corrections = {}
        
        if "terminal" in issue.description.lower():
            # Suggest increasing max steps
            suggestions.append(Suggestion(
                parameter="max_steps",
                current_value=result.metrics.get("max_steps", 500),
                suggested_value=int(result.metrics.get("max_steps", 500) * 1.5),
                reason="More steps allow growth to reach target terminal count",
                priority=SuggestionPriority.MEDIUM,
                expected_impact="Better terminal count but longer generation",
                confidence=0.8,
            ))
            
            # Suggest adjusting kill radius
            suggestions.append(Suggestion(
                parameter="kill_radius",
                current_value=result.metrics.get("kill_radius", 0.002),
                suggested_value=result.metrics.get("kill_radius", 0.002) * 0.8,
                reason="Smaller kill radius allows more branching",
                priority=SuggestionPriority.MEDIUM,
                expected_impact="More terminals but denser network",
                confidence=0.75,
            ))
        
        if "murray" in issue.description.lower():
            suggestions.append(Suggestion(
                parameter="murray_gamma",
                current_value=result.metrics.get("murray_gamma", 3.0),
                suggested_value=3.0,
                reason="Use standard Murray's law exponent",
                priority=SuggestionPriority.LOW,
                expected_impact="Physiologically correct branching ratios",
                confidence=0.9,
            ))
        
        return suggestions, auto_corrections
    
    def _handle_performance_issue(
        self,
        issue: Issue,
        result: GenerationResult
    ) -> Tuple[List[Suggestion], Dict[str, Any]]:
        """Handle performance issues."""
        suggestions = []
        auto_corrections = {}
        
        # Suggest reducing complexity
        suggestions.append(Suggestion(
            parameter="target_terminals",
            current_value=result.metrics.get("target_terminals", 50),
            suggested_value=int(result.metrics.get("target_terminals", 50) * 0.8),
            reason="Fewer terminals reduce generation time",
            priority=SuggestionPriority.LOW,
            expected_impact="Faster generation but lower coverage",
            confidence=0.7,
        ))
        
        # Suggest increasing step size
        suggestions.append(Suggestion(
            parameter="step_size",
            current_value=result.metrics.get("step_size", 0.001),
            suggested_value=result.metrics.get("step_size", 0.001) * 1.2,
            reason="Larger steps reduce iteration count",
            priority=SuggestionPriority.LOW,
            expected_impact="Faster generation but coarser structure",
            confidence=0.7,
        ))
        
        return suggestions, auto_corrections
    
    def _generate_summary(self, issues: List[Issue], suggestions: List[Suggestion]) -> str:
        """Generate a summary of the analysis."""
        if not issues:
            return "No issues detected. Generation completed successfully."
        
        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        
        lines = []
        lines.append(f"Analysis found {error_count} error(s) and {warning_count} warning(s).")
        
        if error_count > 0:
            lines.append("\nCritical issues:")
            for issue in issues:
                if issue.severity == "error":
                    lines.append(f"  - [{issue.category.value}] {issue.description}")
        
        if suggestions:
            lines.append(f"\n{len(suggestions)} suggestion(s) available.")
            top_suggestion = suggestions[0]
            lines.append(f"Top suggestion: Adjust {top_suggestion.parameter} from "
                        f"{top_suggestion.current_value} to {top_suggestion.suggested_value}")
        
        return "\n".join(lines)
    
    def apply_auto_corrections(
        self,
        spec_dict: Dict[str, Any],
        corrections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply auto-corrections to a spec dictionary.
        
        Parameters
        ----------
        spec_dict : Dict[str, Any]
            The current spec as a dictionary
        corrections : Dict[str, Any]
            The corrections to apply
            
        Returns
        -------
        Dict[str, Any]
            The updated spec dictionary
        """
        updated = spec_dict.copy()
        
        for param, value in corrections.items():
            # Handle nested parameters
            if "." in param:
                parts = param.split(".")
                current = updated
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                updated[param] = value
        
        return updated
    
    def get_adaptation_history(self) -> List[AdaptationResult]:
        """Get the history of adaptations."""
        return self._adaptation_history.copy()
    
    def predict_impact(
        self,
        current_metrics: Dict[str, Any],
        suggested_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict the impact of suggested changes.
        
        Parameters
        ----------
        current_metrics : Dict[str, Any]
            Current generation metrics
        suggested_changes : Dict[str, Any]
            Proposed parameter changes
            
        Returns
        -------
        Dict[str, Any]
            Predicted metrics after changes
        """
        predicted = current_metrics.copy()
        
        # Simple heuristic predictions
        if "target_terminals" in suggested_changes:
            ratio = suggested_changes["target_terminals"] / current_metrics.get("target_terminals", 50)
            predicted["coverage"] = min(current_metrics.get("coverage", 0.5) * ratio ** 0.5, 0.98)
            predicted["generation_time_seconds"] = current_metrics.get("generation_time_seconds", 60) * ratio
        
        if "min_radius" in suggested_changes:
            ratio = suggested_changes["min_radius"] / current_metrics.get("min_radius", 0.0001)
            predicted["mesh_quality"] = min(current_metrics.get("mesh_quality", 0.7) * ratio ** 0.3, 0.99)
        
        if "step_size" in suggested_changes:
            ratio = suggested_changes["step_size"] / current_metrics.get("step_size", 0.001)
            predicted["generation_time_seconds"] = current_metrics.get("generation_time_seconds", 60) / ratio
        
        return predicted
