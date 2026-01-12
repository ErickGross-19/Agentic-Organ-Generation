"""
Advanced Iteration Strategies Module.

This module provides adaptive iteration strategies for the V5 controller,
including:
- Adaptive iteration budgets based on problem complexity
- Early stopping when convergence is detected
- Hierarchical iteration (coarse -> fine refinement)
- Causal inference for validation failure analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import time


class IterationPhase(Enum):
    """Phases of hierarchical iteration."""
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


class ConvergenceStatus(Enum):
    """Status of convergence detection."""
    NOT_STARTED = "not_started"
    IMPROVING = "improving"
    PLATEAUED = "plateaued"
    CONVERGED = "converged"
    DIVERGING = "diverging"


@dataclass
class IterationMetrics:
    """Metrics tracked during iteration."""
    iteration: int = 0
    phase: IterationPhase = IterationPhase.COARSE
    validation_score: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    fields_complete: int = 0
    fields_total: int = 0
    time_elapsed: float = 0.0
    convergence_status: ConvergenceStatus = ConvergenceStatus.NOT_STARTED
    
    def completeness_ratio(self) -> float:
        """Get the ratio of complete fields."""
        if self.fields_total == 0:
            return 0.0
        return self.fields_complete / self.fields_total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "phase": self.phase.value,
            "validation_score": self.validation_score,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "fields_complete": self.fields_complete,
            "fields_total": self.fields_total,
            "time_elapsed": self.time_elapsed,
            "convergence_status": self.convergence_status.value,
            "completeness_ratio": self.completeness_ratio(),
        }


@dataclass
class IterationBudget:
    """Budget for iterations."""
    max_iterations: int = 50
    max_time_seconds: float = 300.0
    max_errors_per_phase: int = 5
    min_improvement_threshold: float = 0.01
    convergence_window: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "max_time_seconds": self.max_time_seconds,
            "max_errors_per_phase": self.max_errors_per_phase,
            "min_improvement_threshold": self.min_improvement_threshold,
            "convergence_window": self.convergence_window,
        }


@dataclass
class IterationHistory:
    """History of iteration metrics."""
    metrics: List[IterationMetrics] = field(default_factory=list)
    
    def add(self, metric: IterationMetrics) -> None:
        """Add a metric to history."""
        self.metrics.append(metric)
    
    def get_recent(self, n: int = 5) -> List[IterationMetrics]:
        """Get the n most recent metrics."""
        return self.metrics[-n:] if self.metrics else []
    
    def get_score_trend(self, window: int = 3) -> List[float]:
        """Get the trend of validation scores."""
        recent = self.get_recent(window)
        return [m.validation_score for m in recent]
    
    def is_improving(self, threshold: float = 0.01) -> bool:
        """Check if scores are improving."""
        trend = self.get_score_trend()
        if len(trend) < 2:
            return True
        return trend[-1] - trend[0] > threshold
    
    def is_plateaued(self, threshold: float = 0.01, window: int = 3) -> bool:
        """Check if scores have plateaued."""
        trend = self.get_score_trend(window)
        if len(trend) < window:
            return False
        return max(trend) - min(trend) < threshold


class ComplexityEstimator:
    """
    Estimate problem complexity to determine iteration budget.
    
    Complexity factors:
    - Topology type (dual_trees > tree > path)
    - Domain size (larger = more complex)
    - Number of constraints
    - Target terminal count
    """
    
    TOPOLOGY_COMPLEXITY = {
        "dual_trees": 1.5,
        "tree": 1.0,
        "backbone": 0.8,
        "loop": 0.9,
        "path": 0.5,
    }
    
    @classmethod
    def estimate_complexity(cls, spec: Dict[str, Any]) -> float:
        """
        Estimate problem complexity from specification.
        
        Parameters
        ----------
        spec : dict
            Design specification
            
        Returns
        -------
        float
            Complexity score (0.0 to 2.0+)
        """
        complexity = 0.0
        
        topology = spec.get("topology", {}).get("kind", "tree")
        complexity += cls.TOPOLOGY_COMPLEXITY.get(topology, 1.0)
        
        domain_size = spec.get("domain", {}).get("size", [20, 60, 30])
        if isinstance(domain_size, (list, tuple)) and len(domain_size) >= 3:
            volume = domain_size[0] * domain_size[1] * domain_size[2]
            complexity += min(0.5, volume / 100000)
        
        target_terminals = spec.get("topology", {}).get("target_terminals", 50)
        complexity += min(0.5, target_terminals / 200)
        
        constraints = spec.get("constraints", [])
        complexity += len(constraints) * 0.1
        
        return complexity
    
    @classmethod
    def get_recommended_budget(cls, spec: Dict[str, Any]) -> IterationBudget:
        """
        Get recommended iteration budget based on complexity.
        
        Parameters
        ----------
        spec : dict
            Design specification
            
        Returns
        -------
        IterationBudget
            Recommended budget
        """
        complexity = cls.estimate_complexity(spec)
        
        base_iterations = 30
        base_time = 180.0
        
        return IterationBudget(
            max_iterations=int(base_iterations * (1 + complexity * 0.5)),
            max_time_seconds=base_time * (1 + complexity * 0.3),
            max_errors_per_phase=max(3, int(5 * complexity)),
            min_improvement_threshold=0.01 / (1 + complexity * 0.2),
            convergence_window=max(2, int(3 * complexity)),
        )


class ConvergenceDetector:
    """
    Detect convergence in iteration process.
    
    Uses multiple signals:
    - Validation score plateau
    - Error count stability
    - Field completeness
    """
    
    def __init__(self, threshold: float = 0.01, window: int = 3):
        """
        Initialize convergence detector.
        
        Parameters
        ----------
        threshold : float
            Minimum improvement threshold
        window : int
            Number of iterations to consider
        """
        self.threshold = threshold
        self.window = window
    
    def detect(self, history: IterationHistory) -> ConvergenceStatus:
        """
        Detect convergence status from history.
        
        Parameters
        ----------
        history : IterationHistory
            Iteration history
            
        Returns
        -------
        ConvergenceStatus
            Current convergence status
        """
        if len(history.metrics) < 2:
            return ConvergenceStatus.NOT_STARTED
        
        recent = history.get_recent(self.window)
        
        if len(recent) < self.window:
            return ConvergenceStatus.IMPROVING
        
        scores = [m.validation_score for m in recent]
        
        if all(s >= 0.95 for s in scores):
            return ConvergenceStatus.CONVERGED
        
        if max(scores) - min(scores) < self.threshold:
            return ConvergenceStatus.PLATEAUED
        
        if scores[-1] < scores[0] - self.threshold:
            return ConvergenceStatus.DIVERGING
        
        return ConvergenceStatus.IMPROVING
    
    def should_stop_early(self, history: IterationHistory) -> Tuple[bool, str]:
        """
        Determine if iteration should stop early.
        
        Parameters
        ----------
        history : IterationHistory
            Iteration history
            
        Returns
        -------
        Tuple[bool, str]
            (should_stop, reason)
        """
        status = self.detect(history)
        
        if status == ConvergenceStatus.CONVERGED:
            return True, "Converged: validation score reached target"
        
        if status == ConvergenceStatus.PLATEAUED:
            recent = history.get_recent(self.window)
            if recent and recent[-1].validation_score > 0.8:
                return True, "Plateaued at acceptable quality"
        
        if status == ConvergenceStatus.DIVERGING:
            return True, "Diverging: quality is decreasing"
        
        return False, ""


class HierarchicalIterator:
    """
    Implement hierarchical iteration (coarse -> fine refinement).
    
    Phases:
    1. COARSE: Focus on topology and major structure
    2. MEDIUM: Refine positions and connections
    3. FINE: Optimize radii and fine details
    """
    
    PHASE_PRIORITIES = {
        IterationPhase.COARSE: [
            "topology.kind",
            "domain.type",
            "domain.size",
            "inlet.face",
            "outlet.face",
        ],
        IterationPhase.MEDIUM: [
            "inlet.position",
            "outlet.position",
            "inlet.radius",
            "outlet.radius",
            "topology.target_terminals",
        ],
        IterationPhase.FINE: [
            "topology.segment_budget",
            "topology.density_level",
            "constraints",
        ],
    }
    
    def __init__(self):
        """Initialize the hierarchical iterator."""
        self.current_phase = IterationPhase.COARSE
        self.phase_iterations = {phase: 0 for phase in IterationPhase}
    
    def get_current_phase(self) -> IterationPhase:
        """Get the current iteration phase."""
        return self.current_phase
    
    def get_priority_fields(self) -> List[str]:
        """Get priority fields for current phase."""
        return self.PHASE_PRIORITIES.get(self.current_phase, [])
    
    def should_advance_phase(self, metrics: IterationMetrics) -> bool:
        """
        Determine if should advance to next phase.
        
        Parameters
        ----------
        metrics : IterationMetrics
            Current iteration metrics
            
        Returns
        -------
        bool
            True if should advance
        """
        if self.current_phase == IterationPhase.FINE:
            return False
        
        if metrics.error_count == 0 and metrics.completeness_ratio() > 0.8:
            return True
        
        if self.phase_iterations[self.current_phase] > 10:
            return True
        
        return False
    
    def advance_phase(self) -> Optional[IterationPhase]:
        """
        Advance to the next phase.
        
        Returns
        -------
        Optional[IterationPhase]
            New phase, or None if already at finest
        """
        phases = list(IterationPhase)
        current_idx = phases.index(self.current_phase)
        
        if current_idx < len(phases) - 1:
            self.current_phase = phases[current_idx + 1]
            return self.current_phase
        
        return None
    
    def record_iteration(self) -> None:
        """Record an iteration in the current phase."""
        self.phase_iterations[self.current_phase] += 1


class ValidationFailureAnalyzer:
    """
    Analyze validation failures to determine root causes.
    
    Uses causal inference patterns to identify:
    - Which parameters are likely causing failures
    - Suggested fixes based on failure patterns
    """
    
    FAILURE_PATTERNS = {
        "collision": {
            "likely_causes": ["inlet.radius too large", "domain.size too small", "topology.target_terminals too high"],
            "suggested_fixes": ["Reduce inlet radius", "Increase domain size", "Reduce target terminals"],
        },
        "connectivity": {
            "likely_causes": ["inlet/outlet positions incompatible", "topology mismatch"],
            "suggested_fixes": ["Check inlet/outlet face alignment", "Verify topology supports flow path"],
        },
        "manufacturability": {
            "likely_causes": ["radii below minimum", "wall thickness insufficient"],
            "suggested_fixes": ["Increase minimum radius", "Adjust embedding parameters"],
        },
        "flow": {
            "likely_causes": ["Murray's law violation", "pressure gradient issues"],
            "suggested_fixes": ["Adjust branching radii", "Check inlet/outlet radius ratio"],
        },
    }
    
    @classmethod
    def analyze_failure(cls, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a validation failure.
        
        Parameters
        ----------
        validation_result : dict
            Validation result with errors
            
        Returns
        -------
        dict
            Analysis with likely causes and suggested fixes
        """
        errors = validation_result.get("errors", [])
        checks = validation_result.get("checks", [])
        
        analysis = {
            "error_count": len(errors),
            "failure_categories": [],
            "likely_causes": [],
            "suggested_fixes": [],
            "priority_fixes": [],
        }
        
        for check in checks:
            if not check.get("passed", True):
                check_name = check.get("name", "").lower()
                
                for category, patterns in cls.FAILURE_PATTERNS.items():
                    if category in check_name:
                        analysis["failure_categories"].append(category)
                        analysis["likely_causes"].extend(patterns["likely_causes"])
                        analysis["suggested_fixes"].extend(patterns["suggested_fixes"])
        
        analysis["likely_causes"] = list(set(analysis["likely_causes"]))
        analysis["suggested_fixes"] = list(set(analysis["suggested_fixes"]))
        
        if analysis["suggested_fixes"]:
            analysis["priority_fixes"] = analysis["suggested_fixes"][:3]
        
        return analysis
    
    @classmethod
    def get_fix_priority(cls, analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get prioritized list of fixes.
        
        Parameters
        ----------
        analysis : dict
            Failure analysis
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (fix, priority_score) tuples
        """
        fixes = analysis.get("suggested_fixes", [])
        categories = analysis.get("failure_categories", [])
        
        priority_scores = []
        for fix in fixes:
            score = 1.0
            
            if "radius" in fix.lower():
                score += 0.3
            if "domain" in fix.lower():
                score += 0.2
            
            if "collision" in categories:
                if "radius" in fix.lower() or "size" in fix.lower():
                    score += 0.5
            
            priority_scores.append((fix, score))
        
        return sorted(priority_scores, key=lambda x: x[1], reverse=True)


class AdaptiveIterationController:
    """
    Main controller for adaptive iteration strategies.
    
    Combines:
    - Complexity-based budget estimation
    - Convergence detection with early stopping
    - Hierarchical iteration phases
    - Failure analysis and recovery
    """
    
    def __init__(self, spec: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive iteration controller.
        
        Parameters
        ----------
        spec : dict, optional
            Initial specification for budget estimation
        """
        self.spec = spec or {}
        self.budget = ComplexityEstimator.get_recommended_budget(self.spec)
        self.history = IterationHistory()
        self.convergence_detector = ConvergenceDetector(
            threshold=self.budget.min_improvement_threshold,
            window=self.budget.convergence_window,
        )
        self.hierarchical_iterator = HierarchicalIterator()
        self.start_time = time.time()
    
    def update_spec(self, spec: Dict[str, Any]) -> None:
        """Update specification and recalculate budget."""
        self.spec = spec
        self.budget = ComplexityEstimator.get_recommended_budget(spec)
        self.convergence_detector = ConvergenceDetector(
            threshold=self.budget.min_improvement_threshold,
            window=self.budget.convergence_window,
        )
    
    def record_iteration(
        self,
        validation_score: float,
        error_count: int,
        warning_count: int,
        fields_complete: int,
        fields_total: int,
    ) -> IterationMetrics:
        """
        Record an iteration and get updated metrics.
        
        Parameters
        ----------
        validation_score : float
            Current validation score (0-1)
        error_count : int
            Number of errors
        warning_count : int
            Number of warnings
        fields_complete : int
            Number of complete fields
        fields_total : int
            Total number of fields
            
        Returns
        -------
        IterationMetrics
            Updated metrics
        """
        metrics = IterationMetrics(
            iteration=len(self.history.metrics) + 1,
            phase=self.hierarchical_iterator.get_current_phase(),
            validation_score=validation_score,
            error_count=error_count,
            warning_count=warning_count,
            fields_complete=fields_complete,
            fields_total=fields_total,
            time_elapsed=time.time() - self.start_time,
            convergence_status=self.convergence_detector.detect(self.history),
        )
        
        self.history.add(metrics)
        self.hierarchical_iterator.record_iteration()
        
        if self.hierarchical_iterator.should_advance_phase(metrics):
            self.hierarchical_iterator.advance_phase()
        
        return metrics
    
    def should_continue(self) -> Tuple[bool, str]:
        """
        Determine if iteration should continue.
        
        Returns
        -------
        Tuple[bool, str]
            (should_continue, reason_if_stopping)
        """
        if len(self.history.metrics) >= self.budget.max_iterations:
            return False, f"Reached maximum iterations ({self.budget.max_iterations})"
        
        elapsed = time.time() - self.start_time
        if elapsed >= self.budget.max_time_seconds:
            return False, f"Reached time limit ({self.budget.max_time_seconds}s)"
        
        should_stop, reason = self.convergence_detector.should_stop_early(self.history)
        if should_stop:
            return False, reason
        
        return True, ""
    
    def get_priority_fields(self) -> List[str]:
        """Get priority fields for current phase."""
        return self.hierarchical_iterator.get_priority_fields()
    
    def analyze_failure(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a validation failure."""
        return ValidationFailureAnalyzer.analyze_failure(validation_result)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current iteration status."""
        recent = self.history.get_recent(1)
        current_metrics = recent[0] if recent else IterationMetrics()
        
        return {
            "iteration": current_metrics.iteration,
            "phase": self.hierarchical_iterator.get_current_phase().value,
            "convergence": current_metrics.convergence_status.value,
            "validation_score": current_metrics.validation_score,
            "completeness": current_metrics.completeness_ratio(),
            "time_elapsed": current_metrics.time_elapsed,
            "budget_remaining": {
                "iterations": self.budget.max_iterations - current_metrics.iteration,
                "time": self.budget.max_time_seconds - current_metrics.time_elapsed,
            },
            "priority_fields": self.get_priority_fields(),
        }
