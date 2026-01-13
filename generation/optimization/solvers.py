"""
Shared NLP solver utilities for vascular network optimization.

Provides common solver infrastructure that can be used by different
optimization problems (global geometry optimization, bifurcation point
optimization, etc.).

Supported solvers:
- IPOPT (via cyipopt) - Interior point method, recommended for large problems
- scipy.optimize - SLSQP, trust-constr, L-BFGS-B methods

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, Protocol
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class SolverConfig:
    """Configuration for NLP solvers."""
    
    solver: str = "scipy"
    method: str = "SLSQP"
    tolerance: float = 1e-6
    max_iterations: int = 1000
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "solver": self.solver,
            "method": self.method,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
        }


@dataclass
class SolverResult:
    """Result from NLP solver."""
    
    x: np.ndarray
    success: bool
    iterations: int
    objective_value: float
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "iterations": self.iterations,
            "objective_value": self.objective_value,
            "message": self.message,
        }


class NLPProblemInterface(Protocol):
    """Protocol for NLP problem interface."""
    
    def objective(self, x: np.ndarray) -> float:
        """Compute objective function value."""
        ...
    
    def objective_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function."""
        ...
    
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Compute constraint values."""
        ...


def solve_nlp(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    constraints: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    config: Optional[SolverConfig] = None,
) -> SolverResult:
    """
    Solve an NLP problem using the configured solver.
    
    Parameters
    ----------
    objective : callable
        Objective function f(x) -> float
    x0 : np.ndarray
        Initial guess
    bounds : tuple, optional
        (lower_bounds, upper_bounds) arrays
    gradient : callable, optional
        Gradient function grad_f(x) -> np.ndarray
    constraints : callable, optional
        Constraint function g(x) -> np.ndarray (equality constraints = 0)
    config : SolverConfig, optional
        Solver configuration
        
    Returns
    -------
    SolverResult
        Optimization result
    """
    if config is None:
        config = SolverConfig()
    
    if config.solver == "ipopt":
        try:
            return _solve_with_ipopt(
                objective, x0, bounds, gradient, constraints, config
            )
        except ImportError:
            config = SolverConfig(
                solver="scipy",
                method=config.method,
                tolerance=config.tolerance,
                max_iterations=config.max_iterations,
                verbose=config.verbose,
            )
    
    return _solve_with_scipy(
        objective, x0, bounds, gradient, constraints, config
    )


def _solve_with_scipy(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    gradient: Optional[Callable[[np.ndarray], np.ndarray]],
    constraints: Optional[Callable[[np.ndarray], np.ndarray]],
    config: SolverConfig,
) -> SolverResult:
    """Solve NLP using scipy.optimize."""
    from scipy.optimize import minimize, Bounds
    
    scipy_bounds = None
    if bounds is not None:
        lb, ub = bounds
        scipy_bounds = Bounds(lb, ub)
    
    scipy_constraints = None
    if constraints is not None:
        scipy_constraints = {
            'type': 'eq',
            'fun': constraints,
        }
    
    method = config.method
    options: Dict[str, Any] = {
        'maxiter': config.max_iterations,
    }
    
    if method == "SLSQP":
        options['ftol'] = config.tolerance
        options['disp'] = config.verbose
    elif method == "trust-constr":
        options['gtol'] = config.tolerance
        options['verbose'] = 2 if config.verbose else 0
    elif method == "L-BFGS-B":
        options['ftol'] = config.tolerance
        options['disp'] = config.verbose
    else:
        options['ftol'] = config.tolerance
    
    result = minimize(
        objective,
        x0,
        method=method,
        jac=gradient,
        bounds=scipy_bounds,
        constraints=scipy_constraints,
        options=options,
    )
    
    return SolverResult(
        x=result.x,
        success=result.success,
        iterations=result.nit if hasattr(result, 'nit') else 0,
        objective_value=result.fun,
        message=result.message if hasattr(result, 'message') else "",
    )


def _solve_with_ipopt(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    gradient: Optional[Callable[[np.ndarray], np.ndarray]],
    constraints: Optional[Callable[[np.ndarray], np.ndarray]],
    config: SolverConfig,
) -> SolverResult:
    """Solve NLP using IPOPT via cyipopt."""
    import cyipopt
    
    n_vars = len(x0)
    
    if bounds is not None:
        lb, ub = bounds
    else:
        lb = np.full(n_vars, -np.inf)
        ub = np.full(n_vars, np.inf)
    
    class IPOPTProblem:
        def __init__(self):
            self._gradient = gradient
            self._constraints = constraints
        
        def objective(self, x: np.ndarray) -> float:
            return objective(x)
        
        def gradient(self, x: np.ndarray) -> np.ndarray:
            if self._gradient is not None:
                return self._gradient(x)
            eps = 1e-8
            grad = np.zeros(len(x))
            f0 = objective(x)
            for i in range(len(x)):
                x_pert = x.copy()
                x_pert[i] += eps
                grad[i] = (objective(x_pert) - f0) / eps
            return grad
        
        def constraints(self, x: np.ndarray) -> np.ndarray:
            if self._constraints is not None:
                return self._constraints(x)
            return np.array([])
        
        def jacobian(self, x: np.ndarray) -> np.ndarray:
            if self._constraints is None:
                return np.array([])
            
            c0 = self._constraints(x)
            n_constraints = len(c0)
            n_vars = len(x)
            jac = np.zeros((n_constraints, n_vars))
            
            eps = 1e-8
            for i in range(n_vars):
                x_pert = x.copy()
                x_pert[i] += eps
                c1 = self._constraints(x_pert)
                jac[:, i] = (c1 - c0) / eps
            
            return jac.flatten()
    
    ipopt_problem = IPOPTProblem()
    
    if constraints is not None:
        n_constraints = len(constraints(x0))
        cl = np.zeros(n_constraints)
        cu = np.zeros(n_constraints)
    else:
        n_constraints = 0
        cl = np.array([])
        cu = np.array([])
    
    nlp = cyipopt.Problem(
        n=n_vars,
        m=n_constraints,
        problem_obj=ipopt_problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    
    nlp.add_option('tol', config.tolerance)
    nlp.add_option('max_iter', config.max_iterations)
    nlp.add_option('print_level', 5 if config.verbose else 0)
    
    x_opt, info = nlp.solve(x0)
    
    success = info['status'] == 0
    iterations = info.get('iter_count', 0)
    
    return SolverResult(
        x=x_opt,
        success=success,
        iterations=iterations,
        objective_value=objective(x_opt),
        message=f"IPOPT status: {info['status']}",
    )


def solve_bounded_optimization(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    method: str = "SLSQP",
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    prefer_ipopt: bool = True,
) -> SolverResult:
    """
    Convenience function for bounded optimization without constraints.
    
    This is useful for simple optimization problems like bifurcation point
    optimization where we only have box constraints.
    
    Parameters
    ----------
    objective : callable
        Objective function f(x) -> float
    x0 : np.ndarray
        Initial guess
    lower_bounds : np.ndarray
        Lower bounds for each variable
    upper_bounds : np.ndarray
        Upper bounds for each variable
    method : str
        Optimization method ("SLSQP", "trust-constr", "L-BFGS-B") for scipy fallback
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum number of iterations
    prefer_ipopt : bool
        If True, try IPOPT first before falling back to scipy. Default: True.
        IPOPT generally provides better convergence for nonlinear problems.
        
    Returns
    -------
    SolverResult
        Optimization result
    """
    import logging
    logger = logging.getLogger(__name__)
    
    solver_type = "scipy"
    if prefer_ipopt and is_ipopt_available():
        solver_type = "ipopt"
        logger.debug("Using IPOPT solver for bounded optimization")
    elif prefer_ipopt:
        logger.warning("NLP warning: IPOPT not available, using scipy fallback")
    
    config = SolverConfig(
        solver=solver_type,
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    
    result = solve_nlp(
        objective=objective,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        config=config,
    )
    
    if not result.success and solver_type == "scipy":
        logger.error("NLP error: Optimization did not converge")
    
    return result


def is_ipopt_available() -> bool:
    """Check if IPOPT (cyipopt) is available."""
    try:
        import cyipopt
        return True
    except ImportError:
        return False
