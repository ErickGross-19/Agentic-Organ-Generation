"""End-to-end experiment runner for vascular network design.

This module provides a high-level run_experiment() function that combines
network generation and evaluation into a single workflow with automatic
file saving and logging.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json
import time

from ..specs.design_spec import DesignSpec
from ..specs.eval_result import EvalResult
from .design import _design_from_spec_impl
from .evaluate import evaluate_network, EvalConfig


def run_experiment(
    spec: DesignSpec,
    output_dir: Optional[str] = None,
    eval_config: Optional[EvalConfig] = None,
    save_network: bool = True,
    save_spec: bool = True,
    save_eval: bool = True,
) -> Dict[str, Any]:
    """
    Run complete vascular network design experiment.
    
    This function:
    1. Creates network from DesignSpec
    2. Evaluates network quality
    3. Saves all artifacts to output directory
    4. Returns comprehensive results
    
    Parameters
    ----------
    spec : DesignSpec
        Design specification
    output_dir : str, optional
        Output directory for artifacts. If None, uses "./output"
    eval_config : EvalConfig, optional
        Evaluation configuration
    save_network : bool
        Whether to save network to JSON
    save_spec : bool
        Whether to save spec to JSON
    save_eval : bool
        Whether to save evaluation results to JSON
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - network: VascularNetwork
        - eval_result: EvalResult
        - paths: dict of saved file paths
        - timing: dict of timing information
        - metadata: dict of experiment metadata
    """
    if output_dir is None:
        output_dir = "./output"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    timing = {}
    
    design_start = time.time()
    network = _design_from_spec_impl(spec)
    timing['design'] = time.time() - design_start
    
    eval_start = time.time()
    
    if spec.tree is not None:
        tissue_points = spec.tree.colonization.tissue_points
    elif spec.dual_tree is not None:
        tissue_points = spec.dual_tree.arterial.colonization.tissue_points
    else:
        raise ValueError("Spec must have either tree or dual_tree")
    
    eval_result = evaluate_network(network, tissue_points, eval_config)
    timing['evaluation'] = time.time() - eval_start
    
    save_start = time.time()
    paths = {}
    
    if save_spec:
        spec_path = output_path / "design_spec.json"
        spec.to_json(str(spec_path))
        paths['spec'] = str(spec_path)
    
    if save_network:
        network_path = output_path / "network.json"
        network_dict = {
            'nodes': {nid: node.to_dict() for nid, node in network.nodes.items()},
            'segments': {sid: seg.to_dict() for sid, seg in network.segments.items()},
            'metadata': {
                'num_nodes': len(network.nodes),
                'num_segments': len(network.segments),
                'domain_type': spec.domain.type,
            }
        }
        with open(network_path, 'w') as f:
            json.dump(network_dict, f, indent=2)
        paths['network'] = str(network_path)
    
    if save_eval:
        eval_path = output_path / "evaluation.json"
        eval_result.to_json(str(eval_path))
        paths['evaluation'] = str(eval_path)
    
    summary_path = output_path / "summary.json"
    summary = {
        'experiment': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': time.time() - start_time,
            'timing': timing,
        },
        'network': {
            'num_nodes': len(network.nodes),
            'num_segments': len(network.segments),
            'num_inlets': sum(1 for n in network.nodes.values() if n.node_type == 'inlet'),
            'num_outlets': sum(1 for n in network.nodes.values() if n.node_type == 'outlet'),
            'num_terminals': sum(1 for n in network.nodes.values() if n.node_type == 'terminal'),
        },
        'evaluation': {
            'overall_score': eval_result.scores.overall_score,
            'coverage_score': eval_result.scores.coverage_score,
            'flow_score': eval_result.scores.flow_score,
            'structure_score': eval_result.scores.structure_score,
            'coverage_fraction': eval_result.coverage.coverage_fraction,
            'flow_balance_error': eval_result.flow.flow_balance_error,
            'murray_deviation': eval_result.structure.murray_deviation,
        },
        'paths': paths,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    paths['summary'] = str(summary_path)
    
    timing['saving'] = time.time() - save_start
    timing['total'] = time.time() - start_time
    
    return {
        'network': network,
        'eval_result': eval_result,
        'paths': paths,
        'timing': timing,
        'metadata': {
            'output_dir': str(output_path),
            'num_nodes': len(network.nodes),
            'num_segments': len(network.segments),
            'overall_score': eval_result.scores.overall_score,
        }
    }
