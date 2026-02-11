"""End-to-end experiment runner for vascular network design.

This module provides a high-level run_experiment() function that combines
network generation and evaluation into a single workflow with automatic
file saving and logging.

It also provides run_odc_experiment() for running ODC-specific experiments
with anti-starburst branching, flexible tissue distributions, and multi-tree
coordination.
"""

from typing import Optional, Dict, Any, List
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


def run_odc_experiment(
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    save_results: bool = True,
) -> Dict[str, Any]:
    """Run an ODC-specific experiment with v2 features.

    Exercises anti-starburst branching, flexible tissue distributions,
    expanded search space, and (optionally) multi-tree coordination.

    Parameters
    ----------
    params : dict, optional
        Override any parameter accepted by ``test.odc_runner.run_odc``.
        Additional keys recognised here:

        * ``anti_starburst`` (bool, default True) – enable anti-starburst
        * ``tissue_distribution`` (str) – distribution type for tissue
          points (e.g. "uniform", "poisson_disk", "liver_lobule")
        * ``multi_tree`` (bool, default False) – run multi-tree mode
        * ``preset`` (str) – named ODC policy preset to apply first
    output_dir : str, optional
        Where to write JSON artefacts.  Defaults to ``"./odc_output"``.
    save_results : bool
        Persist summary JSON to *output_dir*.

    Returns
    -------
    dict
        ``network``, ``stats``, ``odc_features`` summary, ``timing``,
        and ``paths`` (when *save_results* is True).
    """
    if params is None:
        params = {}
    if output_dir is None:
        output_dir = "./odc_output"

    from aog_policies.odc import get_odc_preset

    preset_name = params.pop("preset", None)
    if preset_name is not None:
        preset = get_odc_preset(preset_name)
        preset_dict = preset.to_dict()
        anti_starburst_dict = preset_dict.get("anti_starburst", {})
        for k, v in anti_starburst_dict.items():
            params.setdefault(k, v)

    enable_anti_starburst = params.pop("anti_starburst", True)
    tissue_distribution = params.pop("tissue_distribution", None)
    enable_multi_tree = params.pop("multi_tree", False)

    if enable_anti_starburst:
        params.setdefault("min_generations_before_tissue", 2)
        params.setdefault("max_initial_branches", 3)
        params.setdefault("force_bifurcation_depth", 3)

    start_time = time.time()
    timing: Dict[str, float] = {}

    tissue_pts = None
    if tissue_distribution is not None:
        from ..tissue.distributions import TissueDistributionSpec
        from ..core.domain import BoxDomain

        domain_type = params.get("domain_type", "cylinder")
        if domain_type == "box":
            domain_for_dist = BoxDomain(
                x_min=params.get("x_min", -0.005),
                x_max=params.get("x_max", 0.005),
                y_min=params.get("y_min", -0.005),
                y_max=params.get("y_max", 0.005),
                z_min=params.get("z_min", -0.005),
                z_max=params.get("z_max", 0.005),
            )
        else:
            from ..core.domain import CylinderDomain
            from ..core.types import Point3D
            domain_for_dist = CylinderDomain(
                radius=params.get("domain_radius", 0.005),
                height=params.get("domain_height", 0.010),
                center=Point3D(*params.get("domain_center", [0.0, 0.0, 0.0])),
            )

        dist_start = time.time()
        dist_kwargs: Dict[str, Any] = {
            "distribution_type": tissue_distribution,
            "n_points": params.get("tissue_n_points", 200),
            "seed": params.get("seed"),
        }
        if params.get("gaussian_centers"):
            dist_kwargs["gaussian_centers"] = [tuple(c) for c in params["gaussian_centers"]]
        if params.get("gaussian_sigmas"):
            dist_kwargs["gaussian_sigmas"] = [tuple(s) for s in params["gaussian_sigmas"]]
        if params.get("gaussian_weights"):
            dist_kwargs["gaussian_weights"] = params["gaussian_weights"]
        if params.get("depth_axis") is not None:
            dist_kwargs["depth_axis"] = params["depth_axis"]
        if params.get("depth_power") is not None:
            dist_kwargs["depth_power"] = params["depth_power"]
        if params.get("depth_distribution"):
            dist_kwargs["depth_distribution"] = params["depth_distribution"]
        if params.get("depth_beta_params"):
            dist_kwargs["depth_beta_params"] = tuple(params["depth_beta_params"])
        if params.get("min_distance") is not None:
            dist_kwargs["min_distance"] = params["min_distance"]
        spec = TissueDistributionSpec(**dist_kwargs)
        tissue_pts = spec.generate(domain_for_dist)
        timing["tissue_distribution"] = time.time() - dist_start

        params["tissue_levels"] = [
            {
                "priority": 1,
                "points": tissue_pts.tolist(),
                "label": tissue_distribution,
                "weight": 1.0,
                "coverage_threshold": 0.003,
            }
        ]

    gen_start = time.time()

    import sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from test.odc_runner import run_odc

    multi_tree_result = None
    if enable_multi_tree:
        tree_configs_raw = params.pop("tree_configs", None)
        if tree_configs_raw is not None:
            from ..ops.multi_tree_odc import run_multi_tree_odc, TreeConfig
            from ..tissue.hierarchical import TissueLevel, HierarchicalTissueSpec
            from ..core.domain import CylinderDomain, BoxDomain
            from ..core.types import Point3D

            domain_type = params.get("domain_type", "cylinder")
            if domain_type == "box":
                mt_domain = BoxDomain(
                    x_min=params.get("x_min", -0.005),
                    x_max=params.get("x_max", 0.005),
                    y_min=params.get("y_min", -0.005),
                    y_max=params.get("y_max", 0.005),
                    z_min=params.get("z_min", -0.005),
                    z_max=params.get("z_max", 0.005),
                )
            else:
                mt_domain = CylinderDomain(
                    radius=params.get("domain_radius", 0.005),
                    height=params.get("domain_height", 0.010),
                    center=Point3D(*params.get("domain_center", [0.0, 0.0, 0.0])),
                )

            tree_cfgs = [TreeConfig.from_dict(tc) for tc in tree_configs_raw]

            if tissue_pts is not None:
                tissue_levels = [
                    TissueLevel(
                        priority=1,
                        points=tissue_pts,
                        label=tissue_distribution or "custom",
                        weight=1.0,
                        coverage_threshold=0.003,
                    )
                ]
            else:
                from ..tissue.samplers import generate_hierarchical_from_strategy
                tissue_spec_mt = generate_hierarchical_from_strategy(
                    mt_domain,
                    n_levels=params.get("auto_n_levels", 3),
                    points_per_level=params.get("auto_points_per_level", 200),
                    seed=params.get("seed"),
                )
                tissue_levels = tissue_spec_mt.levels

            mt_tissue_spec = HierarchicalTissueSpec(levels=tissue_levels)

            multi_tree_result = run_multi_tree_odc(
                domain=mt_domain,
                tissue_spec=mt_tissue_spec,
                tree_configs=tree_cfgs,
                collision_radius=params.get("collision_radius", 0.001),
                interleave_strategy=params.get("interleave_strategy", "sequential"),
                seed=params.get("seed"),
            )

    if multi_tree_result is not None:
        first_tid = list(multi_tree_result.networks.keys())[0]
        network = multi_tree_result.networks[first_tid]
        stats = {
            "trees": {
                tid: {
                    "nodes": len(net.nodes),
                    "segments": len(net.segments),
                    "iterations_used": multi_tree_result.tree_results[tid].iterations_used,
                }
                for tid, net in multi_tree_result.networks.items()
            },
            "collision_count": multi_tree_result.collision_count,
        }
        timing["generation"] = time.time() - gen_start
    else:
        network, stats = run_odc(params)
        timing["generation"] = time.time() - gen_start

    odc_features: Dict[str, Any] = {
        "anti_starburst_enabled": enable_anti_starburst,
        "tissue_distribution": tissue_distribution,
        "multi_tree_enabled": enable_multi_tree,
        "preset": preset_name,
    }

    if enable_anti_starburst:
        odc_features["min_generations_before_tissue"] = params.get(
            "min_generations_before_tissue", 2
        )
        odc_features["max_initial_branches"] = params.get(
            "max_initial_branches", 3
        )
        odc_features["force_bifurcation_depth"] = params.get(
            "force_bifurcation_depth", 3
        )

    timing["total"] = time.time() - start_time

    result: Dict[str, Any] = {
        "network": network,
        "stats": stats,
        "odc_features": odc_features,
        "timing": timing,
    }

    if multi_tree_result is not None:
        result["multi_tree_result"] = multi_tree_result
        result["all_networks"] = multi_tree_result.networks

    if save_results:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment_type": "odc_v2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {k: v for k, v in stats.items() if not isinstance(v, (dict, list))},
            "odc_features": odc_features,
            "timing": timing,
        }
        summary_path = out_path / "odc_experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        result["paths"] = {"summary": str(summary_path)}

    return result
