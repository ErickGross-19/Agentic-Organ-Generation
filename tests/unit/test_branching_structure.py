"""
Tests that validate actual tree branching structure produced by Space Colonization.

These tests go beyond junction counts — they verify that bifurcated branches
actually survive and grow (both children have meaningful subtree depth),
producing visible tree-like structures rather than linear strands.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from test.space_colonization_runner import run_space_colonization


def _build_children_map(network):
    children = {}
    for seg in network.segments.values():
        children.setdefault(seg.start_node_id, []).append(seg.end_node_id)
    return children


def _subtree_size(children_map, root):
    count = 0
    stack = [root]
    while stack:
        n = stack.pop()
        count += 1
        stack.extend(children_map.get(n, []))
    return count


def _branching_metrics(network):
    children_map = _build_children_map(network)
    branch_nodes = [nid for nid, kids in children_map.items() if len(kids) >= 2]

    smaller_child_sizes = []
    for nid in branch_nodes:
        sizes = sorted(
            [_subtree_size(children_map, k) for k in children_map[nid]],
            reverse=True,
        )
        smaller_child_sizes.append(sizes[1])

    positions = np.array(
        [[n.position.x, n.position.y, n.position.z] for n in network.nodes.values()]
    )
    spread = positions.max(axis=0) - positions.min(axis=0)

    return {
        "total_nodes": len(network.nodes),
        "total_segments": len(network.segments),
        "branch_nodes": len(branch_nodes),
        "smaller_child_sizes": smaller_child_sizes,
        "surviving_branches": sum(1 for s in smaller_child_sizes if s > 10),
        "dead_on_arrival": sum(1 for s in smaller_child_sizes if s == 1),
        "spatial_spread": spread,
    }


SC_PARAMS_100 = {
    "domain_type": "cylinder",
    "domain_radius": 0.1,
    "domain_height": 0.3,
    "domain_center": [0.0, 0.0, 0.0],
    "inlet_position": [0.0, 0.0, 0.15],
    "inlet_radius": 0.001,
    "vessel_type": "arterial",
    "num_attractors": 15000,
    "attraction_distance": 0.015,
    "kill_distance": 0.003,
    "step_size": 0.0001,
    "max_iterations": 100,
    "max_steps": 100,
    "branch_angle_deg": 35.0,
    "directional_bias": 0.85,
    "max_deviation_deg": 25.0,
    "encourage_bifurcation": True,
    "max_children_per_node": 2,
    "bifurcation_probability": 0.8,
    "min_attractions_for_bifurcation": 3,
    "bifurcation_angle_threshold_deg": 55.0,
    "min_radius": 0.0001,
    "taper_factor": 0.95,
    "progress": False,
    "check_collisions": True,
    "collision_clearance": 0.0002,
    "collision_merge_distance": 0.0003,
    "collision_mode": "break",
    "seed": 42,
    "num_outlets": 50,
    "apply_murray": False,
    "tissue_sampling": {
        "enabled": True,
        "n_points": 15000,
        "strategy": "uniform",
        "depth_reference": {"mode": "face", "face": "top"},
        "depth_distribution": "power",
        "depth_power": 2.0,
        "seed": 42,
    },
}


class TestBranchingStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        logging.disable(logging.CRITICAL)
        net, _stats = run_space_colonization(SC_PARAMS_100)
        logging.disable(logging.NOTSET)
        cls.network = net
        cls.metrics = _branching_metrics(net)

    def test_has_real_branch_nodes(self):
        self.assertGreaterEqual(
            self.metrics["branch_nodes"],
            3,
            "Should have at least 3 nodes with >=2 children",
        )

    def test_branches_survive(self):
        self.assertGreaterEqual(
            self.metrics["surviving_branches"],
            2,
            "At least 2 branch nodes should have both children surviving >10 nodes",
        )

    def test_no_mass_die_off(self):
        if self.metrics["branch_nodes"] == 0:
            self.skipTest("No branch nodes to evaluate")
        dead_ratio = self.metrics["dead_on_arrival"] / self.metrics["branch_nodes"]
        self.assertLess(
            dead_ratio,
            0.5,
            f"Less than 50% of branches should die immediately "
            f"({self.metrics['dead_on_arrival']}/{self.metrics['branch_nodes']})",
        )

    def test_spatial_spread(self):
        spread = self.metrics["spatial_spread"]
        self.assertTrue(
            np.any(spread > 0.005),
            f"Tree should spread at least 5mm in some dimension, got {spread*1000}mm",
        )

    def test_sufficient_growth(self):
        self.assertGreaterEqual(
            self.metrics["total_nodes"],
            50,
            "Should grow at least 50 nodes in 100 steps",
        )


if __name__ == "__main__":
    unittest.main()
