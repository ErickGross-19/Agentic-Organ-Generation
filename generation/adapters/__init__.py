"""
Adapters for integrating vascular_lib with existing vascular_network package.
"""

from .networkx_adapter import to_networkx_graph, from_networkx_graph
from .mesh_adapter import to_trimesh, to_hollow_tube_mesh, export_hollow_tube_stl
from .report_adapter import make_full_report
from .liver_adapter import (
    liver_tree_to_network,
    network_to_liver_tree,
    merge_liver_trees_to_network,
)

__all__ = [
    "to_networkx_graph",
    "from_networkx_graph",
    "to_trimesh",
    "to_hollow_tube_mesh",
    "export_hollow_tube_stl",
    "make_full_report",
    "liver_tree_to_network",
    "network_to_liver_tree",
    "merge_liver_trees_to_network",
]
