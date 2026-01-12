"""
Text-Based Visualizations Module.

This module provides ASCII art and text-based visualizations for
vascular network structures, including:
- ASCII art network previews
- Text-based domain cross-sections
- Tabular structure summaries
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Point3D:
    """Simple 3D point for visualization."""
    x: float
    y: float
    z: float


class ASCIINetworkVisualizer:
    """
    Generate ASCII art representations of vascular networks.
    
    This visualizer creates text-based previews of network topology
    that can be displayed in the chat interface.
    """
    
    def __init__(self, width: int = 60, height: int = 30):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        width : int
            Width of the ASCII canvas in characters
        height : int
            Height of the ASCII canvas in characters
        """
        self.width = width
        self.height = height
    
    def visualize_topology(self, topology: str, inlet_face: str = "bottom", outlet_face: str = "top") -> str:
        """
        Generate ASCII art for a topology type.
        
        Parameters
        ----------
        topology : str
            Topology type (tree, dual_trees, path, backbone, loop)
        inlet_face : str
            Face where inlet is located
        outlet_face : str
            Face where outlet is located
            
        Returns
        -------
        str
            ASCII art representation
        """
        if topology == "tree":
            return self._draw_tree_topology(inlet_face, outlet_face)
        elif topology == "dual_trees":
            return self._draw_dual_trees_topology(inlet_face, outlet_face)
        elif topology == "path":
            return self._draw_path_topology(inlet_face, outlet_face)
        elif topology == "backbone":
            return self._draw_backbone_topology(inlet_face, outlet_face)
        elif topology == "loop":
            return self._draw_loop_topology(inlet_face, outlet_face)
        else:
            return f"(Unknown topology: {topology})"
    
    def _draw_tree_topology(self, inlet_face: str, outlet_face: str) -> str:
        """Draw a tree topology."""
        return """
TREE TOPOLOGY
=============
Single inlet branching to multiple terminals

           [Inlet]
              |
              |
         _____|_____
        |     |     |
        |     |     |
      __|__   |   __|__
     |  |  |  |  |  |  |
     *  *  *  *  *  *  *
        (terminals)

Flow: Inlet -> Branches -> Terminals
"""
    
    def _draw_dual_trees_topology(self, inlet_face: str, outlet_face: str) -> str:
        """Draw a dual trees topology."""
        return """
DUAL TREES TOPOLOGY
===================
Two interleaved trees meeting in capillary bed

  [Inlet A]              [Inlet B]
      |                      |
      |                      |
   ___|___                ___|___
  |   |   |              |   |   |
  |   |   |              |   |   |
  *   *   *    <--->     *   *   *
      \\       /   \\       /
       \\     /     \\     /
        \\   /       \\   /
         \\ /         \\ /
    [Capillary Bed / Meeting Shell]

Flow: Inlet A -> Tree A -> Capillaries -> Tree B -> Inlet B
      (or bidirectional perfusion)
"""
    
    def _draw_path_topology(self, inlet_face: str, outlet_face: str) -> str:
        """Draw a path topology."""
        return """
PATH TOPOLOGY
=============
Simple inlet-to-outlet channel

    [Inlet]
       |
       |
       |
       |
       |
       |
       |
       |
    [Outlet]

Flow: Direct path from inlet to outlet
"""
    
    def _draw_backbone_topology(self, inlet_face: str, outlet_face: str) -> str:
        """Draw a backbone topology."""
        return """
BACKBONE TOPOLOGY
=================
Main trunk with side branches

    [Inlet]
       |
    ___|___
   |   |   |
   *   |   *
       |
    ___|___
   |   |   |
   *   |   *
       |
    ___|___
   |   |   |
   *   |   *
       |
    [Outlet]

Flow: Main trunk with perpendicular branches
"""
    
    def _draw_loop_topology(self, inlet_face: str, outlet_face: str) -> str:
        """Draw a loop topology."""
        return """
LOOP TOPOLOGY
=============
Circular/recirculating network

       [Inlet]
          |
    ______|______
   |             |
   |             |
   |             |
   |_____________|
          |
       [Outlet]

Flow: Recirculating loop structure
"""
    
    def visualize_domain(
        self,
        domain_type: str,
        size: Tuple[float, float, float],
        inlet_face: str = "bottom",
        outlet_face: str = "top",
    ) -> str:
        """
        Generate ASCII art for a domain with inlet/outlet markers.
        
        Parameters
        ----------
        domain_type : str
            Domain type (box, ellipsoid, cylinder)
        size : tuple
            Domain dimensions (width, depth, height)
        inlet_face : str
            Face where inlet is located
        outlet_face : str
            Face where outlet is located
            
        Returns
        -------
        str
            ASCII art representation
        """
        w, d, h = size
        
        if domain_type == "box":
            return self._draw_box_domain(w, d, h, inlet_face, outlet_face)
        elif domain_type == "ellipsoid":
            return self._draw_ellipsoid_domain(w, d, h, inlet_face, outlet_face)
        elif domain_type == "cylinder":
            return self._draw_cylinder_domain(w, d, h, inlet_face, outlet_face)
        else:
            return f"(Unknown domain type: {domain_type})"
    
    def _draw_box_domain(
        self,
        w: float, d: float, h: float,
        inlet_face: str, outlet_face: str
    ) -> str:
        """Draw a box domain."""
        inlet_marker = self._get_face_marker(inlet_face, "I")
        outlet_marker = self._get_face_marker(outlet_face, "O")
        
        return f"""
BOX DOMAIN ({w:.1f} x {d:.1f} x {h:.1f} mm)
{'=' * 40}

        +Z (top)
           ^
           |
    +------{outlet_marker if outlet_face == 'top' else '-'}------+
   /|             /|
  / |            / |
 +--{outlet_marker if outlet_face == 'back' else '-'}---{outlet_marker if outlet_face == 'right' else '-'}------+  |
 |  |           |  |
 |  +------{inlet_marker if inlet_face == 'bottom' else '-'}---|--+ --> +Y (back)
 | /            | /
 |/             |/
 +------{inlet_marker if inlet_face == 'front' else '-'}------+
 |
 v
+X (right)

Inlet:  {inlet_face} face (marked 'I')
Outlet: {outlet_face} face (marked 'O')
"""
    
    def _draw_ellipsoid_domain(
        self,
        w: float, d: float, h: float,
        inlet_face: str, outlet_face: str
    ) -> str:
        """Draw an ellipsoid domain."""
        return f"""
ELLIPSOID DOMAIN ({w:.1f} x {d:.1f} x {h:.1f} mm)
{'=' * 40}

        +Z (top)
           ^
           |
        ___O___        (O = outlet if top)
      /    |    \\
     /     |     \\
    |      |      |
    |------+------|---> +Y
    |     /       |
     \\   /       /
      \\_I______/        (I = inlet if bottom)
        |
        v
       +X

Semi-axes: a={w/2:.1f}, b={d/2:.1f}, c={h/2:.1f} mm
Inlet:  {inlet_face}
Outlet: {outlet_face}
"""
    
    def _draw_cylinder_domain(
        self,
        w: float, d: float, h: float,
        inlet_face: str, outlet_face: str
    ) -> str:
        """Draw a cylinder domain."""
        return f"""
CYLINDER DOMAIN ({w:.1f} x {d:.1f} x {h:.1f} mm)
{'=' * 40}

        +Z (top)
           ^
           |
        ___O___
       /   |   \\
      |    |    |
      |    |    |
      |    |    |
      |    |    |
      |    |    |
       \\___I___/
           |
           v
          +X

Radius: {min(w, d)/2:.1f} mm
Height: {h:.1f} mm
Inlet:  {inlet_face}
Outlet: {outlet_face}
"""
    
    def _get_face_marker(self, face: str, marker: str) -> str:
        """Get marker character for a face."""
        return marker


class CrossSectionVisualizer:
    """Generate text-based cross-section views."""
    
    def __init__(self, width: int = 40, height: int = 20):
        """Initialize the cross-section visualizer."""
        self.width = width
        self.height = height
    
    def visualize_xy_cross_section(
        self,
        domain_type: str,
        size: Tuple[float, float, float],
        z_level: float = 0.5,
    ) -> str:
        """
        Generate XY cross-section at given Z level.
        
        Parameters
        ----------
        domain_type : str
            Domain type
        size : tuple
            Domain dimensions
        z_level : float
            Z level as fraction (0-1) of domain height
            
        Returns
        -------
        str
            ASCII cross-section
        """
        w, d, h = size
        z_pos = z_level * h
        
        lines = [f"XY Cross-Section at Z = {z_pos:.1f} mm ({z_level*100:.0f}% height)"]
        lines.append("=" * 50)
        lines.append("")
        
        if domain_type == "box":
            lines.append(self._draw_rectangle_cross_section(w, d))
        elif domain_type == "ellipsoid":
            lines.append(self._draw_ellipse_cross_section(w, d, z_level))
        elif domain_type == "cylinder":
            lines.append(self._draw_circle_cross_section(min(w, d)))
        
        lines.append("")
        lines.append(f"  ^ +Y (depth: {d:.1f} mm)")
        lines.append("  |")
        lines.append("  +---> +X (width: {:.1f} mm)".format(w))
        
        return "\n".join(lines)
    
    def _draw_rectangle_cross_section(self, w: float, d: float) -> str:
        """Draw rectangular cross-section."""
        aspect = d / w if w > 0 else 1
        char_width = min(self.width, 40)
        char_height = int(char_width * aspect * 0.5)
        char_height = max(5, min(char_height, 15))
        
        lines = []
        lines.append("+" + "-" * (char_width - 2) + "+")
        for _ in range(char_height - 2):
            lines.append("|" + " " * (char_width - 2) + "|")
        lines.append("+" + "-" * (char_width - 2) + "+")
        
        return "\n".join(lines)
    
    def _draw_ellipse_cross_section(self, w: float, d: float, z_level: float) -> str:
        """Draw elliptical cross-section (varies with z_level)."""
        scale = 1.0 - abs(2 * z_level - 1) ** 2
        scale = max(0.1, scale)
        
        char_width = int(min(self.width, 40) * scale)
        char_height = int(char_width * (d / w) * 0.5) if w > 0 else 5
        char_height = max(3, min(char_height, 12))
        
        lines = []
        for i in range(char_height):
            if i == 0 or i == char_height - 1:
                padding = char_width // 4
                inner = char_width - 2 * padding
                lines.append(" " * padding + "_" * inner)
            else:
                t = abs(2 * i / (char_height - 1) - 1)
                indent = int(char_width * t * 0.2)
                inner_width = char_width - 2 * indent - 2
                if i < char_height // 2:
                    lines.append(" " * indent + "/" + " " * inner_width + "\\")
                else:
                    lines.append(" " * indent + "\\" + " " * inner_width + "/")
        
        return "\n".join(lines)
    
    def _draw_circle_cross_section(self, diameter: float) -> str:
        """Draw circular cross-section."""
        char_size = min(self.width, 30)
        
        lines = []
        lines.append("    " + "_" * (char_size - 8))
        for i in range(char_size // 3):
            t = i / (char_size // 3)
            indent = int((1 - t) * 3)
            inner = char_size - 2 * indent - 2
            lines.append(" " * indent + "/" + " " * inner + "\\")
        for i in range(char_size // 3):
            t = i / (char_size // 3)
            indent = int(t * 3)
            inner = char_size - 2 * indent - 2
            lines.append(" " * indent + "\\" + " " * inner + "/")
        lines.append("    " + "-" * (char_size - 8))
        
        return "\n".join(lines)


class NetworkSummaryTable:
    """Generate tabular summaries of network structures."""
    
    @staticmethod
    def format_spec_table(spec: Dict[str, Any]) -> str:
        """
        Format specification as a table.
        
        Parameters
        ----------
        spec : dict
            Specification dictionary
            
        Returns
        -------
        str
            Formatted table
        """
        rows = []
        
        domain = spec.get("domain", {})
        rows.append(["Domain Type", str(domain.get("type", "not set"))])
        rows.append(["Domain Size", str(domain.get("size", "not set"))])
        
        topology = spec.get("topology", {})
        rows.append(["Topology", str(topology.get("kind", "not set"))])
        rows.append(["Target Terminals", str(topology.get("target_terminals", "not set"))])
        
        inlet = spec.get("inlet", {})
        rows.append(["Inlet Face", str(inlet.get("face", "not set"))])
        rows.append(["Inlet Radius", str(inlet.get("radius", "not set"))])
        
        outlet = spec.get("outlet", {})
        rows.append(["Outlet Face", str(outlet.get("face", "not set"))])
        rows.append(["Outlet Radius", str(outlet.get("radius", "not set"))])
        
        return NetworkSummaryTable._format_table(["Parameter", "Value"], rows)
    
    @staticmethod
    def format_network_stats(stats: Dict[str, Any]) -> str:
        """
        Format network statistics as a table.
        
        Parameters
        ----------
        stats : dict
            Network statistics
            
        Returns
        -------
        str
            Formatted table
        """
        rows = []
        for key, value in stats.items():
            if isinstance(value, float):
                rows.append([key, f"{value:.4f}"])
            else:
                rows.append([key, str(value)])
        
        return NetworkSummaryTable._format_table(["Statistic", "Value"], rows)
    
    @staticmethod
    def _format_table(headers: List[str], rows: List[List[str]]) -> str:
        """Format data as a text table."""
        if not rows:
            return "(empty table)"
        
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        def format_row(cells: List[str]) -> str:
            formatted = []
            for i, cell in enumerate(cells):
                width = col_widths[i] if i < len(col_widths) else len(str(cell))
                formatted.append(str(cell).ljust(width))
            return "| " + " | ".join(formatted) + " |"
        
        separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        
        lines = [separator]
        lines.append(format_row(headers))
        lines.append(separator)
        for row in rows:
            lines.append(format_row(row))
        lines.append(separator)
        
        return "\n".join(lines)


def visualize_network_preview(
    topology: str,
    domain_type: str = "box",
    domain_size: Tuple[float, float, float] = (20, 60, 30),
    inlet_face: str = "bottom",
    outlet_face: str = "top",
) -> str:
    """
    Generate a complete text-based network preview.
    
    Parameters
    ----------
    topology : str
        Network topology type
    domain_type : str
        Domain shape type
    domain_size : tuple
        Domain dimensions (w, d, h)
    inlet_face : str
        Inlet face location
    outlet_face : str
        Outlet face location
        
    Returns
    -------
    str
        Complete ASCII visualization
    """
    viz = ASCIINetworkVisualizer()
    cross = CrossSectionVisualizer()
    
    lines = []
    lines.append("=" * 60)
    lines.append("VASCULAR NETWORK PREVIEW".center(60))
    lines.append("=" * 60)
    lines.append("")
    
    lines.append(viz.visualize_topology(topology, inlet_face, outlet_face))
    lines.append("")
    
    lines.append(viz.visualize_domain(domain_type, domain_size, inlet_face, outlet_face))
    lines.append("")
    
    lines.append(cross.visualize_xy_cross_section(domain_type, domain_size, 0.5))
    
    return "\n".join(lines)


def get_topology_comparison_table() -> str:
    """
    Get a comparison table of all topology types.
    
    Returns
    -------
    str
        Formatted comparison table
    """
    headers = ["Topology", "Inlets", "Outlets", "Use Case"]
    rows = [
        ["tree", "1", "Many", "Single-supply organs, simple perfusion"],
        ["dual_trees", "2", "2", "Liver, kidney (arterial + venous)"],
        ["path", "1", "1", "Simple channels, microfluidics"],
        ["backbone", "1", "1", "Spine-like structures, manifolds"],
        ["loop", "1", "1", "Recirculating systems"],
    ]
    
    return NetworkSummaryTable._format_table(headers, rows)
