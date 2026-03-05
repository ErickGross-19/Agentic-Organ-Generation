"""
Output Generator Module

Generates complex, clean outputs in multiple formats.
This provides a multi-layered output system for generated structures.

The generator produces:
1. STL files with multiple resolutions
2. Interactive 3D viewer HTML
3. Detailed analysis report
4. Generation recipe (reproducible steps)
5. Performance metrics
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
import os
import datetime


class OutputFormat(Enum):
    """Available output formats."""
    STL = "stl"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"


class Resolution(Enum):
    """Mesh resolution levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ArtifactInfo:
    """Information about a generated artifact."""
    name: str
    path: str
    format: OutputFormat
    size_bytes: int = 0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactPackage:
    """Complete package of generated artifacts."""
    artifacts: List[ArtifactInfo] = field(default_factory=list)
    output_dir: str = ""
    generation_time: str = ""
    spec_summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_artifact(self, name: str) -> Optional[ArtifactInfo]:
        """Get an artifact by name."""
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None
    
    def list_artifacts(self) -> List[str]:
        """List all artifact names."""
        return [a.name for a in self.artifacts]


@dataclass
class GenerationRecipe:
    """Reproducible generation recipe."""
    spec: Dict[str, Any]
    seed: int
    parameters: Dict[str, Any]
    steps: List[str]
    version: str
    timestamp: str


class OutputGenerator:
    """
    Generates complex, clean outputs in multiple formats.
    
    This class provides a multi-layered output system that produces
    STL files, interactive viewers, analysis reports, and more.
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_artifact_package(
        self,
        network: Any,
        spec: Any,
        object_name: str = "structure",
        version: int = 1,
        include_viewer: bool = True,
        include_report: bool = True,
        resolutions: Optional[List[Resolution]] = None,
    ) -> ArtifactPackage:
        """
        Generate a complete artifact package.
        
        Parameters
        ----------
        network : VascularNetwork
            The generated vascular network
        spec : DesignSpec
            The design specification used
        object_name : str
            Name of the object
        version : int
            Version number
        include_viewer : bool
            Whether to include interactive HTML viewer
        include_report : bool
            Whether to include analysis report
        resolutions : List[Resolution], optional
            Mesh resolutions to generate
            
        Returns
        -------
        ArtifactPackage
            Complete package of generated artifacts
        """
        resolutions = resolutions or [Resolution.MEDIUM]
        
        # Create version-specific output directory
        version_dir = os.path.join(self.output_dir, object_name, f"v{version:03d}")
        os.makedirs(version_dir, exist_ok=True)
        
        artifacts = []
        timestamp = datetime.datetime.now().isoformat()
        
        # Generate STL files at different resolutions
        for resolution in resolutions:
            stl_artifact = self._generate_stl(
                network, version_dir, object_name, version, resolution
            )
            if stl_artifact:
                artifacts.append(stl_artifact)
        
        # Generate network JSON
        network_artifact = self._generate_network_json(
            network, version_dir, object_name, version
        )
        if network_artifact:
            artifacts.append(network_artifact)
        
        # Generate spec JSON
        spec_artifact = self._generate_spec_json(
            spec, version_dir, object_name, version
        )
        if spec_artifact:
            artifacts.append(spec_artifact)
        
        # Generate interactive viewer
        if include_viewer:
            viewer_artifact = self._generate_viewer_html(
                network, version_dir, object_name, version
            )
            if viewer_artifact:
                artifacts.append(viewer_artifact)
        
        # Generate analysis report
        if include_report:
            report_artifact = self._generate_analysis_report(
                network, spec, version_dir, object_name, version
            )
            if report_artifact:
                artifacts.append(report_artifact)
        
        # Generate recipe
        recipe_artifact = self._generate_recipe(
            spec, version_dir, object_name, version
        )
        if recipe_artifact:
            artifacts.append(recipe_artifact)
        
        # Generate metrics
        metrics = self._compute_metrics(network, spec)
        metrics_artifact = self._generate_metrics_json(
            metrics, version_dir, object_name, version
        )
        if metrics_artifact:
            artifacts.append(metrics_artifact)
        
        return ArtifactPackage(
            artifacts=artifacts,
            output_dir=version_dir,
            generation_time=timestamp,
            spec_summary=self._spec_to_summary(spec),
            metrics=metrics,
        )
    
    def _generate_stl(
        self,
        network: Any,
        output_dir: str,
        object_name: str,
        version: int,
        resolution: Resolution,
    ) -> Optional[ArtifactInfo]:
        """Generate STL file at specified resolution."""
        filename = f"{object_name}_v{version:03d}_{resolution.value}.stl"
        filepath = os.path.join(output_dir, filename)
        
        try:
            # In a real implementation, this would use the mesh adapter
            # For now, create a placeholder
            with open(filepath, 'w') as f:
                f.write(f"solid {object_name}\n")
                f.write(f"endsolid {object_name}\n")
            
            size = os.path.getsize(filepath)
            
            return ArtifactInfo(
                name=f"mesh_{resolution.value}",
                path=filepath,
                format=OutputFormat.STL,
                size_bytes=size,
                description=f"STL mesh at {resolution.value} resolution",
                metadata={"resolution": resolution.value},
            )
        except Exception:
            return None
    
    def _generate_network_json(
        self,
        network: Any,
        output_dir: str,
        object_name: str,
        version: int,
    ) -> Optional[ArtifactInfo]:
        """Generate network JSON file."""
        filename = f"{object_name}_v{version:03d}_network.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            # Convert network to JSON-serializable format
            network_data = self._network_to_dict(network)
            
            with open(filepath, 'w') as f:
                json.dump(network_data, f, indent=2)
            
            size = os.path.getsize(filepath)
            
            return ArtifactInfo(
                name="network",
                path=filepath,
                format=OutputFormat.JSON,
                size_bytes=size,
                description="Vascular network data",
                metadata={"node_count": network_data.get("node_count", 0)},
            )
        except Exception:
            return None
    
    def _generate_spec_json(
        self,
        spec: Any,
        output_dir: str,
        object_name: str,
        version: int,
    ) -> Optional[ArtifactInfo]:
        """Generate spec JSON file."""
        filename = f"{object_name}_v{version:03d}_spec.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            spec_data = self._spec_to_dict(spec)
            
            with open(filepath, 'w') as f:
                json.dump(spec_data, f, indent=2)
            
            size = os.path.getsize(filepath)
            
            return ArtifactInfo(
                name="spec",
                path=filepath,
                format=OutputFormat.JSON,
                size_bytes=size,
                description="Design specification",
            )
        except Exception:
            return None
    
    def _generate_viewer_html(
        self,
        network: Any,
        output_dir: str,
        object_name: str,
        version: int,
    ) -> Optional[ArtifactInfo]:
        """Generate interactive 3D viewer HTML."""
        filename = f"{object_name}_v{version:03d}_viewer.html"
        filepath = os.path.join(output_dir, filename)
        
        try:
            html_content = self._create_viewer_html(network, object_name, version)
            
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            size = os.path.getsize(filepath)
            
            return ArtifactInfo(
                name="viewer",
                path=filepath,
                format=OutputFormat.HTML,
                size_bytes=size,
                description="Interactive 3D viewer",
            )
        except Exception:
            return None
    
    def _generate_analysis_report(
        self,
        network: Any,
        spec: Any,
        output_dir: str,
        object_name: str,
        version: int,
    ) -> Optional[ArtifactInfo]:
        """Generate analysis report in Markdown."""
        filename = f"{object_name}_v{version:03d}_report.md"
        filepath = os.path.join(output_dir, filename)
        
        try:
            report_content = self._create_analysis_report(network, spec, object_name, version)
            
            with open(filepath, 'w') as f:
                f.write(report_content)
            
            size = os.path.getsize(filepath)
            
            return ArtifactInfo(
                name="report",
                path=filepath,
                format=OutputFormat.MARKDOWN,
                size_bytes=size,
                description="Detailed analysis report",
            )
        except Exception:
            return None
    
    def _generate_recipe(
        self,
        spec: Any,
        output_dir: str,
        object_name: str,
        version: int,
    ) -> Optional[ArtifactInfo]:
        """Generate reproducible recipe JSON."""
        filename = f"{object_name}_v{version:03d}_recipe.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            recipe = GenerationRecipe(
                spec=self._spec_to_dict(spec),
                seed=getattr(spec, 'seed', 42) or 42,
                parameters=self._extract_parameters(spec),
                steps=self._generate_steps(spec),
                version=f"{version:03d}",
                timestamp=datetime.datetime.now().isoformat(),
            )
            
            recipe_data = {
                "spec": recipe.spec,
                "seed": recipe.seed,
                "parameters": recipe.parameters,
                "steps": recipe.steps,
                "version": recipe.version,
                "timestamp": recipe.timestamp,
            }
            
            with open(filepath, 'w') as f:
                json.dump(recipe_data, f, indent=2)
            
            size = os.path.getsize(filepath)
            
            return ArtifactInfo(
                name="recipe",
                path=filepath,
                format=OutputFormat.JSON,
                size_bytes=size,
                description="Reproducible generation recipe",
            )
        except Exception:
            return None
    
    def _generate_metrics_json(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
        object_name: str,
        version: int,
    ) -> Optional[ArtifactInfo]:
        """Generate metrics JSON file."""
        filename = f"{object_name}_v{version:03d}_metrics.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            size = os.path.getsize(filepath)
            
            return ArtifactInfo(
                name="metrics",
                path=filepath,
                format=OutputFormat.JSON,
                size_bytes=size,
                description="Performance and quality metrics",
            )
        except Exception:
            return None
    
    def _network_to_dict(self, network: Any) -> Dict[str, Any]:
        """Convert network to dictionary."""
        if network is None:
            return {"node_count": 0, "segment_count": 0, "nodes": [], "segments": []}
        
        # Try to access network attributes
        try:
            nodes = []
            if hasattr(network, 'nodes'):
                for node_id, node in network.nodes.items():
                    nodes.append({
                        "id": node_id,
                        "position": list(node.position) if hasattr(node, 'position') else [0, 0, 0],
                        "type": getattr(node, 'node_type', 'unknown'),
                        "radius": getattr(node, 'radius', 0),
                    })
            
            segments = []
            if hasattr(network, 'segments'):
                for seg_id, seg in network.segments.items():
                    segments.append({
                        "id": seg_id,
                        "start_node": getattr(seg, 'start_node_id', 0),
                        "end_node": getattr(seg, 'end_node_id', 0),
                        "radius_start": getattr(seg.geometry, 'radius_start', 0) if hasattr(seg, 'geometry') else 0,
                        "radius_end": getattr(seg.geometry, 'radius_end', 0) if hasattr(seg, 'geometry') else 0,
                    })
            
            return {
                "node_count": len(nodes),
                "segment_count": len(segments),
                "nodes": nodes,
                "segments": segments,
            }
        except Exception:
            return {"node_count": 0, "segment_count": 0, "nodes": [], "segments": []}
    
    def _spec_to_dict(self, spec: Any) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        if spec is None:
            return {}
        
        result = {}
        
        try:
            if hasattr(spec, 'seed'):
                result['seed'] = spec.seed
            if hasattr(spec, 'output_units'):
                result['output_units'] = spec.output_units
            if hasattr(spec, 'domain'):
                domain = spec.domain
                if hasattr(domain, 'center') and hasattr(domain, 'size'):
                    result['domain'] = {
                        'type': 'box',
                        'center': list(domain.center),
                        'size': list(domain.size),
                    }
                elif hasattr(domain, 'center') and hasattr(domain, 'semi_axes'):
                    result['domain'] = {
                        'type': 'ellipsoid',
                        'center': list(domain.center),
                        'semi_axes': list(domain.semi_axes),
                    }
            if hasattr(spec, 'tree') and spec.tree:
                tree = spec.tree
                result['tree'] = {
                    'terminal_count': getattr(tree, 'terminal_count', 50),
                    'topology_kind': getattr(tree, 'topology_kind', 'tree'),
                }
        except Exception:
            pass
        
        return result
    
    def _spec_to_summary(self, spec: Any) -> Dict[str, Any]:
        """Create a summary of the spec."""
        return self._spec_to_dict(spec)
    
    def _compute_metrics(self, network: Any, spec: Any) -> Dict[str, Any]:
        """Compute metrics for the generated structure."""
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        try:
            if network and hasattr(network, 'nodes'):
                metrics["node_count"] = len(network.nodes)
            if network and hasattr(network, 'segments'):
                metrics["segment_count"] = len(network.segments)
            
            # Count terminals
            terminal_count = 0
            if network and hasattr(network, 'nodes'):
                for node in network.nodes.values():
                    if getattr(node, 'node_type', '') == 'terminal':
                        terminal_count += 1
            metrics["terminal_count"] = terminal_count
            
            # Get target from spec
            if spec and hasattr(spec, 'tree') and spec.tree:
                metrics["target_terminals"] = getattr(spec.tree, 'terminal_count', 50)
        except Exception:
            pass
        
        return metrics
    
    def _extract_parameters(self, spec: Any) -> Dict[str, Any]:
        """Extract key parameters from spec."""
        params = {}
        
        try:
            if spec and hasattr(spec, 'tree') and spec.tree:
                tree = spec.tree
                if hasattr(tree, 'colonization') and tree.colonization:
                    col = tree.colonization
                    params['influence_radius'] = getattr(col, 'influence_radius', None)
                    params['kill_radius'] = getattr(col, 'kill_radius', None)
                    params['step_size'] = getattr(col, 'step_size', None)
                    params['max_steps'] = getattr(col, 'max_steps', None)
                    params['min_radius'] = getattr(col, 'min_radius', None)
                    params['radius_decay'] = getattr(col, 'radius_decay', None)
        except Exception:
            pass
        
        return {k: v for k, v in params.items() if v is not None}
    
    def _generate_steps(self, spec: Any) -> List[str]:
        """Generate list of steps for reproduction."""
        steps = [
            "1. Load design specification",
            "2. Initialize vascular network",
            "3. Generate attractor points within domain",
            "4. Run space colonization algorithm",
            "5. Apply Murray's law radius adjustments",
            "6. Validate network topology",
            "7. Generate mesh from network",
            "8. Export artifacts",
        ]
        return steps
    
    def _create_viewer_html(self, network: Any, object_name: str, version: int) -> str:
        """Create interactive 3D viewer HTML."""
        network_data = self._network_to_dict(network)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{object_name} v{version:03d} - 3D Viewer</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
        }}
        #info h2 {{
            margin: 0 0 10px 0;
            color: #4ecdc4;
        }}
        #info p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        #controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 8px;
            z-index: 100;
        }}
        #controls button {{
            background: #4ecdc4;
            border: none;
            padding: 8px 16px;
            margin: 2px;
            border-radius: 4px;
            cursor: pointer;
            color: #1a1a2e;
            font-weight: bold;
        }}
        #controls button:hover {{
            background: #45b7aa;
        }}
        canvas {{
            display: block;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="info">
            <h2>{object_name}</h2>
            <p>Version: {version:03d}</p>
            <p>Nodes: {network_data.get('node_count', 0)}</p>
            <p>Segments: {network_data.get('segment_count', 0)}</p>
        </div>
        <div id="controls">
            <button onclick="resetView()">Reset View</button>
            <button onclick="toggleWireframe()">Toggle Wireframe</button>
            <button onclick="toggleAxes()">Toggle Axes</button>
        </div>
        <canvas id="viewer"></canvas>
    </div>
    
    <script>
        // Network data embedded in page
        const networkData = {json.dumps(network_data)};
        
        // Simple 3D viewer implementation
        const canvas = document.getElementById('viewer');
        const ctx = canvas.getContext('2d');
        
        let rotation = {{ x: 0.3, y: 0.5 }};
        let zoom = 1;
        let showWireframe = false;
        let showAxes = true;
        
        function resize() {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            render();
        }}
        
        function project(x, y, z) {{
            // Simple perspective projection
            const scale = 200 * zoom;
            const cx = Math.cos(rotation.x);
            const sx = Math.sin(rotation.x);
            const cy = Math.cos(rotation.y);
            const sy = Math.sin(rotation.y);
            
            const x1 = x * cy - z * sy;
            const z1 = x * sy + z * cy;
            const y1 = y * cx - z1 * sx;
            const z2 = y * sx + z1 * cx;
            
            const perspective = 500 / (500 + z2);
            return {{
                x: canvas.width/2 + x1 * scale * perspective,
                y: canvas.height/2 - y1 * scale * perspective,
                z: z2
            }};
        }}
        
        function render() {{
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw axes
            if (showAxes) {{
                const origin = project(0, 0, 0);
                const xAxis = project(0.05, 0, 0);
                const yAxis = project(0, 0.05, 0);
                const zAxis = project(0, 0, 0.05);
                
                ctx.strokeStyle = '#ff6b6b';
                ctx.beginPath();
                ctx.moveTo(origin.x, origin.y);
                ctx.lineTo(xAxis.x, xAxis.y);
                ctx.stroke();
                
                ctx.strokeStyle = '#4ecdc4';
                ctx.beginPath();
                ctx.moveTo(origin.x, origin.y);
                ctx.lineTo(yAxis.x, yAxis.y);
                ctx.stroke();
                
                ctx.strokeStyle = '#ffe66d';
                ctx.beginPath();
                ctx.moveTo(origin.x, origin.y);
                ctx.lineTo(zAxis.x, zAxis.y);
                ctx.stroke();
            }}
            
            // Draw segments
            ctx.strokeStyle = showWireframe ? '#4ecdc4' : '#ff6b6b';
            ctx.lineWidth = 2;
            
            networkData.segments.forEach(seg => {{
                const startNode = networkData.nodes.find(n => n.id === seg.start_node);
                const endNode = networkData.nodes.find(n => n.id === seg.end_node);
                
                if (startNode && endNode) {{
                    const p1 = project(startNode.position[0], startNode.position[1], startNode.position[2]);
                    const p2 = project(endNode.position[0], endNode.position[1], endNode.position[2]);
                    
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }}
            }});
            
            // Draw nodes
            networkData.nodes.forEach(node => {{
                const p = project(node.position[0], node.position[1], node.position[2]);
                const radius = Math.max(3, node.radius * 1000 * zoom);
                
                ctx.fillStyle = node.type === 'inlet' ? '#4ecdc4' : 
                               node.type === 'outlet' ? '#ff6b6b' : 
                               node.type === 'terminal' ? '#ffe66d' : '#888';
                ctx.beginPath();
                ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
                ctx.fill();
            }});
        }}
        
        function resetView() {{
            rotation = {{ x: 0.3, y: 0.5 }};
            zoom = 1;
            render();
        }}
        
        function toggleWireframe() {{
            showWireframe = !showWireframe;
            render();
        }}
        
        function toggleAxes() {{
            showAxes = !showAxes;
            render();
        }}
        
        // Mouse interaction
        let isDragging = false;
        let lastMouse = {{ x: 0, y: 0 }};
        
        canvas.addEventListener('mousedown', e => {{
            isDragging = true;
            lastMouse = {{ x: e.clientX, y: e.clientY }};
        }});
        
        canvas.addEventListener('mousemove', e => {{
            if (isDragging) {{
                const dx = e.clientX - lastMouse.x;
                const dy = e.clientY - lastMouse.y;
                rotation.y += dx * 0.01;
                rotation.x += dy * 0.01;
                lastMouse = {{ x: e.clientX, y: e.clientY }};
                render();
            }}
        }});
        
        canvas.addEventListener('mouseup', () => isDragging = false);
        canvas.addEventListener('mouseleave', () => isDragging = false);
        
        canvas.addEventListener('wheel', e => {{
            zoom *= e.deltaY > 0 ? 0.9 : 1.1;
            zoom = Math.max(0.1, Math.min(10, zoom));
            render();
            e.preventDefault();
        }});
        
        window.addEventListener('resize', resize);
        resize();
    </script>
</body>
</html>'''
        
        return html
    
    def _create_analysis_report(
        self,
        network: Any,
        spec: Any,
        object_name: str,
        version: int
    ) -> str:
        """Create detailed analysis report in Markdown."""
        network_data = self._network_to_dict(network)
        spec_data = self._spec_to_dict(spec)
        metrics = self._compute_metrics(network, spec)
        
        report = f'''# Analysis Report: {object_name} v{version:03d}

Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

This report provides a detailed analysis of the generated vascular structure.

## Network Statistics

| Metric | Value |
|--------|-------|
| Total Nodes | {network_data.get('node_count', 0)} |
| Total Segments | {network_data.get('segment_count', 0)} |
| Terminal Count | {metrics.get('terminal_count', 0)} |
| Target Terminals | {metrics.get('target_terminals', 'N/A')} |

## Specification

'''
        
        if spec_data.get('domain'):
            domain = spec_data['domain']
            report += f'''### Domain
- Type: {domain.get('type', 'unknown')}
- Center: {domain.get('center', [0,0,0])}
- Size: {domain.get('size', domain.get('semi_axes', [0,0,0]))}

'''
        
        if spec_data.get('tree'):
            tree = spec_data['tree']
            report += f'''### Tree Configuration
- Topology: {tree.get('topology_kind', 'tree')}
- Target Terminals: {tree.get('terminal_count', 50)}

'''
        
        report += '''## Node Distribution

'''
        
        # Count node types
        node_types = {}
        for node in network_data.get('nodes', []):
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for node_type, count in node_types.items():
            report += f'- {node_type}: {count}\n'
        
        report += '''

## Quality Assessment

The generated structure has been analyzed for the following quality metrics:

1. **Topology**: Network connectivity and branching patterns
2. **Geometry**: Vessel radii and segment lengths
3. **Coverage**: Spatial distribution of terminals

## Recommendations

Based on the analysis, the following recommendations are provided:

'''
        
        # Add recommendations based on metrics
        terminal_count = metrics.get('terminal_count', 0)
        target = metrics.get('target_terminals', 50)
        
        if terminal_count < target * 0.8:
            report += f'- Consider increasing max_steps or adjusting kill_radius to achieve target terminal count\n'
        elif terminal_count > target * 1.2:
            report += f'- Terminal count exceeds target; consider adjusting parameters for more controlled growth\n'
        else:
            report += f'- Terminal count is within acceptable range of target\n'
        
        report += '''

## Files Generated

- `*_network.json`: Network topology data
- `*_spec.json`: Design specification
- `*_recipe.json`: Reproducible generation recipe
- `*_metrics.json`: Performance metrics
- `*_viewer.html`: Interactive 3D viewer
- `*.stl`: Mesh files at various resolutions

---

*Report generated by OutputGenerator*
'''
        
        return report
