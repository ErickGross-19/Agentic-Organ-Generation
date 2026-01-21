#!/usr/bin/env python3
"""
Scaffold Agent - LLM-based STL Generator

An agentic system that uses natural language to generate vascular scaffold STL files.
Interprets prompts, determines parameters, generates geometry, and exports.

Usage:
    python scaffold_agent.py "Create a dense vascular network with 9 inlets and 4 branching levels"
    python scaffold_agent.py --interactive
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import manifold3d as m3d
from pathlib import Path

# Try to import Anthropic SDK
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: anthropic package not installed. Run: pip install anthropic")


# =============================================================================
# SCAFFOLD GENERATION (extracted from scaffold_web_collision.py)
# =============================================================================

def generate_scaffold(params: dict) -> tuple:
    """
    Generate scaffold geometry based on parameters.
    Returns (scaffold_manifold, channels_manifold, result_manifold)
    """
    # Extract parameters with defaults
    inlets = params.get('inlets', 4)
    levels = params.get('levels', 2)
    splits = params.get('splits', 2)
    spread = params.get('spread', 0.35)
    ratio = params.get('ratio', 0.79)
    cone_angle = np.radians(params.get('cone_angle', 60))
    curvature = params.get('curvature', 0.3)
    seed = params.get('seed', 42)
    tips_down = params.get('tips_down', False)
    deterministic = params.get('deterministic', False)

    # Geometry parameters
    outer_r = params.get('outer_radius', 4.875)
    inner_r = params.get('inner_radius', 4.575)
    height = params.get('height', 2.0)
    scaffold_h = params.get('scaffold_height', 1.92)
    resolution = params.get('resolution', 16)  # Higher for export quality

    net_r = inner_r - 0.12
    net_top = scaffold_h
    net_bot = 0.06
    inlet_r = 0.35

    # Build scaffold body
    outer = m3d.Manifold.cylinder(height, outer_r, outer_r, 48)
    inner_cut = m3d.Manifold.cylinder(height + 0.02, inner_r, inner_r, 48).translate([0, 0, -0.01])
    ring = outer - inner_cut
    body = m3d.Manifold.cylinder(scaffold_h, inner_r, inner_r, 48)
    scaffold_body = ring + body

    # Generate inlet positions
    n = inlets
    if n == 1:
        inlet_pos = [(0.0, 0.0)]
    elif n <= 4:
        r = net_r * 0.45
        inlet_pos = [(r * np.cos(np.pi/4 + i * np.pi/2),
                      r * np.sin(np.pi/4 + i * np.pi/2)) for i in range(n)]
    elif n == 9:
        sp = net_r * 0.5
        inlet_pos = [(i * sp, j * sp) for i in range(-1, 2) for j in range(-1, 2)]
    else:
        g = np.pi * (3 - np.sqrt(5))
        inlet_pos = [(net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.cos(i * g),
                     net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.sin(i * g)) for i in range(n)]

    channels = []
    rng = np.random.default_rng(seed)
    res = resolution

    # Randomize inlet positions (unless deterministic)
    if deterministic:
        randomized_inlets = inlet_pos
    else:
        randomized_inlets = []
        for ix, iy in inlet_pos:
            jitter = 0.12 * rng.uniform(0.5, 1.5)
            ang = rng.uniform(0, 2 * np.pi)
            nx = ix + jitter * np.cos(ang)
            ny = iy + jitter * np.sin(ang)
            d = np.sqrt(nx*nx + ny*ny)
            if d > net_r - 0.5:
                scale = (net_r - 0.5) / d
                nx *= scale
                ny *= scale
            randomized_inlets.append((nx, ny))

    def make_cylinder(x1, y1, z1, x2, y2, z2, r1, r2):
        """Create tapered cylinder between two points."""
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 0.01:
            return None
        cyl = m3d.Manifold.cylinder(length, r2, r1, res)
        h = np.sqrt(dx*dx + dy*dy)
        if h > 0.001 or abs(dz) > 0.001:
            tilt = np.arctan2(h, -dz) * 180 / np.pi
            azim = np.arctan2(-dy, -dx) * 180 / np.pi
            cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])
        return cyl.translate([x2, y2, z2])

    # Generate branches for each inlet
    for ix, iy in randomized_inlets:
        # Inlet port
        port = m3d.Manifold.cylinder(height - net_top + 0.03, inlet_r, inlet_r, res)
        port = port.translate([ix, iy, net_top - 0.01])
        channels.append(port)

        # Initial angle
        if deterministic:
            out_ang = np.arctan2(iy, ix) if (abs(ix) > 0.01 or abs(iy) > 0.01) else 0
        else:
            out_ang = rng.uniform(0, 2 * np.pi)

        def branch(x, y, z, r, ang, remaining_levels, parent_dir=None):
            if r < 0.03 or z <= net_bot + 0.02:
                return

            is_terminal = remaining_levels == 0

            # Calculate target z
            if remaining_levels > 0:
                z_step = (z - net_bot) / (remaining_levels + 1)
                if not deterministic:
                    z_step *= rng.uniform(0.7, 1.3)
                nz = max(z - z_step, net_bot + 0.02)
            else:
                nz = net_bot + 0.02

            # Calculate spread
            if deterministic:
                sp = spread if remaining_levels < levels else 0
                safe_ang = ang
            else:
                sp = spread * rng.uniform(0.7, 1.3) if remaining_levels < levels else 0
                safe_ang = ang + rng.uniform(-0.4, 0.4)

            nx = x + sp * np.cos(safe_ang)
            ny = y + sp * np.sin(safe_ang)

            # Keep within bounds
            d = np.sqrt(nx*nx + ny*ny)
            if d > net_r - 0.1:
                scale = (net_r - 0.1) / d
                nx *= scale
                ny *= scale

            # Child radius
            cr = r * ratio if deterministic else r * ratio * rng.uniform(0.85, 1.15)

            # Direction vectors
            if parent_dir is None:
                start_dir = np.array([0.0, 0.0, -1.0])
            else:
                start_dir = np.array(parent_dir)

            # Tips down: smooth bezier curve ending straight down
            if tips_down and is_terminal:
                dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
                ctrl_dist = dist * 0.4

                p0 = np.array([x, y, z])
                p1 = p0 + start_dir * ctrl_dist
                p3 = np.array([nx, ny, nz])
                p2 = p3 + np.array([0, 0, dist * 0.35])

                prev_pt = None
                prev_r = r
                for i in range(5):
                    t = i / 4
                    mt = 1 - t
                    pt = mt**3 * p0 + 3*mt**2*t * p1 + 3*mt*t**2 * p2 + t**3 * p3
                    cur_r = r + (cr - r) * t

                    if prev_pt is not None:
                        seg = make_cylinder(prev_pt[0], prev_pt[1], prev_pt[2],
                                           pt[0], pt[1], pt[2], prev_r, cur_r)
                        if seg:
                            channels.append(seg)
                        channels.append(m3d.Manifold.sphere(cur_r * 1.02, res).translate([pt[0], pt[1], pt[2]]))

                    prev_pt = pt
                    prev_r = cur_r
            else:
                # Normal branch: curved path
                dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
                if dist > 0.02:
                    ctrl_dist = dist * (0.4 + curvature * 0.5)

                    c1x = x + start_dir[0] * ctrl_dist
                    c1y = y + start_dir[1] * ctrl_dist
                    c1z = z + start_dir[2] * ctrl_dist - curvature * dist * 0.15

                    outward = np.array([np.cos(safe_ang), np.sin(safe_ang), 0.0])
                    end_dir = np.array([outward[0] * 0.6, outward[1] * 0.6, -0.5])
                    end_dir = end_dir / np.linalg.norm(end_dir)

                    c2x = nx - end_dir[0] * ctrl_dist * 0.8
                    c2y = ny - end_dir[1] * ctrl_dist * 0.8
                    c2z = nz + ctrl_dist * 0.3 * curvature

                    # Sample bezier curve
                    n_pts = max(8, int(dist / 0.1))
                    for i in range(n_pts + 1):
                        t = i / n_pts
                        t2, t3 = t*t, t*t*t
                        mt, mt2, mt3 = 1-t, (1-t)**2, (1-t)**3

                        cx = mt3*x + 3*mt2*t*c1x + 3*mt*t2*c2x + t3*nx
                        cy = mt3*y + 3*mt2*t*c1y + 3*mt*t2*c2y + t3*ny
                        cz = mt3*z + 3*mt2*t*c1z + 3*mt*t2*c2z + t3*nz
                        cur_r = r + (cr - r) * t

                        channels.append(m3d.Manifold.sphere(cur_r, res).translate([cx, cy, cz]))

            # Continue branching
            if remaining_levels > 0 and nz > net_bot + 0.05:
                channels.append(m3d.Manifold.sphere(cr * 1.15, res).translate([nx, ny, nz]))

                if deterministic:
                    if splits == 1:
                        child_angles = [safe_ang]
                    else:
                        child_angles = [i * 2 * np.pi / splits for i in range(splits)]
                else:
                    base_spacing = 2 * np.pi / splits
                    start_rot = rng.uniform(0, base_spacing)
                    child_angles = [start_rot + i * base_spacing + rng.uniform(-0.3, 0.3) for i in range(splits)]

                for child_ang in child_angles:
                    # Calculate end direction for smooth continuity
                    outward = np.array([np.cos(safe_ang), np.sin(safe_ang), 0.0])
                    new_end_dir = np.array([outward[0] * 0.6, outward[1] * 0.6, -0.5])
                    new_end_dir = new_end_dir / np.linalg.norm(new_end_dir)
                    branch(nx, ny, nz, cr, child_ang, remaining_levels - 1, tuple(new_end_dir))

        start_r = inlet_r * 1.1
        branch(ix, iy, net_top, start_r, out_ang, levels, None)

    if not channels:
        return scaffold_body, None, scaffold_body

    # Combine channels using tree reduction
    def tree_union(manifolds):
        if not manifolds:
            return None
        if len(manifolds) == 1:
            return manifolds[0]
        current = list(manifolds)
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    next_level.append(current[i] + current[i + 1])
                else:
                    next_level.append(current[i])
            current = next_level
        return current[0]

    print(f"  Combining {len(channels)} channel segments...")
    combined = tree_union(channels)

    print(f"  Performing boolean subtraction...")
    result = scaffold_body - combined

    return scaffold_body, combined, result


def export_stl(manifold, filename: str):
    """Export manifold to STL file."""
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    tris = np.array(mesh.tri_verts)

    # Write binary STL
    with open(filename, 'wb') as f:
        import struct
        # Header (80 bytes)
        f.write(b'\0' * 80)
        # Number of triangles
        f.write(struct.pack('<I', len(tris)))

        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            # Calculate normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])

            # Write normal
            f.write(struct.pack('<fff', *normal))
            # Write vertices
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            # Attribute byte count
            f.write(struct.pack('<H', 0))

    print(f"  Exported: {filename} ({len(tris):,} triangles)")


# =============================================================================
# LLM AGENT
# =============================================================================

SYSTEM_PROMPT = """You are a scaffold design assistant that helps create vascular network scaffolds for biomedical applications.

When the user describes what they want, extract the relevant parameters and return them as JSON.

Available parameters:
- inlets: Number of inlet ports (1-25, default 4)
- levels: Branching depth/levels (0-8, default 2)
- splits: Branches per junction (1-6, default 2)
- spread: Horizontal spread factor (0.1-0.8, default 0.35)
- ratio: Child/parent radius ratio, Murray's law is 0.79 (0.5-0.95, default 0.79)
- curvature: How curved the branches are (0-1, default 0.3)
- cone_angle: Spread angle in degrees (10-180, default 60)
- seed: Random seed for reproducibility (any integer, default 42)
- tips_down: Whether branch tips point straight down (true/false, default false)
- deterministic: No randomness, even grid-like pattern (true/false, default false)
- resolution: Geometry resolution for export (8-32, default 16)

Interpret natural language descriptions:
- "dense" or "complex" -> more inlets (9-16), more levels (3-5)
- "simple" or "minimal" -> fewer inlets (1-4), fewer levels (1-2)
- "organic" or "natural" -> higher curvature (0.5-0.8), not deterministic
- "regular" or "uniform" -> deterministic=true
- "fine" or "detailed" -> smaller ratio (0.6-0.7), more splits
- "thick" or "coarse" -> larger ratio (0.85-0.9)
- "dripping" or "vertical tips" -> tips_down=true

Return ONLY valid JSON with the parameters. Example:
{"inlets": 9, "levels": 3, "splits": 2, "curvature": 0.5, "tips_down": true}
"""

def extract_parameters_with_llm(prompt: str, api_key: str = None) -> dict:
    """Use Claude to extract scaffold parameters from natural language."""
    if not HAS_ANTHROPIC:
        print("Error: anthropic package required. Install with: pip install anthropic")
        return get_default_params()

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: No ANTHROPIC_API_KEY found. Using defaults.")
        return get_default_params()

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Generate scaffold parameters for: {prompt}"}
            ]
        )

        response_text = message.content[0].text.strip()

        # Extract JSON from response
        if '{' in response_text:
            json_start = response_text.index('{')
            json_end = response_text.rindex('}') + 1
            json_str = response_text[json_start:json_end]
            params = json.loads(json_str)
        else:
            params = {}

        # Merge with defaults
        defaults = get_default_params()
        defaults.update(params)
        return defaults

    except Exception as e:
        print(f"LLM error: {e}")
        return get_default_params()


def get_default_params() -> dict:
    """Return default scaffold parameters."""
    return {
        'inlets': 4,
        'levels': 2,
        'splits': 2,
        'spread': 0.35,
        'ratio': 0.79,
        'curvature': 0.3,
        'cone_angle': 60,
        'seed': 42,
        'tips_down': False,
        'deterministic': False,
        'resolution': 16,
    }


def run_agent(prompt: str, output_dir: str = ".", api_key: str = None) -> str:
    """
    Main agent function: interpret prompt, generate scaffold, export STL.
    Returns the path to the generated STL file.
    """
    print("=" * 60)
    print("SCAFFOLD AGENT")
    print("=" * 60)
    print(f"\nPrompt: {prompt}\n")

    # Step 1: Extract parameters
    print("Step 1: Interpreting prompt...")
    params = extract_parameters_with_llm(prompt, api_key)
    print(f"  Parameters: {json.dumps(params, indent=2)}")

    # Step 2: Generate scaffold
    print("\nStep 2: Generating scaffold geometry...")
    t0 = time.time()
    scaffold, channels, result = generate_scaffold(params)
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.2f}s")

    if result is None:
        print("Error: Failed to generate scaffold")
        return None

    # Step 3: Export STL
    print("\nStep 3: Exporting STL files...")

    # Create output filename based on key params
    base_name = f"scaffold_{params['inlets']}in_{params['levels']}lvl_{params['splits']}sp"
    if params.get('tips_down'):
        base_name += "_tips"
    if params.get('deterministic'):
        base_name += "_det"
    base_name += f"_s{params['seed']}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = str(output_path / f"{base_name}.stl")
    channels_file = str(output_path / f"{base_name}_channels.stl")

    export_stl(result, result_file)
    if channels:
        export_stl(channels, channels_file)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Scaffold: {result_file}")
    print(f"Channels: {channels_file}")

    return result_file


def interactive_mode(output_dir: str = ".", api_key: str = None):
    """Run agent in interactive mode."""
    print("=" * 60)
    print("SCAFFOLD AGENT - Interactive Mode")
    print("=" * 60)
    print("\nDescribe the scaffold you want to create.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            run_agent(prompt, output_dir, api_key)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scaffold Agent - Generate vascular scaffold STLs from natural language"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Natural language description of the scaffold to generate"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "-o", "--output",
        default=".",
        help="Output directory for STL files (default: current directory)"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--params",
        help="Direct JSON parameters (bypass LLM)"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.output, args.api_key)
    elif args.params:
        # Direct parameter mode
        params = json.loads(args.params)
        defaults = get_default_params()
        defaults.update(params)

        print(f"Parameters: {json.dumps(defaults, indent=2)}")
        scaffold, channels, result = generate_scaffold(defaults)

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = f"scaffold_{defaults['inlets']}in_{defaults['levels']}lvl"
        if defaults.get('tips_down'):
            base_name += "_tips"
        if defaults.get('deterministic'):
            base_name += "_det"
        base_name += f"_s{defaults['seed']}"

        export_stl(result, str(output_path / f"{base_name}.stl"))
        if channels:
            export_stl(channels, str(output_path / f"{base_name}_channels.stl"))
    elif args.prompt:
        run_agent(args.prompt, args.output, args.api_key)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
