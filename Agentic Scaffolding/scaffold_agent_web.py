#!/usr/bin/env python3
"""
Scaffold Agent Web - Interactive Scaffold Designer
Works with or without LLM - intelligent rule-based parsing as fallback.
"""

import json
import os
import time
import re
import numpy as np
import manifold3d as m3d
from pathlib import Path

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3
from trame.decorators import TrameApp

import pyvista as pv
from pyvista.trame.ui import plotter_ui

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# SCAFFOLD GENERATION
# =============================================================================

def generate_scaffold(params: dict, progress_callback=None) -> tuple:
    inlets = params.get('inlets', 4)
    levels = params.get('levels', 2)
    splits = params.get('splits', 2)
    spread = params.get('spread', 0.35)
    ratio = params.get('ratio', 0.79)
    curvature = params.get('curvature', 0.3)
    seed = params.get('seed', 42)
    tips_down = params.get('tips_down', False)
    deterministic = params.get('deterministic', False)
    res = params.get('resolution', 12)

    # Geometry params - can be customized
    outer_r = params.get('outer_radius', 4.875)
    inner_r = params.get('inner_radius', 4.575)
    height = params.get('height', 2.0)
    scaffold_h = params.get('scaffold_height', 1.92)

    net_r = inner_r - 0.12
    net_top, net_bot = scaffold_h, 0.06
    inlet_r = params.get('inlet_radius', 0.35)

    if progress_callback:
        progress_callback("Building body...")

    outer = m3d.Manifold.cylinder(height, outer_r, outer_r, 48)
    inner_cut = m3d.Manifold.cylinder(height + 0.02, inner_r, inner_r, 48).translate([0, 0, -0.01])
    body = m3d.Manifold.cylinder(scaffold_h, inner_r, inner_r, 48)
    scaffold_body = (outer - inner_cut) + body

    n = inlets
    if n == 1:
        inlet_pos = [(0.0, 0.0)]
    elif n <= 4:
        r = net_r * 0.45
        inlet_pos = [(r * np.cos(np.pi/4 + i * np.pi/2), r * np.sin(np.pi/4 + i * np.pi/2)) for i in range(n)]
    elif n == 9:
        sp = net_r * 0.5
        inlet_pos = [(i * sp, j * sp) for i in range(-1, 2) for j in range(-1, 2)]
    else:
        g = np.pi * (3 - np.sqrt(5))
        inlet_pos = [(net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.cos(i * g),
                     net_r * 0.7 * np.sqrt((i + 0.5) / n) * np.sin(i * g)) for i in range(n)]

    channels = []
    rng = np.random.default_rng(seed)

    if deterministic:
        randomized_inlets = inlet_pos
    else:
        randomized_inlets = []
        for ix, iy in inlet_pos:
            j = 0.12 * rng.uniform(0.5, 1.5)
            a = rng.uniform(0, 2 * np.pi)
            nx = ix + j * np.cos(a)
            ny = iy + j * np.sin(a)
            d = np.sqrt(nx*nx + ny*ny)
            if d > net_r - 0.5:
                nx, ny = nx * (net_r - 0.5) / d, ny * (net_r - 0.5) / d
            randomized_inlets.append((nx, ny))

    def make_cyl(x1, y1, z1, x2, y2, z2, r1, r2):
        dx, dy, dz = x2-x1, y2-y1, z2-z1
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 0.01:
            return None
        cyl = m3d.Manifold.cylinder(length, r2, r1, res)
        h = np.sqrt(dx*dx + dy*dy)
        if h > 0.001 or abs(dz) > 0.001:
            cyl = cyl.rotate([0, np.arctan2(h, -dz) * 180 / np.pi, 0]).rotate([0, 0, np.arctan2(-dy, -dx) * 180 / np.pi])
        return cyl.translate([x2, y2, z2])

    if progress_callback:
        progress_callback(f"Generating {len(randomized_inlets)} branches...")

    for ix, iy in randomized_inlets:
        channels.append(m3d.Manifold.cylinder(height - net_top + 0.03, inlet_r, inlet_r, res).translate([ix, iy, net_top - 0.01]))
        out_ang = np.arctan2(iy, ix) if deterministic and (abs(ix) > 0.01 or abs(iy) > 0.01) else rng.uniform(0, 2 * np.pi)

        def branch(x, y, z, r, ang, rem, pdir=None):
            if r < 0.03 or z <= net_bot + 0.02:
                return
            is_term = rem == 0
            z_step = (z - net_bot) / (rem + 1) * (1 if deterministic else rng.uniform(0.7, 1.3)) if rem > 0 else z - net_bot - 0.02
            nz = max(z - z_step, net_bot + 0.02)
            sp = spread * (1 if deterministic else rng.uniform(0.7, 1.3)) if rem < levels else 0
            sa = ang if deterministic else ang + rng.uniform(-0.4, 0.4)
            nx, ny = x + sp * np.cos(sa), y + sp * np.sin(sa)
            d = np.sqrt(nx*nx + ny*ny)
            if d > net_r - 0.1:
                nx, ny = nx * (net_r - 0.1) / d, ny * (net_r - 0.1) / d
            cr = r * ratio * (1 if deterministic else rng.uniform(0.85, 1.15))
            sdir = np.array(pdir) if pdir else np.array([0., 0., -1.])

            if tips_down and is_term:
                dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
                p0, p3 = np.array([x, y, z]), np.array([nx, ny, nz])
                p1, p2 = p0 + sdir * dist * 0.4, p3 + np.array([0, 0, dist * 0.35])
                prev_pt, prev_r = None, r
                for i in range(5):
                    t, mt = i / 4, 1 - i / 4
                    pt = mt**3 * p0 + 3*mt**2*t * p1 + 3*mt*t**2 * p2 + t**3 * p3
                    cur_r = r + (cr - r) * t
                    if prev_pt is not None:
                        seg = make_cyl(prev_pt[0], prev_pt[1], prev_pt[2], pt[0], pt[1], pt[2], prev_r, cur_r)
                        if seg:
                            channels.append(seg)
                        channels.append(m3d.Manifold.sphere(cur_r * 1.02, res).translate([pt[0], pt[1], pt[2]]))
                    prev_pt, prev_r = pt, cur_r
            else:
                dist = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
                if dist > 0.02:
                    cd = dist * (0.4 + curvature * 0.5)
                    c1 = np.array([x + sdir[0]*cd, y + sdir[1]*cd, z + sdir[2]*cd - curvature*dist*0.15])
                    out = np.array([np.cos(sa), np.sin(sa), 0.])
                    edir = np.array([out[0]*0.6, out[1]*0.6, -0.5])
                    edir /= np.linalg.norm(edir)
                    c2 = np.array([nx - edir[0]*cd*0.8, ny - edir[1]*cd*0.8, nz + cd*0.3*curvature])
                    for i in range(max(8, int(dist/0.1)) + 1):
                        t = i / max(8, int(dist/0.1))
                        mt, mt2, mt3 = 1-t, (1-t)**2, (1-t)**3
                        c = mt3*np.array([x,y,z]) + 3*mt2*t*c1 + 3*mt*t*t*c2 + t**3*np.array([nx,ny,nz])
                        channels.append(m3d.Manifold.sphere(r + (cr-r)*t, res).translate([c[0], c[1], c[2]]))

            if rem > 0 and nz > net_bot + 0.05:
                channels.append(m3d.Manifold.sphere(cr * 1.15, res).translate([nx, ny, nz]))
                if deterministic:
                    ca = [sa] if splits == 1 else [i * 2 * np.pi / splits for i in range(splits)]
                else:
                    bs = 2 * np.pi / splits
                    sr = rng.uniform(0, bs)
                    ca = [sr + i * bs + rng.uniform(-0.3, 0.3) for i in range(splits)]
                for child_ang in ca:
                    out = np.array([np.cos(sa), np.sin(sa), 0.])
                    ned = np.array([out[0]*0.6, out[1]*0.6, -0.5])
                    ned /= np.linalg.norm(ned)
                    branch(nx, ny, nz, cr, child_ang, rem - 1, tuple(ned))

        branch(ix, iy, net_top, inlet_r * 1.1, out_ang, levels, None)

    if not channels:
        return scaffold_body, None, scaffold_body

    if progress_callback:
        progress_callback(f"Combining {len(channels)} segments...")

    current = list(channels)
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            next_level.append(current[i] + current[i+1] if i+1 < len(current) else current[i])
        current = next_level
    combined = current[0]

    if progress_callback:
        progress_callback("Finalizing...")
    return scaffold_body, combined, scaffold_body - combined


def manifold_to_pyvista(manifold):
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    tris = np.array(mesh.tri_verts)
    return pv.PolyData(verts, np.hstack([np.full((len(tris), 1), 3), tris]).flatten())


def export_stl(manifold, filename: str):
    mesh = manifold.to_mesh()
    verts, tris = np.array(mesh.vert_properties)[:, :3], np.array(mesh.tri_verts)
    import struct
    with open(filename, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(tris)))
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nm = np.linalg.norm(n)
            n = n / nm if nm > 0 else np.array([0, 0, 1])
            f.write(struct.pack('<fff', *n) + struct.pack('<fff', *v0) + struct.pack('<fff', *v1) + struct.pack('<fff', *v2) + struct.pack('<H', 0))
    return len(tris)


# =============================================================================
# INTELLIGENT AGENT (works without LLM)
# =============================================================================

class ScaffoldAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        if HAS_ANTHROPIC and api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
            except:
                pass
        self.current_params = None
        self.history = []
        self.waiting_for = None  # Track what question we asked

    def default_params(self):
        return {
            'inlets': 4, 'levels': 2, 'splits': 2, 'spread': 0.35, 'ratio': 0.79,
            'curvature': 0.3, 'seed': int(time.time()) % 10000, 'tips_down': False,
            'deterministic': False, 'resolution': 12, 'height': 2.0,
            'outer_radius': 4.875, 'inner_radius': 4.575, 'inlet_radius': 0.35
        }

    def process(self, msg: str) -> dict:
        msg_lower = msg.lower().strip()

        # Handle responses to our questions
        if self.waiting_for:
            return self._handle_response(msg_lower)

        # Check for greetings / starting fresh
        if any(w in msg_lower for w in ['hello', 'hi', 'hey', 'start', 'begin', 'help']):
            return {
                'action': 'ask',
                'message': "Let's design a scaffold! What kind are you looking for?",
                'suggestions': ['Simple basic scaffold', 'Complex dense network', 'Organic flowing design', 'Let me describe it']
            }

        # Parse the message for parameters
        params, changes, questions = self._parse_message(msg_lower, msg)

        # If we have questions to ask, ask them
        if questions:
            self.waiting_for = questions[0]
            return {
                'action': 'ask',
                'message': questions[0]['question'],
                'suggestions': questions[0]['options']
            }

        # If we parsed meaningful changes, apply them
        if changes:
            if self.current_params is None:
                self.current_params = self.default_params()

            for key, value in changes.items():
                self.current_params[key] = value

            # Generate new seed for variation unless explicitly set
            if 'seed' not in changes:
                self.current_params['seed'] = int(time.time()) % 10000

            change_desc = ', '.join(f"{k}={v}" for k, v in changes.items())
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': f"Creating scaffold with: {change_desc}"
            }

        # No clear intent - ask for clarification
        return {
            'action': 'ask',
            'message': "I'm not sure what you'd like to change. What aspect should I modify?",
            'suggestions': ['More inlets', 'More branching', 'More organic curves', 'Tips pointing down']
        }

    def _handle_response(self, msg: str) -> dict:
        q = self.waiting_for
        self.waiting_for = None

        if self.current_params is None:
            self.current_params = self.default_params()

        param = q.get('param')

        # Match response to options
        for i, opt in enumerate(q.get('options', [])):
            if opt.lower() in msg or msg in opt.lower() or str(i+1) in msg:
                value = q.get('values', [None] * (i+1))[i] if 'values' in q else None
                if value is not None and param:
                    self.current_params[param] = value
                    return {
                        'action': 'generate',
                        'params': self.current_params.copy(),
                        'message': f"Setting {param} to {value}"
                    }

        # Try to parse as direct value
        nums = re.findall(r'\d+', msg)
        if nums and param:
            self.current_params[param] = int(nums[0])
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': f"Setting {param} to {nums[0]}"
            }

        # Re-parse as regular message
        return self.process(msg)

    def _parse_message(self, msg_lower: str, msg_orig: str) -> tuple:
        """Parse message and return (params, changes, questions)"""
        changes = {}
        questions = []

        if self.current_params is None:
            base = self.default_params()
        else:
            base = self.current_params.copy()

        # === EXACT NUMBER EXTRACTION ===

        # Inlets: "3 inlets", "inlets: 5", "with 5 inlet"
        for pattern in [r'(\d+)\s*inlet', r'inlet[s]?\s*[:=]?\s*(\d+)', r'inlet[s]?\s*(?:to|of|:)?\s*(\d+)']:
            m = re.search(pattern, msg_lower)
            if m:
                changes['inlets'] = min(25, max(1, int(m.group(1))))
                break

        # Levels/Layers/Depth: "4 levels", "4 layers", "depth 3"
        for pattern in [r'(\d+)\s*(?:level|layer|depth)', r'(?:level|layer|depth)[s]?\s*[:=]?\s*(\d+)']:
            m = re.search(pattern, msg_lower)
            if m:
                changes['levels'] = min(8, max(0, int(m.group(1))))
                break

        # Splits/Branches per node: "3 splits", "split into 3", "3 branches each"
        for pattern in [r'(\d+)\s*split', r'split[s]?\s*(?:into|of|:)?\s*(\d+)', r'(\d+)\s*branch.*(?:each|per)']:
            m = re.search(pattern, msg_lower)
            if m:
                changes['splits'] = min(6, max(1, int(m.group(1))))
                break

        # Height: "2mm", "2 mm tall", "height 3", "3mm height"
        for pattern in [r'(\d+(?:\.\d+)?)\s*mm', r'height\s*[:=]?\s*(\d+(?:\.\d+)?)', r'(\d+(?:\.\d+)?)\s*(?:tall|high)']:
            m = re.search(pattern, msg_lower)
            if m:
                changes['height'] = float(m.group(1))
                changes['scaffold_height'] = float(m.group(1)) * 0.96  # Proportional
                break

        # Seed
        m = re.search(r'seed\s*[:=]?\s*(\d+)', msg_lower)
        if m:
            changes['seed'] = int(m.group(1))

        # === RELATIVE MODIFIERS (thicker, taller, etc.) ===

        # Thickness: thicker/thinner, fatter/thinner
        if any(w in msg_lower for w in ['thicker', 'fatter', 'wider channels', 'bigger channels']):
            changes['ratio'] = min(base.get('ratio', 0.79) + 0.08, 0.95)
            changes['inlet_radius'] = min(base.get('inlet_radius', 0.35) + 0.08, 0.6)
        elif any(w in msg_lower for w in ['thinner', 'skinnier', 'narrower channels', 'smaller channels']):
            changes['ratio'] = max(base.get('ratio', 0.79) - 0.08, 0.5)
            changes['inlet_radius'] = max(base.get('inlet_radius', 0.35) - 0.08, 0.15)
        elif any(w in msg_lower for w in ['thick', 'fat', 'wide channel', 'big channel']) and 'ratio' not in changes:
            changes['ratio'] = 0.9
            changes['inlet_radius'] = 0.5
        elif any(w in msg_lower for w in ['thin', 'skinny', 'narrow', 'fine', 'delicate']) and 'ratio' not in changes:
            changes['ratio'] = 0.65
            changes['inlet_radius'] = 0.25

        # Height: taller/shorter
        if 'taller' in msg_lower or 'higher' in msg_lower:
            changes['height'] = base.get('height', 2.0) + 0.5
            changes['scaffold_height'] = changes['height'] * 0.96
        elif 'shorter' in msg_lower or 'lower' in msg_lower:
            changes['height'] = max(base.get('height', 2.0) - 0.5, 0.5)
            changes['scaffold_height'] = changes['height'] * 0.96
        elif 'tall' in msg_lower and 'height' not in changes:
            changes['height'] = 3.0
            changes['scaffold_height'] = 2.88
        elif 'short' in msg_lower and 'height' not in changes:
            changes['height'] = 1.0
            changes['scaffold_height'] = 0.96

        # Spread: broader/narrower, wider/tighter
        if any(w in msg_lower for w in ['broader', 'wider spread', 'more spread', 'spread out']):
            changes['spread'] = min(base.get('spread', 0.35) + 0.15, 0.8)
        elif any(w in msg_lower for w in ['narrower', 'tighter', 'less spread', 'compact']):
            changes['spread'] = max(base.get('spread', 0.35) - 0.1, 0.1)
        elif 'broad' in msg_lower or 'wide' in msg_lower and 'spread' not in changes:
            changes['spread'] = 0.6
        elif 'narrow' in msg_lower or 'tight' in msg_lower and 'spread' not in changes:
            changes['spread'] = 0.2

        # Curvature: curvier/straighter
        if any(w in msg_lower for w in ['curvier', 'more curve', 'more curved', 'bendier']):
            changes['curvature'] = min(base.get('curvature', 0.3) + 0.2, 1.0)
        elif any(w in msg_lower for w in ['straighter', 'less curve', 'more straight']):
            changes['curvature'] = max(base.get('curvature', 0.3) - 0.2, 0.0)
        elif any(w in msg_lower for w in ['curved', 'curvy', 'organic', 'flowing']) and 'curvature' not in changes:
            changes['curvature'] = 0.7
        elif any(w in msg_lower for w in ['straight', 'linear']) and 'curvature' not in changes:
            changes['curvature'] = 0.1

        # === MORE/LESS MODIFIERS ===

        if 'more' in msg_lower:
            if 'inlet' in msg_lower and 'inlets' not in changes:
                changes['inlets'] = min(base.get('inlets', 4) + 3, 25)
            if any(w in msg_lower for w in ['branch', 'level', 'layer', 'depth']) and 'levels' not in changes:
                changes['levels'] = min(base.get('levels', 2) + 1, 8)
            if 'split' in msg_lower and 'splits' not in changes:
                changes['splits'] = min(base.get('splits', 2) + 1, 6)

        if 'less' in msg_lower or 'fewer' in msg_lower:
            if 'inlet' in msg_lower and 'inlets' not in changes:
                changes['inlets'] = max(base.get('inlets', 4) - 2, 1)
            if any(w in msg_lower for w in ['branch', 'level', 'layer', 'depth']) and 'levels' not in changes:
                changes['levels'] = max(base.get('levels', 2) - 1, 0)
            if 'split' in msg_lower and 'splits' not in changes:
                changes['splits'] = max(base.get('splits', 2) - 1, 1)

        # === STYLE KEYWORDS ===

        if any(w in msg_lower for w in ['dense', 'complex', 'intricate', 'detailed', 'many', 'lots']):
            if 'inlets' not in changes:
                changes['inlets'] = min(base.get('inlets', 4) + 5, 16)
            if 'levels' not in changes:
                changes['levels'] = min(base.get('levels', 2) + 2, 6)

        if any(w in msg_lower for w in ['simple', 'basic', 'minimal', 'sparse']):
            if 'inlets' not in changes:
                changes['inlets'] = max(4, base.get('inlets', 4) - 2)
            if 'levels' not in changes:
                changes['levels'] = max(1, base.get('levels', 2) - 1)

        if any(w in msg_lower for w in ['regular', 'uniform', 'grid', 'symmetric', 'even', 'deterministic']):
            changes['deterministic'] = True

        if any(w in msg_lower for w in ['random', 'varied', 'chaotic', 'irregular', 'natural']):
            changes['deterministic'] = False

        # Tips down
        if any(w in msg_lower for w in ['tips down', 'drip', 'dripping', 'hanging', 'point down', 'vertical tip']):
            changes['tips_down'] = True
        if any(w in msg_lower for w in ['tips up', 'no drip', 'not drip']):
            changes['tips_down'] = False

        # === ASK QUESTIONS IF COMPLETELY VAGUE ===

        if len(msg_lower.split()) <= 2 and not changes:
            if any(w in msg_lower for w in ['scaffold', 'build', 'create', 'make', 'start', 'new']):
                questions.append({
                    'question': 'What kind of scaffold? How complex?',
                    'options': ['Simple (4 inlets)', 'Medium (9 inlets)', 'Complex (16 inlets)'],
                    'param': 'inlets',
                    'values': [4, 9, 16]
                })

        return base, changes, questions


# =============================================================================
# WEB APP
# =============================================================================

@TrameApp()
class ScaffoldAgentWeb:
    def __init__(self, server=None, api_key=None):
        self.server = get_server(server, client_type="vue3")
        self.state = self.server.state
        self.ctrl = self.server.controller

        self.state.messages = []
        self.state.user_input = ""
        self.state.generating = False
        self.state.status = "Ready - describe what you want to create"
        self.state.has_scaffold = False
        self.state.params_text = ""
        self.state.api_key_input = ""
        self.state.show_api_dialog = False

        self.agent = ScaffoldAgent(api_key)
        self.result_manifold = None

        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background('#1e1e1e')
        self.plotter.enable_anti_aliasing('ssaa')

        self._build_ui()

        # Welcome message
        self._add_msg("assistant",
            "Describe your scaffold. I understand:\n" +
            "- Exact values: '3 inlets', '4 layers', '2mm tall'\n" +
            "- Relative: 'thicker', 'taller', 'broader', 'curvier'\n" +
            "- Style: 'organic', 'uniform', 'tips down'\n" +
            "- Comparisons: 'more inlets', 'less branching'",
            ["3 inlets, 2 layers", "9 inlets, 4 layers, organic", "thick channels, tips down"]
        )

    def _add_msg(self, role, content, suggestions=None):
        msgs = list(self.state.messages)
        msgs.append({"role": role, "content": content, "suggestions": suggestions or [], "id": len(msgs)})
        self.state.messages = msgs

    def send_message(self):
        txt = self.state.user_input.strip()
        if not txt or self.state.generating:
            return
        self.state.user_input = ""
        self._add_msg("user", txt)
        self.state.generating = True
        self.state.status = "Processing..."

        result = self.agent.process(txt)

        if result.get('action') == 'generate':
            self._add_msg("assistant", result.get('message', 'Generating...'))
            self._generate(result['params'])
        elif result.get('action') == 'ask':
            self._add_msg("assistant", result.get('message', ''), result.get('suggestions', []))
            self.state.generating = False
            self.state.status = "Waiting for your input"
        else:
            self._add_msg("assistant", result.get('message', 'What would you like to create?'))
            self.state.generating = False
            self.state.status = "Ready"

    def click_suggestion(self, text):
        self.state.user_input = text
        self.send_message()

    def _generate(self, params):
        self.state.status = "Generating..."
        try:
            t0 = time.time()
            _, channels, result = generate_scaffold(params, lambda m: setattr(self.state, 'status', m))
            elapsed = time.time() - t0

            self.result_manifold = result
            self.plotter.clear()
            if result:
                self.plotter.add_mesh(manifold_to_pyvista(result), color='#e74c3c', opacity=0.95, smooth_shading=True)
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self.ctrl.view_update()

            self.state.has_scaffold = True
            self.state.status = f"Done in {elapsed:.1f}s"
            p = params
            self.state.params_text = f"Inlets: {p['inlets']} | Layers: {p['levels']} | Height: {p.get('height', 2.0):.1f}mm | Spread: {p.get('spread', 0.35):.2f} | Thickness: {p.get('ratio', 0.79):.2f} | Tips: {'down' if p['tips_down'] else 'normal'}"

            self._add_msg("assistant",
                "Here's your scaffold! What would you like to change?",
                ["More inlets", "More branching", "Add curves", "Tips down", "New variation"]
            )
        except Exception as e:
            self.state.status = f"Error: {e}"
            self._add_msg("assistant", f"Error generating: {str(e)}")
        self.state.generating = False

    def export_stl_file(self):
        if not self.result_manifold:
            return
        Path("./output").mkdir(exist_ok=True)
        fn = f"./output/scaffold_{int(time.time())}.stl"
        tris = export_stl(self.result_manifold, fn)
        self._add_msg("assistant", f"Exported to {fn} ({tris:,} triangles)")

    def new_variation(self):
        if self.agent.current_params:
            p = self.agent.current_params.copy()
            p['seed'] = int(time.time()) % 100000
            self.agent.current_params = p
            self._add_msg("user", "New variation please")
            self.state.generating = True
            self._generate(p)

    def _build_ui(self):
        self.state.trame__title = "Scaffold Designer"

        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("Scaffold Designer")

            with layout.toolbar:
                v3.VSpacer()
                v3.VBtn("Variation", click=self.new_variation, disabled=("!has_scaffold || generating",),
                       variant="text", size="small", prepend_icon="mdi-refresh")
                v3.VBtn("Export", click=self.export_stl_file, disabled=("!has_scaffold || generating",),
                       variant="text", size="small", prepend_icon="mdi-download")

            with layout.content:
                with v3.VContainer(fluid=True, classes="fill-height pa-0"):
                    with v3.VRow(classes="fill-height ma-0"):
                        # Left: 3D View
                        with v3.VCol(cols=7, classes="pa-0"):
                            with v3.VCard(flat=True, classes="fill-height d-flex flex-column", color="#1e1e1e"):
                                with v3.VCardText(classes="flex-grow-1 pa-0"):
                                    view = plotter_ui(self.plotter, mode='client')
                                    self.ctrl.view_update = view.update
                                v3.VCardText("{{ status }}", classes="text-caption pa-2",
                                            style="background: rgba(0,0,0,0.3); color: #aaa;")

                        # Right: Chat
                        with v3.VCol(cols=5, classes="pa-0"):
                            with v3.VCard(flat=True, classes="fill-height d-flex flex-column", color="#fafafa"):
                                # Messages area
                                with v3.VCardText(classes="flex-grow-1 overflow-y-auto pa-4",
                                                 style="max-height: calc(100vh - 180px);"):
                                    with v3.VSheet(v_for="msg in messages", key="msg.id", classes="mb-3", color="transparent"):
                                        # User message
                                        with v3.VCard(v_if="msg.role === 'user'", color="primary",
                                                     classes="ml-8", rounded="lg"):
                                            v3.VCardText("{{ msg.content }}", classes="text-white pa-3")

                                        # Assistant message
                                        with v3.VCard(v_if="msg.role === 'assistant'", variant="outlined",
                                                     classes="mr-8", rounded="lg"):
                                            v3.VCardText("{{ msg.content }}", classes="pa-3",
                                                        style="white-space: pre-wrap;")
                                            with v3.VCardActions(v_if="msg.suggestions && msg.suggestions.length",
                                                               classes="px-3 pb-3 pt-0 flex-wrap"):
                                                v3.VBtn("{{ s }}", v_for="s in msg.suggestions",
                                                       click=(self.click_suggestion, "[s]"),
                                                       size="small", variant="tonal", color="primary",
                                                       classes="ma-1")

                                    # Loading indicator
                                    with v3.VCard(v_if="generating", variant="outlined", classes="mr-8", rounded="lg"):
                                        with v3.VCardText(classes="pa-3 d-flex align-center"):
                                            v3.VProgressCircular(indeterminate=True, size=20, width=2, color="primary")
                                            v3.VLabel("Generating...", classes="ml-3")

                                # Input area
                                with v3.VCardActions(classes="pa-4", style="border-top: 1px solid #e0e0e0; background: white;"):
                                    with v3.VRow(no_gutters=True, align="center"):
                                        with v3.VCol():
                                            v3.VTextField(
                                                v_model=("user_input",),
                                                placeholder="Describe changes or type a command...",
                                                variant="outlined",
                                                density="comfortable",
                                                hide_details=True,
                                                disabled=("generating",),
                                                v_on_keyup_enter=self.send_message,
                                            )
                                        with v3.VCol(cols="auto", classes="pl-2"):
                                            v3.VBtn(icon="mdi-send", color="primary", click=self.send_message,
                                                   disabled=("generating || !user_input.trim()",))

                                # Params display
                                v3.VCardText("{{ params_text }}", v_if="has_scaffold",
                                            classes="text-caption text-grey pa-2 text-center",
                                            style="background: #f0f0f0; border-top: 1px solid #e0e0e0;")

    def run(self, **kwargs):
        self.server.start(**kwargs)


def main():
    print("\n  Scaffold Designer")
    print("  " + "=" * 40)
    print("  Interactive scaffold design tool")
    print("  " + "=" * 40 + "\n")

    app = ScaffoldAgentWeb()
    app.run(port=8081)


if __name__ == "__main__":
    main()
