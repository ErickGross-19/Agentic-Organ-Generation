#!/usr/bin/env python3
"""
Scaffold Agent - Core scaffold generation and LLM-based parameter parsing.

An agentic system that uses natural language to generate vascular scaffold STL files.
Works with or without LLM - intelligent rule-based parsing as fallback.

Usage:
    python scaffold_agent.py "Create a dense vascular network with 9 inlets and 4 branching levels"
    python scaffold_agent.py --interactive
"""

import argparse
import json
import os
import re
import time
import numpy as np
import manifold3d as m3d
from pathlib import Path

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

# Supported LLM providers
LLM_PROVIDERS = {
    'anthropic': {'name': 'Anthropic (Claude)', 'available': HAS_ANTHROPIC, 'env_key': 'ANTHROPIC_API_KEY'},
    'openai': {'name': 'OpenAI (GPT)', 'available': HAS_OPENAI, 'env_key': 'OPENAI_API_KEY'},
    'google': {'name': 'Google (Gemini)', 'available': HAS_GOOGLE, 'env_key': 'GOOGLE_API_KEY'},
}


# =============================================================================
# SCAFFOLD GENERATION
# =============================================================================

def generate_scaffold(params: dict, progress_callback=None) -> tuple:
    """
    Generate scaffold geometry based on parameters.
    Returns (scaffold_body, channels, result) where result = scaffold_body - channels.
    """
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

    # Geometry params
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
    """Convert a Manifold3D object to PyVista PolyData."""
    import pyvista as pv
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    tris = np.array(mesh.tri_verts)
    return pv.PolyData(verts, np.hstack([np.full((len(tris), 1), 3), tris]).flatten())


def export_stl(manifold, filename: str) -> int:
    """Export manifold to binary STL file. Returns triangle count."""
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
# SCAFFOLD AGENT (LLM-powered)
# =============================================================================

SYSTEM_PROMPT = """You are an expert assistant for designing vascular scaffold geometries for biomedical tissue engineering applications.

You help users create scaffolds by understanding their needs and translating them into specific parameters. You should be conversational, helpful, and ask clarifying questions when needed.

## Available Parameters

You control these scaffold parameters:
- **inlets** (1-25): Number of inlet ports/entry points for the vascular network. Default: 4
- **levels** (0-8): Branching depth - how many times the channels split. Default: 2
- **splits** (1-6): How many branches at each junction. Default: 2
- **spread** (0.1-0.8): How far branches spread horizontally. Default: 0.35
- **ratio** (0.5-0.95): Child/parent radius ratio. Murray's law optimal is 0.79. Higher = thicker child branches. Default: 0.79
- **curvature** (0-1): How curved/organic the branches are. 0=straight, 1=very curved. Default: 0.3
- **tips_down** (true/false): Whether terminal branches point straight down (dripping effect). Default: false
- **deterministic** (true/false): If true, creates uniform grid-like pattern. If false, organic randomness. Default: false
- **height** (mm): Total scaffold height. Default: 2.0
- **inlet_radius** (mm): Radius of inlet channels. Default: 0.35

## Your Responses

You MUST respond with valid JSON in this exact format:
```json
{
  "message": "Your conversational response to the user",
  "action": "generate" or "chat",
  "params": { ... only include parameters to change ... },
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}
```

- **action: "generate"** - Use when you have enough info to create/modify a scaffold. Include params to set.
- **action: "chat"** - Use when asking questions, explaining, or need more info. No params needed.
- **suggestions** - Always provide 2-4 clickable suggestions for what the user might want next.
- **params** - Only include parameters you want to change. Omit unchanged ones.

## Guidelines

1. Be conversational and friendly, but concise
2. If the user's request is vague, ask clarifying questions (action: "chat")
3. Explain what you're doing when you generate
4. Suggest logical next steps
5. Understand relative terms: "denser" = more inlets/levels, "organic" = high curvature + not deterministic
6. Remember context from the conversation
7. If user says "more X" or "less X", adjust relative to current values

## Examples

User: "I want something that looks like blood vessels"
Response: {"message": "Blood vessels have an organic, branching structure! I'll create a scaffold with curved branches and natural randomness. How dense should the network be - just a few main branches, or a complex capillary-like network?", "action": "chat", "suggestions": ["Simple - few branches", "Medium density", "Dense capillary network"]}

User: "Dense capillary network"
Response: {"message": "Creating a dense vascular network with 12 inlets, 4 branching levels, organic curves, and natural variation.", "action": "generate", "params": {"inlets": 12, "levels": 4, "curvature": 0.6, "deterministic": false}, "suggestions": ["Make it denser", "Add dripping tips", "Make branches thicker", "Try uniform pattern"]}

User: "make the channels thicker"
Response: {"message": "Increasing the channel thickness by raising the radius ratio.", "action": "generate", "params": {"ratio": 0.88, "inlet_radius": 0.45}, "suggestions": ["Even thicker", "Back to normal thickness", "More inlets", "Export STL"]}
"""


class ScaffoldAgent:
    """
    Multi-provider LLM-powered scaffold design agent.
    Supports Anthropic (Claude), OpenAI (GPT), and Google (Gemini).
    Falls back to guided wizard mode if no API key available.
    """

    # Wizard steps for guided mode
    WIZARD_STEPS = [
        {'param': 'inlets', 'question': 'How many inlets (entry points)?', 'min': 1, 'max': 25, 'step': 3},
        {'param': 'levels', 'question': 'How many branching levels (depth)?', 'min': 0, 'max': 8, 'step': 1},
        {'param': 'splits', 'question': 'How many branches at each junction?', 'min': 1, 'max': 6, 'step': 1},
        {'param': 'spread', 'question': 'How spread out should branches be?', 'min': 0.1, 'max': 0.8, 'step': 0.15, 'labels': ['Tight', 'Medium', 'Wide']},
        {'param': 'curvature', 'question': 'How curved should the channels be?', 'min': 0.0, 'max': 1.0, 'step': 0.3, 'labels': ['Straight', 'Slightly curved', 'Very organic']},
        {'param': 'ratio', 'question': 'Channel thickness ratio (child/parent)?', 'min': 0.5, 'max': 0.95, 'step': 0.1, 'labels': ['Thin children', 'Murray optimal (0.79)', 'Thick children']},
        {'param': 'tips_down', 'question': 'Should branch tips point straight down?', 'type': 'bool'},
        {'param': 'deterministic', 'question': 'Pattern style?', 'type': 'bool', 'labels': ['Organic/random', 'Uniform/grid']},
    ]

    def __init__(self, api_key=None, provider='anthropic'):
        self.provider = provider
        self.api_key = api_key
        self.client = None
        self._init_client()
        self.current_params = self.default_params()
        self.history = []  # Conversation history for context
        self.wizard_step = -1  # -1 = not in wizard, 0+ = current step
        self.wizard_active = False

    def _init_client(self):
        """Initialize the LLM client based on provider."""
        self.client = None

        if self.provider == 'anthropic':
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if HAS_ANTHROPIC and key:
                try:
                    self.client = anthropic.Anthropic(api_key=key)
                    self._provider_name = "Claude"
                except Exception as e:
                    print(f"Failed to initialize Anthropic client: {e}")

        elif self.provider == 'openai':
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if HAS_OPENAI and key:
                try:
                    self.client = openai.OpenAI(api_key=key)
                    self._provider_name = "GPT"
                except Exception as e:
                    print(f"Failed to initialize OpenAI client: {e}")

        elif self.provider == 'google':
            key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            if HAS_GOOGLE and key:
                try:
                    genai.configure(api_key=key)
                    self.client = genai.GenerativeModel('gemini-1.5-flash')
                    self._provider_name = "Gemini"
                except Exception as e:
                    print(f"Failed to initialize Google client: {e}")

    def set_provider(self, provider, api_key=None):
        """Switch to a different LLM provider."""
        self.provider = provider
        if api_key:
            self.api_key = api_key
        self._init_client()
        # Clear history when switching providers
        self.history = []

    def default_params(self):
        return {
            'inlets': 4, 'levels': 2, 'splits': 2, 'spread': 0.35, 'ratio': 0.79,
            'curvature': 0.3, 'seed': int(time.time()) % 10000, 'tips_down': False,
            'deterministic': False, 'resolution': 12, 'height': 2.0,
            'outer_radius': 4.875, 'inner_radius': 4.575, 'inlet_radius': 0.35
        }

    def process(self, msg: str) -> dict:
        """
        Process a user message using the configured LLM.
        Returns: {'action': 'generate'|'chat', 'params': {...}, 'message': '...', 'suggestions': [...]}
        """
        # Add user message to history
        self.history.append({"role": "user", "content": msg})

        # If we have an LLM client, use it
        if self.client:
            result = self._process_with_llm(msg)
        else:
            result = self._process_fallback(msg)

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": result.get('message', '')})

        # Apply any parameter changes
        if result.get('action') == 'generate' and result.get('params'):
            for key, value in result['params'].items():
                if key in self.current_params:
                    self.current_params[key] = value
            # Always generate new seed unless explicitly set
            if 'seed' not in result['params']:
                self.current_params['seed'] = int(time.time()) % 10000
            result['params'] = self.current_params.copy()

        return result

    def _process_with_llm(self, msg: str) -> dict:
        """Use the configured LLM to process the message."""
        try:
            # Build context about current state
            context = f"\n\nCurrent scaffold parameters: {json.dumps(self.current_params, indent=2)}"
            full_system = SYSTEM_PROMPT + context

            # Build messages with history (keep last 10 exchanges for context)
            messages = self.history[-20:]

            if self.provider == 'anthropic':
                response_text = self._call_anthropic(full_system, messages)
            elif self.provider == 'openai':
                response_text = self._call_openai(full_system, messages)
            elif self.provider == 'google':
                response_text = self._call_google(full_system, messages)
            else:
                return self._process_fallback(msg)

            return self._parse_llm_response(response_text)

        except Exception as e:
            print(f"LLM error: {e}")
            return self._process_fallback(msg)

    def _call_anthropic(self, system: str, messages: list) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=messages
        )
        return response.content[0].text.strip()

    def _call_openai(self, system: str, messages: list) -> str:
        """Call OpenAI GPT API."""
        openai_messages = [{"role": "system", "content": system}]
        for m in messages:
            openai_messages.append({"role": m["role"], "content": m["content"]})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=openai_messages
        )
        return response.choices[0].message.content.strip()

    def _call_google(self, system: str, messages: list) -> str:
        """Call Google Gemini API."""
        # Gemini uses a different format - combine system + conversation
        prompt_parts = [system + "\n\n"]
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {m['content']}\n")
        prompt_parts.append("Assistant: ")

        response = self.client.generate_content("".join(prompt_parts))
        return response.text.strip()

    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse JSON response from any LLM."""
        try:
            # Find JSON in response
            if '{' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                # Ensure required fields
                if 'message' not in result:
                    result['message'] = "Here's your scaffold!"
                if 'action' not in result:
                    result['action'] = 'chat'
                if 'suggestions' not in result:
                    result['suggestions'] = []

                # Map 'chat' action to 'ask' for compatibility with web UI
                if result['action'] == 'chat':
                    result['action'] = 'ask'

                return result
            else:
                # No JSON found, treat as chat
                return {
                    'action': 'ask',
                    'message': response_text,
                    'suggestions': ['Create a scaffold', 'Help me design', 'Show options']
                }
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {
                'action': 'ask',
                'message': response_text.split('{')[0].strip() if '{' in response_text else response_text,
                'suggestions': ['Simple scaffold', 'Complex network', 'Help']
            }

    def start_wizard(self):
        """Start the guided wizard flow."""
        self.wizard_active = True
        self.wizard_step = 0
        self.current_params = self.default_params()
        self.current_params['seed'] = int(time.time()) % 10000
        result = self._wizard_ask_current()
        result['action'] = 'generate'
        result['params'] = self.current_params.copy()
        return result

    def use_defaults(self):
        """Use default parameters and generate."""
        self.wizard_active = False
        self.wizard_step = -1
        self.current_params = self.default_params()
        self.current_params['seed'] = int(time.time()) % 10000
        return {
            'action': 'generate',
            'params': self.current_params.copy(),
            'message': "Using defaults: 4 inlets, 2 levels. Adjust as needed!",
            'suggestions': ['More inlets', 'More levels', 'Organic style', 'Start from scratch']
        }

    # Word to number mapping
    WORD_NUMBERS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'twenty-five': 25, 'twentyfive': 25
    }

    def _parse_number(self, text):
        """Parse a number from text, including word numbers and decimals."""
        text_lower = text.lower().strip()

        # Check for word numbers first
        for word, num in self.WORD_NUMBERS.items():
            if word in text_lower:
                return num

        # Check for decimal numbers (e.g., 0.5, 0.79, .5)
        decimals = re.findall(r'\d*\.?\d+', text)
        if decimals:
            # Find the most likely number (prefer decimals for float params)
            for d in decimals:
                if '.' in d:
                    return float(d)
            # No decimal found, return first number
            return float(decimals[0]) if '.' in text else int(decimals[0])

        return None

    def _wizard_ask_current(self):
        """Generate the question for the current wizard step."""
        if self.wizard_step >= len(self.WIZARD_STEPS):
            # Wizard complete - generate!
            self.wizard_active = False
            self.wizard_step = -1
            self.current_params['seed'] = int(time.time()) % 10000
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': f"Creating your scaffold: {self.current_params['inlets']} inlets, {self.current_params['levels']} levels, {self.current_params['splits']} splits per junction",
                'suggestions': ['More inlets', 'Fewer inlets', 'New variation', 'Start over']
            }

        step = self.WIZARD_STEPS[self.wizard_step]
        param = step['param']
        current_val = self.current_params[param]

        # Build suggestions based on parameter type
        if step.get('type') == 'bool':
            if param == 'tips_down':
                suggestions = ["Yes (dripping tips)", "No (normal)", "Skip"]
            elif param == 'deterministic':
                suggestions = ["Organic (natural)", "Uniform (grid)", "Skip"]
            else:
                suggestions = ["Yes", "No", "Skip"]
        else:
            # Numeric parameter - show specific number options
            min_val, max_val = step['min'], step['max']

            if param == 'inlets':
                suggestions = ["1", "4", "9", "16"]
            elif param == 'levels':
                suggestions = ["0", "2", "4", "6"]
            elif param == 'splits':
                suggestions = ["2", "3", "4"]
            elif param == 'spread':
                suggestions = ["0.2 (tight)", "0.35 (medium)", "0.6 (wide)"]
            elif param == 'curvature':
                suggestions = ["0 (straight)", "0.3 (slight)", "0.7 (organic)"]
            elif param == 'ratio':
                suggestions = ["0.65 (thin)", "0.79 (optimal)", "0.9 (thick)"]
            else:
                suggestions = [str(min_val), str(current_val), str(max_val)]

        return {
            'action': 'ask',
            'message': f"Step {self.wizard_step + 1}/{len(self.WIZARD_STEPS)}: {step['question']}\n\nCurrent: {self._format_param_value(param, current_val)}",
            'suggestions': suggestions
        }

    def _format_param_value(self, param, value):
        """Format a parameter value for display."""
        if param == 'inlets':
            return f"{value} inlet{'s' if value != 1 else ''}"
        elif param == 'levels':
            return f"{value} level{'s' if value != 1 else ''}"
        elif param == 'splits':
            return f"{value} branch{'es' if value != 1 else ''} per junction"
        elif param == 'spread':
            if value < 0.25:
                return f"{value:.2f} (tight)"
            elif value > 0.55:
                return f"{value:.2f} (wide)"
            else:
                return f"{value:.2f} (medium)"
        elif param == 'curvature':
            if value < 0.2:
                return f"{value:.2f} (straight)"
            elif value > 0.6:
                return f"{value:.2f} (very curved)"
            else:
                return f"{value:.2f} (moderate)"
        elif param == 'ratio':
            return f"{value:.2f} ({'thin' if value < 0.7 else 'thick' if value > 0.85 else 'optimal'} child branches)"
        elif param == 'tips_down':
            return "Yes (dripping)" if value else "No (normal)"
        elif param == 'deterministic':
            return "Uniform grid" if value else "Organic random"
        else:
            return str(value)

    def _process_wizard_response(self, msg: str) -> dict:
        """Process a response during wizard mode."""
        msg_lower = msg.lower().strip()
        step = self.WIZARD_STEPS[self.wizard_step]
        param = step['param']

        # Check for skip
        if 'skip' in msg_lower:
            self.wizard_step += 1
            return self._wizard_ask_current()

        # Check for restart
        if 'start over' in msg_lower or 'restart' in msg_lower:
            return self.start_wizard()

        # Handle boolean parameters
        if step.get('type') == 'bool':
            if param == 'tips_down':
                if 'yes' in msg_lower or 'drip' in msg_lower:
                    self.current_params[param] = True
                else:
                    self.current_params[param] = False
            elif param == 'deterministic':
                if 'uniform' in msg_lower or 'grid' in msg_lower:
                    self.current_params[param] = True
                else:
                    self.current_params[param] = False
            else:
                self.current_params[param] = 'yes' in msg_lower

            set_val = "Yes" if self.current_params[param] else "No"
            self.wizard_step += 1
            self.current_params['seed'] = int(time.time()) % 10000
            next_q = self._wizard_ask_current()
            next_q['action'] = 'generate'
            next_q['params'] = self.current_params.copy()
            next_q['message'] = f"✓ Set {param} to {set_val}\n\n" + next_q['message']
            return next_q

        # Handle numeric parameters
        min_val, max_val = step['min'], step['max']

        # FIRST: Try to parse an exact number (digit or word)
        parsed_num = self._parse_number(msg)
        if parsed_num is not None:
            # Check if in valid range
            if parsed_num < min_val or parsed_num > max_val:
                return {
                    'action': 'ask',
                    'message': f"'{parsed_num}' is out of range.\n\nPlease choose a value between {min_val} and {max_val}.",
                    'suggestions': self._wizard_ask_current()['suggestions']
                }

            # User specified an exact number - use it!
            if isinstance(min_val, float):
                new_val = float(parsed_num)
            else:
                new_val = int(parsed_num)
            self.current_params[param] = new_val
            set_value = self.current_params[param]
            self.wizard_step += 1
            self.current_params['seed'] = int(time.time()) % 10000
            next_q = self._wizard_ask_current()
            next_q['action'] = 'generate'
            next_q['params'] = self.current_params.copy()
            next_q['message'] = f"✓ Set {param} to {set_value}\n\n" + next_q['message']
            return next_q

        # Check for descriptive words
        changed = False
        if param == 'spread':
            if 'tight' in msg_lower or 'narrow' in msg_lower:
                self.current_params[param] = 0.2
                changed = True
            elif 'wide' in msg_lower:
                self.current_params[param] = 0.6
                changed = True
            elif 'medium' in msg_lower:
                self.current_params[param] = 0.35
                changed = True
        elif param == 'curvature':
            if 'straight' in msg_lower:
                self.current_params[param] = 0.0
                changed = True
            elif 'organic' in msg_lower or 'curved' in msg_lower:
                self.current_params[param] = 0.7
                changed = True
            elif 'slight' in msg_lower:
                self.current_params[param] = 0.3
                changed = True
        elif param == 'ratio':
            if 'thin' in msg_lower:
                self.current_params[param] = 0.65
                changed = True
            elif 'thick' in msg_lower:
                self.current_params[param] = 0.9
                changed = True
            elif 'optimal' in msg_lower:
                self.current_params[param] = 0.79
                changed = True

        # If nothing was recognized, show error
        if not changed:
            return {
                'action': 'ask',
                'message': f"Sorry, I didn't understand that.\n\nPlease type a number or select an option below.",
                'suggestions': self._wizard_ask_current()['suggestions']
            }

        self.wizard_step += 1
        self.current_params['seed'] = int(time.time()) % 10000
        next_q = self._wizard_ask_current()
        next_q['action'] = 'generate'
        next_q['params'] = self.current_params.copy()
        next_q['message'] = f"✓ Set {param} to {self.current_params[param]}\n\n" + next_q['message']
        return next_q

    def _process_fallback(self, msg: str) -> dict:
        """
        Fallback mode - uses guided wizard for step-by-step configuration.
        """
        msg_lower = msg.lower().strip()

        # If wizard is active, process wizard response
        if self.wizard_active:
            return self._process_wizard_response(msg)

        # Check for defaults
        if 'default' in msg_lower or 'use default' in msg_lower:
            return self.use_defaults()

        # Check for wizard start commands
        if any(phrase in msg_lower for phrase in ['start', 'new', 'create', 'begin', 'scratch', 'fresh', 'wizard']):
            return self.start_wizard()

        # Check for quick presets
        if 'blood vessel' in msg_lower or 'vascular' in msg_lower:
            self.current_params.update({'inlets': 9, 'levels': 4, 'curvature': 0.6, 'deterministic': False, 'seed': int(time.time()) % 10000})
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': "Creating blood vessel network: 9 inlets, 4 levels, organic curves",
                'suggestions': ['More inlets', 'Fewer levels', 'Start over', 'New variation']
            }
        elif 'simple' in msg_lower:
            self.current_params.update({'inlets': 4, 'levels': 2, 'curvature': 0.3, 'deterministic': False, 'seed': int(time.time()) % 10000})
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': "Creating simple scaffold: 4 inlets, 2 levels",
                'suggestions': ['More inlets', 'More levels', 'Start over', 'New variation']
            }
        elif 'dense' in msg_lower or 'capillary' in msg_lower:
            self.current_params.update({'inlets': 16, 'levels': 5, 'curvature': 0.5, 'ratio': 0.75, 'deterministic': False, 'seed': int(time.time()) % 10000})
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': "Creating dense capillary network: 16 inlets, 5 levels",
                'suggestions': ['Fewer inlets', 'Fewer levels', 'Start over', 'New variation']
            }

        # Check for adjustment commands on existing scaffold
        if self.current_params and any(w in msg_lower for w in ['more', 'less', 'fewer', 'increase', 'decrease', 'adjust']):
            return self._process_adjustment(msg_lower)

        # Check for variation
        if 'variation' in msg_lower or 'random' in msg_lower:
            self.current_params['seed'] = int(time.time()) % 10000
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': "Generating new variation",
                'suggestions': ['Another variation', 'Adjust inlets', 'Start over']
            }

        # Check for direct number + parameter (e.g., "4 inlets please", "6 levels")
        parsed_num = self._parse_number(msg_lower)
        if parsed_num is not None:
            # Figure out what parameter they're setting
            if any(w in msg_lower for w in ['inlet', 'port', 'entry', 'opening']):
                self.current_params['inlets'] = max(1, min(25, parsed_num))
                self.current_params['seed'] = int(time.time()) % 10000
                return {
                    'action': 'generate',
                    'params': self.current_params.copy(),
                    'message': f"Set inlets to {self.current_params['inlets']}",
                    'suggestions': ['More levels', 'Fewer levels', 'New variation', 'Start over']
                }
            elif any(w in msg_lower for w in ['level', 'depth']):
                self.current_params['levels'] = max(0, min(8, parsed_num))
                self.current_params['seed'] = int(time.time()) % 10000
                return {
                    'action': 'generate',
                    'params': self.current_params.copy(),
                    'message': f"Set levels to {self.current_params['levels']}",
                    'suggestions': ['More inlets', 'Fewer inlets', 'New variation', 'Start over']
                }
            elif any(w in msg_lower for w in ['split', 'branch', 'fork']):
                self.current_params['splits'] = max(1, min(6, parsed_num))
                self.current_params['seed'] = int(time.time()) % 10000
                return {
                    'action': 'generate',
                    'params': self.current_params.copy(),
                    'message': f"Set splits to {self.current_params['splits']}",
                    'suggestions': ['More inlets', 'More levels', 'New variation', 'Start over']
                }

        # Default - show options
        return {
            'action': 'ask',
            'message': "How would you like to start?",
            'suggestions': ['Use defaults', 'Start from scratch']
        }

    def _process_adjustment(self, msg_lower: str) -> dict:
        """Process adjustment commands like 'more inlets', 'less levels'."""
        is_increase = any(w in msg_lower for w in ['more', 'increase', 'greater', 'higher', 'add'])
        delta = 1 if is_increase else -1
        changed = []

        # Check what to adjust
        if any(w in msg_lower for w in ['inlet', 'port', 'entry', 'opening']):
            new_val = self.current_params['inlets'] + (3 * delta)
            self.current_params['inlets'] = max(1, min(25, new_val))
            changed.append(f"inlets: {self.current_params['inlets']}")

        if any(w in msg_lower for w in ['level', 'depth', 'branch']):
            new_val = self.current_params['levels'] + delta
            self.current_params['levels'] = max(0, min(8, new_val))
            changed.append(f"levels: {self.current_params['levels']}")

        if any(w in msg_lower for w in ['split', 'fork', 'junction']):
            new_val = self.current_params['splits'] + delta
            self.current_params['splits'] = max(1, min(6, new_val))
            changed.append(f"splits: {self.current_params['splits']}")

        if any(w in msg_lower for w in ['curve', 'organic', 'bend']):
            new_val = self.current_params['curvature'] + (0.2 * delta)
            self.current_params['curvature'] = max(0, min(1, new_val))
            changed.append(f"curvature: {self.current_params['curvature']:.2f}")

        if any(w in msg_lower for w in ['thick', 'thin', 'radius', 'channel']):
            new_val = self.current_params['ratio'] + (0.08 * delta)
            self.current_params['ratio'] = max(0.5, min(0.95, new_val))
            changed.append(f"ratio: {self.current_params['ratio']:.2f}")

        if any(w in msg_lower for w in ['spread', 'wide', 'narrow']):
            new_val = self.current_params['spread'] + (0.1 * delta)
            self.current_params['spread'] = max(0.1, min(0.8, new_val))
            changed.append(f"spread: {self.current_params['spread']:.2f}")

        if changed:
            self.current_params['seed'] = int(time.time()) % 10000
            return {
                'action': 'generate',
                'params': self.current_params.copy(),
                'message': f"Adjusted: {', '.join(changed)}",
                'suggestions': ['More inlets', 'Fewer levels', 'New variation', 'Start over']
            }
        else:
            return {
                'action': 'ask',
                'message': "What would you like to adjust? (inlets, levels, splits, curvature, thickness, spread)",
                'suggestions': ['More inlets', 'Fewer levels', 'More curved', 'Start over']
            }


# =============================================================================
# CLI
# =============================================================================

def run_agent(prompt: str, output_dir: str = ".", api_key: str = None) -> str:
    """
    CLI agent function: interpret prompt, generate scaffold, export STL.
    Returns the path to the generated STL file.
    """
    print("=" * 60)
    print("SCAFFOLD AGENT")
    print("=" * 60)
    print(f"\nPrompt: {prompt}\n")

    # Use the ScaffoldAgent to parse the prompt
    agent = ScaffoldAgent(api_key)
    result = agent.process(prompt)

    if result.get('action') == 'ask':
        # For CLI, if it asks a question, just use defaults
        print(f"Agent asked: {result.get('message')}")
        print("Using default parameters for CLI mode.")
        params = agent.default_params()
    else:
        params = result.get('params', agent.default_params())

    print(f"Parameters: {json.dumps(params, indent=2)}")

    # Generate scaffold
    print("\nGenerating scaffold geometry...")
    t0 = time.time()
    scaffold, channels, final = generate_scaffold(params, lambda m: print(f"  {m}"))
    gen_time = time.time() - t0
    print(f"Generation time: {gen_time:.2f}s")

    if final is None:
        print("Error: Failed to generate scaffold")
        return None

    # Export STL
    print("\nExporting STL files...")
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

    tris = export_stl(final, result_file)
    print(f"  Exported: {result_file} ({tris:,} triangles)")

    if channels:
        tris = export_stl(channels, channels_file)
        print(f"  Exported: {channels_file} ({tris:,} triangles)")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    return result_file


def interactive_mode(output_dir: str = ".", api_key: str = None):
    """Run agent in interactive mode."""
    print("=" * 60)
    print("SCAFFOLD AGENT - Interactive Mode")
    print("=" * 60)
    print("\nDescribe the scaffold you want to create.")
    print("Type 'quit' or 'exit' to stop.\n")

    agent = ScaffoldAgent(api_key)

    while True:
        try:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            result = agent.process(prompt)

            if result.get('action') == 'ask':
                print(f"\n{result.get('message')}")
                if result.get('suggestions'):
                    print("Suggestions:", ', '.join(result['suggestions']))
            elif result.get('action') == 'generate':
                print(f"\n{result.get('message')}")
                params = result['params']
                print(f"Parameters: {json.dumps(params, indent=2)}")

                print("\nGenerating...")
                t0 = time.time()
                scaffold, channels, final = generate_scaffold(params, lambda m: print(f"  {m}"))
                print(f"Done in {time.time() - t0:.2f}s")

                # Export
                base_name = f"scaffold_{params['inlets']}in_{params['levels']}lvl"
                if params.get('tips_down'):
                    base_name += "_tips"
                base_name += f"_s{params['seed']}"

                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                result_file = str(output_path / f"{base_name}.stl")
                tris = export_stl(final, result_file)
                print(f"Exported: {result_file} ({tris:,} triangles)")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


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
        default="./output",
        help="Output directory for STL files (default: ./output)"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--params",
        help="Direct JSON parameters (bypass parsing)"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.output, args.api_key)
    elif args.params:
        params = json.loads(args.params)
        defaults = ScaffoldAgent().default_params()
        defaults.update(params)

        print(f"Parameters: {json.dumps(defaults, indent=2)}")
        scaffold, channels, result = generate_scaffold(defaults)

        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = f"scaffold_{defaults['inlets']}in_{defaults['levels']}lvl"
        if defaults.get('tips_down'):
            base_name += "_tips"
        base_name += f"_s{defaults['seed']}"

        tris = export_stl(result, str(output_path / f"{base_name}.stl"))
        print(f"Exported: {base_name}.stl ({tris:,} triangles)")

        if channels:
            tris = export_stl(channels, str(output_path / f"{base_name}_channels.stl"))
            print(f"Exported: {base_name}_channels.stl ({tris:,} triangles)")
    elif args.prompt:
        run_agent(args.prompt, args.output, args.api_key)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
