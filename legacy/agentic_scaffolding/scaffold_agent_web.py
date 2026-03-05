#!/usr/bin/env python3
"""
Scaffold Agent Web - Interactive Web UI for Scaffold Designer
Rebuilt with reliable rendering and fast preview mode.
"""

import os
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3
from pyvista.trame import PyVistaRemoteView

from scaffold_agent import (
    generate_scaffold,
    manifold_to_pyvista,
    export_stl,
    ScaffoldAgent,
    LLM_PROVIDERS,
)


# Global state
server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

# Initialize state
state.messages = []
state.user_input = ""
state.generating = False
state.status = "Ready"
state.has_scaffold = False
state.params_text = ""
state.api_key_input = ""
state.selected_provider = "anthropic"
state.show_api_dialog = True
state.llm_status = "LLM: Disabled"
state.provider_options = [
    {"value": k, "title": v['name'], "disabled": not v['available']}
    for k, v in LLM_PROVIDERS.items()
]

# Agent and geometry
agent = ScaffoldAgent(None)
result_manifold = None
current_mesh = None

# Plotter setup - matching collision version for reliability
plotter = pv.Plotter(off_screen=True)
plotter.set_background('#1e1e1e')
plotter.enable_anti_aliasing('ssaa')

# Add initial placeholder
plotter.add_mesh(pv.Sphere(radius=0.5), color='gray', opacity=0.3, name='placeholder')
plotter.view_isometric()


def add_message(role, content, suggestions=None):
    """Add a message to the chat."""
    msgs = list(state.messages)
    msgs.append({
        "role": role,
        "content": content,
        "suggestions": suggestions or [],
        "id": f"msg_{len(msgs)}_{int(time.time()*1000)}"
    })
    state.messages = msgs


def update_view():
    """Update the 3D visualization."""
    global current_mesh
    plotter.clear_actors()

    if current_mesh is not None:
        plotter.add_mesh(current_mesh, color='#e74c3c', opacity=0.95, smooth_shading=True)
        plotter.add_axes()
        plotter.view_isometric()
    else:
        plotter.add_mesh(pv.Sphere(radius=0.5), color='gray', opacity=0.3, name='placeholder')

    ctrl.view_update()


def do_generate(params):
    """Generate scaffold with given parameters."""
    global result_manifold, current_mesh

    state.generating = True
    state.status = "Generating geometry..."

    try:
        t0 = time.time()

        # Use lower resolution for faster preview
        preview_params = params.copy()
        preview_params['resolution'] = 8  # Fast preview

        def progress(msg):
            state.status = msg

        _, channels, result = generate_scaffold(preview_params, progress)
        elapsed = time.time() - t0

        result_manifold = result
        if result:
            current_mesh = manifold_to_pyvista(result)
        else:
            current_mesh = None

        update_view()

        state.has_scaffold = True
        state.status = f"Done ({elapsed:.1f}s)"
        state.params_text = f"{params['inlets']} inlets | {params['levels']} levels"

    except Exception as e:
        state.status = f"Error: {e}"
        add_message("assistant", f"Generation failed: {str(e)}", ["Try simpler settings", "Reset to defaults"])

    state.generating = False


@ctrl.add("send_message")
def send_message():
    """Process user input."""
    txt = state.user_input.strip()
    if not txt or state.generating:
        return

    # Handle special commands
    if txt == "Enter API Key":
        state.show_api_dialog = True
        state.user_input = ""
        return
    if txt == "Continue without LLM":
        skip_api_key()
        state.user_input = ""
        return

    state.user_input = ""
    add_message("user", txt)
    state.status = "Processing..."

    try:
        result = agent.process(txt)

        if result.get('action') == 'generate':
            add_message("assistant", result.get('message', 'Generating...'), result.get('suggestions', []))
            do_generate(result['params'])
        else:
            add_message("assistant", result.get('message', ''), result.get('suggestions', []))
            state.status = "Ready"
    except Exception as e:
        add_message("assistant", f"Error: {str(e)}", ["Try again", "Reset"])
        state.status = "Ready"


@ctrl.add("click_suggestion")
def click_suggestion(text):
    """Handle suggestion button click."""
    if state.generating:
        return
    state.user_input = text
    send_message()


@ctrl.add("set_api_key")
def set_api_key():
    """Set API key and initialize LLM."""
    global agent
    key = state.api_key_input.strip()
    provider = state.selected_provider
    if key:
        agent = ScaffoldAgent(key, provider=provider)
        state.show_api_dialog = False
        provider_name = LLM_PROVIDERS.get(provider, {}).get('name', 'LLM')
        state.llm_status = f"{provider_name}: Ready"
        state.api_key_input = ""
        state.messages = []
        add_message("assistant", f"Connected to {provider_name}! Describe the scaffold you'd like.",
            ["Blood vessel network", "Dense capillary", "Simple 4-inlet"])


@ctrl.add("skip_api_key")
def skip_api_key():
    """Continue without LLM."""
    state.show_api_dialog = False
    state.llm_status = "Basic Mode"
    state.messages = []
    add_message("assistant",
        "How would you like to start?",
        ["Use defaults", "Start from scratch"])


@ctrl.add("new_variation")
def new_variation():
    """Generate new random variation."""
    if agent.current_params and not state.generating:
        p = agent.current_params.copy()
        p['seed'] = int(time.time()) % 100000
        agent.current_params = p
        add_message("user", "New variation")
        do_generate(p)


@ctrl.add("export_stl")
def export_stl_file():
    """Export current scaffold to STL."""
    global result_manifold
    if not result_manifold:
        return

    # Re-generate at higher resolution for export
    state.status = "Exporting (high quality)..."

    try:
        export_params = agent.current_params.copy()
        export_params['resolution'] = 16  # High quality for export

        _, _, export_result = generate_scaffold(export_params, lambda m: None)

        Path("./output").mkdir(exist_ok=True)
        fn = f"./output/scaffold_{int(time.time())}.stl"
        tris = export_stl(export_result, fn)
        add_message("assistant", f"Exported: {fn} ({tris:,} triangles)")
        state.status = "Exported!"
    except Exception as e:
        add_message("assistant", f"Export failed: {str(e)}")
        state.status = "Export failed"


# Check for existing API key
for provider, info in LLM_PROVIDERS.items():
    key = os.environ.get(info['env_key'])
    if key and info['available']:
        agent = ScaffoldAgent(key, provider=provider)
        state.show_api_dialog = False
        state.llm_status = f"{info['name']}: Ready"
        state.selected_provider = provider
        break

# Initial welcome message
if state.show_api_dialog:
    add_message("assistant",
        "Welcome! Configure an LLM for smart assistance, or continue in basic mode.",
        ["Enter API Key", "Continue without LLM"])
else:
    add_message("assistant",
        "How would you like to start?",
        ["Use defaults", "Start from scratch"])


# ============================================================================
# UI LAYOUT
# ============================================================================

with SinglePageLayout(server) as layout:
    layout.title.set_text("Scaffold Designer")

    # Toolbar
    with layout.toolbar:
        v3.VChip("{{ llm_status }}", size="small", variant="outlined", classes="mr-2")
        v3.VSpacer()
        v3.VBtn("Variation", click=ctrl.new_variation, disabled=("!has_scaffold || generating",),
                variant="text", size="small")
        v3.VBtn("Export STL", click=ctrl.export_stl, disabled=("!has_scaffold || generating",),
                variant="text", size="small")
        v3.VBtn(icon="mdi-key", click="show_api_dialog = true", variant="text", size="small")

    # Content
    with layout.content:
        # API Key Dialog
        with v3.VDialog(v_model=("show_api_dialog",), max_width="450", persistent=False):
            with v3.VCard():
                v3.VCardTitle("Configure LLM")
                with v3.VCardText():
                    v3.VSelect(
                        v_model=("selected_provider",),
                        items=("provider_options",),
                        label="Provider",
                        variant="outlined",
                        density="compact",
                        classes="mb-3"
                    )
                    v3.VTextField(
                        v_model=("api_key_input",),
                        label="API Key",
                        variant="outlined",
                        density="compact",
                        type="password"
                    )
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn("Skip", variant="text", click=ctrl.skip_api_key)
                    v3.VBtn("Connect", color="primary", click=ctrl.set_api_key)

        # Main layout
        with v3.VContainer(fluid=True, classes="fill-height pa-0"):
            with v3.VRow(classes="fill-height ma-0"):

                # LEFT: 3D Viewer
                with v3.VCol(cols=7, classes="pa-0"):
                    view = PyVistaRemoteView(plotter, interactive_ratio=1)
                    ctrl.view_update = view.update

                # RIGHT: Chat Panel
                with v3.VCol(cols=5, classes="pa-0 d-flex flex-column",
                             style="background: #f5f5f5; border-left: 1px solid #ddd;"):

                    # Header with status
                    with v3.VToolbar(density="compact", flat=True, color="white",
                                     style="border-bottom: 1px solid #eee;"):
                        v3.VIcon("mdi-chat", size="small", classes="mr-2")
                        v3.VToolbarTitle("{{ status }}", classes="text-body-2")
                        v3.VSpacer()
                        v3.VChip("{{ params_text }}", v_if="has_scaffold", size="x-small",
                                 variant="flat", color="grey-lighten-2")

                    # Loading bar
                    v3.VProgressLinear(indeterminate=True, v_if="generating", color="primary", height=3)

                    # Messages
                    with v3.VList(classes="flex-grow-1 overflow-y-auto pa-2",
                                  style="background: transparent;"):
                        # User messages
                        with v3.VListItem(
                            v_for="(msg, idx) in messages",
                            v_bind_key="msg.id",
                            classes="pa-1"
                        ):
                            # User message (right-aligned, blue)
                            with v3.VCard(
                                v_if="msg.role === 'user'",
                                color="primary",
                                classes="ml-auto",
                                rounded="lg",
                                style="max-width: 85%;"
                            ):
                                v3.VCardText("{{ msg.content }}", classes="pa-2 text-white text-body-2")

                            # Assistant message (left-aligned, white)
                            with v3.VCard(
                                v_if="msg.role === 'assistant'",
                                variant="flat",
                                classes="mr-auto",
                                rounded="lg",
                                color="white",
                                style="max-width: 85%;"
                            ):
                                v3.VCardText("{{ msg.content }}", classes="pa-2 text-body-2", style="white-space: pre-wrap;")
                                # Suggestion buttons
                                with v3.VCardActions(
                                    v_if="msg.suggestions && msg.suggestions.length > 0",
                                    classes="pa-2 pt-0 flex-wrap",
                                    style="gap: 4px;"
                                ):
                                    v3.VBtn(
                                        "{{ s }}",
                                        v_for="(s, i) in msg.suggestions",
                                        v_bind_key="i",
                                        click=(ctrl.click_suggestion, "[s]"),
                                        size="small",
                                        variant="outlined",
                                        color="primary",
                                        density="compact"
                                    )

                        # Loading indicator
                        with v3.VListItem(v_if="generating", classes="pa-1"):
                            with v3.VCard(variant="flat", color="white", rounded="lg",
                                          classes="mr-auto", style="max-width: 85%;"):
                                with v3.VCardText(classes="pa-2 d-flex align-center"):
                                    v3.VProgressCircular(indeterminate=True, size=16, width=2,
                                                         color="primary", classes="mr-2")
                                    v3.VLabel("{{ status }}", classes="text-body-2 text-grey")

                    # Input area
                    with v3.VCard(flat=True, classes="pa-2",
                                  style="border-top: 1px solid #ddd; background: white;"):
                        with v3.VRow(no_gutters=True, align="center"):
                            with v3.VCol():
                                v3.VTextField(
                                    v_model=("user_input",),
                                    placeholder="Describe your scaffold...",
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                    disabled=("generating",),
                                    v_on_keyup_enter=ctrl.send_message
                                )
                            with v3.VCol(cols="auto", classes="pl-2"):
                                v3.VBtn(
                                    icon="mdi-send",
                                    color="primary",
                                    variant="flat",
                                    size="small",
                                    click=ctrl.send_message,
                                    disabled=("generating || !user_input.trim()",)
                                )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    if args.api_key:
        agent = ScaffoldAgent(args.api_key)
        state.show_api_dialog = False
        state.llm_status = "LLM: Ready"

    print("\n  Scaffold Designer")
    print("  " + "=" * 40)
    print(f"  http://localhost:{args.port}")
    print("  " + "=" * 40 + "\n")

    server.start(port=args.port)
