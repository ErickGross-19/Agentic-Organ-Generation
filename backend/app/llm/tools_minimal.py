"""
Minimal tool definitions for LLM endpoints with small context windows.
Only includes the most common scaffold types with condensed descriptions.
"""

from typing import Any

MINIMAL_SCAFFOLD_TOOLS: list[dict[str, Any]] = [
    {
        "name": "generate_vascular_network",
        "description": "Branching vascular network for tissue perfusion and blood vessels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "inlets": {"type": "integer", "description": "Number of inlets (1-25)", "minimum": 1, "maximum": 25},
                "levels": {"type": "integer", "description": "Branching depth (0-8)", "minimum": 0, "maximum": 8},
                "curvature": {"type": "number", "description": "Curviness 0-1", "minimum": 0, "maximum": 1},
            },
            "required": ["inlets", "levels"],
        },
    },
    {
        "name": "generate_porous_disc",
        "description": "Flat disc with uniform pores for cell culture and drug delivery.",
        "input_schema": {
            "type": "object",
            "properties": {
                "diameter_mm": {"type": "number", "description": "Diameter in mm", "minimum": 1, "maximum": 50},
                "height_mm": {"type": "number", "description": "Thickness in mm", "minimum": 0.5, "maximum": 10},
                "pore_diameter_um": {"type": "number", "description": "Pore size in microns", "minimum": 50, "maximum": 1000},
            },
            "required": ["diameter_mm", "height_mm", "pore_diameter_um"],
        },
    },
    {
        "name": "generate_lattice",
        "description": "3D lattice scaffold for bone and load-bearing applications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "unit_cell": {"type": "string", "enum": ["cubic", "bcc", "fcc", "octet", "kelvin"], "description": "Lattice type"},
                "cell_size_mm": {"type": "number", "description": "Unit cell size in mm", "minimum": 0.5, "maximum": 10},
                "strut_diameter_mm": {"type": "number", "description": "Strut thickness in mm", "minimum": 0.1, "maximum": 2},
            },
            "required": ["unit_cell", "cell_size_mm"],
        },
    },
    {
        "name": "generate_tubular_conduit",
        "description": "Hollow tube for nerve conduits and vascular grafts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outer_diameter_mm": {"type": "number", "description": "Outer diameter in mm", "minimum": 1, "maximum": 20},
                "wall_thickness_mm": {"type": "number", "description": "Wall thickness in mm", "minimum": 0.2, "maximum": 5},
                "length_mm": {"type": "number", "description": "Length in mm", "minimum": 5, "maximum": 100},
            },
            "required": ["outer_diameter_mm", "length_mm"],
        },
    },
    {
        "name": "generate_gyroid",
        "description": "Triply periodic minimal surface (TPMS) scaffold with excellent mechanical properties.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_size_mm": {"type": "number", "description": "Unit cell size in mm", "minimum": 1, "maximum": 20},
                "wall_thickness_mm": {"type": "number", "description": "Wall thickness in mm", "minimum": 0.1, "maximum": 2},
                "porosity": {"type": "number", "description": "Target porosity 0-1", "minimum": 0.3, "maximum": 0.9},
            },
            "required": ["cell_size_mm"],
        },
    },
    {
        "name": "ask_clarification",
        "description": "Ask user for more details when the request is unclear.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Clarifying question"},
                "options": {"type": "array", "items": {"type": "string"}, "description": "Suggested options"},
            },
            "required": ["question"],
        },
    },
]

# Mapping from tool name to scaffold type for minimal tools
MINIMAL_TOOL_TO_SCAFFOLD_TYPE = {
    "generate_vascular_network": "vascular_network",
    "generate_porous_disc": "porous_disc",
    "generate_lattice": "lattice",
    "generate_tubular_conduit": "tubular_conduit",
    "generate_gyroid": "gyroid",
}
