"""
Claude function calling tool definitions for scaffold generation.
"""

from typing import Any

SCAFFOLD_TOOLS: list[dict[str, Any]] = [
    {
        "name": "generate_vascular_network",
        "description": "Generate a branching vascular network scaffold for tissue perfusion. Use for blood vessel scaffolds, liver sinusoids, kidney tubules, or any branching channel network that requires fluid flow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "inlets": {
                    "type": "integer",
                    "description": "Number of inlet channels (1-25). More inlets create a denser network.",
                    "minimum": 1,
                    "maximum": 25,
                },
                "levels": {
                    "type": "integer",
                    "description": "Branching depth levels (0-8). 0 = no branching, higher = more complex tree.",
                    "minimum": 0,
                    "maximum": 8,
                },
                "splits": {
                    "type": "integer",
                    "description": "Number of branches at each junction (1-6). 2 = binary tree.",
                    "minimum": 1,
                    "maximum": 6,
                },
                "spread": {
                    "type": "number",
                    "description": "Horizontal spread angle (0.1-0.8). Low = tight/vertical, high = wide/spreading.",
                    "minimum": 0.1,
                    "maximum": 0.8,
                },
                "ratio": {
                    "type": "number",
                    "description": "Child/parent radius ratio (0.5-0.95). 0.79 = Murray's law optimal for efficient flow.",
                    "minimum": 0.5,
                    "maximum": 0.95,
                },
                "curvature": {
                    "type": "number",
                    "description": "Branch curviness (0-1). 0 = straight lines, 1 = very organic/curved.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "tips_down": {
                    "type": "boolean",
                    "description": "Whether terminal branches point straight down (dripping effect).",
                },
                "deterministic": {
                    "type": "boolean",
                    "description": "If true, creates uniform grid-like pattern. If false, organic randomness.",
                },
                "outer_radius_mm": {
                    "type": "number",
                    "description": "Outer boundary radius in mm.",
                    "exclusiveMinimum": 0,
                },
                "height_mm": {
                    "type": "number",
                    "description": "Total scaffold height in mm.",
                    "exclusiveMinimum": 0,
                },
                "inlet_radius_mm": {
                    "type": "number",
                    "description": "Channel radius at inlets in mm.",
                    "exclusiveMinimum": 0,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (optional).",
                },
            },
            "required": ["inlets", "levels"],
        },
    },
    {
        "name": "generate_porous_disc",
        "description": "Generate a flat disc scaffold with uniform pores. Good for cell seeding experiments, drug delivery patches, wound healing scaffolds, or simple tissue scaffolds requiring nutrient diffusion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "diameter_mm": {
                    "type": "number",
                    "description": "Disc diameter in mm (1-50).",
                    "minimum": 1.0,
                    "maximum": 50.0,
                },
                "height_mm": {
                    "type": "number",
                    "description": "Disc thickness in mm (0.5-10).",
                    "minimum": 0.5,
                    "maximum": 10.0,
                },
                "pore_diameter_um": {
                    "type": "number",
                    "description": "Pore diameter in micrometers (50-500). ~100-200um for most cells, larger for cell clusters.",
                    "minimum": 50.0,
                    "maximum": 500.0,
                },
                "pore_spacing_um": {
                    "type": "number",
                    "description": "Center-to-center pore spacing in micrometers (100-1000). Must be greater than pore_diameter.",
                    "minimum": 100.0,
                    "maximum": 1000.0,
                },
                "pore_pattern": {
                    "type": "string",
                    "description": "Pore arrangement pattern. 'hexagonal' = denser packing, 'grid' = rectangular array.",
                    "enum": ["hexagonal", "grid"],
                },
                "porosity_target": {
                    "type": "number",
                    "description": "Target porosity/void fraction (0.2-0.8). Higher = more porous.",
                    "minimum": 0.2,
                    "maximum": 0.8,
                },
            },
            "required": ["diameter_mm", "pore_diameter_um"],
        },
    },
    {
        "name": "generate_tubular_conduit",
        "description": "Generate a hollow tubular scaffold. Use for nerve conduits (with grooves for axon guidance), vascular grafts, tracheal scaffolds, bile ducts, or urethral stents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outer_diameter_mm": {
                    "type": "number",
                    "description": "Outer tube diameter in mm (1-20).",
                    "minimum": 1.0,
                    "maximum": 20.0,
                },
                "wall_thickness_mm": {
                    "type": "number",
                    "description": "Wall thickness in mm (0.3-5). Must be less than outer_diameter/2.",
                    "minimum": 0.3,
                    "maximum": 5.0,
                },
                "length_mm": {
                    "type": "number",
                    "description": "Total tube length in mm (1-100).",
                    "minimum": 1.0,
                    "maximum": 100.0,
                },
                "inner_texture": {
                    "type": "string",
                    "description": "Inner surface texture. 'smooth' = plain, 'grooved' = longitudinal grooves for cell guidance, 'porous' = small pores for nutrient diffusion.",
                    "enum": ["smooth", "grooved", "porous"],
                },
                "groove_count": {
                    "type": "integer",
                    "description": "Number of longitudinal grooves (2-32). Required if inner_texture is 'grooved'.",
                    "minimum": 2,
                    "maximum": 32,
                },
            },
            "required": ["outer_diameter_mm", "length_mm"],
        },
    },
    {
        "name": "generate_lattice",
        "description": "Generate a 3D lattice scaffold with repeating unit cells. Good for bone scaffolds, load-bearing implants, cartilage scaffolds, or any application requiring mechanical strength with porosity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bounding_box": {
                    "type": "object",
                    "description": "Bounding box dimensions in mm.",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "X dimension in mm.",
                            "exclusiveMinimum": 0,
                        },
                        "y": {
                            "type": "number",
                            "description": "Y dimension in mm.",
                            "exclusiveMinimum": 0,
                        },
                        "z": {
                            "type": "number",
                            "description": "Z dimension in mm.",
                            "exclusiveMinimum": 0,
                        },
                    },
                    "required": ["x", "y", "z"],
                },
                "unit_cell": {
                    "type": "string",
                    "description": "Lattice unit cell type. 'cubic' = simple cubic, 'bcc' = body-centered cubic (stronger).",
                    "enum": ["cubic", "bcc"],
                },
                "cell_size_mm": {
                    "type": "number",
                    "description": "Unit cell size in mm (0.5-5). Smaller = denser lattice.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "strut_diameter_mm": {
                    "type": "number",
                    "description": "Strut/beam diameter in mm (0.2-1). Must be less than cell_size.",
                    "minimum": 0.2,
                    "maximum": 1.0,
                },
            },
            "required": ["bounding_box"],
        },
    },
    {
        "name": "generate_primitive",
        "description": "Generate a basic geometric shape with optional modifications. Use as a starting point for custom scaffolds, simple prototypes, or educational models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "shape": {
                    "type": "string",
                    "description": "Primitive shape type.",
                    "enum": ["cylinder", "sphere", "box", "cone"],
                },
                "dimensions": {
                    "type": "object",
                    "description": "Shape-specific dimensions in mm. Cylinder: {radius, height}. Sphere: {radius}. Box: {x, y, z}. Cone: {bottom_radius, top_radius, height}.",
                    "additionalProperties": {"type": "number"},
                },
                "modifications": {
                    "type": "array",
                    "description": "Optional modifications to apply (holes, shells).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "Modification type: 'hole' to add a cylindrical hole, 'shell' to hollow out.",
                                "enum": ["hole", "shell"],
                            },
                            "params": {
                                "type": "object",
                                "description": "Operation parameters. Hole: {diameter, depth, position: {x, y, z}}. Shell: {wall_thickness}.",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["operation", "params"],
                    },
                },
            },
            "required": ["shape", "dimensions"],
        },
    },
    # Skeletal scaffolds (7)
    {
        "name": "generate_trabecular_bone",
        "description": "Generate a trabecular bone scaffold with interconnected porous network mimicking cancellous bone structure. Use for bone tissue engineering requiring trabecular architecture.",
        "input_schema": {
            "type": "object",
            "properties": {
                "porosity": {
                    "type": "number",
                    "description": "Void fraction (0.0-1.0). Higher = more porous. Default 0.7.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "pore_size_um": {
                    "type": "number",
                    "description": "Pore size in micrometers (50-1000um typical). Default 400.",
                    "minimum": 50.0,
                    "maximum": 1000.0,
                },
                "strut_thickness_um": {
                    "type": "number",
                    "description": "Strut thickness in micrometers (50-500um). Default 200.",
                    "minimum": 50.0,
                    "maximum": 500.0,
                },
                "anisotropy_ratio": {
                    "type": "number",
                    "description": "Anisotropy ratio for load direction (1.0 = isotropic, >1.0 = stronger in Z). Default 1.5.",
                    "minimum": 1.0,
                    "maximum": 3.0,
                },
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (4-16). Default 8.",
                    "minimum": 4,
                    "maximum": 16,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility. Default 42.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_osteochondral",
        "description": "Generate an osteochondral scaffold with gradient structure transitioning from bone to cartilage. Use for bone-cartilage interface tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bone_depth": {
                    "type": "number",
                    "description": "Thickness of bone zone in mm (1-10mm). Default 3.0.",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "cartilage_depth": {
                    "type": "number",
                    "description": "Thickness of cartilage zone in mm (0.5-5mm). Default 2.0.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "transition_width": {
                    "type": "number",
                    "description": "Width of transition zone in mm (0.2-3mm). Default 1.0.",
                    "minimum": 0.2,
                    "maximum": 3.0,
                },
                "gradient_type": {
                    "type": "string",
                    "description": "Gradient function: 'linear' or 'exponential'. Default 'linear'.",
                    "enum": ["linear", "exponential"],
                },
                "diameter": {
                    "type": "number",
                    "description": "Scaffold diameter in mm (3-20mm). Default 8.0.",
                    "minimum": 3.0,
                    "maximum": 20.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Circular resolution (8-32). Default 16.",
                    "minimum": 8,
                    "maximum": 32,
                },
                "bone_porosity": {
                    "type": "number",
                    "description": "Bone zone porosity (0.5-0.9). Default 0.7.",
                    "minimum": 0.5,
                    "maximum": 0.9,
                },
                "cartilage_porosity": {
                    "type": "number",
                    "description": "Cartilage zone porosity (0.7-0.95). Default 0.85.",
                    "minimum": 0.7,
                    "maximum": 0.95,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_articular_cartilage",
        "description": "Generate an articular cartilage scaffold with three-zone architecture (superficial, middle, deep). Use for cartilage tissue engineering with zonal pore structures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "total_thickness": {
                    "type": "number",
                    "description": "Total scaffold thickness in mm (0.5-5mm). Default 2.0.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "zone_ratios": {
                    "type": "array",
                    "description": "Zone thickness ratios [superficial, middle, deep]. Must sum to 1.0. Default [0.2, 0.4, 0.4].",
                    "items": {"type": "number", "minimum": 0.05, "maximum": 0.8},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "pore_gradient": {
                    "type": "array",
                    "description": "Pore radius per zone in mm [superficial, middle, deep]. Default [0.15, 0.25, 0.35].",
                    "items": {"type": "number", "minimum": 0.05, "maximum": 0.5},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "diameter": {
                    "type": "number",
                    "description": "Scaffold diameter in mm (3-20mm). Default 8.0.",
                    "minimum": 3.0,
                    "maximum": 20.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Circular resolution (8-32). Default 16.",
                    "minimum": 8,
                    "maximum": 32,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_meniscus",
        "description": "Generate a meniscus scaffold with wedge-shaped C-shaped geometry and radial/circumferential fiber zones. Use for knee meniscus tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "inner_radius": {
                    "type": "number",
                    "description": "Inner radius of C-shape in mm (5-20mm). Default 12.0.",
                    "minimum": 5.0,
                    "maximum": 20.0,
                },
                "outer_radius": {
                    "type": "number",
                    "description": "Outer radius of C-shape in mm (10-30mm). Must be > inner_radius. Default 20.0.",
                    "minimum": 10.0,
                    "maximum": 30.0,
                },
                "height": {
                    "type": "number",
                    "description": "Meniscus height in mm (3-15mm). Default 8.0.",
                    "minimum": 3.0,
                    "maximum": 15.0,
                },
                "wedge_angle_deg": {
                    "type": "number",
                    "description": "Wedge slope angle in degrees (10-45). Default 20.0.",
                    "minimum": 10.0,
                    "maximum": 45.0,
                },
                "zone_count": {
                    "type": "integer",
                    "description": "Number of radial zones (2-5). Default 3.",
                    "minimum": 2,
                    "maximum": 5,
                },
                "fiber_diameter": {
                    "type": "number",
                    "description": "Fiber diameter in mm (0.05-0.5mm). Default 0.2.",
                    "minimum": 0.05,
                    "maximum": 0.5,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-64). Default 32.",
                    "minimum": 8,
                    "maximum": 64,
                },
                "arc_degrees": {
                    "type": "number",
                    "description": "Arc extent in degrees (180-360). 300 = typical C-shape. Default 300.0.",
                    "minimum": 180.0,
                    "maximum": 360.0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_tendon_ligament",
        "description": "Generate a tendon/ligament scaffold with aligned crimped fiber bundles. Use for tendon/ligament tissue engineering with characteristic crimp patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fiber_diameter": {
                    "type": "number",
                    "description": "Fiber diameter in mm (0.05-0.3mm). Default 0.15.",
                    "minimum": 0.05,
                    "maximum": 0.3,
                },
                "fiber_spacing": {
                    "type": "number",
                    "description": "Spacing between fiber centers in mm (0.1-1mm). Default 0.4.",
                    "minimum": 0.1,
                    "maximum": 1.0,
                },
                "crimp_amplitude": {
                    "type": "number",
                    "description": "Crimp wave amplitude in mm (0.05-1mm). Default 0.3.",
                    "minimum": 0.05,
                    "maximum": 1.0,
                },
                "crimp_wavelength": {
                    "type": "number",
                    "description": "Crimp wavelength in mm (0.5-5mm). Default 2.0.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "bundle_count": {
                    "type": "integer",
                    "description": "Number of fiber bundles (1-10). Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
                "length": {
                    "type": "number",
                    "description": "Scaffold length in mm (5-100mm). Default 20.0.",
                    "minimum": 5.0,
                    "maximum": 100.0,
                },
                "width": {
                    "type": "number",
                    "description": "Scaffold width in mm (1-20mm). Default 5.0.",
                    "minimum": 1.0,
                    "maximum": 20.0,
                },
                "thickness": {
                    "type": "number",
                    "description": "Scaffold thickness in mm (0.5-10mm). Default 2.0.",
                    "minimum": 0.5,
                    "maximum": 10.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6-24). Default 12.",
                    "minimum": 6,
                    "maximum": 24,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_intervertebral_disc",
        "description": "Generate an intervertebral disc scaffold with annulus fibrosus rings and nucleus pulposus core. Use for spinal disc tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "disc_diameter": {
                    "type": "number",
                    "description": "Disc diameter in mm (20-60mm). Default 40.0.",
                    "minimum": 20.0,
                    "maximum": 60.0,
                },
                "disc_height": {
                    "type": "number",
                    "description": "Disc height in mm (3-20mm). Default 10.0.",
                    "minimum": 3.0,
                    "maximum": 20.0,
                },
                "af_ring_count": {
                    "type": "integer",
                    "description": "Number of annulus fibrosus rings (3-10). Default 5.",
                    "minimum": 3,
                    "maximum": 10,
                },
                "np_diameter": {
                    "type": "number",
                    "description": "Nucleus pulposus diameter in mm (5-25mm). Default 15.0.",
                    "minimum": 5.0,
                    "maximum": 25.0,
                },
                "af_layer_angle": {
                    "type": "number",
                    "description": "Fiber angle in AF layers (degrees, 10-45). Alternates ±. Default 30.0.",
                    "minimum": 10.0,
                    "maximum": 45.0,
                },
                "fiber_diameter": {
                    "type": "number",
                    "description": "Fiber diameter in mm (0.05-0.3mm). Default 0.15.",
                    "minimum": 0.05,
                    "maximum": 0.3,
                },
                "np_porosity": {
                    "type": "number",
                    "description": "Nucleus pulposus porosity (0.7-0.95). Default 0.9.",
                    "minimum": 0.7,
                    "maximum": 0.95,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_haversian_bone",
        "description": "Generate a Haversian bone scaffold with cortical bone microstructure and longitudinal vascular canals (osteons). Use for cortical bone tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "canal_diameter_um": {
                    "type": "number",
                    "description": "Haversian canal diameter in micrometers (30-100um). Default 50.0.",
                    "minimum": 30.0,
                    "maximum": 100.0,
                },
                "canal_spacing_um": {
                    "type": "number",
                    "description": "Spacing between canal centers in micrometers (150-500um). Default 250.0.",
                    "minimum": 150.0,
                    "maximum": 500.0,
                },
                "cortical_thickness": {
                    "type": "number",
                    "description": "Cortical wall thickness in mm (2-10mm). Default 5.0.",
                    "minimum": 2.0,
                    "maximum": 10.0,
                },
                "osteon_pattern": {
                    "type": "string",
                    "description": "Canal arrangement: 'hexagonal' or 'random'. Default 'hexagonal'.",
                    "enum": ["hexagonal", "random"],
                },
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-24). Default 12.",
                    "minimum": 8,
                    "maximum": 24,
                },
            },
            "required": [],
        },
    },
    # Organ scaffolds (6)
    {
        "name": "generate_hepatic_lobule",
        "description": "Generate a hepatic lobule scaffold with hexagonal unit cells and central/portal veins. Use for liver tissue engineering with lobule architecture.",
        "input_schema": {
            "type": "object",
            "properties": {
                "num_lobules": {
                    "type": "integer",
                    "description": "Number of hexagonal lobules (1-19). Default 1.",
                    "minimum": 1,
                    "maximum": 19,
                },
                "lobule_radius": {
                    "type": "number",
                    "description": "Lobule radius in mm (0.5-3mm). Default 1.5.",
                    "minimum": 0.5,
                    "maximum": 3.0,
                },
                "lobule_height": {
                    "type": "number",
                    "description": "Lobule height in mm (1-10mm). Default 3.0.",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "wall_thickness": {
                    "type": "number",
                    "description": "Hexagon wall thickness in mm (0.05-0.3mm). Default 0.1.",
                    "minimum": 0.05,
                    "maximum": 0.3,
                },
                "central_vein_radius": {
                    "type": "number",
                    "description": "Central vein radius in mm (0.05-0.5mm). Default 0.15.",
                    "minimum": 0.05,
                    "maximum": 0.5,
                },
                "portal_vein_radius": {
                    "type": "number",
                    "description": "Portal vein radius at corners in mm (0.05-0.3mm). Default 0.12.",
                    "minimum": 0.05,
                    "maximum": 0.3,
                },
                "show_sinusoids": {
                    "type": "boolean",
                    "description": "Include radial sinusoidal channels. Default false.",
                },
                "sinusoid_radius": {
                    "type": "number",
                    "description": "Sinusoid channel radius in mm (0.01-0.1mm). Default 0.025.",
                    "minimum": 0.01,
                    "maximum": 0.1,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6-16). Default 8.",
                    "minimum": 6,
                    "maximum": 16,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_cardiac_patch",
        "description": "Generate a cardiac patch scaffold with aligned microfibers for cardiomyocyte culture. Use for cardiac tissue engineering with fiber guidance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fiber_spacing": {
                    "type": "number",
                    "description": "Fiber spacing in micrometers (100-500um). Default 300.",
                    "minimum": 100.0,
                    "maximum": 500.0,
                },
                "fiber_diameter": {
                    "type": "number",
                    "description": "Fiber diameter in micrometers (50-200um). Default 100.",
                    "minimum": 50.0,
                    "maximum": 200.0,
                },
                "patch_size": {
                    "type": "object",
                    "description": "Patch dimensions in mm (x, y, z).",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "layer_count": {
                    "type": "integer",
                    "description": "Number of fiber layers (1-5). Default 3.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "alignment_angle": {
                    "type": "number",
                    "description": "Angular offset between layers in degrees (0-45). Default 0.",
                    "minimum": 0.0,
                    "maximum": 45.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6-16). Default 8.",
                    "minimum": 6,
                    "maximum": 16,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_kidney_tubule",
        "description": "Generate a kidney tubule scaffold with convoluted hollow tube mimicking nephron. Use for kidney tissue engineering with perfusable tubules.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tubule_diameter": {
                    "type": "number",
                    "description": "Tubule outer diameter in micrometers (50-200um). Default 100.",
                    "minimum": 50.0,
                    "maximum": 200.0,
                },
                "wall_thickness": {
                    "type": "number",
                    "description": "Tubule wall thickness in micrometers (10-30um). Default 15.",
                    "minimum": 10.0,
                    "maximum": 30.0,
                },
                "convolution_amplitude": {
                    "type": "number",
                    "description": "Convolution amplitude in micrometers (200-1000um). Default 500.",
                    "minimum": 200.0,
                    "maximum": 1000.0,
                },
                "convolution_frequency": {
                    "type": "number",
                    "description": "Oscillations per length unit (1-10). Default 3.0.",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "length": {
                    "type": "number",
                    "description": "Total tubule length in mm (3-30mm). Default 10.0.",
                    "minimum": 3.0,
                    "maximum": 30.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-32). Default 16.",
                    "minimum": 8,
                    "maximum": 32,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_lung_alveoli",
        "description": "Generate a lung alveoli scaffold with branching airways terminating in alveolar sacs. Use for lung tissue engineering with airway structures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch_generations": {
                    "type": "integer",
                    "description": "Branching depth levels (1-5). Default 3.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "alveoli_diameter": {
                    "type": "number",
                    "description": "Terminal alveoli diameter in micrometers (100-300um). Default 200.",
                    "minimum": 100.0,
                    "maximum": 300.0,
                },
                "airway_diameter": {
                    "type": "number",
                    "description": "Initial airway diameter in mm (0.5-3mm). Default 1.0.",
                    "minimum": 0.5,
                    "maximum": 3.0,
                },
                "branch_angle": {
                    "type": "number",
                    "description": "Angle between branches in degrees (20-60). Default 35.",
                    "minimum": 20.0,
                    "maximum": 60.0,
                },
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-24). Default 12.",
                    "minimum": 8,
                    "maximum": 24,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_pancreatic_islet",
        "description": "Generate a pancreatic islet scaffold with spherical porous clusters. Use for pancreatic islet tissue engineering with shell architecture.",
        "input_schema": {
            "type": "object",
            "properties": {
                "islet_diameter": {
                    "type": "number",
                    "description": "Islet diameter in micrometers (100-300um). Default 200.",
                    "minimum": 100.0,
                    "maximum": 300.0,
                },
                "shell_thickness": {
                    "type": "number",
                    "description": "Shell thickness in micrometers (20-100um). Default 50.",
                    "minimum": 20.0,
                    "maximum": 100.0,
                },
                "pore_size": {
                    "type": "number",
                    "description": "Pore size in micrometers (10-50um). Default 20.",
                    "minimum": 10.0,
                    "maximum": 50.0,
                },
                "cluster_count": {
                    "type": "integer",
                    "description": "Number of islets in cluster (1-20). Default 7.",
                    "minimum": 1,
                    "maximum": 20,
                },
                "spacing": {
                    "type": "number",
                    "description": "Distance between islet centers in micrometers (150-600um). Default 300.",
                    "minimum": 150.0,
                    "maximum": 600.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-32). Default 16.",
                    "minimum": 8,
                    "maximum": 32,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_liver_sinusoid",
        "description": "Generate a liver sinusoid scaffold with fenestrated hollow tube for blood flow. Use for liver tissue engineering with sinusoidal channels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sinusoid_diameter": {
                    "type": "number",
                    "description": "Sinusoid outer diameter in micrometers (10-30um). Default 20.",
                    "minimum": 10.0,
                    "maximum": 30.0,
                },
                "length": {
                    "type": "number",
                    "description": "Sinusoid length in mm (2-15mm). Default 5.0.",
                    "minimum": 2.0,
                    "maximum": 15.0,
                },
                "fenestration_size": {
                    "type": "number",
                    "description": "Fenestration pore size in micrometers (0.5-3um, scaled for printing). Default 1.0.",
                    "minimum": 0.5,
                    "maximum": 3.0,
                },
                "fenestration_density": {
                    "type": "number",
                    "description": "Fraction of surface with fenestrations (0.0-1.0). Default 0.2.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-32). Default 16.",
                    "minimum": 8,
                    "maximum": 32,
                },
            },
            "required": [],
        },
    },
    # Soft tissue scaffolds (4)
    {
        "name": "generate_multilayer_skin",
        "description": "Generate a multilayer skin scaffold with epidermis, dermis, and hypodermis layers with optional vascular channels. Use for skin tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "epidermis_thickness_mm": {
                    "type": "number",
                    "description": "Epidermis layer thickness in mm (0.05-0.5mm). Default 0.15.",
                    "minimum": 0.05,
                    "maximum": 0.5,
                },
                "dermis_thickness_mm": {
                    "type": "number",
                    "description": "Dermis layer thickness in mm (0.5-5mm). Default 1.5.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "hypodermis_thickness_mm": {
                    "type": "number",
                    "description": "Hypodermis layer thickness in mm (1-10mm). Default 3.0.",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "diameter_mm": {
                    "type": "number",
                    "description": "Scaffold diameter in mm (5-30mm). Default 10.0.",
                    "minimum": 5.0,
                    "maximum": 30.0,
                },
                "pore_gradient": {
                    "type": "array",
                    "description": "Porosity per layer [epidermis, dermis, hypodermis] (0.0-1.0). Default [0.3, 0.5, 0.7].",
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "vascular_channel_diameter_mm": {
                    "type": "number",
                    "description": "Vascular channel diameter in mm (0.1-0.5mm). Default 0.2.",
                    "minimum": 0.1,
                    "maximum": 0.5,
                },
                "vascular_channel_count": {
                    "type": "integer",
                    "description": "Number of vertical vascular channels (0-16). Default 8.",
                    "minimum": 0,
                    "maximum": 16,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-32). Default 16.",
                    "minimum": 8,
                    "maximum": 32,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_skeletal_muscle",
        "description": "Generate a skeletal muscle scaffold with aligned myofiber channels. Supports parallel, unipennate, or bipennate fiber architectures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fiber_diameter_um": {
                    "type": "number",
                    "description": "Fiber diameter in micrometers (50-150um). Default 75.",
                    "minimum": 50.0,
                    "maximum": 150.0,
                },
                "fiber_spacing_um": {
                    "type": "number",
                    "description": "Spacing between fibers in micrometers (75-300um). Default 150.",
                    "minimum": 75.0,
                    "maximum": 300.0,
                },
                "pennation_angle_deg": {
                    "type": "number",
                    "description": "Pennation angle in degrees (0-30). Default 15.",
                    "minimum": 0.0,
                    "maximum": 30.0,
                },
                "fascicle_count": {
                    "type": "integer",
                    "description": "Number of fascicles (1-10). Default 3.",
                    "minimum": 1,
                    "maximum": 10,
                },
                "architecture_type": {
                    "type": "string",
                    "description": "Fiber architecture: 'parallel', 'unipennate', or 'bipennate'. Default 'parallel'.",
                    "enum": ["parallel", "unipennate", "bipennate"],
                },
                "length_mm": {
                    "type": "number",
                    "description": "Scaffold length in mm (5-50mm). Default 10.0.",
                    "minimum": 5.0,
                    "maximum": 50.0,
                },
                "width_mm": {
                    "type": "number",
                    "description": "Scaffold width in mm (2-20mm). Default 5.0.",
                    "minimum": 2.0,
                    "maximum": 20.0,
                },
                "height_mm": {
                    "type": "number",
                    "description": "Scaffold height in mm (2-20mm). Default 5.0.",
                    "minimum": 2.0,
                    "maximum": 20.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6-16). Default 8.",
                    "minimum": 6,
                    "maximum": 16,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_cornea",
        "description": "Generate a cornea scaffold with curved dome geometry and lamella layers. Use for corneal tissue engineering with optical clarity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "diameter_mm": {
                    "type": "number",
                    "description": "Cornea diameter in mm (8-14mm). Default 11.0.",
                    "minimum": 8.0,
                    "maximum": 14.0,
                },
                "thickness_mm": {
                    "type": "number",
                    "description": "Cornea thickness in mm (0.3-1.0mm). Default 0.55.",
                    "minimum": 0.3,
                    "maximum": 1.0,
                },
                "radius_of_curvature_mm": {
                    "type": "number",
                    "description": "Radius of curvature in mm (6-10mm). Default 7.8.",
                    "minimum": 6.0,
                    "maximum": 10.0,
                },
                "lamella_count": {
                    "type": "integer",
                    "description": "Number of stromal lamella layers (2-10). Default 5.",
                    "minimum": 2,
                    "maximum": 10,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64, higher for smoothness). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_adipose",
        "description": "Generate an adipose tissue scaffold with honeycomb cell structure and optional vascular channels. Use for fat tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cell_size_um": {
                    "type": "number",
                    "description": "Adipocyte cell diameter in micrometers (100-200um). Default 150.",
                    "minimum": 100.0,
                    "maximum": 200.0,
                },
                "wall_thickness_um": {
                    "type": "number",
                    "description": "Cell wall thickness in micrometers (20-50um). Default 30.",
                    "minimum": 20.0,
                    "maximum": 50.0,
                },
                "vascular_channel_spacing_mm": {
                    "type": "number",
                    "description": "Spacing between perfusion channels in mm (1-5mm). Default 2.0.",
                    "minimum": 1.0,
                    "maximum": 5.0,
                },
                "vascular_channel_diameter_mm": {
                    "type": "number",
                    "description": "Vascular channel diameter in mm (0.2-0.6mm). Default 0.3.",
                    "minimum": 0.2,
                    "maximum": 0.6,
                },
                "height_mm": {
                    "type": "number",
                    "description": "Scaffold height in mm (2-15mm). Default 5.0.",
                    "minimum": 2.0,
                    "maximum": 15.0,
                },
                "diameter_mm": {
                    "type": "number",
                    "description": "Scaffold diameter in mm (5-30mm). Default 10.0.",
                    "minimum": 5.0,
                    "maximum": 30.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6 for hexagons). Default 6.",
                    "minimum": 6,
                    "maximum": 6,
                },
            },
            "required": [],
        },
    },
    # Tubular scaffolds (5)
    {
        "name": "generate_blood_vessel",
        "description": "Generate a blood vessel scaffold with tri-layer wall (intima, media, adventitia). Optional Y-bifurcation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "inner_diameter_mm": {
                    "type": "number",
                    "description": "Lumen inner diameter in mm (2-15mm). Default 5.0.",
                    "minimum": 2.0,
                    "maximum": 15.0,
                },
                "wall_thickness_mm": {
                    "type": "number",
                    "description": "Total wall thickness in mm (0.5-3mm). Default 1.0.",
                    "minimum": 0.5,
                    "maximum": 3.0,
                },
                "layer_ratios": {
                    "type": "array",
                    "description": "Layer thickness ratios [intima, media, adventitia]. Must sum to 1.0. Default [0.1, 0.5, 0.4].",
                    "items": {"type": "number", "minimum": 0.05, "maximum": 0.8},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "length_mm": {
                    "type": "number",
                    "description": "Vessel length in mm (10-100mm). Default 30.0.",
                    "minimum": 10.0,
                    "maximum": 100.0,
                },
                "bifurcation_angle_deg": {
                    "type": "number",
                    "description": "Bifurcation angle in degrees (20-60). Null = no bifurcation. Default null.",
                    "minimum": 20.0,
                    "maximum": 60.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Circular resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_nerve_conduit",
        "description": "Generate a nerve conduit with internal microchannels for axon guidance. Multi-channel design.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outer_diameter_mm": {
                    "type": "number",
                    "description": "Outer conduit diameter in mm (2-8mm). Default 4.0.",
                    "minimum": 2.0,
                    "maximum": 8.0,
                },
                "channel_count": {
                    "type": "integer",
                    "description": "Number of guidance channels (1-19). Default 7 (1 central + 6 ring).",
                    "minimum": 1,
                    "maximum": 19,
                },
                "channel_diameter_um": {
                    "type": "number",
                    "description": "Channel diameter in micrometers (50-500um). Default 200.",
                    "minimum": 50.0,
                    "maximum": 500.0,
                },
                "wall_thickness_mm": {
                    "type": "number",
                    "description": "Outer wall thickness in mm (0.3-2mm). Default 0.5.",
                    "minimum": 0.3,
                    "maximum": 2.0,
                },
                "length_mm": {
                    "type": "number",
                    "description": "Conduit length in mm (5-100mm). Default 20.0.",
                    "minimum": 5.0,
                    "maximum": 100.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-32). Default 16.",
                    "minimum": 8,
                    "maximum": 32,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_spinal_cord",
        "description": "Generate a spinal cord scaffold with butterfly gray matter core and white matter channels. Use for spinal cord tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cord_diameter_mm": {
                    "type": "number",
                    "description": "Spinal cord diameter in mm (8-18mm). Human ~10-15mm. Default 12.0.",
                    "minimum": 8.0,
                    "maximum": 18.0,
                },
                "channel_diameter_um": {
                    "type": "number",
                    "description": "White matter channel diameter in micrometers (100-500um). Default 300.",
                    "minimum": 100.0,
                    "maximum": 500.0,
                },
                "channel_count": {
                    "type": "integer",
                    "description": "Number of white matter channels (8-48). Default 24.",
                    "minimum": 8,
                    "maximum": 48,
                },
                "gray_matter_ratio": {
                    "type": "number",
                    "description": "Gray matter cross-section fraction (0.2-0.6). Default 0.4.",
                    "minimum": 0.2,
                    "maximum": 0.6,
                },
                "length_mm": {
                    "type": "number",
                    "description": "Scaffold length in mm (10-100mm). Default 50.0.",
                    "minimum": 10.0,
                    "maximum": 100.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_bladder",
        "description": "Generate a bladder scaffold with multi-layer dome-shaped wall. Layers represent urothelium, lamina propria, and detrusor muscle.",
        "input_schema": {
            "type": "object",
            "properties": {
                "diameter_mm": {
                    "type": "number",
                    "description": "Bladder diameter in mm (40-120mm). Default 70.0.",
                    "minimum": 40.0,
                    "maximum": 120.0,
                },
                "wall_thickness_mm": {
                    "type": "number",
                    "description": "Wall thickness in mm (1.5-6mm). Default 3.0.",
                    "minimum": 1.5,
                    "maximum": 6.0,
                },
                "layer_count": {
                    "type": "integer",
                    "description": "Number of wall layers (2-4). Default 3.",
                    "minimum": 2,
                    "maximum": 4,
                },
                "dome_curvature": {
                    "type": "number",
                    "description": "Dome curvature (0.3-1.0). 0.5=hemisphere, 1.0=sphere, <0.5=flatter. Default 0.7.",
                    "minimum": 0.3,
                    "maximum": 1.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_trachea",
        "description": "Generate a trachea scaffold with C-shaped cartilage rings and posterior gap. Use for tracheal tissue engineering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outer_diameter_mm": {
                    "type": "number",
                    "description": "Outer trachea diameter in mm (12-30mm). Human ~15-25mm. Default 20.0.",
                    "minimum": 12.0,
                    "maximum": 30.0,
                },
                "ring_thickness_mm": {
                    "type": "number",
                    "description": "Cartilage ring thickness in mm (1-5mm). Default 3.0.",
                    "minimum": 1.0,
                    "maximum": 5.0,
                },
                "ring_spacing_mm": {
                    "type": "number",
                    "description": "Spacing between rings in mm (2-7mm). Default 4.0.",
                    "minimum": 2.0,
                    "maximum": 7.0,
                },
                "ring_count": {
                    "type": "integer",
                    "description": "Number of C-shaped rings (3-20). Default 10.",
                    "minimum": 3,
                    "maximum": 20,
                },
                "length_mm": {
                    "type": "number",
                    "description": "Total trachea length in mm (auto-calculated if null). Default null.",
                    "minimum": 10.0,
                    "maximum": 150.0,
                },
                "posterior_gap_angle_deg": {
                    "type": "number",
                    "description": "Posterior membrane gap angle in degrees (45-135). Default 90.",
                    "minimum": 45.0,
                    "maximum": 135.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    # Dental scaffolds (3)
    {
        "name": "generate_dentin_pulp",
        "description": "Generate a dentin-pulp tooth scaffold with crown and root geometry. Use for dental tissue engineering with hollow pulp chamber.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tooth_height": {
                    "type": "number",
                    "description": "Total tooth height in mm (8-20mm). Default 12.0.",
                    "minimum": 8.0,
                    "maximum": 20.0,
                },
                "crown_diameter": {
                    "type": "number",
                    "description": "Crown diameter in mm (6-15mm). Default 10.0.",
                    "minimum": 6.0,
                    "maximum": 15.0,
                },
                "root_length": {
                    "type": "number",
                    "description": "Root length in mm (6-20mm). Default 12.0.",
                    "minimum": 6.0,
                    "maximum": 20.0,
                },
                "root_diameter": {
                    "type": "number",
                    "description": "Root tip diameter in mm (2-8mm). Default 5.0.",
                    "minimum": 2.0,
                    "maximum": 8.0,
                },
                "pulp_chamber_size": {
                    "type": "number",
                    "description": "Relative pulp chamber size (0.2-0.7). Default 0.4.",
                    "minimum": 0.2,
                    "maximum": 0.7,
                },
                "dentin_thickness": {
                    "type": "number",
                    "description": "Dentin shell thickness in mm (1-5mm). Default 3.0.",
                    "minimum": 1.0,
                    "maximum": 5.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_ear_auricle",
        "description": "Generate an ear auricle scaffold with curved cartilage framework. Use for ear reconstruction with helix and antihelix features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scale_factor": {
                    "type": "number",
                    "description": "Overall size multiplier (0.5-1.5). 1.0 = adult size. Default 1.0.",
                    "minimum": 0.5,
                    "maximum": 1.5,
                },
                "thickness": {
                    "type": "number",
                    "description": "Cartilage thickness in mm (1-4mm). Default 2.0.",
                    "minimum": 1.0,
                    "maximum": 4.0,
                },
                "helix_definition": {
                    "type": "number",
                    "description": "Prominence of helix rim (0-1). 0=subtle, 1=prominent. Default 0.7.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "antihelix_depth": {
                    "type": "number",
                    "description": "Depth of antihelix ridge (0-1). 0=none, 1=deep. Default 0.3.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_nasal_septum",
        "description": "Generate a nasal septum scaffold with curved cartilage sheet. Supports single curve or S-curve deviation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "length": {
                    "type": "number",
                    "description": "Anterior-posterior length in mm (20-60mm). Default 40.0.",
                    "minimum": 20.0,
                    "maximum": 60.0,
                },
                "height": {
                    "type": "number",
                    "description": "Superior-inferior height in mm (15-50mm). Default 30.0.",
                    "minimum": 15.0,
                    "maximum": 50.0,
                },
                "thickness": {
                    "type": "number",
                    "description": "Cartilage thickness in mm (1-6mm). Default 3.0.",
                    "minimum": 1.0,
                    "maximum": 6.0,
                },
                "curvature_radius": {
                    "type": "number",
                    "description": "Radius of curvature in mm (30-200mm). Larger = less curved. Default 75.0.",
                    "minimum": 30.0,
                    "maximum": 200.0,
                },
                "curve_type": {
                    "type": "string",
                    "description": "Deviation type: 'single' or 's_curve'. Default 'single'.",
                    "enum": ["single", "s_curve"],
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (16-64). Default 32.",
                    "minimum": 16,
                    "maximum": 64,
                },
            },
            "required": [],
        },
    },
    # Advanced lattice scaffolds (5)
    {
        "name": "generate_gyroid",
        "description": "Generate a gyroid TPMS scaffold with smooth interconnected pores. Excellent for bone and high-flow applications. Requires scikit-image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "cell_size_mm": {
                    "type": "number",
                    "description": "Unit cell period in mm (0.5-10mm). Default 2.0.",
                    "minimum": 0.5,
                    "maximum": 10.0,
                },
                "wall_thickness_mm": {
                    "type": "number",
                    "description": "Wall thickness in mm (0.1-2mm). Default 0.3.",
                    "minimum": 0.1,
                    "maximum": 2.0,
                },
                "isovalue": {
                    "type": "number",
                    "description": "Isosurface level (-1 to 1). Controls porosity. Default 0.0.",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "samples_per_cell": {
                    "type": "integer",
                    "description": "Grid resolution per cell (10-40). Higher = smoother. Default 20.",
                    "minimum": 10,
                    "maximum": 40,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_schwarz_p",
        "description": "Generate a Schwarz P TPMS scaffold with cubic symmetry. Good for directional flow. Requires scikit-image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "cell_size_mm": {
                    "type": "number",
                    "description": "Unit cell period in mm (0.5-10mm). Default 2.0.",
                    "minimum": 0.5,
                    "maximum": 10.0,
                },
                "wall_thickness_mm": {
                    "type": "number",
                    "description": "Wall thickness in mm (0.1-2mm). Default 0.3.",
                    "minimum": 0.1,
                    "maximum": 2.0,
                },
                "isovalue": {
                    "type": "number",
                    "description": "Isosurface level (-3 to 3). Controls porosity. Default 0.0.",
                    "minimum": -3.0,
                    "maximum": 3.0,
                },
                "samples_per_cell": {
                    "type": "integer",
                    "description": "Grid resolution per cell (10-40). Higher = smoother. Default 20.",
                    "minimum": 10,
                    "maximum": 40,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_octet_truss",
        "description": "Generate an octet truss lattice with exceptional strength-to-weight ratio. Stretch-dominated structure ideal for load-bearing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "cell_size_mm": {
                    "type": "number",
                    "description": "Unit cell size in mm (1-10mm). Default 2.0.",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "strut_diameter_mm": {
                    "type": "number",
                    "description": "Strut diameter in mm (0.2-2mm). Must be < cell_size. Default 0.3.",
                    "minimum": 0.2,
                    "maximum": 2.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6-16). Default 8.",
                    "minimum": 6,
                    "maximum": 16,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_voronoi",
        "description": "Generate a Voronoi lattice with organic randomized structure mimicking trabecular bone. Requires scipy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "cell_count": {
                    "type": "integer",
                    "description": "Number of Voronoi seed points (10-100). Default 30.",
                    "minimum": 10,
                    "maximum": 100,
                },
                "strut_diameter_mm": {
                    "type": "number",
                    "description": "Strut diameter in mm (0.2-1.5mm). Default 0.3.",
                    "minimum": 0.2,
                    "maximum": 1.5,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility. Default null (random).",
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6-16). Default 8.",
                    "minimum": 6,
                    "maximum": 16,
                },
                "margin_factor": {
                    "type": "number",
                    "description": "Margin around bbox for seed points (0.1-0.5). Default 0.2.",
                    "minimum": 0.1,
                    "maximum": 0.5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_honeycomb",
        "description": "Generate a honeycomb lattice with hexagonal cells. High in-plane stiffness and strength.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "cell_size_mm": {
                    "type": "number",
                    "description": "Hexagon flat-to-flat size in mm (1-10mm). Default 2.0.",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "wall_thickness_mm": {
                    "type": "number",
                    "description": "Wall thickness in mm (0.2-2mm). Default 0.3.",
                    "minimum": 0.2,
                    "maximum": 2.0,
                },
                "height_mm": {
                    "type": "number",
                    "description": "Extrusion height in mm (uses bbox.z if null). Default null.",
                    "exclusiveMinimum": 0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Segments per hexagon side (1 for sharp edges). Default 1.",
                    "minimum": 1,
                    "maximum": 4,
                },
            },
            "required": [],
        },
    },
    # Microfluidic scaffolds (3)
    {
        "name": "generate_organ_on_chip",
        "description": "Generate an organ-on-chip microfluidic device with tissue chambers and channels. Use for organ-on-chip applications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_width_mm": {
                    "type": "number",
                    "description": "Microfluidic channel width in mm (0.1-1mm). Default 0.3.",
                    "minimum": 0.1,
                    "maximum": 1.0,
                },
                "channel_depth_mm": {
                    "type": "number",
                    "description": "Channel depth in mm (0.05-0.5mm). Default 0.1.",
                    "minimum": 0.05,
                    "maximum": 0.5,
                },
                "chamber_size": {
                    "type": "object",
                    "description": "Tissue chamber dimensions in mm (x, y, z).",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "chamber_count": {
                    "type": "integer",
                    "description": "Number of tissue chambers (1-5). Default 2.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "inlet_count": {
                    "type": "integer",
                    "description": "Number of inlet channels (1-5). Default 2.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "chip_size": {
                    "type": "object",
                    "description": "Overall chip dimensions in mm (x, y, z).",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (6-16). Default 8.",
                    "minimum": 6,
                    "maximum": 16,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_gradient_scaffold",
        "description": "Generate a scaffold with continuous porosity gradient. Linear, exponential, or sigmoid gradients along specified axis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dimensions": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm (x, y, z).",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "gradient_direction": {
                    "type": "string",
                    "description": "Gradient axis: 'x', 'y', or 'z'. Default 'z'.",
                    "enum": ["x", "y", "z"],
                },
                "start_porosity": {
                    "type": "number",
                    "description": "Porosity at start (0.0-1.0). 0=solid, 1=empty. Default 0.2.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "end_porosity": {
                    "type": "number",
                    "description": "Porosity at end (0.0-1.0). Default 0.8.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "gradient_type": {
                    "type": "string",
                    "description": "Gradient function: 'linear', 'exponential', or 'sigmoid'. Default 'linear'.",
                    "enum": ["linear", "exponential", "sigmoid"],
                },
                "pore_base_size_mm": {
                    "type": "number",
                    "description": "Base pore diameter in mm (0.2-2mm). Default 0.5.",
                    "minimum": 0.2,
                    "maximum": 2.0,
                },
                "grid_spacing_mm": {
                    "type": "number",
                    "description": "Distance between pore centers in mm (0.5-5mm). Default 1.5.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-24). Default 12.",
                    "minimum": 8,
                    "maximum": 24,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_perfusable_network",
        "description": "Generate a perfusable vascular network following Murray's law. Use for thick tissue constructs requiring internal vascularization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "inlet_diameter_mm": {
                    "type": "number",
                    "description": "Inlet vessel diameter in mm (0.5-5mm). Default 2.0.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                },
                "branch_generations": {
                    "type": "integer",
                    "description": "Branching depth levels (1-5). Default 3.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "murray_ratio": {
                    "type": "number",
                    "description": "Murray's law ratio (child/parent radius). Default 0.79.",
                    "minimum": 0.6,
                    "maximum": 0.9,
                },
                "network_type": {
                    "type": "string",
                    "description": "Network type: 'arterial', 'venous', or 'capillary'. Default 'arterial'.",
                    "enum": ["arterial", "venous", "capillary"],
                },
                "bounding_box": {
                    "type": "object",
                    "description": "Scaffold dimensions in mm.",
                    "properties": {
                        "x": {"type": "number", "exclusiveMinimum": 0},
                        "y": {"type": "number", "exclusiveMinimum": 0},
                        "z": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["x", "y", "z"],
                },
                "branching_angle_deg": {
                    "type": "number",
                    "description": "Angle between parent and child branches (20-60). Default 30.",
                    "minimum": 20.0,
                    "maximum": 60.0,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Mesh resolution (8-24). Default 12.",
                    "minimum": 8,
                    "maximum": 24,
                },
            },
            "required": [],
        },
    },
    {
        "name": "ask_clarification",
        "description": "Ask the user for clarification when the request is ambiguous, multiple scaffold types could work, or critical information is missing. Always prefer asking over guessing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "description": "Optional list of clickable options for the user to choose from.",
                    "items": {"type": "string"},
                },
            },
            "required": ["question"],
        },
    },
]

# Tool name to scaffold type mapping
TOOL_TO_SCAFFOLD_TYPE: dict[str, str] = {
    "generate_vascular_network": "vascular_network",
    "generate_porous_disc": "porous_disc",
    "generate_tubular_conduit": "tubular_conduit",
    "generate_lattice": "lattice",
    "generate_primitive": "primitive",
    # Skeletal (7)
    "generate_trabecular_bone": "trabecular_bone",
    "generate_osteochondral": "osteochondral",
    "generate_articular_cartilage": "articular_cartilage",
    "generate_meniscus": "meniscus",
    "generate_tendon_ligament": "tendon_ligament",
    "generate_intervertebral_disc": "intervertebral_disc",
    "generate_haversian_bone": "haversian_bone",
    # Organ (6)
    "generate_hepatic_lobule": "hepatic_lobule",
    "generate_cardiac_patch": "cardiac_patch",
    "generate_kidney_tubule": "kidney_tubule",
    "generate_lung_alveoli": "lung_alveoli",
    "generate_pancreatic_islet": "pancreatic_islet",
    "generate_liver_sinusoid": "liver_sinusoid",
    # Soft Tissue (4)
    "generate_multilayer_skin": "multilayer_skin",
    "generate_skeletal_muscle": "skeletal_muscle",
    "generate_cornea": "cornea",
    "generate_adipose": "adipose",
    # Tubular (5)
    "generate_blood_vessel": "blood_vessel",
    "generate_nerve_conduit": "nerve_conduit",
    "generate_spinal_cord": "spinal_cord",
    "generate_bladder": "bladder",
    "generate_trachea": "trachea",
    # Dental (3)
    "generate_dentin_pulp": "dentin_pulp",
    "generate_ear_auricle": "ear_auricle",
    "generate_nasal_septum": "nasal_septum",
    # Advanced Lattice (5)
    "generate_gyroid": "gyroid",
    "generate_schwarz_p": "schwarz_p",
    "generate_octet_truss": "octet_truss",
    "generate_voronoi": "voronoi",
    "generate_honeycomb": "honeycomb",
    # Microfluidic (3)
    "generate_organ_on_chip": "organ_on_chip",
    "generate_gradient_scaffold": "gradient_scaffold",
    "generate_perfusable_network": "perfusable_network",
}

# Scaffold type to default suggestions mapping
DEFAULT_SUGGESTIONS: dict[str, list[str]] = {
    "vascular_network": [
        "Increase branching levels",
        "Add more inlets",
        "Make it more organic",
        "Use uniform pattern",
    ],
    "porous_disc": [
        "Increase pore size",
        "Change to hexagonal pattern",
        "Make it thicker",
        "Increase porosity",
    ],
    "tubular_conduit": [
        "Add grooves for cell guidance",
        "Make it longer",
        "Increase wall thickness",
        "Make the lumen larger",
    ],
    "lattice": [
        "Use BCC unit cell for strength",
        "Make cells smaller",
        "Increase strut diameter",
        "Change bounding box shape",
    ],
    "primitive": [
        "Add a hole through the center",
        "Shell it out (make hollow)",
        "Try a different shape",
        "Add modifications",
    ],
    # Skeletal
    "trabecular_bone": [
        "Increase porosity",
        "Adjust strut thickness",
        "Add anisotropy for load direction",
        "Change pore size",
    ],
    "osteochondral": [
        "Adjust bone/cartilage layer ratio",
        "Change gradient type",
        "Modify transition width",
        "Increase diameter",
    ],
    "articular_cartilage": [
        "Adjust zone thicknesses",
        "Change pore gradient",
        "Increase overall thickness",
        "Modify zone ratios",
    ],
    "meniscus": [
        "Adjust wedge angle",
        "Change fiber diameter",
        "Modify arc extent",
        "Add more zones",
    ],
    "tendon_ligament": [
        "Adjust crimp amplitude",
        "Change fiber spacing",
        "Increase bundle count",
        "Modify crimp wavelength",
    ],
    "intervertebral_disc": [
        "Add more AF rings",
        "Adjust disc height",
        "Change NP diameter",
        "Modify fiber angle",
    ],
    "haversian_bone": [
        "Adjust canal spacing",
        "Change osteon pattern",
        "Modify canal diameter",
        "Increase thickness",
    ],
    # Organ
    "hepatic_lobule": [
        "Add more lobules",
        "Include sinusoids",
        "Adjust lobule size",
        "Change height",
    ],
    "cardiac_patch": [
        "Adjust fiber spacing",
        "Change alignment angle",
        "Add more layers",
        "Increase patch size",
    ],
    "kidney_tubule": [
        "Increase convolution",
        "Adjust tubule diameter",
        "Change length",
        "Modify amplitude",
    ],
    "lung_alveoli": [
        "Add more generations",
        "Change branch angle",
        "Adjust alveoli size",
        "Modify airway diameter",
    ],
    "pancreatic_islet": [
        "Add more islets",
        "Change cluster arrangement",
        "Adjust pore size",
        "Modify islet diameter",
    ],
    "liver_sinusoid": [
        "Increase fenestration density",
        "Adjust diameter",
        "Change length",
        "Modify fenestration size",
    ],
    # Soft Tissue
    "multilayer_skin": [
        "Adjust layer thicknesses",
        "Add vascular channels",
        "Change porosity gradient",
        "Increase diameter",
    ],
    "skeletal_muscle": [
        "Change pennation angle",
        "Adjust fiber diameter",
        "Switch architecture type",
        "Modify dimensions",
    ],
    "cornea": [
        "Adjust curvature",
        "Change thickness",
        "Add more lamellae",
        "Modify diameter",
    ],
    "adipose_tissue": [
        "Adjust cell size",
        "Add vascular channels",
        "Change spacing",
        "Modify dimensions",
    ],
    # Tubular
    "blood_vessel": [
        "Add bifurcation",
        "Adjust wall thickness",
        "Change diameter",
        "Increase length",
    ],
    "nerve_conduit": [
        "Add more channels",
        "Adjust channel diameter",
        "Change length",
        "Modify outer diameter",
    ],
    "spinal_cord": [
        "Add more channels",
        "Adjust cord diameter",
        "Change length",
        "Modify gray matter ratio",
    ],
    "bladder": [
        "Adjust dome curvature",
        "Change wall thickness",
        "Add more layers",
        "Increase diameter",
    ],
    "trachea": [
        "Add more rings",
        "Adjust ring spacing",
        "Change gap angle",
        "Modify diameter",
    ],
    # Dental
    "dentin_pulp": [
        "Adjust dentin thickness",
        "Change crown/root ratio",
        "Modify pulp chamber size",
        "Increase overall size",
    ],
    "ear_auricle": [
        "Adjust thickness",
        "Change helix definition",
        "Modify scale factor",
        "Add antihelix ridge",
    ],
    "nasal_septum": [
        "Change curvature",
        "Adjust thickness",
        "Switch curve type",
        "Modify dimensions",
    ],
    # Advanced Lattice
    "gyroid": [
        "Adjust cell size",
        "Change wall thickness",
        "Modify bounding box",
        "Alter isovalue",
    ],
    "schwarz_p": [
        "Adjust cell size",
        "Change wall thickness",
        "Modify bounding box",
        "Alter isovalue",
    ],
    "octet_truss": [
        "Adjust strut diameter",
        "Change cell size",
        "Modify bounding box",
        "Increase resolution",
    ],
    "voronoi": [
        "Change cell count",
        "Adjust strut diameter",
        "Modify seed",
        "Change bounding box",
    ],
    "honeycomb": [
        "Adjust cell size",
        "Change wall thickness",
        "Modify height",
        "Alter bounding box",
    ],
    # Microfluidic
    "organ_on_chip": [
        "Add more chambers",
        "Adjust channel width",
        "Change chip size",
        "Add inlet channels",
    ],
    "gradient_scaffold": [
        "Change gradient direction",
        "Adjust porosity range",
        "Switch gradient type",
        "Modify dimensions",
    ],
    "perfusable_network": [
        "Add more generations",
        "Adjust inlet diameter",
        "Change network type",
        "Modify branching angle",
    ],
}

# General suggestions for chat-only responses
GENERAL_SUGGESTIONS: list[str] = [
    "Create a vascular network",
    "Make a porous disc",
    "Design a tubular conduit",
    "Generate a lattice scaffold",
    "Start with a simple shape",
    "Build a bone scaffold",
    "Create an organ-specific scaffold",
    "Design a multi-layer tissue scaffold",
    "Generate a TPMS structure",
    "Make a microfluidic device",
]
