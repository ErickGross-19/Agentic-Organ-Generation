# Agentic Scaffolding

LLM-powered scaffold generation tools for creating vascular network scaffolds.

## Files

### scaffold_agent.py
Command-line agent for generating scaffold STL files from natural language descriptions.

```bash
# With LLM (requires ANTHROPIC_API_KEY)
python scaffold_agent.py "Create a dense vascular network with 9 inlets"

# Direct params (no LLM needed)
python scaffold_agent.py --params '{"inlets": 9, "levels": 4, "tips_down": true}'

# Interactive mode
python scaffold_agent.py --interactive
```

### scaffold_agent_web.py
Web-based interactive scaffold designer with 3D preview and conversational interface.

```bash
python scaffold_agent_web.py
# Opens at http://localhost:8081
```

**Features:**
- Natural language input: "3 inlets", "4 layers", "2mm tall"
- Relative modifiers: "thicker", "taller", "broader", "curvier"
- Style keywords: "organic", "uniform", "tips down"
- Live 3D preview with rotation
- STL export

### scaffold_web_collision.py
Original web-based scaffold generator with slider controls and collision detection.

```bash
python scaffold_web_collision.py
# Opens at http://localhost:8080
```

## Requirements

```bash
pip install numpy manifold3d pyvista trame trame-vuetify trame-vtk anthropic
```

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| inlets | 1-25 | Number of inlet ports |
| levels | 0-8 | Branching depth (layers) |
| splits | 1-6 | Branches per junction |
| spread | 0.1-0.8 | Horizontal spread |
| ratio | 0.5-0.95 | Child/parent radius ratio |
| curvature | 0-1 | Branch curvature |
| tips_down | bool | Tips point straight down |
| deterministic | bool | No randomness |
