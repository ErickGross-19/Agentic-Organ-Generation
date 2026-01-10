# Organ Generator GUI

A graphical user interface for the Agentic Organ Generation system. Provides workflow selection, agent configuration, and 3D STL visualization.

## Features

The GUI wraps the existing command-line workflows with a professional interface:

**Workflow Selection** - Single Agent Organ Generator V4 with interactive, topology-first questioning.

**Agent Configuration** - Configure LLM providers including OpenAI, Anthropic, Google Gemini, Mistral, xAI (Grok), Groq, and local OpenAI-compatible endpoints. API keys are stored securely with machine-specific encryption.

**Three-Panel Layout** - The main interface provides a chat panel for workflow interaction, an output panel for logs and status, and an integrated STL viewer for visualizing generated meshes.

**STL Viewer** - 3D visualization using trimesh and matplotlib with rotation, zoom, wireframe/solid toggle, and mesh statistics display.

## Requirements

The GUI requires Python 3.8+ with tkinter support (included with standard Python installations on most desktop systems). Additional dependencies:

```
matplotlib>=3.5.0
trimesh>=3.10.0
numpy>=1.20.0
```

Optional for system keyring integration:
```
keyring>=23.0.0
```

## Usage

### Launch the GUI

```bash
# From the repository root
python main.py

# Or as a module
python -m gui
```

### Command Line Options

```bash
python main.py --help     # Show help
python main.py --version  # Show version
python main.py --cli      # Run CLI mode instead of GUI
```

### Quick Start

1. Launch the GUI with `python main.py`
2. Click "Select Workflow" to start the Single Agent workflow
3. Click "Agent Config" to configure your LLM provider and API key
4. Click "Start" to begin the workflow
5. Interact with the workflow through the chat panel
6. View generated STL files in the integrated viewer

## Components

### MainWindow (`main_window.py`)

The primary application window with:
- Menu bar (File, Workflow, Help)
- Toolbar with workflow controls
- Three-panel layout (Chat, Output, STL Viewer)
- Status bar with progress indicator
- Keyboard shortcuts (Ctrl+N for new workflow, Ctrl+Q to quit)

### WorkflowManager (`workflow_manager.py`)

Orchestrates workflow execution with:
- Thread-safe message passing between workflow and GUI
- User input handling via queue-based communication
- Support for SingleAgentOrganGeneratorV4 workflow

### STLViewer (`stl_viewer.py`)

3D visualization component with:
- STL file loading via trimesh
- 3D rendering via matplotlib
- Wireframe/solid display toggle
- Mesh statistics (vertices, faces, size, volume, watertight status)
- Image export (PNG, PDF, SVG)

### AgentConfigPanel (`agent_config.py`)

LLM configuration panel with:
- Provider selection dropdown
- Model selection (provider-specific options)
- API key input with show/hide toggle
- Secure key storage
- Temperature and max tokens settings

### SecureConfig (`security.py`)

Secure configuration storage with:
- System keyring integration (when available)
- Fallback to encrypted file storage
- Machine-specific key derivation using PBKDF2
- Separate storage for API keys and general config

## Building an Executable

Use PyInstaller to create a standalone executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller build.spec

# The executable will be in dist/OrganGenerator/
```

The `build.spec` file is pre-configured to include all necessary modules and dependencies.

## Architecture

```
gui/
├── __init__.py          # Module exports with graceful fallback
├── __main__.py          # Entry point for python -m gui
├── main_window.py       # Primary application window
├── workflow_manager.py  # Workflow orchestration
├── stl_viewer.py        # 3D STL visualization
├── agent_config.py      # LLM configuration panel
├── security.py          # Encrypted API key storage
└── README.md            # This file
```

## Configuration Storage

Configuration is stored in `~/.organ_generator/`:
- `config.json` - General settings (window geometry, last provider, etc.)
- `credentials.enc` - Encrypted API keys (when keyring unavailable)

## Troubleshooting

**"No module named '_tkinter'"** - Your Python installation doesn't include tkinter. On Ubuntu/Debian: `sudo apt-get install python3-tk`. On macOS with Homebrew: `brew install python-tk`.

**"matplotlib not available"** - Install matplotlib: `pip install matplotlib`. The STL viewer will show a fallback message without it.

**"trimesh library not available"** - Install trimesh: `pip install trimesh`. Required for loading STL files.

**API key not saving** - Check that `~/.organ_generator/` directory is writable. The GUI will fall back to file-based encrypted storage if system keyring is unavailable.
