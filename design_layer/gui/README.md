# GUI Module

Desktop and web interfaces for the Agentic Organ Generation system.

## Web Interface (Primary)

The primary interface is the **Next.js frontend** in `frontend/` paired with the **FastAPI backend** in `backend/app/`. This provides:

- **Two modes**: Direct Generate (scaffold) and DesignSpec Agent (conversational spec editing)
- **44 scaffold types** organized into 9 categories with parameter panels
- **3D viewer** using Three.js for real-time mesh visualization
- **Chat panel** for LLM-driven DesignSpec editing with patch approval UI
- **SpecViewer** showing current DesignSpec JSON with expand/collapse
- **STL export** for generated scaffolds

### Quick Start (Web)

```bash
# Backend
cd backend && pip install -r requirements.txt && python run.py

# Frontend
cd frontend && npm install && npm run dev
```

Then open http://localhost:3000.

See `frontend/README.md` and `backend/README.md` for detailed setup.

## Desktop GUI (Legacy)

The tkinter-based desktop GUI is still available but is superseded by the web interface.

### Components

| Module | Purpose |
|--------|---------|
| `app.py` | Main entry point (`launch_gui()`) |
| `configuration_wizard.py` | Setup wizard for LLM provider, project name, workflow mode |
| `designspec_workflow_manager.py` | DesignSpec workflow orchestration with patch/run approval |
| `workflow_manager.py` | V5/V4 workflow orchestration |
| `stl_viewer.py` | 3D STL visualization using trimesh + matplotlib |
| `agent_config.py` | LLM provider configuration panel |
| `security.py` | Encrypted API key storage (keyring or PBKDF2 fallback) |

### Launch

```bash
python main.py          # Launch GUI
python -m gui           # As module
python main.py --cli    # CLI mode instead
```

### DesignSpec Workflow Manager

The `DesignSpecWorkflowManager` bridges the DesignSpec pipeline to the GUI:

- Project creation and loading
- Patch proposal and approval flow (GUI panels for spec, patches, run, artifacts)
- Compile and run controls with event-based callbacks
- States: IDLE, PROCESSING, WAITING_PATCH_APPROVAL, WAITING_RUN_APPROVAL, RUNNING, ERROR

This same workflow manager is wrapped by the FastAPI `DesignSpecBridge` for the web interface.

### Configuration Storage

Desktop GUI config is stored in `~/.organ_generator/`:
- `config.json` - General settings
- `credentials.enc` - Encrypted API keys

## Architecture

```
gui/
├── __init__.py
├── __main__.py
├── app.py                         # Desktop GUI entry point
├── configuration_wizard.py        # Setup wizard
├── designspec_workflow_manager.py  # DesignSpec workflow (shared by web + desktop)
├── workflow_manager.py            # V5/V4 workflow orchestration
├── stl_viewer.py                  # 3D STL viewer (desktop)
├── agent_config.py                # LLM config panel (desktop)
├── security.py                    # Encrypted key storage
└── _legacy/                       # Deprecated components
```
