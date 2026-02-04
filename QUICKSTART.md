# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Run the setup script
./setup.sh
```

### 2. Start Backend

```bash
cd backend
source venv/Scripts/activate  # Windows Git Bash: venv/Scripts/activate
                               # Windows CMD: venv\Scripts\activate
                               # macOS/Linux: source venv/bin/activate
python -m app.main
```

Keep this terminal running.

### 3. Start Frontend

Open a **new terminal**:

```bash
cd frontend
npm run dev
```

### 4. Open Browser

Navigate to: `http://localhost:3000`

### 5. Generate Your First Vascular Network

1. Go to the generator page
2. Select "Space Colonization" from dropdown
3. Click "Generate"
4. Wait ~10 seconds
5. **See network statistics overlay in top-left!** ★NEW
6. See your 3D vascular network!

### 6. Try a Preset

1. Look for "Presets" dropdown
2. Select "Single Inlet Vascular Network"
3. Click "Generate"

### 7. Export

1. Click "Export STL"
2. Open in 3D viewer
3. Ready to 3D print!

---

## What's New in This Version

### ✨ Vascular Statistics Overlay (Phase 3)

When you generate vascular scaffolds, you now see:
- 🔷 Network nodes and segments
- 📏 Total vessel length
- 📐 Min/max vessel radius
- 🎯 Terminal node count
- ⚙️ Configuration details
- 🔺 Mesh quality metrics

The overlay appears automatically in the top-left corner of the viewport!

---

## New Scaffold Types

### Space Colonization
- Organic vascular growth
- Multi-inlet support
- 50,000+ attraction points
- Blended/partitioned/forest modes

### Bifurcating Tree
- Regular geometric trees
- 1-10 branching levels
- Murray's law optimization
- Optional variation

---

## Next Steps

- Read [START_HERE.md](START_HERE.md) for overview
- See [docs/HOW_TO_RUN_AND_CREATE_REPO.md](docs/HOW_TO_RUN_AND_CREATE_REPO.md) for detailed setup
- Check [docs/SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md) for troubleshooting
- Read [docs/PHASE3_VISUALIZATION_COMPLETE.md](docs/PHASE3_VISUALIZATION_COMPLETE.md) for overlay details

**Enjoy creating vascular scaffolds! 🧬**
