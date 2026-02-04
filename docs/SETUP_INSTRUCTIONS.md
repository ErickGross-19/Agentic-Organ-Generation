# MorphoStruct + AOG Setup Instructions

## Quick Start Guide

Follow these steps to get the updated MorphoStruct with AOG vascular generation running on your computer.

---

## Prerequisites

### Required Software
- **Python 3.10+** (check: `python --version` or `python3 --version`)
- **Node.js 18+** (check: `node --version`)
- **npm** or **pnpm** (check: `npm --version`)
- **Git** (check: `git --version`)

### System Requirements
- **OS:** Windows, macOS, or Linux
- **RAM:** 8GB minimum (16GB recommended for large networks)
- **Disk:** 2GB free space

---

## Step 1: Verify Directory Structure

Ensure both repositories exist:

```bash
# Check MorphoStruct
ls C:\Users\Erick\MorphoStruct

# Check AOG
ls C:\Users\Erick\organ-agent-generation\repo
```

You should see:
- `MorphoStruct/` with `backend/` and `frontend/` directories
- `organ-agent-generation/repo/` with `generation/` and `aog_policies/` directories

---

## Step 2: Backend Setup

### A. Create Virtual Environment (Recommended)

```bash
cd C:\Users\Erick\MorphoStruct\backend

# Create virtual environment
python -m venv venv

# Activate it
# On Windows (Git Bash):
source venv/Scripts/activate
# On Windows (CMD):
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### B. Install Dependencies

```bash
# Make sure you're in the backend directory with venv activated
cd C:\Users\Erick\MorphoStruct\backend

# Install requirements
pip install -r requirements.txt

# This will install:
# - FastAPI, uvicorn, manifold3d, numpy, etc.
# - trimesh, scipy, scikit-learn (for AOG)
# - AOG package as editable install from local path
```

### C. Verify AOG Installation

```bash
# Test AOG imports
python -c "from generation.backends.space_colonization_backend import SpaceColonizationBackend; print('✅ Space Colonization OK')"
python -c "from generation.backends.scaffold_topdown_backend import ScaffoldTopDownBackend; print('✅ Bifurcating Tree OK')"
python -c "from aog_policies import MeshSynthesisPolicy; print('✅ Policies OK')"
```

If you see errors, the AOG path in `requirements.txt` might need adjustment. Check:
```bash
grep "organ-agent-generation" requirements.txt
```

Should show:
```
-e C:\Users\Erick\organ-agent-generation\repo
```

### D. Test Backend

```bash
# Start the FastAPI server
cd C:\Users\Erick\MorphoStruct\backend
python -m app.main

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

Keep this terminal open!

---

## Step 3: Frontend Setup

### A. Install Node Dependencies

Open a **new terminal** (keep backend running):

```bash
cd C:\Users\Erick\MorphoStruct\frontend

# Install dependencies
npm install

# This will install React, Next.js, Three.js, and UI components
```

### B. Check for Missing UI Components

The implementation uses these UI components:
- `Button`, `Label`, `Slider`, `Switch`, `Input`, `Select`
- **`Tabs`** (TabsContent, TabsList, TabsTrigger)

If `Tabs` component doesn't exist, you'll need to add it:

```bash
# Check if shadcn/ui is configured
cat components.json

# If Tabs component is missing, add it:
npx shadcn-ui@latest add tabs
```

### C. Type Check

```bash
# Verify TypeScript types are correct
npm run type-check

# Should show no errors (or only warnings)
```

### D. Start Frontend

```bash
cd C:\Users\Erick\MorphoStruct\frontend

# Start dev server
npm run dev

# You should see:
# ▲ Next.js 14.x.x
# - Local:   http://localhost:3000
```

---

## Step 4: Test the Integration

### A. Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

Then go to the generator page:
```
http://localhost:3000/generator
```
(or whatever the scaffold design page route is)

### B. Test Space Colonization

1. From the scaffold type dropdown, select **"Space Colonization"**
2. You should see the `AdvancedVascularControls` component with tabs
3. Configure parameters:
   - Leave defaults or set `num_attractors` to `10000` (faster for testing)
   - Set `max_iterations` to `100`
4. Click **"Generate"**
5. Wait 5-10 seconds
6. You should see a 3D vascular network in the viewport

### C. Test Bifurcating Tree

1. Select **"Bifurcating Tree"** from dropdown
2. Configure:
   - `branching_levels`: `4`
   - `branches_per_node`: `2`
   - `radius_mode`: `"murray"`
3. Click **"Generate"**
4. Should render in <2 seconds

### D. Test Presets

1. Look for a presets dropdown or button
2. Load "Single Inlet Vascular Network" preset
3. Click "Generate"
4. Verify it works

### E. Test Export

1. After generating a scaffold, click "Export STL"
2. Download should start
3. Open the STL file in a 3D viewer (e.g., Windows 3D Viewer, MeshLab)
4. Verify the vascular network geometry looks correct

---

## Step 5: Troubleshooting

### Backend Issues

#### Error: "No module named 'generation'"

**Problem:** AOG not installed correctly.

**Solution:**
```bash
cd C:\Users\Erick\organ-agent-generation\repo
pip install -e .
```

#### Error: "ModuleNotFoundError: No module named 'trimesh'"

**Problem:** Missing dependency.

**Solution:**
```bash
pip install trimesh scipy scikit-learn
```

#### Error: "manifold3d not found"

**Problem:** manifold3d not installed.

**Solution:**
```bash
pip install manifold3d
```

#### Backend won't start: "Address already in use"

**Problem:** Port 8000 already in use.

**Solution:**
```bash
# Kill the process on port 8000 (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or start on different port
uvicorn app.main:app --port 8001
```

### Frontend Issues

#### Error: "Module not found: Can't resolve '@/components/ui/tabs'"

**Problem:** Tabs component doesn't exist.

**Solution:**
```bash
cd frontend
npx shadcn-ui@latest add tabs
```

#### Error: TypeScript errors in AdvancedVascularControls.tsx

**Problem:** UI component types don't match.

**Solution:**
Check that all imported components exist:
```bash
ls components/ui/
```

If missing, install them:
```bash
npx shadcn-ui@latest add button label slider switch input select
```

#### Frontend won't start: "Port 3000 already in use"

**Problem:** Port 3000 in use.

**Solution:**
```bash
# Kill process (Windows)
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use different port
PORT=3001 npm run dev
```

#### Error: "Scaffold type not appearing in dropdown"

**Problem:** ScaffoldType enum not updated or component not registered.

**Solution:**
1. Verify `lib/types/scaffolds/base.ts` has `SPACE_COLONIZATION` and `BIFURCATING_TREE`
2. Verify `components/controls/ParameterPanel.tsx` has conditional rendering for new types
3. Restart dev server: `npm run dev`

### Generation Issues

#### Network generates but doesn't display

**Problem:** Mesh conversion failed or Three.js rendering issue.

**Solution:**
1. Check browser console (F12) for errors
2. Verify backend response has `mesh_data` with `vertices` and `indices`
3. Check network in backend logs for errors

#### "Network generation failed" error

**Problem:** AOG generation crashed.

**Solution:**
1. Check backend terminal for Python traceback
2. Reduce `num_attractors` or `branching_levels`
3. Verify inlet positions are inside domain
4. Check backend logs in `backend/logs/` (if logging enabled)

#### Generation is very slow

**Problem:** Too many attractors or iterations.

**Solution:**
1. Use **Preview Mode** toggle (if available)
2. Reduce `num_attractors` to 10,000-20,000 for testing
3. Reduce `max_iterations` to 100-200
4. Lower `radial_resolution` to 8

---

## Performance Tips

### Fast Iteration During Development

**Space Colonization:**
- Use 10,000 attractors (generates in ~5 seconds)
- Max 100-200 iterations
- Radial resolution: 8

**Bifurcating Tree:**
- Max 5 levels
- Binary branching (2 branches/node)
- No variation

### Production Quality

**Space Colonization:**
- 50,000-100,000 attractors
- 300-500 iterations
- Radial resolution: 12-16

**Bifurcating Tree:**
- 6-7 levels
- Ternary branching (3 branches/node)
- With variation

---

## Next Steps

Once everything is working:

1. **Explore Parameters:**
   - Try multi-inlet space colonization (add 3-5 inlets)
   - Test different radius modes for bifurcating trees
   - Experiment with variation settings

2. **Create Custom Presets:**
   - Modify `backend/app/api/scaffolds.py` PRESETS list
   - Add your own parameter combinations
   - Restart backend to load new presets

3. **Integrate into Your Workflow:**
   - Export STL files
   - Import into slicing software
   - 3D print vascular scaffolds!

4. **Optional: Add Visualization (Phase 3):**
   - Create `VascularOverlay.tsx` component (see implementation plan)
   - Display network statistics
   - Add color-by-radius visualization

---

## Getting Help

If you encounter issues:

1. **Check Implementation Summary:**
   - Read `MORPHOSTRUCT_AOG_IMPLEMENTATION_SUMMARY.md`
   - Contains detailed troubleshooting section

2. **Check Logs:**
   - Backend: Terminal output or `backend/logs/`
   - Frontend: Browser console (F12)

3. **Verify File Changes:**
   - Make sure all files listed in implementation summary were modified/created
   - Check git status: `git status` in MorphoStruct directory

4. **Test AOG Standalone:**
   ```bash
   cd C:\Users\Erick\organ-agent-generation\repo
   python examples/example_space_colonization.py
   # (if such an example exists)
   ```

---

## Success Checklist

- [ ] Backend starts without errors
- [ ] Frontend starts without errors
- [ ] Can navigate to generator page
- [ ] "Space Colonization" appears in scaffold type dropdown
- [ ] "Bifurcating Tree" appears in scaffold type dropdown
- [ ] Space Colonization controls display with tabs
- [ ] Can add/remove inlets in Space Colonization UI
- [ ] Generating Space Colonization network succeeds
- [ ] Generating Bifurcating Tree succeeds
- [ ] 3D mesh displays in viewport
- [ ] Can export STL file
- [ ] Presets load and generate correctly

---

**You're all set! Enjoy creating advanced vascular scaffolds! 🎉**
