# How to Run the Updated MorphoStruct and Create an Integrated Repo

## Option 1: Run the Updated MorphoStruct (In-Place)

This runs the updated code directly from your existing directories.

### Step 1: Install Backend Dependencies

```bash
cd C:\Users\Erick\MorphoStruct\backend

# Create virtual environment
python -m venv venv

# Activate it (Git Bash)
source venv/Scripts/activate

# Install dependencies (including AOG)
pip install -r requirements.txt
```

**Important:** The `requirements.txt` now includes:
```
-e C:\Users\Erick\organ-agent-generation\repo
```

This installs AOG as an editable package from your local directory.

### Step 2: Install Frontend Dependencies

```bash
cd C:\Users\Erick\MorphoStruct\frontend
npm install
```

**Note:** You might need to install the Tabs component if it's missing:
```bash
npx shadcn-ui@latest add tabs
```

### Step 3: Start Backend (Terminal 1)

```bash
cd C:\Users\Erick\MorphoStruct\backend
source venv/Scripts/activate
python -m app.main
```

You should see: `Uvicorn running on http://0.0.0.0:8000`

### Step 4: Start Frontend (Terminal 2)

```bash
cd C:\Users\Erick\MorphoStruct\frontend
npm run dev
```

You should see: `Local: http://localhost:3000`

### Step 5: Test It!

1. Open browser: `http://localhost:3000`
2. Go to the generator/scaffold design page
3. Select "Space Colonization" from dropdown
4. Click "Generate"
5. You should see a 3D vascular network!

---

## Option 2: Create an Integrated Repository (Recommended)

This creates a **new, standalone repository** with everything in one place.

### Why Create an Integrated Repo?

✅ **Single repository** - easier to manage and share
✅ **Self-contained** - AOG included as a local package
✅ **Comprehensive docs** - README, quickstart, troubleshooting
✅ **Git-ready** - initialized with proper .gitignore
✅ **Easy deployment** - simple setup script included

### How to Create It

#### Step 1: Run the Creation Script

```bash
cd C:\Users\Erick

# Make script executable
chmod +x create_integrated_repo.sh

# Run it (creates repo at default location)
./create_integrated_repo.sh

# OR specify custom location:
./create_integrated_repo.sh ~/projects/my-vascular-tool
```

The script will:
- ✅ Copy MorphoStruct backend and frontend
- ✅ Copy AOG library to `backend/aog/`
- ✅ Update `requirements.txt` to use local AOG
- ✅ Create comprehensive README.md
- ✅ Create QUICKSTART.md guide
- ✅ Create setup.sh automation script
- ✅ Create .gitignore
- ✅ Initialize git repository (optional)

#### Step 2: Navigate to New Repo

```bash
cd C:\Users\Erick\morphostruct-aog-integrated
# OR your custom path
```

#### Step 3: Run Setup Script

```bash
./setup.sh
```

This automatically:
- Creates Python virtual environment
- Installs all backend dependencies
- Installs AOG from `backend/aog/`
- Installs all frontend dependencies

#### Step 4: Start the Application

**Terminal 1 (Backend):**
```bash
cd backend
source venv/Scripts/activate
python -m app.main
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

#### Step 5: Test It!

Open browser: `http://localhost:3000`

#### Step 6: Push to GitHub (Optional)

```bash
# Create a new repository on GitHub (via web interface)
# Then:

git remote add origin https://github.com/yourusername/morphostruct-aog.git
git push -u origin main
```

Now you have a public repository you can share!

---

## Comparison: In-Place vs Integrated Repo

| Feature | In-Place | Integrated Repo |
|---------|----------|-----------------|
| Setup Speed | Fast | Medium (one-time) |
| Organization | Two separate dirs | One directory |
| AOG Dependency | Absolute path | Local package |
| Portability | Not portable | Fully portable |
| Documentation | Scattered | Comprehensive |
| Git-ready | No | Yes |
| Shareable | Difficult | Easy |
| Deployment | Manual | Automated |

**Recommendation:** Use **Integrated Repo** if you want to:
- Share with others
- Deploy to a server
- Keep everything organized
- Have comprehensive documentation

Use **In-Place** if you just want to:
- Quickly test the implementation
- Continue development in existing setup

---

## What Each File Does

### Files Created on Your Computer

**`C:\Users\Erick\SETUP_INSTRUCTIONS.md`**
- Detailed setup guide for running in-place
- Troubleshooting section
- Performance tips

**`C:\Users\Erick\MORPHOSTRUCT_AOG_IMPLEMENTATION_SUMMARY.md`**
- Complete technical documentation
- Architecture diagrams
- API usage examples
- Future enhancement roadmap

**`C:\Users\Erick\create_integrated_repo.sh`**
- Automated script to create integrated repository
- Copies files, updates configs, creates docs

**`C:\Users\Erick\HOW_TO_RUN_AND_CREATE_REPO.md`** (this file)
- Simple guide for getting started

### Files Modified in MorphoStruct

**Backend:**
- `backend/requirements.txt` - Added AOG dependencies
- `backend/app/vascular/` - NEW: AOG integration package
- `backend/app/models/scaffold.py` - Added 2 new scaffold types
- `backend/app/api/scaffolds.py` - Added endpoints + presets

**Frontend:**
- `frontend/lib/types/scaffolds/` - Added vascular types
- `frontend/components/controls/` - Added AdvancedVascularControls

---

## Troubleshooting

### "python command not found"

Try:
```bash
python3 --version
# OR
py --version
```

Use whichever works for subsequent commands.

### "Module 'generation' not found"

AOG not installed correctly. Try:
```bash
cd C:\Users\Erick\organ-agent-generation\repo
pip install -e .
```

### "Module '@/components/ui/tabs' not found"

Install Tabs component:
```bash
cd frontend
npx shadcn-ui@latest add tabs
```

### "Port 8000 already in use"

Kill the process:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
uvicorn app.main:app --port 8001
```

### "Scaffold type not appearing"

1. Check backend is running: `curl http://localhost:8000/api/health`
2. Check browser console (F12) for errors
3. Restart both servers
4. Clear browser cache

---

## Next Steps After Setup

### 1. Explore Parameters

**Space Colonization:**
- Try adding multiple inlets (3-5)
- Test different multi-inlet modes (blended, partitioned, forest)
- Adjust bifurcation probability

**Bifurcating Tree:**
- Try ternary branching (3 branches/node)
- Test Murray's law vs linear radius mode
- Add variation

### 2. Create Custom Presets

Edit `backend/app/api/scaffolds.py` and add to the `PRESETS` list:

```python
PresetInfo(
    id="my_custom_network",
    name="My Custom Vascular Network",
    type=ScaffoldType.SPACE_COLONIZATION,
    description="Custom configuration",
    category="vascular",
    params={
        # Your custom parameters
    },
),
```

### 3. Export and 3D Print

1. Generate a scaffold
2. Click "Export STL"
3. Import into slicing software (Cura, PrusaSlicer)
4. Adjust print settings for bioprinting
5. Print!

### 4. Integrate into Your Workflow

- Export meshes for FEA analysis
- Import into CAD software
- Use as templates for tissue engineering
- Create libraries of common configurations

---

## Getting Help

1. **Read the docs:**
   - `SETUP_INSTRUCTIONS.md` - Detailed setup
   - `MORPHOSTRUCT_AOG_IMPLEMENTATION_SUMMARY.md` - Technical details
   - Integrated repo's `README.md` - Complete guide

2. **Check logs:**
   - Backend: Terminal output
   - Frontend: Browser console (F12)

3. **Verify installation:**
   ```bash
   # Backend
   cd backend
   source venv/Scripts/activate
   python -c "from generation.backends.space_colonization_backend import SpaceColonizationBackend; print('OK')"

   # Frontend
   cd frontend
   npm run type-check
   ```

---

## Summary

### To Run Updated Code (In-Place):
```bash
# Terminal 1: Backend
cd C:\Users\Erick\MorphoStruct\backend
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python -m app.main

# Terminal 2: Frontend
cd C:\Users\Erick\MorphoStruct\frontend
npm install
npm run dev

# Browser
Open http://localhost:3000
```

### To Create Integrated Repo:
```bash
cd C:\Users\Erick
./create_integrated_repo.sh
cd morphostruct-aog-integrated
./setup.sh

# Then start as above
```

**You're all set! 🚀**
