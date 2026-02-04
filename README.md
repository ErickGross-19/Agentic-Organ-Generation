# MorphoStruct + AOG: Advanced Vascular Scaffold Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)

**Integrated repository combining MorphoStruct web platform with AOG (Agentic Organ Generation) vascular algorithms.**

Generate anatomically-accurate 3D vascular scaffolds using:
- 🌿 **Space Colonization**: Organic vascular growth driven by tissue attraction
- 🌳 **Bifurcating Trees**: Regular geometric trees with Murray's law optimization

---

## 🎯 What's New

This integrated version adds **2 new scaffold types** powered by AOG:

### Space Colonization
- Multi-inlet organic vascular networks
- 50,000+ attraction points for realistic angiogenesis
- Blended/partitioned/forest multi-inlet modes
- Configurable bifurcation probability
- Murray's law radius tapering

### Bifurcating Tree
- Regular binary/ternary branching trees
- 1-10 branching levels
- Murray's law, linear, or fixed radius modes
- Optional random variation
- Length and radius tapering

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- 8GB RAM (16GB recommended)

### Installation

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd morphostruct-aog

# 2. Backend setup
cd backend
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# OR: venv\Scripts\activate    # Windows CMD
# OR: source venv/bin/activate # macOS/Linux

pip install -r requirements.txt

# 3. Frontend setup (in new terminal)
cd frontend
npm install

# 4. Start backend (terminal 1)
cd backend
source venv/Scripts/activate
python -m app.main
# → http://0.0.0.0:8000

# 5. Start frontend (terminal 2)
cd frontend
npm run dev
# → http://localhost:3000
```

### First Vascular Network

1. Navigate to `http://localhost:3000/generator`
2. Select **"Space Colonization"** from scaffold type dropdown
3. Configure parameters or use defaults
4. Click **"Generate"**
5. Wait 10-20 seconds
6. View your vascular network in 3D!
7. Export to STL

---

## 📚 Features

### Space Colonization
- **Multi-inlet support**: Add/remove inlets via UI
- **Organic growth**: Mimics natural angiogenesis
- **Blended mode**: Smooth merging of inlet territories
- **Configurable bifurcation**: Control branching probability
- **Radius tapering**: Murray's law optimization

### Bifurcating Tree
- **Flexible branching**: 2-5 branches per node
- **Murray's law**: Optimal flow distribution
- **Variation modes**: Add natural randomness
- **Geometric control**: Precise angle and length control
- **Fast generation**: <2 seconds for 5 levels

### 4 Built-in Presets
- Single Inlet Vascular Network
- 5-Inlet Vascular Network (cross pattern)
- Standard Bifurcating Tree (binary, Murray's law)
- Ternary Bifurcating Tree (3-way, with variation)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│     MORPHOSTRUCT WEB APP                │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────┐   ┌───────────────┐  │
│  │  Frontend    │   │  Backend      │  │
│  │  (Next.js)   │◄──┤  (FastAPI)    │  │
│  │              │   │               │  │
│  │ • React UI   │   │ • Vascular    │  │
│  │ • Three.js   │   │   Endpoints   │  │
│  │ • Controls   │   │ • AOG Wrapper │  │
│  └──────────────┘   └───────┬───────┘  │
│                             │          │
│                             ▼          │
│                  ┌──────────────────┐  │
│                  │  AOG Library     │  │
│                  │  (backend/aog/)  │  │
│                  │                  │  │
│                  │ • Space          │  │
│                  │   Colonization   │  │
│                  │ • Bifurcating    │  │
│                  │   Tree           │  │
│                  │ • Mesh Synthesis │  │
│                  └──────────────────┘  │
└─────────────────────────────────────────┘
```

---

## 📖 Documentation

- **[SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md)**: Detailed setup guide
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)**: Technical implementation details
- **[API_REFERENCE.md](docs/API_REFERENCE.md)**: REST API documentation
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues and solutions

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Next.js 14, React 18, TypeScript, Three.js, Tailwind CSS |
| **Backend** | FastAPI, Python 3.10+, Manifold3D, NumPy |
| **Vascular** | AOG (Space Colonization, Bifurcating Trees) |
| **Mesh** | Trimesh, Manifold3D |
| **AI** | Anthropic Claude, OpenAI GPT (optional) |

---

## 🎨 Usage Examples

### Via Web UI
1. Select scaffold type: "Space Colonization"
2. Add multiple inlets (click "Add Inlet")
3. Set attraction points: 50,000
4. Set iterations: 300
5. Choose multi-inlet mode: "Blended"
6. Click "Generate"
7. Export to STL

### Via API

**Space Colonization:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "space_colonization",
    "params": {
      "num_attractors": 50000,
      "max_iterations": 300,
      "inlets": [{
        "position": [0.0, 0.0, 0.001],
        "radius": 0.0002,
        "direction": [0.0, 0.0, -1.0]
      }]
    }
  }'
```

**Bifurcating Tree:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "bifurcating_tree",
    "params": {
      "branching_levels": 5,
      "branches_per_node": 2,
      "radius_mode": "murray"
    }
  }'
```

---

## ⚡ Performance

### Expected Generation Times

| Scaffold Type | Parameters | Time |
|--------------|------------|------|
| Space Colonization | 10k attractors | ~5s |
| Space Colonization | 50k attractors | ~15s |
| Space Colonization | 100k attractors | ~40s |
| Bifurcating Tree | 5 levels, binary | <1s |
| Bifurcating Tree | 7 levels, binary | ~3s |
| Bifurcating Tree | 8 levels, ternary | ~10s |

### Optimization Tips
- Use **Preview Mode** for fast iteration
- Start with 10k attractors for testing
- Reduce radial resolution (8) for previews
- Use 12-16 resolution for final exports

---

## 🔧 Development

### Project Structure
```
morphostruct-aog/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI endpoints
│   │   ├── models/        # Pydantic models
│   │   ├── geometry/      # Geometry generators
│   │   └── vascular/      # AOG integration
│   ├── aog/               # AOG library (local)
│   │   ├── generation/    # Core algorithms
│   │   └── aog_policies/  # Generation policies
│   └── requirements.txt
├── frontend/
│   ├── components/
│   │   └── controls/      # Parameter controls
│   ├── lib/
│   │   └── types/         # TypeScript types
│   └── package.json
└── docs/
```

### Adding New Features
1. Backend: Add generator to `backend/app/vascular/`
2. Models: Add params to `backend/app/models/scaffold.py`
3. API: Add endpoint to `backend/app/api/scaffolds.py`
4. Frontend: Add types to `frontend/lib/types/scaffolds/`
5. UI: Add controls to `frontend/components/controls/`

---

## 🐛 Troubleshooting

### Backend won't start
- Check Python version: `python --version` (need 3.10+)
- Verify venv activated: `which python` should show venv path
- Check AOG installation: `pip list | grep aog`

### Frontend won't start
- Check Node version: `node --version` (need 18+)
- Clear cache: `rm -rf .next`
- Reinstall: `rm -rf node_modules && npm install`

### Scaffold types not appearing
- Verify backend running: `curl http://localhost:8000/api/health`
- Check browser console for errors (F12)
- Restart both frontend and backend

### Generation fails
- Reduce `num_attractors` to 10,000
- Check backend logs for errors
- Verify inlet positions inside domain
- Try a preset first

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

---

## 📊 Stats & Metrics

### Scaffold Library
- **Total Scaffolds:** 41 types
- **Categories:** 9 (Skeletal, Organ, Soft Tissue, Vascular, etc.)
- **Vascular Types:** 5 (Legacy + 2 AOG-powered)

### Vascular Capabilities
- **Max Attractors:** 500,000
- **Max Inlets:** 10+
- **Max Branching Levels:** 10
- **Mesh Resolution:** 6-32 segments/circle

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **MorphoStruct**: Original scaffold design platform
- **AOG**: Advanced vascular generation algorithms
- **Manifold3D**: Robust geometry kernel
- **Trimesh**: Mesh processing library

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check documentation in `docs/`
- Read troubleshooting guide

---

**Happy scaffold designing! 🧬**
