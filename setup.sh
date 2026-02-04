#!/bin/bash
# Quick setup script for MorphoStruct + AOG

set -e

echo "🚀 Setting up MorphoStruct + AOG..."

# Backend
echo "📦 Installing backend dependencies..."
cd backend
python -m venv venv
source venv/Scripts/activate || source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# Frontend
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Terminal 1: cd backend && source venv/Scripts/activate && python -m app.main"
echo "  2. Terminal 2: cd frontend && npm run dev"
echo "  3. Open http://localhost:3000"
