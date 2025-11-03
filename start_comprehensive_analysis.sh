#!/bin/bash
# Quick Start Script for Comprehensive Shot Analysis API

set -e

echo "🏀 Basketball AI - Comprehensive Shot Analysis"
echo "=============================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
cd "$(dirname "$0")"
pip install -q opencv-python-headless mediapipe numpy scipy fastapi uvicorn python-multipart aiofiles

# Verify imports
echo ""
echo "✅ Verifying service imports..."
python3 -c "
from services.pose_processor import PoseProcessor
from services.phase_detector import PhaseDetector
from services.metrics_calculator import MetricsCalculator
from services.shot_analysis_service import ShotAnalysisService
print('✅ All services imported successfully')
"

# Check baseline data
echo ""
echo "📁 Checking baseline data..."
if [ -f "baselines/stephen_curry.json" ]; then
    echo "✅ Stephen Curry baseline found"
else
    echo "⚠️  Stephen Curry baseline not found"
fi

# Start server
echo ""
echo "🚀 Starting FastAPI server..."
echo ""
echo "📡 API Endpoints:"
echo "  - POST /analyze/comprehensive"
echo "  - GET  /baselines/available"
echo "  - GET  /baselines/{player_name}"
echo ""
echo "📖 Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 main.py
