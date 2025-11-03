#!/bin/bash
# Test script for Comprehensive Shot Analysis API

echo "🧪 Testing Comprehensive Shot Analysis API"
echo "=========================================="
echo ""

BASE_URL="http://localhost:8000"

# Test 1: Check server health
echo "Test 1: Server Health Check"
echo "----------------------------"
curl -s "$BASE_URL/" | jq -r '.message, .status, .version'
echo ""

# Test 2: Get available baselines
echo ""
echo "Test 2: Available Baselines"
echo "----------------------------"
curl -s "$BASE_URL/baselines/available" | jq '.available_baselines[]'
echo ""

# Test 3: Get Stephen Curry baseline
echo ""
echo "Test 3: Stephen Curry Baseline"
echo "-------------------------------"
curl -s "$BASE_URL/baselines/Stephen%20Curry" | jq -r '.player_name, .baseline_data.total_frames, .baseline_data.duration'
echo ""

# Test 4: Comprehensive analysis (if video file exists)
if [ -f "baselines/StephCurryShot.mp4" ]; then
    echo ""
    echo "Test 4: Comprehensive Analysis"
    echo "-------------------------------"
    echo "Analyzing StephCurryShot.mp4..."
    
    RESPONSE=$(curl -s -X POST "$BASE_URL/analyze/comprehensive" \
        -F "video=@baselines/StephCurryShot.mp4" \
        -F "baseline_player=Stephen Curry" \
        -F "frame_skip=3")
    
    echo "$RESPONSE" | jq '{
        success: .success,
        overall_score: .overall_score,
        confidence: .confidence,
        player: .comparison.player,
        similarity: .comparison.overall_similarity,
        strengths: .comparison.strengths,
        cues: [.coaching_cues[].cue]
    }'
else
    echo ""
    echo "⚠️  Test 4 skipped: Video file not found"
fi

echo ""
echo "✅ Tests complete!"
