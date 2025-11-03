#!/usr/bin/env python3
"""
Quick verification test for the Basketball AI Backend
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoints():
    print("🏀 Basketball AI Backend - Quick Test")
    print("=" * 60)
    
    # Test 1: Root endpoint
    print("\n1️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server running - Version {data['version']}")
            print(f"   Available baselines: {len(data['available_baselines'])}")
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure the server is running:")
        print("   cd backend && uvicorn main:app --reload")
        return False
    
    # Test 2: List baselines
    print("\n2️⃣ Testing baselines endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/baselines/list")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} baseline(s):")
            for baseline in data['baselines']:
                print(f"   - {baseline['name']}")
                print(f"     Release Angle: {baseline['metrics']['release_angle']:.1f}°")
                print(f"     Follow Through: {baseline['metrics']['follow_through']:.1f}")
                print(f"     Balance: {baseline['metrics']['balance']:.1f}")
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 3: Health check
    print("\n3️⃣ Testing health status...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status', 'unknown')}")
        else:
            print("⚠️  No health endpoint (optional)")
    except:
        print("⚠️  No health endpoint (optional)")
    
    print("\n" + "=" * 60)
    print("✨ All core endpoints are working!")
    print("=" * 60)
    print("\n📝 Next Steps:")
    print("1. Upload a test video:")
    print("   curl -X POST -F 'file=@video.mp4' http://localhost:8000/upload/")
    print("\n2. Analyze the video:")
    print("   curl -X POST http://localhost:8000/analyze/shooting \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"video_id\": \"video.mp4\", \"baseline_name\": \"stephen_curry_form_shot\"}'")
    print("\n3. Update React Native app:")
    print("   Set isOfflineMode = false in src/services/aiAnalysisService.js")
    print()
    
    return True

if __name__ == "__main__":
    success = test_endpoints()
    exit(0 if success else 1)
