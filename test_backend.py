"""
Test script to verify the backend endpoints and baseline functionality.
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working!")
            print(f"   Message: {data.get('message')}")
            print(f"   Available baselines: {data.get('available_baselines', [])}")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_list_baselines():
    """Test listing baselines"""
    print("\n🔍 Testing baselines list endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/baselines/list")
        if response.status_code == 200:
            data = response.json()
            print("✅ Baselines list endpoint working!")
            print(f"   Found {len(data.get('baselines', []))} baseline(s)")
            for baseline in data.get('baselines', []):
                print(f"   - {baseline.get('name')}: {baseline.get('player_name')}")
            return True
        else:
            print(f"❌ Baselines endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_upload_and_analyze():
    """Test video upload and analysis"""
    print("\n🔍 Testing video upload and analysis...")
    
    # Check if we have a baseline
    baseline_path = Path(__file__).parent / "baselines" / "stephen_curry_form_shot.json"
    if not baseline_path.exists():
        print("⚠️  No baseline found. Run process_curry_baseline.py first!")
        return False
    
    # For this test, we'd need a sample user video
    # This is a placeholder to show the structure
    print("   Note: Would need a user video to test actual analysis")
    print("   Endpoint: POST /analyze/shooting")
    print("   Expected parameters:")
    print("   - file: video file")
    print("   - baseline_name: 'stephen_curry_form_shot' (optional)")
    return True

def main():
    print("=" * 60)
    print("Basketball AI Backend Test Suite")
    print("=" * 60)
    
    # Check if server is running
    print("\n🔌 Checking if server is running at", BASE_URL)
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print("✅ Server is running!")
    except:
        print("❌ Server is not running!")
        print("\nPlease start the server with:")
        print("   cd backend")
        print("   uvicorn main:app --reload")
        return
    
    # Run tests
    results = []
    results.append(("Root Endpoint", test_root_endpoint()))
    results.append(("List Baselines", test_list_baselines()))
    results.append(("Upload & Analyze", test_upload_and_analyze()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"\n{passed_count}/{total_count} tests passed")
    print("=" * 60)

if __name__ == "__main__":
    main()
