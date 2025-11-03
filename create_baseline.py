#!/usr/bin/env python3
"""
Helper script to create a baseline from a video file.
Usage: python create_baseline.py <video_path> <player_name>
Example: python create_baseline.py curry_shot.mp4 "Stephen Curry"
"""

import sys
import requests
import os
from pathlib import Path

def create_baseline(video_path: str, player_name: str, api_url: str = "http://localhost:8000"):
    """
    Create a baseline from a video file by calling the FastAPI endpoint.
    
    Args:
        video_path: Path to the video file
        player_name: Name of the player (e.g., "Stephen Curry")
        api_url: Base URL of the FastAPI server
    """
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return False
    
    print(f"📹 Processing video: {video_path}")
    print(f"👤 Player name: {player_name}")
    print(f"🔗 API URL: {api_url}")
    print()
    
    # Prepare the file upload
    with open(video_path, 'rb') as video_file:
        files = {
            'video': (os.path.basename(video_path), video_file, 'video/mp4')
        }
        data = {
            'player_name': player_name
        }
        
        print("⏳ Uploading and processing video (this may take a minute)...")
        
        try:
            response = requests.post(
                f"{api_url}/baseline/create",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ Baseline created successfully!")
                print(f"📊 Baseline ID: {result.get('baseline_id')}")
                print(f"👤 Player: {result.get('player_name')}")
                print(f"🎬 Frames analyzed: {result.get('frames_analyzed')}")
                print(f"📁 Saved to: {result.get('baseline_path')}")
                
                # Print metrics summary
                if 'metrics' in result:
                    print("\n📈 Baseline Metrics:")
                    metrics = result['metrics']
                    print(f"  • Release Angle: {metrics.get('release_angle', {}).get('trajectory_angle', 'N/A')}°")
                    print(f"  • Elbow Alignment: {metrics.get('elbow_alignment', {}).get('lateral_offset', 'N/A')}°")
                    print(f"  • Follow Through: {metrics.get('follow_through', {}).get('extension_distance', 'N/A')}")
                    print(f"  • Balance Score: {metrics.get('balance', {}).get('stability_score', 'N/A')}")
                
                return True
            else:
                print(f"\n❌ Error creating baseline: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: Could not connect to the API server.")
            print("Make sure the FastAPI server is running with: uvicorn main:app --reload")
            return False
        except requests.exceptions.Timeout:
            print("\n❌ Error: Request timed out. The video might be too long or complex.")
            return False
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_baseline.py <video_path> <player_name>")
        print('Example: python create_baseline.py curry_shot.mp4 "Stephen Curry"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    player_name = sys.argv[2]
    
    success = create_baseline(video_path, player_name)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
