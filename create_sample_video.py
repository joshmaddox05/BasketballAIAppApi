#!/usr/bin/env python3
"""
Sample video creator for testing baseline functionality.
This creates a simple test video for development purposes.
For production, you should use actual professional player footage.
"""

import cv2
import numpy as np
import os
from datetime import datetime

def create_sample_test_video(output_path="test_shot.mp4", duration_seconds=3):
    """
    Creates a simple test video with a stick figure to test the baseline system.
    This is just for testing - you'll want to replace this with actual footage.
    """
    
    # Video properties
    width, height = 640, 480
    fps = 30
    frames = duration_seconds * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    for i in range(frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(height):
            color_value = int(30 + (y / height) * 50)
            frame[y, :] = (color_value, color_value, color_value)
        
        # Simulate a shooting motion with simple animation
        progress = i / frames
        
        # Body center (moves slightly up during shot)
        center_x = width // 2
        center_y = int(height * 0.6 - progress * 40)
        
        # Head
        cv2.circle(frame, (center_x, center_y - 60), 20, (255, 200, 200), -1)
        
        # Body
        cv2.line(frame, (center_x, center_y - 40), (center_x, center_y + 40), (255, 200, 200), 3)
        
        # Arms (simulate shooting motion)
        arm_angle = -20 + progress * 60  # Arm raises during shot
        arm_length = 50
        
        # Left arm
        left_elbow_x = center_x - 25
        left_elbow_y = center_y
        left_hand_x = int(left_elbow_x - arm_length * np.cos(np.radians(arm_angle)))
        left_hand_y = int(left_elbow_y - arm_length * np.sin(np.radians(arm_angle)))
        cv2.line(frame, (center_x, center_y), (left_elbow_x, left_elbow_y), (255, 200, 200), 3)
        cv2.line(frame, (left_elbow_x, left_elbow_y), (left_hand_x, left_hand_y), (255, 200, 200), 3)
        
        # Right arm (shooting arm)
        right_elbow_x = center_x + 25
        right_elbow_y = center_y
        right_hand_x = int(right_elbow_x + arm_length * np.cos(np.radians(arm_angle)))
        right_hand_y = int(right_elbow_y - arm_length * np.sin(np.radians(arm_angle)))
        cv2.line(frame, (center_x, center_y), (right_elbow_x, right_elbow_y), (255, 200, 200), 3)
        cv2.line(frame, (right_elbow_x, right_elbow_y), (right_hand_x, right_hand_y), (255, 200, 200), 3)
        
        # Legs
        cv2.line(frame, (center_x, center_y + 40), (center_x - 20, center_y + 100), (255, 200, 200), 3)
        cv2.line(frame, (center_x, center_y + 40), (center_x + 20, center_y + 100), (255, 200, 200), 3)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {i+1}/{frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add progress indicator
        cv2.putText(frame, f"Shot Progress: {int(progress * 100)}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {output_path}")
    print(f"📦 File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    return output_path

def print_video_instructions():
    """Print instructions for getting real baseline videos"""
    
    print("\n" + "="*60)
    print("📹 GETTING REAL BASELINE VIDEOS")
    print("="*60)
    print()
    print("The test video above is just for testing the system.")
    print("For real shot analysis, you need actual footage of professional players.")
    print()
    print("Here are some ways to get baseline videos:")
    print()
    print("1. 🎥 RECORD FROM NBA GAMES (PERSONAL USE)")
    print("   - Watch NBA games and record short clips of shots")
    print("   - Focus on clear side-view angles")
    print("   - 3-5 second clips work best")
    print()
    print("2. 📺 YOUTUBE (EDUCATIONAL USE)")
    print("   Search for:")
    print("   - 'Stephen Curry shooting form breakdown'")
    print("   - 'NBA shooting mechanics slow motion'")
    print("   - 'Professional basketball shot analysis'")
    print("   ")
    print("   Use tools like: yt-dlp or youtube-dl")
    print("   Example: yt-dlp -f 'best[height<=720]' [URL]")
    print()
    print("3. 🏀 NBA OFFICIAL CONTENT")
    print("   - NBA.com often has training videos")
    print("   - Check NBA YouTube channel")
    print("   - Look for 'Skills Challenge' or 'Training' playlists")
    print()
    print("4. 📱 RECORD YOUR OWN")
    print("   - Film local pro or college players (with permission)")
    print("   - Use good lighting and stable camera")
    print("   - Get side-view or 45-degree angle")
    print()
    print("="*60)
    print()
    print("Once you have a video, create a baseline with:")
    print('  python create_baseline.py your_video.mp4 "Player Name"')
    print()

if __name__ == "__main__":
    # Create test video
    video_path = create_sample_test_video()
    
    # Print instructions
    print_video_instructions()
    
    print("To test the baseline system with this video:")
    print(f'  python create_baseline.py {video_path} "Test Player"')
    print()
