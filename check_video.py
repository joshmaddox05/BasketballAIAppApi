import cv2
import sys
from pathlib import Path

video_path = Path(__file__).parent / "baselines" / "StephCurryShot.mp4"
print(f"Checking video: {video_path}")
print(f"File exists: {video_path.exists()}")

if video_path.exists():
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        print("✅ Video can be opened!")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Frames: {frame_count}")
        print(f"FPS: {fps}")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {duration:.2f}s")
        
        cap.release()
    else:
        print("❌ Cannot open video!")
else:
    print("❌ File not found!")
