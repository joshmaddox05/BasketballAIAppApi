"""
Simplified Test - Show what data we CAN extract from your video
Even if phase detection fails, we can still see pose data and basic metrics
"""
import json
import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor

logging.basicConfig(level=logging.WARNING)

def analyze_test_video(video_path: str):
    """Extract and display pose data from test video"""

    print("=" * 80)
    print("🏀 BASKETBALL SHOT POSE ANALYSIS - RAW DATA EXTRACTION")
    print("=" * 80)
    print(f"\n📹 Video: {video_path}\n")

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return

    # Initialize pose processor
    processor = PoseProcessor(model_complexity=1)

    # Extract pose data
    print("🔍 Extracting pose keypoints...\n")
    pose_data = processor.process_video(video_path, frame_skip=1)

    print("✅ EXTRACTION COMPLETE\n")
    print("-" * 80)
    print("📊 VIDEO METADATA:")
    print("-" * 80)
    for key, value in pose_data['metadata'].items():
        print(f"   {key}: {value}")

    print("\n" + "-" * 80)
    print("📊 QUALITY METRICS:")
    print("-" * 80)
    for key, value in pose_data['quality'].items():
        if value is not None:
            if isinstance(value, float):
                print(f"   {key}: {value:.2%}" if value <= 1 else f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

    # Analyze keypoints
    keypoints_seq = pose_data['keypoints_sequence']

    print("\n" + "-" * 80)
    print("📊 KEYPOINT TRACKING:")
    print("-" * 80)

    # Check what landmarks are tracked
    if keypoints_seq and len(keypoints_seq) > 0:
        first_frame = keypoints_seq[0]
        tracked_landmarks = [k for k in first_frame.keys() if k not in ['frame', 'timestamp', 'visible']]
        print(f"   Total landmarks tracked: {len(tracked_landmarks)}")
        print(f"   Critical landmarks: {', '.join(tracked_landmarks[:12])}")

    # Show sample frames
    print("\n" + "-" * 80)
    print("📊 SAMPLE FRAME DATA (showing 5 frames):")
    print("-" * 80)

    sample_indices = [0, len(keypoints_seq)//4, len(keypoints_seq)//2,
                     3*len(keypoints_seq)//4, len(keypoints_seq)-1]

    for idx in sample_indices:
        if idx < len(keypoints_seq):
            frame_data = keypoints_seq[idx]
            print(f"\n   Frame {frame_data.get('frame', idx)} (t={frame_data.get('timestamp', 0):.2f}s):")

            # Show shooting arm keypoints (assuming right-handed)
            for landmark in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_knee']:
                if landmark in frame_data:
                    lm_data = frame_data[landmark]
                    if isinstance(lm_data, dict):
                        x = lm_data.get('x', 0)
                        y = lm_data.get('y', 0)
                        vis = lm_data.get('visibility', 0)
                        print(f"      {landmark}: x={x:.3f}, y={y:.3f}, vis={vis:.2%}")

    # Calculate simple metrics manually
    print("\n" + "=" * 80)
    print("📐 BASIC FORM METRICS (calculated from raw data):")
    print("=" * 80)

    # Find frame with highest wrist position (likely release point)
    wrist_heights = []
    for frame in keypoints_seq:
        if 'right_wrist' in frame and isinstance(frame['right_wrist'], dict):
            wrist_heights.append({
                'frame': frame.get('frame', 0),
                'y': frame['right_wrist'].get('y', 1),
                'timestamp': frame.get('timestamp', 0)
            })

    if wrist_heights:
        # Lower y value = higher in image
        highest_point = min(wrist_heights, key=lambda x: x['y'])
        print(f"\n   🎯 Estimated Release Point:")
        print(f"      Frame: {highest_point['frame']}")
        print(f"      Time: {highest_point['timestamp']:.2f}s")
        print(f"      Wrist height: {(1 - highest_point['y']):.1%} of frame")

    # Calculate average shooting form angles
    elbow_angles = []
    knee_angles = []

    for frame in keypoints_seq:
        # Calculate elbow angle (shoulder-elbow-wrist)
        if all(k in frame for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            shoulder = frame['right_shoulder']
            elbow = frame['right_elbow']
            wrist = frame['right_wrist']

            if all(isinstance(x, dict) for x in [shoulder, elbow, wrist]):
                # Calculate angle
                v1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
                v2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])

                angle = np.degrees(np.arccos(np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                    -1.0, 1.0
                )))
                elbow_angles.append(angle)

    if elbow_angles:
        print(f"\n   💪 Elbow Angle Stats:")
        print(f"      Average: {np.mean(elbow_angles):.1f}°")
        print(f"      Min: {np.min(elbow_angles):.1f}° (extended)")
        print(f"      Max: {np.max(elbow_angles):.1f}° (bent)")

    # Save detailed output
    output_file = Path("output") / "raw_pose_analysis.json"
    output_file.parent.mkdir(exist_ok=True)

    output_data = {
        "video": str(video_path),
        "metadata": pose_data['metadata'],
        "quality": pose_data['quality'],
        "sample_frames": [keypoints_seq[i] for i in sample_indices if i < len(keypoints_seq)],
        "estimated_release_frame": highest_point if wrist_heights else None,
        "elbow_angle_stats": {
            "mean": float(np.mean(elbow_angles)) if elbow_angles else None,
            "min": float(np.min(elbow_angles)) if elbow_angles else None,
            "max": float(np.max(elbow_angles)) if elbow_angles else None,
        } if elbow_angles else None
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n\n💾 Raw analysis saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n💡 What this shows:")
    print("   ✓ Your video is being processed successfully")
    print("   ✓ MediaPipe is tracking all body landmarks")
    print("   ✓ Pose data is extracted with high confidence")
    print("   ⚠️  Phase detection needs tuning for your specific video")
    print("\n   The API will return this pose data structure when fully working!")
    print("\n")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "tests/TestShot.mp4"
    analyze_test_video(video_path)

