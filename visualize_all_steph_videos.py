"""
Batch Visualizer - Create annotated videos for all Steph Curry baseline videos
Shows tracking quality and what the model sees in each video
"""
import cv2
import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor

logging.basicConfig(level=logging.WARNING)


def create_quick_visualization(input_video: str, output_video: str):
    """Create a quick visualization showing pose tracking quality"""

    print(f"\n📹 Processing: {Path(input_video).name}")

    if not Path(input_video).exists():
        print(f"   ❌ Not found")
        return False

    try:
        # Initialize pose processor
        processor = PoseProcessor(model_complexity=1)

        # Extract pose data
        pose_data = processor.process_video(input_video, frame_skip=1)

        metadata = pose_data['metadata']
        keypoints_seq = pose_data['keypoints_sequence']
        quality = pose_data['quality']

        print(f"   Quality: {quality['confidence']:.1%}")
        print(f"   Frames: {metadata['processed_frames']}")
        print(f"   Visibility: {quality['visibility_ratio']:.1%}")

        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"   ❌ Cannot open video")
            return False

        # Setup video writer
        fps = metadata['fps']
        width = metadata['width']
        height = metadata['height']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < len(keypoints_seq):
                keypoints = keypoints_seq[frame_idx]

                # Draw skeleton
                frame = draw_skeleton(frame, keypoints, width, height)

                # Draw tracking info
                frame = draw_tracking_info(frame, keypoints, frame_idx, quality, width, height)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        print(f"   ✅ Saved: {Path(output_video).name}")
        return True

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def draw_skeleton(frame, keypoints, width, height):
    """Draw skeleton overlay"""

    # Colors
    SHOOTING_ARM = (255, 0, 0)  # Blue
    OTHER_ARM = (0, 255, 0)     # Green
    BODY = (0, 255, 255)         # Yellow
    LEGS = (255, 0, 255)         # Magenta
    JOINT = (0, 0, 255)          # Red

    def get_point(landmark_name):
        if landmark_name in keypoints:
            lm = keypoints[landmark_name]
            if isinstance(lm, dict):
                vis = lm.get('visibility', 0)
                if vis > 0.3:
                    x = int(lm['x'] * width)
                    y = int(lm['y'] * height)
                    return (x, y), vis
        return None, 0

    # Define connections
    connections = [
        # Shooting arm (right)
        ('right_shoulder', 'right_elbow', SHOOTING_ARM, 5),
        ('right_elbow', 'right_wrist', SHOOTING_ARM, 5),
        # Other arm (left)
        ('left_shoulder', 'left_elbow', OTHER_ARM, 3),
        ('left_elbow', 'left_wrist', OTHER_ARM, 3),
        # Body
        ('left_shoulder', 'right_shoulder', BODY, 3),
        ('left_shoulder', 'left_hip', BODY, 3),
        ('right_shoulder', 'right_hip', BODY, 3),
        ('left_hip', 'right_hip', BODY, 3),
        # Legs
        ('right_hip', 'right_knee', LEGS, 3),
        ('right_knee', 'right_ankle', LEGS, 3),
        ('left_hip', 'left_knee', LEGS, 3),
        ('left_knee', 'left_ankle', LEGS, 3),
    ]

    # Draw lines
    for lm1, lm2, color, thickness in connections:
        pt1, vis1 = get_point(lm1)
        pt2, vis2 = get_point(lm2)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, color, thickness)

    # Draw key joints
    key_joints = ['right_shoulder', 'right_elbow', 'right_wrist',
                  'left_shoulder', 'left_elbow', 'left_wrist',
                  'right_hip', 'left_hip', 'right_knee', 'left_knee',
                  'right_ankle', 'left_ankle']

    for joint in key_joints:
        pt, vis = get_point(joint)
        if pt:
            # Size based on visibility
            radius = int(8 + 4 * vis)
            cv2.circle(frame, pt, radius, (255, 255, 255), -1)
            cv2.circle(frame, pt, radius - 3, JOINT, -1)

    return frame


def draw_tracking_info(frame, keypoints, frame_idx, quality, width, height):
    """Draw tracking information overlay"""

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Title
    cv2.putText(frame, Path(keypoints.get('video_name', 'Video')).stem if 'video_name' in keypoints else 'Steph Curry Analysis',
                (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

    # Frame info
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Overall quality
    conf = quality['confidence']
    conf_color = (0, 255, 0) if conf > 0.8 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
    cv2.putText(frame, f"Quality: {conf:.0%}", (width - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

    # Count visible landmarks
    critical_landmarks = ['right_shoulder', 'right_elbow', 'right_wrist',
                         'left_shoulder', 'right_hip', 'left_hip',
                         'right_knee', 'left_knee']

    visible_count = 0
    for lm in critical_landmarks:
        if lm in keypoints and isinstance(keypoints[lm], dict):
            if keypoints[lm].get('visibility', 0) > 0.3:
                visible_count += 1

    cv2.putText(frame, f"Visible Landmarks: {visible_count}/{len(critical_landmarks)}",
                (width - 300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Legend at bottom
    legend_x = 20
    legend_y = height - 25

    cv2.rectangle(frame, (legend_x, legend_y - 8), (legend_x + 25, legend_y + 8), (255, 0, 0), -1)
    cv2.putText(frame, "Shooting Arm", (legend_x + 35, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(frame, (legend_x + 180, legend_y - 8), (legend_x + 205, legend_y + 8), (0, 255, 255), -1)
    cv2.putText(frame, "Body", (legend_x + 215, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(frame, (legend_x + 310, legend_y - 8), (legend_x + 335, legend_y + 8), (255, 0, 255), -1)
    cv2.putText(frame, "Legs", (legend_x + 345, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def batch_visualize_all():
    """Create visualizations for all Steph Curry videos"""

    baselines_dir = Path("baselines")
    output_dir = Path("output/steph_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎬 CREATING VISUALIZATIONS FOR ALL STEPH CURRY VIDEOS")
    print("=" * 80)

    # Find all videos
    videos = list(baselines_dir.glob("*.mp4"))

    if not videos:
        print("\n❌ No videos found in baselines folder")
        return

    print(f"\nFound {len(videos)} videos\n")

    results = {
        'success': [],
        'failed': []
    }

    # Process each video
    for video in sorted(videos):
        output_path = output_dir / f"{video.stem}_tracked.mp4"

        success = create_quick_visualization(str(video), str(output_path))

        if success:
            results['success'].append({
                'name': video.name,
                'output': str(output_path)
            })
        else:
            results['failed'].append(video.name)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"\n✅ Successfully processed: {len(results['success'])}/{len(videos)}")

    if results['success']:
        print("\n📁 Visualizations saved to: output/steph_visualizations/")
        print("\nGenerated files:")
        for item in results['success']:
            print(f"   • {item['name']} → {Path(item['output']).name}")

    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for name in results['failed']:
            print(f"   • {name}")

    print("\n" + "=" * 80)
    print("🎉 COMPLETE! Review the visualizations to see tracking quality.")
    print("=" * 80)
    print("\n💡 Tips for reviewing:")
    print("   • Look for videos where the full body is visible")
    print("   • Check if shooting arm is clearly tracked (blue lines)")
    print("   • High visibility count = better for baseline")
    print("   • Quality > 80% is ideal for baseline data")
    print()


if __name__ == "__main__":
    batch_visualize_all()

