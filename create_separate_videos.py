"""
Separate Video Visualizer
Creates two separate videos: user shot and Curry shot with pose overlays
OPTIMIZED for faster processing with lower resolution
"""
import cv2
import json
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor
from services.video_handler import VideoHandler

logging.basicConfig(level=logging.WARNING)


class SeparateVideoVisualizer:
    """Create separate annotated videos for user and Curry - OPTIMIZED"""

    COLORS = {
        'user': (255, 100, 100),      # Light blue
        'curry': (100, 255, 100),     # Light green
        'good': (0, 255, 0),          # Green
        'medium': (0, 255, 255),      # Yellow
        'poor': (0, 165, 255),        # Orange
        'bad': (0, 0, 255),           # Red
        'text': (255, 255, 255),      # White
        'bg': (0, 0, 0),              # Black
        'skeleton_user': (255, 0, 0),      # Blue
        'skeleton_curry': (0, 255, 0),     # Green
    }

    def __init__(self):
        # Use highest accuracy model (complexity=2) for best pose tracking
        self.pose_processor = PoseProcessor(model_complexity=2)

    def create_separate_videos(
        self,
        user_video: str,
        curry_video: str,
        comparison_json: str,
        output_user: str,
        output_curry: str
    ):
        """Create two separate annotated videos - OPTIMIZED for Standard Plan"""

        print("=" * 80)
        print("🎬 CREATING SEPARATE ANNOTATED VIDEOS (STANDARD PLAN)")
        print("=" * 80)
        print(f"\n📹 Processing videos with higher quality settings...")

        # Load comparison data
        with open(comparison_json, 'r') as f:
            comparison_data = json.load(f)

        # Process user video
        print(f"\n1️⃣  Creating YOUR shot video...")
        print(f"   Input: {Path(user_video).name}")
        print(f"   Output: {output_user}")

        # Better frame rate for Standard plan
        user_pose = self.pose_processor.process_video(user_video, frame_skip=2)
        user_keypoints = user_pose['keypoints_sequence']

        self._create_single_video(
            user_video,
            user_keypoints,
            comparison_data,
            output_user,
            "YOUR SHOT",
            self.COLORS['skeleton_user'],
            self.COLORS['user'],
            is_user=True
        )

        print(f"   ✅ Saved: {output_user}\n")

        # Process Curry video
        print(f"2️⃣  Creating CURRY shot video...")
        print(f"   Input: {Path(curry_video).name}")
        print(f"   Output: {output_curry}")

        curry_pose = self.pose_processor.process_video(curry_video, frame_skip=2)
        curry_keypoints = curry_pose['keypoints_sequence']

        self._create_single_video(
            curry_video,
            curry_keypoints,
            comparison_data,
            output_curry,
            "STEPH CURRY",
            self.COLORS['skeleton_curry'],
            self.COLORS['curry'],
            is_user=False
        )

        print(f"   ✅ Saved: {output_curry}\n")

        print("=" * 80)
        print("✅ COMPLETE! Two separate videos created:")
        print(f"   📹 Your shot: {output_user}")
        print(f"   📹 Curry shot: {output_curry}")
        print("=" * 80 + "\n")

    def _create_single_video(
        self,
        video_path: str,
        keypoints: list,
        comparison_data: dict,
        output_path: str,
        label: str,
        skeleton_color: tuple,
        text_color: tuple,
        is_user: bool
    ):
        """Create a single annotated video - OPTIMIZED for Standard Plan"""

        # Use VideoHandler to get proper rotation info
        cap, rotation, width, height = VideoHandler.get_video_capture_with_rotation(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Higher resolution for Standard plan (720p instead of 480p)
        target_height = 720  # Increased from 480
        target_width = int(width * target_height / height)

        # Add space for metrics at bottom
        canvas_height = target_height + 150

        # Setup video writer with better codec for Render compatibility
        # Use MP4V codec which is more widely supported than H.264
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from 'avc1' to 'mp4v'
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, canvas_height))

        # Verify writer opened successfully
        if not out.isOpened():
            print(f"   ⚠️  Warning: Failed to open video writer with mp4v, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, canvas_height))

        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")

        # Build mapping from actual video frame -> keypoints index, based on 'frame'
        frame_to_kp_idx = {kp.get('frame', idx): idx for idx, kp in enumerate(keypoints)}

        video_frame_idx = 0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(keypoints)

        # No frame skipping in video generation for Standard plan (best quality)
        processed_frames = 0

        print(f"   Processing {total_video_frames} frames at {fps:.1f} fps (high quality mode)...")
        if rotation != 0:
            print(f"   Applying rotation fix: {rotation}°")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Fix rotation if needed
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Resize frame to target dimensions
            frame = cv2.resize(frame, (target_width, target_height))

            # Create canvas with metrics area
            canvas = np.zeros((canvas_height, target_width, 3), dtype=np.uint8)
            canvas[:target_height, :] = frame

            # Draw pose skeleton if available for this exact video frame
            kp_idx = frame_to_kp_idx.get(video_frame_idx)
            if kp_idx is not None and 0 <= kp_idx < len(keypoints):
                kp = keypoints[kp_idx]
                self._draw_skeleton(canvas, kp, skeleton_color, target_width, target_height)

            # Draw metrics overlay
            canvas = self._draw_metrics(
                canvas,
                comparison_data,
                target_height,
                target_width,
                is_user
            )

            out.write(canvas)
            video_frame_idx += 1
            processed_frames += 1

            # Progress update every 30 frames
            if processed_frames % 30 == 0:
                progress = (processed_frames / max(total_video_frames, 1)) * 100
                print(f"   Progress: {progress:.0f}%")

        cap.release()
        out.release()

        print(f"   ✓ Processed {processed_frames} frames")

    def _draw_skeleton(self, frame, keypoints, color, width, height):
        """Draw pose skeleton on frame"""
        self._draw_pose(frame[:height, :], keypoints, width, height, color)

    def _draw_pose(self, frame, keypoints, width, height, color):
        """Draw pose skeleton on frame"""

        connections = [
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
        ]

        def get_point(landmark_name):
            if landmark_name in keypoints and isinstance(keypoints[landmark_name], dict):
                lm = keypoints[landmark_name]
                if lm.get('visibility', 0) > 0.5:
                    x = int(lm['x'] * width)
                    y = int(lm['y'] * height)
                    return (x, y)
            return None

        # Draw lines
        for lm1, lm2 in connections:
            pt1 = get_point(lm1)
            pt2 = get_point(lm2)
            if pt1 and pt2:
                cv2.line(frame, pt1, pt2, color, 3)

        # Draw joints
        key_joints = ['right_shoulder', 'right_elbow', 'right_wrist',
                     'left_shoulder', 'right_hip', 'left_hip']

        for joint in key_joints:
            pt = get_point(joint)
            if pt:
                cv2.circle(frame, pt, 6, (255, 255, 255), -1)
                cv2.circle(frame, pt, 4, color, -1)

        return frame

    def _draw_metrics(self, canvas, comparison_data, y_offset, width, is_user):
        """Draw metrics overlay at bottom of frame"""

        # Create semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, y_offset), (width, canvas.shape[0]), self.COLORS['bg'], -1)
        canvas = cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0)

        # Overall similarity
        overall_sim = comparison_data['comparison']['overall_similarity'] * 100
        overall_grade = comparison_data['comparison']['overall_grade']

        # Grade color
        if overall_grade == 'A':
            grade_color = self.COLORS['good']
        elif overall_grade == 'B':
            grade_color = self.COLORS['medium']
        elif overall_grade == 'C':
            grade_color = self.COLORS['poor']
        else:
            grade_color = self.COLORS['bad']

        # Title
        if is_user:
            title = f"Your Form vs Curry: {overall_sim:.1f}% (Grade: {overall_grade})"
        else:
            title = f"Curry Baseline Form"

        cv2.putText(canvas, title, (20, y_offset + 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, grade_color if is_user else self.COLORS['curry'], 2)

        # Draw key metrics (only for user video)
        if is_user:
            y_start = y_offset + 70
            x_start = 20
            line_height = 30

            categories = comparison_data['comparison']['categories']

            row = 0
            max_rows = 2  # Show only 2 key metrics

            # Get top metrics to show
            all_metrics = []
            for cat_name, cat_metrics in categories.items():
                for metric_name, metric_data in cat_metrics.items():
                    all_metrics.append((cat_name, metric_name, metric_data))

            # Sort by grade (show worst first)
            grade_order = {'D': 0, 'F': 0, 'C': 1, 'B': 2, 'A': 3}
            all_metrics.sort(key=lambda x: grade_order.get(x[2]['grade'], 0))

            # Display top metrics
            for cat_name, metric_name, metric_data in all_metrics[:max_rows]:
                y = y_start + (row * line_height)

                grade = metric_data['grade']
                color = self._get_grade_color(grade)

                metric_label = metric_name.replace('_', ' ').title()
                user_val = metric_data['user_value']
                curry_val = metric_data['baseline_mean']

                cv2.putText(canvas, f"{metric_label}: You={user_val:.1f}° | Curry={curry_val:.1f}° | Grade: {grade}",
                           (x_start, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                row += 1

        return canvas

    def _get_grade_color(self, grade):
        """Get color based on grade"""
        if grade == 'A':
            return self.COLORS['good']
        elif grade == 'B':
            return self.COLORS['medium']
        elif grade == 'C':
            return self.COLORS['poor']
        else:
            return self.COLORS['bad']


def main():
    """Main function"""

    if len(sys.argv) < 2:
        print("Usage: python3 create_separate_videos.py <user_video> [curry_video] [comparison_json]")
        print("\nExample:")
        print("  python3 create_separate_videos.py tests/TestShot3.mp4")
        return

    user_video = sys.argv[1]

    # Automatically match baseline based on video orientation
    if len(sys.argv) > 2:
        curry_video = sys.argv[2]
    else:
        print("\n🔍 Auto-detecting video orientation...")
        curry_video = VideoHandler.get_matching_baseline(user_video)

    if not curry_video or not Path(curry_video).exists():
        print("❌ No Curry baseline videos found in baselines/")
        return

    # Default comparison JSON
    user_stem = Path(user_video).stem
    comparison_json = sys.argv[3] if len(sys.argv) > 3 else f"output/{user_stem}_vs_curry_comparison.json"

    if not Path(comparison_json).exists():
        print(f"❌ Comparison JSON not found: {comparison_json}")
        print(f"   Run: python3 compare_to_baseline.py {user_video}")
        return

    # Output videos
    output_user = f"output/{user_stem}_annotated.mp4"
    curry_stem = Path(curry_video).stem
    output_curry = f"output/{curry_stem}_baseline_annotated.mp4"

    # Create visualization
    visualizer = SeparateVideoVisualizer()
    visualizer.create_separate_videos(
        user_video,
        curry_video,
        comparison_json,
        output_user,
        output_curry
    )

    print(f"\n💡 Videos created!")
    print(f"   📹 Your shot: {output_user}")
    print(f"   📹 Curry shot: {output_curry}")


if __name__ == "__main__":
    main()
