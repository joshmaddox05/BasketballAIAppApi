"""
Side-by-Side Comparison Visualizer
Creates split-screen video comparing user shot to Steph Curry baseline
Shows real-time metrics comparison overlay
"""
import cv2
import json
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor

logging.basicConfig(level=logging.WARNING)


class SideBySideVisualizer:
    """Create side-by-side comparison video with metrics overlay"""

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
        self.pose_processor = PoseProcessor(model_complexity=1)

    def create_comparison_video(
        self,
        user_video: str,
        curry_video: str,
        comparison_json: str,
        output_video: str
    ):
        """Create side-by-side comparison video"""

        print("=" * 80)
        print("🎬 CREATING SIDE-BY-SIDE COMPARISON")
        print("=" * 80)
        print(f"\n📹 Your shot: {Path(user_video).name}")
        print(f"📹 Curry shot: {Path(curry_video).name}")
        print(f"💾 Output: {output_video}\n")

        # Load comparison data
        with open(comparison_json, 'r') as f:
            comparison_data = json.load(f)

        print("🔍 Step 1: Extracting pose data...")

        # Process user video
        print("   Processing your shot...")
        user_pose = self.pose_processor.process_video(user_video, frame_skip=1)
        user_keypoints = user_pose['keypoints_sequence']

        # Process Curry video
        print("   Processing Curry shot...")
        curry_pose = self.pose_processor.process_video(curry_video, frame_skip=1)
        curry_keypoints = curry_pose['keypoints_sequence']

        print(f"   ✓ User: {len(user_keypoints)} frames")
        print(f"   ✓ Curry: {len(curry_keypoints)} frames")

        # Create side-by-side video
        print("\n🎨 Step 2: Creating side-by-side visualization...")
        self._create_video(
            user_video,
            curry_video,
            user_keypoints,
            curry_keypoints,
            comparison_data,
            output_video
        )

        print(f"\n✅ Comparison video saved to: {output_video}")
        print("=" * 80 + "\n")

    def _create_video(
        self,
        user_video: str,
        curry_video: str,
        user_keypoints: list,
        curry_keypoints: list,
        comparison_data: dict,
        output_path: str
    ):
        """Create the actual side-by-side video"""

        # Open both videos
        cap_user = cv2.VideoCapture(user_video)
        cap_curry = cv2.VideoCapture(curry_video)

        # Get video properties
        fps_user = cap_user.get(cv2.CAP_PROP_FPS)
        fps_curry = cap_curry.get(cv2.CAP_PROP_FPS)
        fps = min(fps_user, fps_curry)  # Use lower FPS

        width_user = int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_user = int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_curry = int(cap_curry.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_curry = int(cap_curry.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Target dimensions (smaller for faster processing)
        target_height = 480  # Reduced from 720
        target_width_user = int(width_user * target_height / height_user)
        target_width_curry = int(width_curry * target_height / height_curry)

        # Combined frame dimensions
        combined_width = target_width_user + target_width_curry
        combined_height = target_height + 200  # Reduced metrics area

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

        frame_idx_user = 0
        frame_idx_curry = 0

        # Use shorter video length for faster processing
        max_frames = min(len(user_keypoints), len(curry_keypoints))

        print(f"   Creating {max_frames} frames at {fps:.1f} fps...")

        while frame_idx_user < max_frames and frame_idx_curry < max_frames:
            # Read frames
            ret_user, frame_user = cap_user.read()
            ret_curry, frame_curry = cap_curry.read()

            if not ret_user or not ret_curry:
                break

            # Resize frames
            frame_user_resized = cv2.resize(frame_user, (target_width_user, target_height))
            frame_curry_resized = cv2.resize(frame_curry, (target_width_curry, target_height))

            # Draw pose overlays
            if frame_idx_user < len(user_keypoints):
                frame_user_resized = self._draw_pose(
                    frame_user_resized,
                    user_keypoints[frame_idx_user],
                    target_width_user,
                    target_height,
                    self.COLORS['skeleton_user']
                )

            if frame_idx_curry < len(curry_keypoints):
                frame_curry_resized = self._draw_pose(
                    frame_curry_resized,
                    curry_keypoints[frame_idx_curry],
                    target_width_curry,
                    target_height,
                    self.COLORS['skeleton_curry']
                )

            # Add labels
            cv2.putText(frame_user_resized, "YOUR SHOT", (20, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, self.COLORS['user'], 2)
            cv2.putText(frame_curry_resized, "STEPH CURRY", (20, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, self.COLORS['curry'], 2)

            # Combine frames horizontally
            combined_frame = np.hstack([frame_user_resized, frame_curry_resized])

            # Create full canvas with metrics area
            canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            canvas[0:target_height, :] = combined_frame

            # Draw metrics overlay at bottom
            canvas = self._draw_metrics_comparison(
                canvas,
                comparison_data,
                target_height,
                combined_width
            )

            out.write(canvas)

            frame_idx_user += 1
            frame_idx_curry += 1

            # Progress
            if frame_idx_user % 10 == 0:
                progress = (frame_idx_user / max_frames) * 100
                print(f"   Progress: {progress:.1f}%", end='\r')

        cap_user.release()
        cap_curry.release()
        out.release()

        print(f"   Progress: 100.0%")

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

    def _draw_metrics_comparison(self, canvas, comparison_data, y_offset, width):
        """Draw metrics comparison at bottom of frame"""

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
        cv2.putText(canvas, "FORM COMPARISON", (20, y_offset + 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, self.COLORS['text'], 2)

        cv2.putText(canvas, f"Overall Similarity: {overall_sim:.1f}% (Grade: {overall_grade})",
                   (width - 380, y_offset + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, grade_color, 2)

        # Draw key metrics
        y_start = y_offset + 70
        x_col1 = 20
        x_col2 = width // 2 + 20
        line_height = 35

        categories = comparison_data['comparison']['categories']

        # Column 1 - Wrist & Head
        row = 0

        # Wrist metrics
        if 'wrist' in categories:
            for metric_name, metric_data in list(categories['wrist'].items())[:2]:
                y = y_start + (row * line_height)

                grade = metric_data['grade']
                color = self._get_grade_color(grade)

                metric_label = metric_name.replace('_', ' ').title()
                user_val = metric_data['user_value']
                curry_val = metric_data['baseline_mean']

                cv2.putText(canvas, f"{metric_label}:", (x_col1, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

                cv2.putText(canvas, f"You: {user_val:.1f}", (x_col1 + 180, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['user'], 1)

                cv2.putText(canvas, f"Curry: {curry_val:.1f}", (x_col1 + 300, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['curry'], 1)

                cv2.putText(canvas, grade, (x_col1 + 430, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                row += 1

        # Column 2 - Head & Body
        row = 0

        # Head metrics
        if 'head' in categories:
            for metric_name, metric_data in categories['head'].items():
                y = y_start + (row * line_height)

                grade = metric_data['grade']
                color = self._get_grade_color(grade)

                metric_label = metric_name.replace('_', ' ').title()
                user_val = metric_data['user_value']
                curry_val = metric_data['baseline_mean']

                cv2.putText(canvas, f"{metric_label}:", (x_col2, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

                cv2.putText(canvas, f"You: {user_val:.1f}", (x_col2 + 180, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['user'], 1)

                cv2.putText(canvas, f"Curry: {curry_val:.1f}", (x_col2 + 300, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['curry'], 1)

                cv2.putText(canvas, grade, (x_col2 + 430, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                row += 1

        # Add body metrics to column 2
        if 'body' in categories and row < 3:
            for metric_name, metric_data in list(categories['body'].items())[:1]:
                y = y_start + (row * line_height)

                grade = metric_data['grade']
                color = self._get_grade_color(grade)

                metric_label = metric_name.replace('_', ' ').title()
                user_val = metric_data['user_value']
                curry_val = metric_data['baseline_mean']

                cv2.putText(canvas, f"{metric_label}:", (x_col2, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

                cv2.putText(canvas, f"You: {user_val:.1f}", (x_col2 + 180, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['user'], 1)

                cv2.putText(canvas, f"Curry: {curry_val:.1f}", (x_col2 + 300, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['curry'], 1)

                cv2.putText(canvas, grade, (x_col2 + 430, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                row += 1

        # Legend at bottom
        legend_y = canvas.shape[0] - 20
        cv2.rectangle(canvas, (20, legend_y - 10), (40, legend_y + 5), self.COLORS['user'], -1)
        cv2.putText(canvas, "Your Form", (45, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

        cv2.rectangle(canvas, (150, legend_y - 10), (170, legend_y + 5), self.COLORS['curry'], -1)
        cv2.putText(canvas, "Curry Form", (175, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

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
        print("Usage: python3 create_sidebyside_comparison.py <user_video> [curry_video] [comparison_json]")
        print("\nExample:")
        print("  python3 create_sidebyside_comparison.py tests/TestShot3.mp4")
        return

    user_video = sys.argv[1]

    # Default Curry video (use first one in baselines)
    baselines_dir = Path("baselines")
    curry_videos = list(baselines_dir.glob("*.mp4"))
    curry_video = sys.argv[2] if len(sys.argv) > 2 else str(curry_videos[0]) if curry_videos else None

    if not curry_video:
        print("❌ No Curry baseline videos found in baselines/")
        return

    # Default comparison JSON
    user_stem = Path(user_video).stem
    comparison_json = sys.argv[3] if len(sys.argv) > 3 else f"output/{user_stem}_vs_curry_comparison.json"

    if not Path(comparison_json).exists():
        print(f"❌ Comparison JSON not found: {comparison_json}")
        print(f"   Run: python3 compare_to_baseline.py {user_video}")
        return

    # Output video
    output_video = f"output/{user_stem}_vs_curry_sidebyside.mp4"

    # Create visualization
    visualizer = SideBySideVisualizer()
    visualizer.create_comparison_video(
        user_video,
        curry_video,
        comparison_json,
        output_video
    )

    print(f"\n💡 Open {output_video} to see the side-by-side comparison!")


if __name__ == "__main__":
    main()
