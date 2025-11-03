"""
Visual Output Generator - Create annotated video with pose overlays
Shows detected poses, key metrics, and tracking confidence in real-time
"""
import cv2
import json
import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor

logging.basicConfig(level=logging.WARNING)


class PoseVisualizer:
    """Create annotated videos with pose overlays"""

    # Colors (BGR format for OpenCV)
    COLORS = {
        'skeleton': (0, 255, 0),       # Green
        'joints': (0, 0, 255),         # Red
        'shooting_arm': (255, 0, 0),   # Blue
        'body': (0, 255, 255),         # Yellow
        'legs': (255, 0, 255),         # Magenta
        'text': (255, 255, 255),       # White
        'bg': (0, 0, 0),               # Black
        'highlight': (0, 255, 255),    # Cyan
    }

    # Skeleton connections
    CONNECTIONS = [
        # Torso
        ('left_shoulder', 'right_shoulder', 'body'),
        ('left_shoulder', 'left_hip', 'body'),
        ('right_shoulder', 'right_hip', 'body'),
        ('left_hip', 'right_hip', 'body'),

        # Right arm (shooting arm)
        ('right_shoulder', 'right_elbow', 'shooting_arm'),
        ('right_elbow', 'right_wrist', 'shooting_arm'),

        # Left arm
        ('left_shoulder', 'left_elbow', 'skeleton'),
        ('left_elbow', 'left_wrist', 'skeleton'),

        # Right leg
        ('right_hip', 'right_knee', 'legs'),
        ('right_knee', 'right_ankle', 'legs'),

        # Left leg
        ('left_hip', 'left_knee', 'legs'),
        ('left_knee', 'left_ankle', 'legs'),
    ]

    # Key joints to highlight
    KEY_JOINTS = ['right_shoulder', 'right_elbow', 'right_wrist',
                  'left_shoulder', 'right_hip', 'left_hip',
                  'right_knee', 'left_knee', 'right_ankle', 'left_ankle']

    def __init__(self):
        self.pose_processor = PoseProcessor(model_complexity=1)

    def create_annotated_video(self, input_video: str, output_video: str, show_metrics: bool = True):
        """
        Create annotated video with pose overlays

        Args:
            input_video: Path to input video
            output_video: Path to save annotated video
            show_metrics: Whether to show metrics overlay
        """
        print("=" * 80)
        print("🎬 CREATING ANNOTATED VIDEO WITH POSE OVERLAYS")
        print("=" * 80)
        print(f"\n📹 Input: {input_video}")
        print(f"💾 Output: {output_video}\n")

        if not Path(input_video).exists():
            print(f"❌ Video not found: {input_video}")
            return

        # Process video to get pose data
        print("🔍 Step 1: Extracting pose data...")
        pose_data = self.pose_processor.process_video(input_video, frame_skip=1)

        metadata = pose_data['metadata']
        keypoints_seq = pose_data['keypoints_sequence']
        quality = pose_data['quality']

        print(f"   ✓ Processed {metadata['processed_frames']} frames")
        print(f"   ✓ Confidence: {quality['confidence']:.1%}")

        # Create mapping from actual video frame index -> keypoints index
        # Ensures we draw keypoints on the exact corresponding video frame
        frame_to_kp_idx = {kp.get('frame', idx): idx for idx, kp in enumerate(keypoints_seq)}

        # Calculate metrics for visualization
        print("\n📐 Step 2: Calculating metrics...")
        metrics = self._calculate_frame_metrics(keypoints_seq)

        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {input_video}")
            return

        # Setup video writer
        fps = metadata['fps']
        width = metadata['width']
        height = metadata['height']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        print(f"\n🎨 Step 3: Rendering annotated frames...")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.1f}")

        video_frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(keypoints_seq)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            kp_idx = frame_to_kp_idx.get(video_frame_idx)
            if kp_idx is not None and 0 <= kp_idx < len(keypoints_seq):
                keypoints = keypoints_seq[kp_idx]
                frame_metrics = metrics[kp_idx] if kp_idx < len(metrics) else {}

                # Draw pose overlay
                frame = self._draw_pose_overlay(frame, keypoints, width, height)

                # Draw metrics overlay
                if show_metrics:
                    frame = self._draw_metrics_overlay(
                        frame, keypoints, frame_metrics,
                        video_frame_idx, total_frames, quality, fps
                    )

            out.write(frame)
            video_frame_idx += 1

            # Progress indicator
            if video_frame_idx % 30 == 0:
                progress = (video_frame_idx / max(total_frames, 1)) * 100
                print(f"   Progress: {progress:.1f}% ({video_frame_idx}/{total_frames} frames)", end='\r')

        cap.release()
        out.release()

        print(f"\n\n✅ Annotated video created successfully!")
        print(f"   📁 Location: {output_video}")
        print(f"   📊 Size: {Path(output_video).stat().st_size / (1024*1024):.1f} MB")
        print("\n" + "=" * 80)
        print("🎉 COMPLETE! You can now view your annotated video.")
        print("=" * 80 + "\n")

    def _draw_pose_overlay(self, frame, keypoints, width, height):
        """Draw skeleton and joints on frame"""

        def get_point(landmark_name):
            """Get pixel coordinates for a landmark"""
            if landmark_name in keypoints:
                lm = keypoints[landmark_name]
                if isinstance(lm, dict) and lm.get('visibility', 0) > 0.3:
                    x = int(lm['x'] * width)
                    y = int(lm['y'] * height)
                    return (x, y), lm['visibility']
            return None, 0

        # Draw connections (skeleton)
        for landmark1, landmark2, color_key in self.CONNECTIONS:
            pt1, vis1 = get_point(landmark1)
            pt2, vis2 = get_point(landmark2)

            if pt1 and pt2 and vis1 > 0.3 and vis2 > 0.3:
                color = self.COLORS[color_key]
                # Thicker line for shooting arm
                thickness = 4 if color_key == 'shooting_arm' else 3
                cv2.line(frame, pt1, pt2, color, thickness)

        # Draw joints
        for landmark_name in self.KEY_JOINTS:
            pt, vis = get_point(landmark_name)
            if pt and vis > 0.3:
                # Outer circle (white)
                cv2.circle(frame, pt, 10, self.COLORS['text'], -1)
                # Inner circle (colored by body part)
                if 'wrist' in landmark_name or 'elbow' in landmark_name or 'shoulder' in landmark_name:
                    color = self.COLORS['shooting_arm'] if 'right' in landmark_name else self.COLORS['skeleton']
                elif 'hip' in landmark_name:
                    color = self.COLORS['body']
                else:
                    color = self.COLORS['legs']
                cv2.circle(frame, pt, 6, color, -1)

                # Add label for key joints
                if landmark_name in ['right_wrist', 'right_elbow', 'right_shoulder']:
                    label = landmark_name.replace('right_', '').replace('_', ' ').title()
                    cv2.putText(frame, label, (pt[0] + 15, pt[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)

        return frame

    def _draw_metrics_overlay(self, frame, keypoints, metrics, frame_idx, total_frames, quality, fps):
        """Draw metrics and info overlay"""
        height, width = frame.shape[:2]

        # Create semi-transparent overlay for text background
        overlay = frame.copy()

        # Top bar background
        cv2.rectangle(overlay, (0, 0), (width, 120), self.COLORS['bg'], -1)

        # Bottom bar background
        cv2.rectangle(overlay, (0, height - 80), (width, height), self.COLORS['bg'], -1)

        # Blend overlay
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Title (using DUPLEX font with thickness 2 for bold effect)
        cv2.putText(frame, "BASKETBALL SHOT ANALYSIS - POSE TRACKING", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, self.COLORS['highlight'], 2)

        # Frame info
        timestamp = frame_idx / max(fps, 1.0)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}  |  Time: {timestamp:.2f}s", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)

        # Quality metrics
        confidence = quality['confidence']
        conf_color = self.COLORS['skeleton'] if confidence > 0.8 else self.COLORS['highlight'] if confidence > 0.5 else self.COLORS['joints']
        cv2.putText(frame, f"Tracking Confidence: {confidence:.1%}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

        # Current frame metrics (right side)
        if metrics:
            x_offset = width - 350
            y_start = 35

            if 'elbow_angle' in metrics:
                cv2.putText(frame, f"Elbow Angle: {metrics['elbow_angle']:.1f}°",
                           (x_offset, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)

            if 'wrist_height' in metrics:
                cv2.putText(frame, f"Wrist Height: {metrics['wrist_height']:.1%}",
                           (x_offset, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)

            if 'knee_angle' in metrics:
                cv2.putText(frame, f"Knee Bend: {metrics['knee_angle']:.1f}°",
                           (x_offset, y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)

        # Progress bar at bottom
        progress = frame_idx / max(total_frames, 1)
        bar_width = int(width * 0.8)
        bar_x = int(width * 0.1)
        bar_y = height - 50

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                     (50, 50, 50), -1)
        # Progress bar
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + int(bar_width * progress), bar_y + 20),
                     self.COLORS['highlight'], -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                     self.COLORS['text'], 2)

        # Legend at bottom
        legend_x = 20
        legend_y = height - 25
        legend_items = [
            ("Shooting Arm", self.COLORS['shooting_arm']),
            ("Body", self.COLORS['body']),
            ("Legs", self.COLORS['legs'])
        ]

        for i, (label, color) in enumerate(legend_items):
            x = legend_x + (i * 150)
            cv2.rectangle(frame, (x, legend_y - 5), (x + 20, legend_y + 5), color, -1)
            cv2.putText(frame, label, (x + 30, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

        return frame

    def _calculate_frame_metrics(self, keypoints_seq):
        """Calculate per-frame metrics for visualization"""
        metrics_list = []

        for frame in keypoints_seq:
            frame_metrics = {}

            # Calculate elbow angle
            if all(k in frame for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                shoulder = frame['right_shoulder']
                elbow = frame['right_elbow']
                wrist = frame['right_wrist']

                if all(isinstance(x, dict) for x in [shoulder, elbow, wrist]):
                    v1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
                    v2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])

                    angle = np.degrees(np.arccos(np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                        -1.0, 1.0
                    )))
                    frame_metrics['elbow_angle'] = angle

            # Wrist height (normalized)
            if 'right_wrist' in frame and isinstance(frame['right_wrist'], dict):
                frame_metrics['wrist_height'] = 1 - frame['right_wrist']['y']

            # Knee angle
            if all(k in frame for k in ['right_hip', 'right_knee', 'right_ankle']):
                hip = frame['right_hip']
                knee = frame['right_knee']
                ankle = frame['right_ankle']

                if all(isinstance(x, dict) for x in [hip, knee, ankle]):
                    v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
                    v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])

                    angle = np.degrees(np.arccos(np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                        -1.0, 1.0
                    )))
                    frame_metrics['knee_angle'] = angle

            metrics_list.append(frame_metrics)

        return metrics_list


def main():
    """Main function"""

    # Parse arguments
    if len(sys.argv) < 2:
        input_video = "tests/TestShot.mp4"
        print("ℹ️  No input video specified, using default: tests/TestShot.mp4")
    else:
        input_video = sys.argv[1]

    # Generate output filename
    input_path = Path(input_video)
    output_video = f"output/{input_path.stem}_annotated.mp4"

    if len(sys.argv) >= 3:
        output_video = sys.argv[2]

    # Create output directory
    Path(output_video).parent.mkdir(exist_ok=True)

    # Create visualizer and process video
    visualizer = PoseVisualizer()
    visualizer.create_annotated_video(input_video, output_video, show_metrics=True)

    print(f"💡 Tip: Play {output_video} to see your pose tracking in action!")


if __name__ == "__main__":
    main()
