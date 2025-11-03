"""
Pose Visualization Script
Shows how the pose processor analyzes basketball shooting form
Press 'q' to quit, SPACE to pause/resume
"""
import cv2
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def visualize_pose_processing(video_path: str, output_path: str = None):
    """
    Process video and display pose landmarks in real-time

    Args:
        video_path: Path to input video
        output_path: Optional path to save output video
    """
    if not os.path.exists(video_path):
        logger.error(f"❌ Video not found: {video_path}")
        return

    logger.info(f"🎬 Opening video: {video_path}")

    # Initialize pose processor
    processor = PoseProcessor(model_complexity=1)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"📹 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Setup video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"💾 Saving output to: {output_path}")

    frame_count = 0
    paused = False

    # Colors for different body parts (BGR format)
    COLORS = {
        'shoulders': (0, 255, 0),      # Green
        'arms': (255, 0, 0),           # Blue
        'torso': (0, 255, 255),        # Yellow
        'legs': (255, 0, 255),         # Magenta
        'joints': (0, 0, 255)          # Red
    }

    logger.info("🎥 Starting playback... Press 'q' to quit, SPACE to pause/resume")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                logger.info("✅ Video finished")
                break

            frame_count += 1

            # Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = processor.pose.process(rgb_frame)

            # Draw pose landmarks if detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Function to get pixel coordinates
                def get_coords(idx):
                    lm = landmarks[idx]
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
                    return x, y, vis

                # Draw connections (skeleton)
                connections = [
                    # Shoulders
                    (11, 12, COLORS['shoulders']),  # Left to right shoulder
                    # Arms
                    (11, 13, COLORS['arms']),  # Left shoulder to elbow
                    (13, 15, COLORS['arms']),  # Left elbow to wrist
                    (12, 14, COLORS['arms']),  # Right shoulder to elbow
                    (14, 16, COLORS['arms']),  # Right elbow to wrist
                    # Torso
                    (11, 23, COLORS['torso']),  # Left shoulder to hip
                    (12, 24, COLORS['torso']),  # Right shoulder to hip
                    (23, 24, COLORS['torso']),  # Left to right hip
                    # Legs
                    (23, 25, COLORS['legs']),  # Left hip to knee
                    (25, 27, COLORS['legs']),  # Left knee to ankle
                    (24, 26, COLORS['legs']),  # Right hip to knee
                    (26, 28, COLORS['legs']),  # Right knee to ankle
                ]

                # Draw lines
                for idx1, idx2, color in connections:
                    x1, y1, vis1 = get_coords(idx1)
                    x2, y2, vis2 = get_coords(idx2)
                    if vis1 > 0.3 and vis2 > 0.3:
                        cv2.line(frame, (x1, y1), (x2, y2), color, 3)

                # Draw critical joints as circles
                critical_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                for idx in critical_indices:
                    x, y, vis = get_coords(idx)
                    if vis > 0.3:
                        # Draw outer circle (white)
                        cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
                        # Draw inner circle (red)
                        cv2.circle(frame, (x, y), 5, COLORS['joints'], -1)

                # Draw visibility info
                avg_visibility = sum(get_coords(i)[2] for i in critical_indices) / len(critical_indices)
                vis_text = f"Visibility: {avg_visibility:.1%}"
                cv2.putText(frame, vis_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No pose detected
                cv2.putText(frame, "No pose detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw legend
            legend_y = 60
            cv2.putText(frame, "Legend:", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.line(frame, (10, legend_y + 10), (50, legend_y + 10), COLORS['arms'], 2)
            cv2.putText(frame, "Arms", (55, legend_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.line(frame, (10, legend_y + 25), (50, legend_y + 25), COLORS['legs'], 2)
            cv2.putText(frame, "Legs", (55, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Save frame if writing
            if writer:
                writer.write(frame)

        # Display frame
        cv2.imshow('Pose Analysis - Press Q to quit, SPACE to pause', frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("⏹️ User stopped playback")
            break
        elif key == ord(' '):
            paused = not paused
            logger.info(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.info(f"✅ Processed {frame_count} frames")


if __name__ == "__main__":
    # Default to Steph Curry video in baselines folder
    default_video = "baselines/StephCurryShot.mp4"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = default_video

    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print("🏀 BASKETBALL POSE ANALYSIS VISUALIZATION")
    print("=" * 60)
    print(f"Video: {video_path}")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press SPACE to pause/resume")
    print("=" * 60)

    visualize_pose_processing(video_path, output_path)

