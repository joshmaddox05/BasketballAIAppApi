"""
Video Visualizer - Create color-coded overlay videos showing shot form quality
Green = Good technique | Yellow = Needs improvement | Red = Poor technique
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoVisualizer:
    """Create annotated videos with color-coded pose overlays"""

    # Color scheme (BGR format for OpenCV)
    COLOR_EXCELLENT = (0, 255, 0)      # Green
    COLOR_GOOD = (0, 200, 100)         # Light green
    COLOR_AVERAGE = (0, 255, 255)      # Yellow
    COLOR_POOR = (0, 165, 255)         # Orange
    COLOR_BAD = (0, 0, 255)            # Red
    COLOR_WHITE = (255, 255, 255)      # White
    COLOR_BLACK = (0, 0, 0)            # Black

    # Pose connections for skeleton visualization
    POSE_CONNECTIONS = [
        # Face
        ('NOSE', 'LEFT_EYE'), ('NOSE', 'RIGHT_EYE'),
        ('LEFT_EYE', 'LEFT_EAR'), ('RIGHT_EYE', 'RIGHT_EAR'),

        # Torso
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_HIP'),
        ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP'),

        # Arms
        ('LEFT_SHOULDER', 'LEFT_ELBOW'),
        ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        ('RIGHT_ELBOW', 'RIGHT_WRIST'),

        # Legs
        ('LEFT_HIP', 'LEFT_KNEE'),
        ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'),
        ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    ]

    def __init__(self, output_dir: str = "output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ VideoVisualizer initialized. Output: {self.output_dir}")

    def _normalize_keypoints(self, keypoints: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize keypoint names to uppercase for consistent lookup"""
        normalized = {}
        for key, value in keypoints.items():
            # Skip non-keypoint fields like 'frame' and 'timestamp'
            if key in ('frame', 'timestamp'):
                normalized[key] = value
            else:
                # Convert to uppercase (e.g., 'right_shoulder' -> 'RIGHT_SHOULDER')
                normalized[key.upper()] = value
        return normalized

    def create_overlay_video(
        self,
        video_path: str,
        keypoints_sequence: List[Dict[str, Any]],
        analysis_results: Dict[str, Any],
        output_name: Optional[str] = None
    ) -> str:
        """
        Create video with color-coded pose overlay

        Args:
            video_path: Path to original video
            keypoints_sequence: Pose keypoints for each frame
            analysis_results: Analysis results with metrics and phases
            output_name: Optional output filename

        Returns:
            Path to output video
        """
        logger.info(f"🎨 Creating visualization overlay for: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video
        if output_name is None:
            output_name = f"overlay_{Path(video_path).stem}.mp4"
        output_path = self.output_dir / output_name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Extract metrics for color coding
        metrics = analysis_results.get('metrics', {})
        phases = analysis_results.get('phases', {})
        overall_score = analysis_results.get('overall_score', 50)
        shooting_hand = phases.get('shooting_hand', 'right')

        frame_idx = 0
        keypoint_idx = 0

        logger.info(f"📹 Processing {total_frames} frames...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get keypoints for this frame
            if keypoint_idx < len(keypoints_sequence):
                kp = keypoints_sequence[keypoint_idx]

                # Check if this is the right frame
                if kp.get('frame', keypoint_idx) == frame_idx:
                    # Determine phase and color
                    phase_name, phase_color = self._get_phase_info(frame_idx, phases)

                    # Draw skeleton overlay
                    frame = self._draw_skeleton(
                        frame, kp, metrics, phase_name, phase_color, shooting_hand
                    )

                    # Draw metrics overlay
                    frame = self._draw_metrics_overlay(
                        frame, metrics, overall_score, phase_name
                    )

                    keypoint_idx += 1

            out.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                logger.info(f"   Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        out.release()

        logger.info(f"✅ Visualization saved: {output_path}")
        return str(output_path)

    def _detect_rotation(self, cap) -> Optional[int]:
        """Detect if video needs rotation based on metadata"""
        # For now, we won't auto-rotate videos to avoid keypoint misalignment
        # The keypoints are already in the original video coordinate system
        # If we rotate the frame, we'd need to also transform all keypoints
        # which adds complexity and potential for errors

        # TODO: If rotation is needed, implement coordinate transformation
        # to transform keypoints according to rotation applied to frames

        return None

    def _get_phase_info(
        self,
        frame_idx: int,
        phases: Dict[str, Any]
    ) -> Tuple[str, Tuple[int, int, int]]:
        """Get current phase name and color for a frame"""

        load_frame = phases.get('load', {}).get('frame') if phases.get('load') else None
        release_frame = phases.get('release', {}).get('frame') if phases.get('release') else None
        follow_through_frame = phases.get('follow_through_end', {}).get('frame') if phases.get('follow_through_end') else None

        if release_frame and frame_idx >= release_frame:
            if follow_through_frame and frame_idx >= follow_through_frame:
                return "Landing", self.COLOR_GOOD
            return "Release/Follow-Through", self.COLOR_EXCELLENT
        elif load_frame and frame_idx >= load_frame:
            return "Loading", self.COLOR_AVERAGE
        else:
            return "Setup", self.COLOR_WHITE

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Any],
        metrics: Dict[str, Any],
        phase_name: str,
        base_color: Tuple[int, int, int],
        shooting_hand: str = 'right'
    ) -> np.ndarray:
        """Draw pose skeleton with color-coded joints based on form quality"""

        height, width = frame.shape[:2]
        overlay = frame.copy()

        # Normalize keypoints to uppercase for consistent lookup with POSE_CONNECTIONS
        keypoints = self._normalize_keypoints(keypoints)

        # Get joint quality colors (using detected shooting hand)
        joint_colors = self._get_joint_colors(keypoints, metrics, base_color, shooting_hand)

        # Draw connections (bones)
        for connection in self.POSE_CONNECTIONS:
            point1_name, point2_name = connection

            if point1_name in keypoints and point2_name in keypoints:
                p1_data = keypoints[point1_name]
                p2_data = keypoints[point2_name]

                if isinstance(p1_data, dict) and isinstance(p2_data, dict):
                    # Convert normalized coordinates to pixel coordinates
                    p1 = (int(p1_data['x'] * width), int(p1_data['y'] * height))
                    p2 = (int(p2_data['x'] * width), int(p2_data['y'] * height))

                    # Use average color of the two joints
                    color1 = joint_colors.get(point1_name, base_color)
                    color2 = joint_colors.get(point2_name, base_color)
                    avg_color = tuple(int((c1 + c2) / 2) for c1, c2 in zip(color1, color2))

                    # Draw line
                    cv2.line(overlay, p1, p2, avg_color, 3)

        # Draw joints (circles)
        for joint_name, joint_data in keypoints.items():
            if isinstance(joint_data, dict) and 'x' in joint_data and 'y' in joint_data:
                x = int(joint_data['x'] * width)
                y = int(joint_data['y'] * height)

                color = joint_colors.get(joint_name, base_color)

                # Draw outer circle
                cv2.circle(overlay, (x, y), 8, color, -1)
                # Draw inner circle (white)
                cv2.circle(overlay, (x, y), 4, self.COLOR_WHITE, -1)

        # Blend overlay with original frame
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    def _get_joint_colors(
        self,
        keypoints: Dict[str, Any],
        metrics: Dict[str, Any],
        base_color: Tuple[int, int, int],
        shooting_hand: str = 'right'
    ) -> Dict[str, Tuple[int, int, int]]:
        """Determine color for each joint based on form quality"""

        joint_colors = {}

        # Color shooting arm based on metrics (using detected shooting hand)
        hand = shooting_hand.upper()
        shooting_arm_joints = [f'{hand}_SHOULDER', f'{hand}_ELBOW', f'{hand}_WRIST']

        # Elbow flare quality
        elbow_flare = metrics.get('elbow_flare', {})
        elbow_quality = elbow_flare.get('quality_score', 5.0)
        elbow_color = self._quality_to_color(elbow_quality)

        for joint in shooting_arm_joints:
            joint_colors[joint] = elbow_color

        # Release angle quality (shoulder-elbow-wrist)
        release_angle = metrics.get('release_angle', {})
        release_quality = release_angle.get('quality_score', 5.0)
        release_color = self._quality_to_color(release_quality)

        joint_colors[f'{hand}_ELBOW'] = release_color  # Override with release quality

        # Knee quality
        knee_load = metrics.get('knee_load', {})
        knee_quality = knee_load.get('quality_score', 5.0)
        knee_color = self._quality_to_color(knee_quality)

        for joint in ['RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_HIP', 'LEFT_HIP']:
            joint_colors[joint] = knee_color

        # Base width quality (ankles)
        base_width = metrics.get('base_width', {})
        base_quality = base_width.get('quality_score', 5.0)
        base_color_result = self._quality_to_color(base_quality)

        for joint in ['RIGHT_ANKLE', 'LEFT_ANKLE']:
            joint_colors[joint] = base_color_result

        return joint_colors

    def _quality_to_color(self, quality_score: float) -> Tuple[int, int, int]:
        """Convert quality score (0-10) to color"""
        if quality_score >= 8.0:
            return self.COLOR_EXCELLENT  # Green
        elif quality_score >= 6.5:
            return self.COLOR_GOOD  # Light green
        elif quality_score >= 5.0:
            return self.COLOR_AVERAGE  # Yellow
        elif quality_score >= 3.0:
            return self.COLOR_POOR  # Orange
        else:
            return self.COLOR_BAD  # Red

    def _draw_metrics_overlay(
        self,
        frame: np.ndarray,
        metrics: Dict[str, Any],
        overall_score: float,
        phase_name: str
    ) -> np.ndarray:
        """Draw metrics information overlay on frame"""

        height, width = frame.shape[:2]

        # Create semi-transparent overlay panel
        panel_height = 180
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background

        # Overall score
        score_color = self._quality_to_color(overall_score / 10)
        cv2.putText(
            panel,
            f"Overall: {overall_score:.1f}/100",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            score_color,
            2
        )

        # Phase
        cv2.putText(
            panel,
            f"Phase: {phase_name}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_WHITE,
            1
        )

        # Key metrics
        y_pos = 90
        metrics_to_show = [
            ('Release Angle', 'release_angle'),
            ('Elbow Form', 'elbow_flare'),
            ('Knee Bend', 'knee_load'),
            ('Balance', 'base_width')
        ]

        for label, metric_key in metrics_to_show:
            metric_data = metrics.get(metric_key, {})
            if 'quality_score' in metric_data:
                quality = metric_data['quality_score']
                color = self._quality_to_color(quality)

                # Draw metric bar
                bar_width = int((quality / 10.0) * 150)
                cv2.rectangle(panel, (150, y_pos - 10), (150 + bar_width, y_pos + 5), color, -1)
                cv2.rectangle(panel, (150, y_pos - 10), (300, y_pos + 5), self.COLOR_WHITE, 1)

                cv2.putText(
                    panel,
                    label,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLOR_WHITE,
                    1
                )

            y_pos += 25

        # Add legend
        legend_y = panel_height - 20
        legend_items = [
            ("Excellent", self.COLOR_EXCELLENT),
            ("Good", self.COLOR_GOOD),
            ("Average", self.COLOR_AVERAGE),
            ("Poor", self.COLOR_POOR),
            ("Bad", self.COLOR_BAD)
        ]

        x_offset = width - 400
        for label, color in legend_items:
            cv2.circle(panel, (x_offset, legend_y), 6, color, -1)
            cv2.putText(
                panel,
                label,
                (x_offset + 15, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.COLOR_WHITE,
                1
            )
            x_offset += 80

        # Blend panel with frame
        alpha = 0.85
        frame[0:panel_height] = cv2.addWeighted(
            panel, alpha, frame[0:panel_height], 1 - alpha, 0
        )

        return frame
