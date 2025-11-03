"""
Wrist Mechanics Calculator - Advanced wrist flick and palm angle analysis
Computes wrist flick velocity, follow-through, and palm orientation
"""
import numpy as np
from typing import Dict, List, Any, Optional
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)


class WristMechanicsCalculator:
    """Calculate advanced wrist mechanics for shooting analysis"""

    def __init__(self):
        """Initialize wrist mechanics calculator"""
        logger.info("✅ WristMechanicsCalculator initialized")

    def compute_wrist_flick_velocity(
        self,
        keypoints_sequence: List[Dict],
        phases: Dict[str, Any],
        fps: float = 30.0,
        hand_data_sequence: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Compute wrist flick angular velocity around release

        Release window: [release-100ms, release+120ms]
        Peak flick velocity measured in deg/s
        """
        if not phases.get('valid'):
            return {'error': 'Invalid phases'}

        release_frame = phases.get('release', {}).get('frame')
        if release_frame is None:
            return {'error': 'No release frame detected'}

        # Define release window (±100ms = ±3 frames at 30fps)
        window_frames = int(0.1 * fps)
        start_frame = max(0, release_frame - window_frames)
        end_frame = min(len(keypoints_sequence), release_frame + int(0.12 * fps))

        try:
            wrist_angles = []
            frame_indices = []

            # Extract wrist angles over time
            for idx in range(start_frame, end_frame):
                if idx >= len(keypoints_sequence):
                    break

                frame = keypoints_sequence[idx]

                # Use hand data if available, otherwise fall back to pose
                if hand_data_sequence and idx < len(hand_data_sequence):
                    hand_data = hand_data_sequence[idx]
                    if hand_data.get('right'):
                        angle = self._compute_wrist_angle_from_hands(hand_data['right'], frame)
                    else:
                        angle = self._compute_wrist_angle_from_pose(frame)
                else:
                    angle = self._compute_wrist_angle_from_pose(frame)

                if angle is not None:
                    wrist_angles.append(angle)
                    frame_indices.append(idx)

            if len(wrist_angles) < 3:
                return {'error': 'Insufficient wrist data in release window'}

            # Smooth angles
            if len(wrist_angles) >= 5:
                wrist_angles_smooth = savgol_filter(wrist_angles, 5, 2)
            else:
                wrist_angles_smooth = wrist_angles

            # Compute angular velocity (deg/frame)
            angular_velocities = np.diff(wrist_angles_smooth)

            # Convert to deg/s
            angular_velocities_deg_s = angular_velocities * fps

            # Find peak flick
            peak_flick_idx = np.argmax(np.abs(angular_velocities_deg_s))
            peak_flick_deg_s = float(angular_velocities_deg_s[peak_flick_idx])

            # Follow-through: how long wrist stays extended after release
            followthrough_frames = self._compute_followthrough_hold(
                wrist_angles_smooth,
                frame_indices,
                release_frame
            )
            followthrough_ms = (followthrough_frames / fps) * 1000

            return {
                'peak_flick_deg_s': abs(peak_flick_deg_s),
                'peak_flick_frame': int(frame_indices[peak_flick_idx]) if peak_flick_idx < len(frame_indices) else release_frame,
                'followthrough_ms': float(followthrough_ms),
                'mean_velocity_deg_s': float(np.mean(np.abs(angular_velocities_deg_s))),
                'angle_range_deg': float(np.max(wrist_angles_smooth) - np.min(wrist_angles_smooth))
            }

        except Exception as e:
            logger.warning(f"Wrist flick velocity computation error: {str(e)}")
            return {'error': str(e)}

    def _compute_wrist_angle_from_pose(self, frame: Dict) -> Optional[float]:
        """Compute wrist angle from pose landmarks only"""
        if not all(k in frame for k in ['right_elbow', 'right_wrist']):
            return None

        try:
            elbow = frame['right_elbow']
            wrist = frame['right_wrist']

            if not all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [elbow, wrist]):
                return None

            # Vector from elbow to wrist
            forearm = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])

            # Angle vs horizontal
            angle = np.degrees(np.arctan2(forearm[1], forearm[0]))

            return float(angle)

        except Exception:
            return None

    def _compute_wrist_angle_from_hands(self, hand_data: Dict, frame: Dict) -> Optional[float]:
        """Compute wrist angle from hand landmarks"""
        if 'wrist' not in hand_data or 'middle_mcp' not in hand_data:
            return self._compute_wrist_angle_from_pose(frame)

        try:
            wrist = hand_data['wrist']
            middle_mcp = hand_data['middle_mcp']

            # Hand pitch angle
            hand_vec = np.array([middle_mcp['x'] - wrist['x'], middle_mcp['y'] - wrist['y']])
            angle = np.degrees(np.arctan2(hand_vec[1], hand_vec[0]))

            return float(angle)

        except Exception:
            return self._compute_wrist_angle_from_pose(frame)

    def _compute_followthrough_hold(
        self,
        angles: np.ndarray,
        frame_indices: List[int],
        release_frame: int
    ) -> int:
        """
        Compute how long wrist stays extended (high angle) after release
        """
        release_idx_in_window = None
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx >= release_frame:
                release_idx_in_window = i
                break

        if release_idx_in_window is None or release_idx_in_window >= len(angles) - 1:
            return 0

        # Find angle at release
        release_angle = angles[release_idx_in_window]

        # Threshold: angle must stay within 20° of release angle
        threshold = release_angle - 20

        # Count frames above threshold
        hold_frames = 0
        for i in range(release_idx_in_window + 1, len(angles)):
            if angles[i] >= threshold:
                hold_frames += 1
            else:
                break

        return hold_frames

    def compute_palm_metrics(
        self,
        hand_data_sequence: List[Dict],
        phases: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute palm orientation metrics at set point and release
        """
        if not phases.get('valid'):
            return {'error': 'Invalid phases'}

        load_frame = phases.get('load', {}).get('frame', 0)
        release_frame = phases.get('release', {}).get('frame')

        if release_frame is None or not hand_data_sequence:
            return {'error': 'No hand data available'}

        try:
            # Get hand data at key moments
            set_point_hands = hand_data_sequence[load_frame] if load_frame < len(hand_data_sequence) else None
            release_hands = hand_data_sequence[release_frame] if release_frame < len(hand_data_sequence) else None

            results = {}

            # Compute at set point
            if set_point_hands and set_point_hands.get('right'):
                palm_angles_set = self._compute_palm_angles(set_point_hands['right'])
                if palm_angles_set:
                    results['set_point'] = palm_angles_set

            # Compute at release
            if release_hands and release_hands.get('right'):
                palm_angles_release = self._compute_palm_angles(release_hands['right'])
                if palm_angles_release:
                    results['release'] = palm_angles_release

            # Average metrics
            if 'set_point' in results and 'release' in results:
                results['average'] = {
                    'palm_angle_to_vertical_deg': (
                        results['set_point']['angle_to_vertical_deg'] +
                        results['release']['angle_to_vertical_deg']
                    ) / 2,
                    'palm_toward_target_deg': (
                        results['set_point']['angle_to_target_deg'] +
                        results['release']['angle_to_target_deg']
                    ) / 2
                }

            return results

        except Exception as e:
            logger.warning(f"Palm metrics computation error: {str(e)}")
            return {'error': str(e)}

    def _compute_palm_angles(self, hand_landmarks: Dict) -> Optional[Dict[str, float]]:
        """Compute palm plane angles from hand landmarks"""
        if not all(k in hand_landmarks for k in ['wrist', 'index_mcp', 'pinky_mcp']):
            return None

        try:
            # Get 3D points
            wrist = np.array([hand_landmarks['wrist'][k] for k in ['x', 'y', 'z']])
            index_mcp = np.array([hand_landmarks['index_mcp'][k] for k in ['x', 'y', 'z']])
            pinky_mcp = np.array([hand_landmarks['pinky_mcp'][k] for k in ['x', 'y', 'z']])

            # Compute palm plane normal
            v1 = index_mcp - wrist
            v2 = pinky_mcp - wrist
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            # Vertical axis (Y-up)
            vertical = np.array([0, 1, 0])

            # Target axis (X-forward, assuming camera faces basket)
            target = np.array([1, 0, 0])

            # Compute angles
            angle_to_vertical = np.degrees(np.arccos(np.clip(
                np.abs(np.dot(normal, vertical)), 0.0, 1.0
            )))

            angle_to_target = np.degrees(np.arccos(np.clip(
                np.abs(np.dot(normal, target)), 0.0, 1.0
            )))

            return {
                'angle_to_vertical_deg': float(angle_to_vertical),
                'angle_to_target_deg': float(angle_to_target),
                'palm_normal': normal.tolist()
            }

        except Exception:
            return None

