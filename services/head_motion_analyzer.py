"""
Head Motion Analyzer - Head stability and gaze tracking
Tracks head orientation and movement for shooting consistency
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)


class HeadMotionAnalyzer:
    """Analyze head stability and orientation during shooting motion"""

    # Head landmarks from MediaPipe Pose
    HEAD_LANDMARKS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']

    def __init__(self):
        """Initialize head motion analyzer"""
        logger.info("✅ HeadMotionAnalyzer initialized")

    def compute_head_orientation(self, keypoints: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Compute head orientation angles (roll, pitch, yaw)

        Roll: ear-to-ear tilt
        Pitch: eye-to-nose angle vs horizontal
        Yaw: face rotation (using ear positions)
        """
        if not all(k in keypoints for k in ['left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear']):
            return None

        try:
            # Extract points
            left_eye = np.array([keypoints['left_eye']['x'], keypoints['left_eye']['y']])
            right_eye = np.array([keypoints['right_eye']['x'], keypoints['right_eye']['y']])
            nose = np.array([keypoints['nose']['x'], keypoints['nose']['y']])
            left_ear = np.array([keypoints['left_ear']['x'], keypoints['left_ear']['y']])
            right_ear = np.array([keypoints['right_ear']['x'], keypoints['right_ear']['y']])

            # Roll: angle of eye-to-eye line vs horizontal
            eye_vector = right_eye - left_eye
            roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

            # Pitch: angle of midpoint-eyes to nose vs vertical
            eye_midpoint = (left_eye + right_eye) / 2
            face_vector = nose - eye_midpoint
            pitch = np.degrees(np.arctan2(face_vector[0], face_vector[1]))

            # Yaw: ear separation (proxy for face rotation)
            ear_vector = right_ear - left_ear
            ear_distance = np.linalg.norm(ear_vector)
            eye_distance = np.linalg.norm(eye_vector)

            # Yaw estimate (when face turns, ear distance decreases relative to eye distance)
            yaw_proxy = (ear_distance / (eye_distance + 1e-6)) - 1.0

            return {
                'roll_deg': float(roll),
                'pitch_deg': float(pitch),
                'yaw_proxy': float(yaw_proxy),
                'head_tilt_deg': float(abs(roll))  # Absolute tilt for stability metric
            }

        except Exception as e:
            logger.warning(f"Head orientation computation error: {str(e)}")
            return None

    def compute_gaze_stability(
        self,
        keypoints_sequence: List[Dict],
        load_frame: int,
        release_frame: int
    ) -> Optional[Dict[str, float]]:
        """
        Compute head/gaze stability metrics between load and release

        Measures:
        - Lateral displacement of nose vs mid-hip
        - Max angular velocity (head jerk)
        """
        if load_frame >= release_frame or load_frame < 0:
            return None

        try:
            nose_positions = []
            mid_hip_positions = []
            head_angles = []

            for frame_idx in range(load_frame, min(release_frame + 1, len(keypoints_sequence))):
                frame = keypoints_sequence[frame_idx]

                # Nose position
                if 'nose' in frame and isinstance(frame['nose'], dict):
                    nose = frame['nose']
                    if nose.get('visibility', 0) > 0.5:
                        nose_positions.append([nose['x'], nose['y']])

                # Mid-hip position (reference)
                if 'left_hip' in frame and 'right_hip' in frame:
                    left_hip = frame['left_hip']
                    right_hip = frame['right_hip']
                    if all(isinstance(h, dict) and h.get('visibility', 0) > 0.5 for h in [left_hip, right_hip]):
                        mid_hip = [(left_hip['x'] + right_hip['x']) / 2, (left_hip['y'] + right_hip['y']) / 2]
                        mid_hip_positions.append(mid_hip)

                # Head orientation for jerk calculation
                head_orient = self.compute_head_orientation(frame)
                if head_orient:
                    head_angles.append(head_orient['roll_deg'])

            if not nose_positions or not mid_hip_positions:
                return None

            # Convert to numpy arrays
            nose_positions = np.array(nose_positions)
            mid_hip_positions = np.array(mid_hip_positions)

            # Compute relative displacement
            min_len = min(len(nose_positions), len(mid_hip_positions))
            nose_rel = nose_positions[:min_len] - mid_hip_positions[:min_len]

            # Lateral displacement (x-axis movement)
            lateral_displacement = np.std(nose_rel[:, 0]) if len(nose_rel) > 0 else 0

            # Total displacement magnitude
            total_displacement = np.linalg.norm(nose_positions[-1] - nose_positions[0]) if len(nose_positions) > 1 else 0

            # Head jerk (angular velocity)
            head_jerk = 0
            if len(head_angles) > 2:
                # Smooth angles
                if len(head_angles) >= 5:
                    head_angles_smooth = savgol_filter(head_angles, min(5, len(head_angles)), 2)
                else:
                    head_angles_smooth = head_angles

                # Compute angular velocity (deg/frame)
                angular_velocities = np.diff(head_angles_smooth)
                head_jerk = float(np.max(np.abs(angular_velocities))) if len(angular_velocities) > 0 else 0

            return {
                'lateral_displacement': float(lateral_displacement),
                'total_displacement': float(total_displacement),
                'head_jerk_deg_per_frame': float(head_jerk),
                'frames_analyzed': min_len
            }

        except Exception as e:
            logger.warning(f"Gaze stability computation error: {str(e)}")
            return None

    def compute_head_stability_score(
        self,
        head_tilt_deg: float,
        head_jerk_deg_s: float,
        gaze_displacement_cm: float
    ) -> Dict[str, Any]:
        """
        Compute overall head stability score and grade

        Targets:
        - Head tilt: 0-8° (good)
        - Head jerk: <50 deg/s (good)
        - Gaze displacement: <3 cm (good)
        """
        scores = []

        # Tilt score
        if head_tilt_deg <= 8:
            tilt_score = 1.0
        elif head_tilt_deg <= 15:
            tilt_score = 0.7
        else:
            tilt_score = 0.4
        scores.append(tilt_score)

        # Jerk score
        if head_jerk_deg_s < 50:
            jerk_score = 1.0
        elif head_jerk_deg_s < 100:
            jerk_score = 0.6
        else:
            jerk_score = 0.3
        scores.append(jerk_score)

        # Displacement score
        if gaze_displacement_cm <= 3:
            displacement_score = 1.0
        elif gaze_displacement_cm <= 6:
            displacement_score = 0.7
        else:
            displacement_score = 0.4
        scores.append(displacement_score)

        # Overall score
        overall_score = np.mean(scores)

        # Grade
        if overall_score >= 0.9:
            grade = 'A'
        elif overall_score >= 0.75:
            grade = 'B'
        elif overall_score >= 0.6:
            grade = 'C'
        else:
            grade = 'D'

        return {
            'score': float(overall_score),
            'grade': grade,
            'tilt_score': float(tilt_score),
            'jerk_score': float(jerk_score),
            'displacement_score': float(displacement_score)
        }

    def analyze_head_motion_sequence(
        self,
        keypoints_sequence: List[Dict],
        phases: Dict[str, Any],
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Analyze head motion throughout the shooting sequence

        Returns comprehensive head stability metrics
        """
        # Relaxed validation - work with fallback phases
        if not phases.get('load') or not phases.get('release'):
            # Use middle portion of video
            total_frames = len(keypoints_sequence)
            load_frame = total_frames // 3
            release_frame = 2 * total_frames // 3
        else:
            load_frame = phases.get('load', {}).get('frame', 0)
            release_frame = phases.get('release', {}).get('frame', len(keypoints_sequence) - 1)

        # Compute head orientation at key moments
        set_point_orient = None
        release_orient = None

        if load_frame < len(keypoints_sequence):
            set_point_orient = self.compute_head_orientation(keypoints_sequence[load_frame])

        if release_frame < len(keypoints_sequence):
            release_orient = self.compute_head_orientation(keypoints_sequence[release_frame])

        # Compute gaze stability
        gaze_stability = self.compute_gaze_stability(keypoints_sequence, load_frame, release_frame)

        # Build results
        results = {
            'set_point_orientation': set_point_orient,
            'release_orientation': release_orient,
            'gaze_stability': gaze_stability
        }

        # Compute overall metrics
        if set_point_orient and release_orient and gaze_stability:
            # Average tilt
            avg_tilt = (set_point_orient['head_tilt_deg'] + release_orient['head_tilt_deg']) / 2

            # Convert head jerk to deg/s (from deg/frame)
            head_jerk_deg_s = gaze_stability['head_jerk_deg_per_frame'] * fps

            # Convert displacement to cm (assuming normalized coords, body height ~170cm)
            body_height_cm = 170
            gaze_displacement_cm = gaze_stability['lateral_displacement'] * body_height_cm

            results['summary'] = {
                'head_tilt_deg': float(avg_tilt),
                'head_yaw_jitter_deg_s': float(head_jerk_deg_s),
                'gaze_stability_cm': float(gaze_displacement_cm)
            }

            # Compute stability score
            stability_score = self.compute_head_stability_score(
                avg_tilt,
                head_jerk_deg_s,
                gaze_displacement_cm
            )
            results['stability_score'] = stability_score

        return results

