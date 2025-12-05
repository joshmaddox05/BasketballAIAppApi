"""
Shot Phase Detection - Identify key moments in basketball shooting motion
Detects: Dip Start, Load, Release, Follow-through, Landing
Uses heuristics with hysteresis and min-duration guards
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class PhaseDetector:
    """Detect shooting phases from pose keypoints"""
    
    # Minimum durations (in seconds)
    MIN_DIP_DURATION = 0.08  # 80ms
    MIN_LOAD_TO_RELEASE = 0.10  # 100ms
    MAX_LOAD_TO_RELEASE = 1.5  # 1500ms - allow for slow motion videos
    
    # Thresholds
    KNEE_ANGLE_FLEX_THRESHOLD = 10  # degrees of flexion to detect dip (reduced from 15)
    WRIST_VELOCITY_THRESHOLD = 0.8  # normalized units/sec (reduced from 1.5)
    ELBOW_EXTENSION_THRESHOLD = 0.5  # normalized velocity (reduced from 0.8)
    
    def __init__(self):
        logger.info("✅ PhaseDetector initialized")

    def _detect_shooting_hand(self, keypoints_sequence: List[Dict], fps: float) -> str:
        """
        Detect shooting hand based on which wrist has more vertical movement.
        The shooting arm is the one that moves the most during the shot.

        Returns:
            'right' or 'left' based on which arm has higher peak velocity
        """
        right_wrist_y = []
        left_wrist_y = []

        for kp in keypoints_sequence:
            # Get right wrist y position
            if 'right_wrist' in kp and isinstance(kp['right_wrist'], dict):
                right_wrist_y.append(kp['right_wrist'].get('y', 0))
            else:
                right_wrist_y.append(right_wrist_y[-1] if right_wrist_y else 0)

            # Get left wrist y position
            if 'left_wrist' in kp and isinstance(kp['left_wrist'], dict):
                left_wrist_y.append(kp['left_wrist'].get('y', 0))
            else:
                left_wrist_y.append(left_wrist_y[-1] if left_wrist_y else 0)

        if len(right_wrist_y) < 2 or len(left_wrist_y) < 2:
            logger.warning("Insufficient data for shooting hand detection, defaulting to right")
            return 'right'

        # Calculate velocities (absolute value of change)
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        right_velocity = np.abs(np.diff(right_wrist_y)) / dt
        left_velocity = np.abs(np.diff(left_wrist_y)) / dt

        # Find peak velocities
        right_peak = np.max(right_velocity) if len(right_velocity) > 0 else 0
        left_peak = np.max(left_velocity) if len(left_velocity) > 0 else 0

        # Also check total movement range
        right_range = np.max(right_wrist_y) - np.min(right_wrist_y)
        left_range = np.max(left_wrist_y) - np.min(left_wrist_y)

        # Combine velocity and range for more robust detection
        right_score = right_peak + right_range * 10
        left_score = left_peak + left_range * 10

        shooting_hand = 'right' if right_score >= left_score else 'left'
        logger.info(f"🎯 Detected shooting hand: {shooting_hand} (right_score={right_score:.2f}, left_score={left_score:.2f})")

        return shooting_hand

    def detect_phases(self, keypoints_sequence: List[Dict], fps: float = 30.0) -> Dict[str, Any]:
        """
        Detect all shooting phases from keypoint sequence
        
        Returns:
            Dictionary with phase timestamps and confidence scores
        """
        if len(keypoints_sequence) < 10:
            return {
                'confidence': 0.0,
                'error': 'Insufficient frames for phase detection'
            }

        # Detect shooting hand first - this determines which arm to analyze
        shooting_hand = self._detect_shooting_hand(keypoints_sequence, fps)

        # Extract time series
        timestamps = [kp['timestamp'] for kp in keypoints_sequence if kp.get('visible', True)]

        # Detect each phase (using the detected shooting hand)
        dip_start = self._detect_dip_start(keypoints_sequence, shooting_hand)
        load_point = self._detect_load(keypoints_sequence, dip_start)
        release_point = self._detect_release(keypoints_sequence, load_point, shooting_hand)
        follow_through_end = self._detect_follow_through_end(keypoints_sequence, release_point, shooting_hand)
        landing = self._detect_landing(keypoints_sequence, release_point)
        
        # Validate phase sequence
        phases_valid = self._validate_phases(dip_start, load_point, release_point)
        
        phases = {
            'dip_start': dip_start,
            'load': load_point,
            'release': release_point,
            'follow_through_end': follow_through_end,
            'landing': landing,
            'shooting_hand': shooting_hand,
            'confidence': self._calculate_confidence(
                dip_start, load_point, release_point, phases_valid
            ),
            'valid': phases_valid
        }
        
        # Calculate timing metrics
        if phases_valid:
            phases['timing'] = self._calculate_timing_metrics(phases)
        
        logger.info(f"📊 Detected phases: {phases}")
        return phases
    
    def _detect_dip_start(self, keypoints_sequence: List[Dict], shooting_hand: str = 'right') -> Optional[Dict]:
        """
        Detect dip start: wrist y decreases and knee angle starts decreasing
        """
        wrist_y_values = []
        knee_angles = []
        timestamps = []
        frame_numbers = []

        # Use the detected shooting hand
        wrist_key = f'{shooting_hand}_wrist'
        other_wrist_key = 'left_wrist' if shooting_hand == 'right' else 'right_wrist'

        for kp in keypoints_sequence:
            if not kp.get('visible', True):
                continue

            # Get wrist position from shooting hand, fallback to other hand
            wrist_y = None
            if wrist_key in kp and isinstance(kp[wrist_key], dict) and 'low_visibility' not in kp[wrist_key]:
                wrist_y = kp[wrist_key]['y']
            elif other_wrist_key in kp and isinstance(kp[other_wrist_key], dict) and 'low_visibility' not in kp[other_wrist_key]:
                wrist_y = kp[other_wrist_key]['y']
            
            # Calculate knee angle
            knee_angle = self._calculate_knee_angle(kp)
            
            if wrist_y is not None and knee_angle is not None:
                wrist_y_values.append(wrist_y)
                knee_angles.append(knee_angle)
                timestamps.append(kp['timestamp'])
                frame_numbers.append(kp['frame'])
        
        if len(wrist_y_values) < 5:
            return None
        
        # Find where wrist starts descending and knees start flexing
        wrist_velocity = np.diff(wrist_y_values)  # Positive = downward
        knee_angle_change = np.diff(knee_angles)  # Negative = flexing
        
        for i in range(1, len(wrist_velocity) - 3):
            # Check if wrist is descending
            if wrist_velocity[i] > 0 and wrist_velocity[i+1] > 0:
                # Check if knees are flexing
                if knee_angle_change[i] < -self.KNEE_ANGLE_FLEX_THRESHOLD:
                    return {
                        'frame': frame_numbers[i],
                        'timestamp': timestamps[i],
                        'wrist_y': wrist_y_values[i],
                        'knee_angle': knee_angles[i]
                    }
        
        # If not found, use first significant knee flexion
        for i in range(len(knee_angles) - 1):
            if knee_angles[i] - knee_angles[i+1] > self.KNEE_ANGLE_FLEX_THRESHOLD:
                return {
                    'frame': frame_numbers[i],
                    'timestamp': timestamps[i],
                    'wrist_y': wrist_y_values[i],
                    'knee_angle': knee_angles[i]
                }
        
        return None
    
    def _detect_load(self, keypoints_sequence: List[Dict], dip_start: Optional[Dict]) -> Optional[Dict]:
        """
        Detect load point: minimum knee angle (maximum flexion)
        """
        knee_angles = []
        timestamps = []
        frame_numbers = []
        
        # Start search from dip_start if available
        start_idx = 0
        if dip_start:
            start_idx = next((i for i, kp in enumerate(keypoints_sequence) 
                            if kp.get('frame') == dip_start['frame']), 0)
        
        for kp in keypoints_sequence[start_idx:]:
            if not kp.get('visible', True):
                continue
            
            knee_angle = self._calculate_knee_angle(kp)
            if knee_angle is not None:
                knee_angles.append(knee_angle)
                timestamps.append(kp['timestamp'])
                frame_numbers.append(kp['frame'])
        
        if len(knee_angles) < 3:
            return None

        # Find minimum knee angle (max flexion)
        min_idx = np.argmin(knee_angles)

        # Validate it's a local minimum (not at edges)
        # More lenient: allow if not at very first or very last frame
        if min_idx > 0 and min_idx < len(knee_angles) - 1:
            return {
                'frame': frame_numbers[min_idx],
                'timestamp': timestamps[min_idx],
                'knee_angle': knee_angles[min_idx],
                'phase': 'load'
            }

        # Fallback: use minimum even if at edge (better than nothing)
        if min_idx == 0 and len(knee_angles) > 5:
            # If at start, use frame 2-3 instead
            min_idx = 2
        elif min_idx == len(knee_angles) - 1 and len(knee_angles) > 5:
            # If at end, use a few frames before
            min_idx = len(knee_angles) - 3

        return {
            'frame': frame_numbers[min_idx],
            'timestamp': timestamps[min_idx],
            'knee_angle': knee_angles[min_idx],
            'phase': 'load',
            'fallback': True
        }
    
    def _detect_release(self, keypoints_sequence: List[Dict], load_point: Optional[Dict], shooting_hand: str = 'right') -> Optional[Dict]:
        """
        Detect release: point of maximum arm extension (elbow angle closest to 180°)
        The release happens when the shooting arm is most extended, not at peak wrist velocity.
        """
        if not load_point:
            return None

        # Start search from load point
        start_idx = next((i for i, kp in enumerate(keypoints_sequence)
                         if kp.get('frame') == load_point['frame']), 0)

        # Use the detected shooting hand
        wrist_key = f'{shooting_hand}_wrist'

        # Collect elbow angles and wrist positions
        elbow_angles = []
        wrist_positions = []
        timestamps = []
        frame_numbers = []

        for kp in keypoints_sequence[start_idx:]:
            if not kp.get('visible', True):
                continue

            # Calculate elbow angle (using shooting hand)
            elbow_angle = self._calculate_elbow_angle(kp, shooting_hand)

            # Get wrist position
            wrist_y = None
            if wrist_key in kp and isinstance(kp[wrist_key], dict) and 'low_visibility' not in kp[wrist_key]:
                wrist_y = kp[wrist_key]['y']

            if elbow_angle is not None and wrist_y is not None:
                elbow_angles.append(elbow_angle)
                wrist_positions.append(wrist_y)
                timestamps.append(kp['timestamp'])
                frame_numbers.append(kp['frame'])

        if len(elbow_angles) < 5:
            return None

        # Strategy: Find the frame with maximum elbow extension (closest to 180°)
        # This represents the point where the arm is most extended during the shot

        # Look for the maximum elbow angle after load
        # Also require the wrist to be relatively high (in upper portion of its range)
        wrist_min = min(wrist_positions)
        wrist_max = max(wrist_positions)
        wrist_range = wrist_max - wrist_min

        # Find candidates where elbow is extended AND wrist is high
        candidates = []
        for i in range(len(elbow_angles)):
            time_since_load = timestamps[i] - load_point['timestamp']

            # Must be at least 100ms after load to be a release
            if time_since_load < 0.1:
                continue

            # Wrist should be in upper 50% of its range (lower y = higher position)
            wrist_height_ratio = (wrist_max - wrist_positions[i]) / (wrist_range + 1e-6)

            if wrist_height_ratio >= 0.5 and elbow_angles[i] >= 100:  # Arm somewhat extended
                candidates.append({
                    'frame': frame_numbers[i],
                    'timestamp': timestamps[i],
                    'elbow_angle': float(elbow_angles[i]),
                    'wrist_y': float(wrist_positions[i]),
                    'wrist_height_ratio': wrist_height_ratio,
                    'time_since_load': time_since_load,
                    # Score: higher elbow angle = more extended = better
                    'score': elbow_angles[i] + (wrist_height_ratio * 20)
                })

        if candidates:
            # Return candidate with highest score (most extended arm + highest wrist)
            best = max(candidates, key=lambda x: x['score'])

            # Calculate wrist velocity at this point for output
            best_idx = frame_numbers.index(best['frame'])
            if best_idx > 0:
                dt = timestamps[best_idx] - timestamps[best_idx - 1]
                dt = max(dt, 0.001)
                wrist_velocity = -(wrist_positions[best_idx] - wrist_positions[best_idx - 1]) / dt
            else:
                wrist_velocity = 0.0

            return {
                'frame': best['frame'],
                'timestamp': best['timestamp'],
                'wrist_velocity': float(wrist_velocity),
                'elbow_angle': best['elbow_angle'],
                'time_since_load': best['time_since_load'],
                'quality': 'high' if best['elbow_angle'] >= 140 else 'medium'
            }

        # Fallback: just find maximum elbow extension
        max_extension_idx = np.argmax(elbow_angles)

        return {
            'frame': frame_numbers[max_extension_idx],
            'timestamp': timestamps[max_extension_idx],
            'wrist_velocity': 0.0,
            'elbow_angle': float(elbow_angles[max_extension_idx]),
            'fallback': True
        }
    
    def _detect_follow_through_end(self, keypoints_sequence: List[Dict], release_point: Optional[Dict], shooting_hand: str = 'right') -> Optional[Dict]:
        """
        Detect end of follow-through: wrist stops moving upward
        """
        if not release_point:
            return None

        start_idx = next((i for i, kp in enumerate(keypoints_sequence)
                         if kp.get('frame') == release_point['frame']), 0)

        # Use the detected shooting hand
        wrist_key = f'{shooting_hand}_wrist'
        other_wrist_key = 'left_wrist' if shooting_hand == 'right' else 'right_wrist'

        wrist_positions = []
        timestamps = []
        frame_numbers = []

        for kp in keypoints_sequence[start_idx:]:
            if not kp.get('visible', True):
                continue

            wrist_y = None
            if wrist_key in kp and isinstance(kp[wrist_key], dict) and 'low_visibility' not in kp[wrist_key]:
                wrist_y = kp[wrist_key]['y']
            elif other_wrist_key in kp and isinstance(kp[other_wrist_key], dict) and 'low_visibility' not in kp[other_wrist_key]:
                wrist_y = kp[other_wrist_key]['y']
            
            if wrist_y is not None:
                wrist_positions.append(wrist_y)
                timestamps.append(kp['timestamp'])
                frame_numbers.append(kp['frame'])
        
        if len(wrist_positions) < 3:
            return None
        
        # Find where wrist velocity becomes near zero or changes direction
        wrist_velocity = -np.diff(wrist_positions)  # Negative = upward
        
        for i in range(len(wrist_velocity)):
            if abs(wrist_velocity[i]) < 0.1:  # Near zero velocity
                return {
                    'frame': frame_numbers[i+1],
                    'timestamp': timestamps[i+1],
                    'wrist_y': wrist_positions[i+1]
                }
        
        # Default to last frame in sequence
        return {
            'frame': frame_numbers[-1],
            'timestamp': timestamps[-1],
            'wrist_y': wrist_positions[-1]
        }
    
    def _detect_landing(self, keypoints_sequence: List[Dict], release_point: Optional[Dict]) -> Optional[Dict]:
        """
        Detect landing: local maximum ankle y after flight
        """
        if not release_point:
            return None
        
        start_idx = next((i for i, kp in enumerate(keypoints_sequence) 
                         if kp.get('frame') == release_point['frame']), 0)
        
        ankle_positions = []
        timestamps = []
        frame_numbers = []
        
        for kp in keypoints_sequence[start_idx:]:
            if not kp.get('visible', True):
                continue
            
            # Average both ankles
            ankle_y = None
            if 'left_ankle' in kp and 'right_ankle' in kp:
                left_ankle = kp['left_ankle']
                right_ankle = kp['right_ankle']
                
                if ('low_visibility' not in left_ankle and 
                    'low_visibility' not in right_ankle):
                    ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
            
            if ankle_y is not None:
                ankle_positions.append(ankle_y)
                timestamps.append(kp['timestamp'])
                frame_numbers.append(kp['frame'])
        
        if len(ankle_positions) < 5:
            return None
        
        # Find peaks (local maxima) in ankle position
        peaks, _ = find_peaks(ankle_positions, distance=3)
        
        if len(peaks) > 0:
            landing_idx = peaks[0]  # First peak after release
            return {
                'frame': frame_numbers[landing_idx],
                'timestamp': timestamps[landing_idx],
                'ankle_y': ankle_positions[landing_idx]
            }
        
        return None
    
    def _calculate_knee_angle(self, keypoint: Dict) -> Optional[float]:
        """Calculate knee angle from hip, knee, ankle"""
        # Use right leg primarily
        if all(k in keypoint for k in ['right_hip', 'right_knee', 'right_ankle']):
            hip = keypoint['right_hip']
            knee = keypoint['right_knee']
            ankle = keypoint['right_ankle']
            
            if all('low_visibility' not in p for p in [hip, knee, ankle]):
                return self._calculate_angle(
                    (hip['x'], hip['y']),
                    (knee['x'], knee['y']),
                    (ankle['x'], ankle['y'])
                )
        
        # Fallback to left leg
        if all(k in keypoint for k in ['left_hip', 'left_knee', 'left_ankle']):
            hip = keypoint['left_hip']
            knee = keypoint['left_knee']
            ankle = keypoint['left_ankle']
            
            if all('low_visibility' not in p for p in [hip, knee, ankle]):
                return self._calculate_angle(
                    (hip['x'], hip['y']),
                    (knee['x'], knee['y']),
                    (ankle['x'], ankle['y'])
                )
        
        return None
    
    def _calculate_elbow_angle(self, keypoint: Dict, shooting_hand: str = 'right') -> Optional[float]:
        """Calculate elbow angle from shoulder, elbow, wrist using the shooting hand"""
        # Use shooting hand primarily
        shoulder_key = f'{shooting_hand}_shoulder'
        elbow_key = f'{shooting_hand}_elbow'
        wrist_key = f'{shooting_hand}_wrist'

        if all(k in keypoint for k in [shoulder_key, elbow_key, wrist_key]):
            shoulder = keypoint[shoulder_key]
            elbow = keypoint[elbow_key]
            wrist = keypoint[wrist_key]

            if all(isinstance(p, dict) and 'low_visibility' not in p for p in [shoulder, elbow, wrist]):
                return self._calculate_angle(
                    (shoulder['x'], shoulder['y']),
                    (elbow['x'], elbow['y']),
                    (wrist['x'], wrist['y'])
                )

        # Fallback to other arm
        other_hand = 'left' if shooting_hand == 'right' else 'right'
        shoulder_key = f'{other_hand}_shoulder'
        elbow_key = f'{other_hand}_elbow'
        wrist_key = f'{other_hand}_wrist'

        if all(k in keypoint for k in [shoulder_key, elbow_key, wrist_key]):
            shoulder = keypoint[shoulder_key]
            elbow = keypoint[elbow_key]
            wrist = keypoint[wrist_key]

            if all(isinstance(p, dict) and 'low_visibility' not in p for p in [shoulder, elbow, wrist]):
                return self._calculate_angle(
                    (shoulder['x'], shoulder['y']),
                    (elbow['x'], elbow['y']),
                    (wrist['x'], wrist['y'])
                )

        return None
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return float(angle)
    
    def _validate_phases(self, dip_start, load_point, release_point) -> bool:
        """Validate phase sequence is logical"""
        # Only require load and release - dip_start is optional
        if not all([load_point, release_point]):
            return False

        # Check temporal order if dip_start exists
        if dip_start:
            if not (dip_start['timestamp'] < load_point['timestamp'] < release_point['timestamp']):
                return False

            # Check timing constraints
            dip_to_load = load_point['timestamp'] - dip_start['timestamp']
            if dip_to_load < self.MIN_DIP_DURATION:
                return False
        else:
            # Without dip_start, just verify load comes before release
            if not (load_point['timestamp'] < release_point['timestamp']):
                return False

        # Check load to release timing
        load_to_release = release_point['timestamp'] - load_point['timestamp']
        if not (self.MIN_LOAD_TO_RELEASE <= load_to_release <= self.MAX_LOAD_TO_RELEASE):
            return False

        return True
    
    def _calculate_confidence(self, dip_start, load_point, release_point, phases_valid) -> float:
        """Calculate confidence score for phase detection"""
        if not phases_valid:
            return 0.0

        confidence = 1.0

        # Penalize if critical phases are missing
        if not load_point:
            confidence *= 0.5
        if not release_point:
            confidence *= 0.3

        # Minor penalty if dip_start is missing (it's optional)
        if not dip_start:
            confidence *= 0.9

        # Penalize fallback detections
        if release_point and release_point.get('fallback'):
            confidence *= 0.85
        if load_point and load_point.get('fallback'):
            confidence *= 0.9

        # Bonus for high-quality release detection
        if release_point and release_point.get('quality') == 'high':
            confidence *= 1.1

        return min(confidence, 1.0)  # Cap at 1.0
    
    def _calculate_timing_metrics(self, phases: Dict) -> Dict[str, float]:
        """Calculate timing between phases"""
        metrics = {}
        
        if phases['dip_start'] and phases['load']:
            metrics['dip_to_load_ms'] = (phases['load']['timestamp'] - 
                                        phases['dip_start']['timestamp']) * 1000
        
        if phases['load'] and phases['release']:
            metrics['load_to_release_ms'] = (phases['release']['timestamp'] - 
                                            phases['load']['timestamp']) * 1000
        
        if phases['release'] and phases['follow_through_end']:
            metrics['release_to_follow_through_ms'] = (phases['follow_through_end']['timestamp'] - 
                                                       phases['release']['timestamp']) * 1000
        
        if phases['dip_start'] and phases['release']:
            metrics['total_shot_duration_ms'] = (phases['release']['timestamp'] - 
                                                 phases['dip_start']['timestamp']) * 1000
        
        return metrics
