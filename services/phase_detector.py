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
    MIN_LOAD_TO_RELEASE = 0.15  # 150ms
    MAX_LOAD_TO_RELEASE = 0.45  # 450ms
    
    # Thresholds
    KNEE_ANGLE_FLEX_THRESHOLD = 15  # degrees of flexion to detect dip
    WRIST_VELOCITY_THRESHOLD = 1.5  # normalized units/sec
    ELBOW_EXTENSION_THRESHOLD = 0.8  # normalized velocity
    
    def __init__(self):
        logger.info("✅ PhaseDetector initialized")
    
    def detect_phases(self, keypoints_sequence: List[Dict]) -> Dict[str, Any]:
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
        
        # Extract time series
        timestamps = [kp['timestamp'] for kp in keypoints_sequence if kp.get('visible', True)]
        
        # Detect each phase
        dip_start = self._detect_dip_start(keypoints_sequence)
        load_point = self._detect_load(keypoints_sequence, dip_start)
        release_point = self._detect_release(keypoints_sequence, load_point)
        follow_through_end = self._detect_follow_through_end(keypoints_sequence, release_point)
        landing = self._detect_landing(keypoints_sequence, release_point)
        
        # Validate phase sequence
        phases_valid = self._validate_phases(dip_start, load_point, release_point)
        
        phases = {
            'dip_start': dip_start,
            'load': load_point,
            'release': release_point,
            'follow_through_end': follow_through_end,
            'landing': landing,
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
    
    def _detect_dip_start(self, keypoints_sequence: List[Dict]) -> Optional[Dict]:
        """
        Detect dip start: wrist y decreases and knee angle starts decreasing
        """
        wrist_y_values = []
        knee_angles = []
        timestamps = []
        frame_numbers = []
        
        for kp in keypoints_sequence:
            if not kp.get('visible', True):
                continue
            
            # Get wrist position (use shooting hand - try right first, then left)
            wrist_y = None
            if 'right_wrist' in kp and 'low_visibility' not in kp['right_wrist']:
                wrist_y = kp['right_wrist']['y']
            elif 'left_wrist' in kp and 'low_visibility' not in kp['left_wrist']:
                wrist_y = kp['left_wrist']['y']
            
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
        if 0 < min_idx < len(knee_angles) - 1:
            return {
                'frame': frame_numbers[min_idx],
                'timestamp': timestamps[min_idx],
                'knee_angle': knee_angles[min_idx],
                'phase': 'load'
            }
        
        return None
    
    def _detect_release(self, keypoints_sequence: List[Dict], load_point: Optional[Dict]) -> Optional[Dict]:
        """
        Detect release: peak wrist angular velocity + elbow extension
        Constrained to occur after load point
        """
        if not load_point:
            return None
        
        # Start search from load point
        start_idx = next((i for i, kp in enumerate(keypoints_sequence) 
                         if kp.get('frame') == load_point['frame']), 0)
        
        # Calculate wrist velocities and elbow extension
        wrist_positions = []
        elbow_angles = []
        timestamps = []
        frame_numbers = []
        
        for kp in keypoints_sequence[start_idx:]:
            if not kp.get('visible', True):
                continue
            
            # Get wrist position
            wrist_y = None
            if 'right_wrist' in kp and 'low_visibility' not in kp['right_wrist']:
                wrist_y = kp['right_wrist']['y']
            elif 'left_wrist' in kp and 'low_visibility' not in kp['left_wrist']:
                wrist_y = kp['left_wrist']['y']
            
            # Calculate elbow angle
            elbow_angle = self._calculate_elbow_angle(kp)
            
            if wrist_y is not None and elbow_angle is not None:
                wrist_positions.append(wrist_y)
                elbow_angles.append(elbow_angle)
                timestamps.append(kp['timestamp'])
                frame_numbers.append(kp['frame'])
        
        if len(wrist_positions) < 5:
            return None
        
        # Calculate velocities
        dt = np.diff(timestamps)
        dt[dt == 0] = 0.001  # Avoid division by zero
        
        wrist_velocity = -np.diff(wrist_positions) / dt  # Negative because upward is negative y
        elbow_extension_velocity = np.diff(elbow_angles) / dt
        
        # Find peak wrist velocity (upward) with elbow extending
        valid_releases = []
        
        for i in range(len(wrist_velocity)):
            if (wrist_velocity[i] > self.WRIST_VELOCITY_THRESHOLD and 
                elbow_extension_velocity[i] > self.ELBOW_EXTENSION_THRESHOLD):
                
                # Check timing constraint (150-450ms after load)
                time_since_load = timestamps[i+1] - load_point['timestamp']
                if self.MIN_LOAD_TO_RELEASE <= time_since_load <= self.MAX_LOAD_TO_RELEASE:
                    valid_releases.append({
                        'frame': frame_numbers[i+1],
                        'timestamp': timestamps[i+1],
                        'wrist_velocity': float(wrist_velocity[i]),
                        'elbow_angle': float(elbow_angles[i+1]),
                        'time_since_load': time_since_load
                    })
        
        if valid_releases:
            # Return release with highest wrist velocity
            return max(valid_releases, key=lambda x: x['wrist_velocity'])
        
        # Fallback: find peak wrist velocity within time window
        time_mask = [(timestamps[i+1] - load_point['timestamp']) >= self.MIN_LOAD_TO_RELEASE 
                    for i in range(len(wrist_velocity))]
        
        if any(time_mask):
            masked_velocities = [v if m else -np.inf for v, m in zip(wrist_velocity, time_mask)]
            peak_idx = np.argmax(masked_velocities)
            
            return {
                'frame': frame_numbers[peak_idx+1],
                'timestamp': timestamps[peak_idx+1],
                'wrist_velocity': float(wrist_velocity[peak_idx]),
                'elbow_angle': float(elbow_angles[peak_idx+1]),
                'fallback': True
            }
        
        return None
    
    def _detect_follow_through_end(self, keypoints_sequence: List[Dict], release_point: Optional[Dict]) -> Optional[Dict]:
        """
        Detect end of follow-through: wrist stops moving upward
        """
        if not release_point:
            return None
        
        start_idx = next((i for i, kp in enumerate(keypoints_sequence) 
                         if kp.get('frame') == release_point['frame']), 0)
        
        wrist_positions = []
        timestamps = []
        frame_numbers = []
        
        for kp in keypoints_sequence[start_idx:]:
            if not kp.get('visible', True):
                continue
            
            wrist_y = None
            if 'right_wrist' in kp and 'low_visibility' not in kp['right_wrist']:
                wrist_y = kp['right_wrist']['y']
            elif 'left_wrist' in kp and 'low_visibility' not in kp['left_wrist']:
                wrist_y = kp['left_wrist']['y']
            
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
    
    def _calculate_elbow_angle(self, keypoint: Dict) -> Optional[float]:
        """Calculate elbow angle from shoulder, elbow, wrist"""
        # Use right arm primarily
        if all(k in keypoint for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            shoulder = keypoint['right_shoulder']
            elbow = keypoint['right_elbow']
            wrist = keypoint['right_wrist']
            
            if all('low_visibility' not in p for p in [shoulder, elbow, wrist]):
                return self._calculate_angle(
                    (shoulder['x'], shoulder['y']),
                    (elbow['x'], elbow['y']),
                    (wrist['x'], wrist['y'])
                )
        
        # Fallback to left arm
        if all(k in keypoint for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            shoulder = keypoint['left_shoulder']
            elbow = keypoint['left_elbow']
            wrist = keypoint['left_wrist']
            
            if all('low_visibility' not in p for p in [shoulder, elbow, wrist]):
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
        if not all([dip_start, load_point, release_point]):
            return False
        
        # Check temporal order
        if not (dip_start['timestamp'] < load_point['timestamp'] < release_point['timestamp']):
            return False
        
        # Check timing constraints
        dip_to_load = load_point['timestamp'] - dip_start['timestamp']
        load_to_release = release_point['timestamp'] - load_point['timestamp']
        
        if dip_to_load < self.MIN_DIP_DURATION:
            return False
        
        if not (self.MIN_LOAD_TO_RELEASE <= load_to_release <= self.MAX_LOAD_TO_RELEASE):
            return False
        
        return True
    
    def _calculate_confidence(self, dip_start, load_point, release_point, phases_valid) -> float:
        """Calculate confidence score for phase detection"""
        if not phases_valid:
            return 0.0
        
        confidence = 1.0
        
        # Penalize if any phase is missing
        if not dip_start:
            confidence *= 0.7
        if not load_point:
            confidence *= 0.7
        if not release_point:
            confidence *= 0.5
        
        # Penalize fallback detections
        if release_point and release_point.get('fallback'):
            confidence *= 0.8
        
        return confidence
    
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
