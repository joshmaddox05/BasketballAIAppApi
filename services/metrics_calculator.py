"""
Biomechanical Metrics Calculator for Shot Analysis
Calculates 7 key shooting metrics with quality scoring
"""
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate biomechanical metrics from pose keypoints"""
    
    def __init__(self):
        # Define optimal ranges for each metric
        self.optimal_ranges = {
            'release_angle': (145, 180),    # degrees - elbow angle at release (extended arm, 180=straight)
            'elbow_flare': (0, 15),         # degrees - horizontal deviation from vertical
            'knee_load': (120, 160),        # degrees - interior angle (180=straight, lower=more bent)
            'hip_shoulder_alignment': (0, 15),  # degrees - rotation difference
            'base_width': (0.10, 0.30),     # ratio of body height - stance width
            'lateral_sway': (0, 0.05),      # ratio of body height
            'arc_trajectory': (30, 90)      # degrees - wrist trajectory angle
        }
        
        # Metric weights for overall score
        self.weights = {
            'release_angle': 0.25,
            'elbow_flare': 0.20,
            'knee_load': 0.15,
            'hip_shoulder_alignment': 0.15,
            'base_width': 0.10,
            'lateral_sway': 0.10,
            'arc_trajectory': 0.05
        }
        
        logger.info("✅ MetricsCalculator initialized")

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

    def calculate_all_metrics(
        self,
        keypoints_sequence: List[Dict[str, Any]],
        phases: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate all biomechanical metrics

        Args:
            keypoints_sequence: List of pose keypoints per frame
            phases: Detected shot phases with frame indices

        Returns:
            Dictionary with all metrics and overall score
        """
        try:
            logger.info("📊 Calculating biomechanical metrics...")

            # Get the detected shooting hand (default to right for backwards compatibility)
            shooting_hand = phases.get('shooting_hand', 'right')
            logger.info(f"🎯 Using shooting hand: {shooting_hand}")

            metrics = {}

            # Extract key frames
            release_frame = phases.get('release', {}).get('frame') if phases.get('release') else None
            load_frame = phases.get('load', {}).get('frame') if phases.get('load') else None
            dip_frame = phases.get('dip_start', {}).get('frame') if phases.get('dip_start') else None

            if release_frame is None or load_frame is None:
                return {
                    'error': 'Missing required phase frames',
                    'confidence': 0.0
                }

            # Get keypoints at key moments
            release_kp = keypoints_sequence[release_frame] if release_frame < len(keypoints_sequence) else None
            load_kp = keypoints_sequence[load_frame] if load_frame < len(keypoints_sequence) else None

            if not release_kp or not load_kp:
                return {
                    'error': 'Missing keypoint data at key frames',
                    'confidence': 0.0
                }

            # Calculate each metric (passing shooting_hand to arm-specific metrics)
            metrics['release_angle'] = self._calculate_release_angle(release_kp, shooting_hand)
            # Elbow flare should be measured at load/set position, not at full extension
            metrics['elbow_flare'] = self._calculate_elbow_flare(load_kp, shooting_hand)
            metrics['knee_load'] = self._calculate_knee_angle(load_kp)
            metrics['hip_shoulder_alignment'] = self._calculate_alignment(load_kp)
            metrics['base_width'] = self._calculate_base_width(load_kp)
            metrics['lateral_sway'] = self._calculate_lateral_sway(keypoints_sequence, phases)
            metrics['arc_trajectory'] = self._calculate_arc_trajectory(keypoints_sequence, phases, shooting_hand)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            metrics['overall_score'] = overall_score
            
            logger.info(f"✅ Metrics calculated. Overall score: {overall_score:.1f}/100")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Metrics calculation failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'confidence': 0.0
            }
    
    def _calculate_release_angle(self, keypoints: Dict[str, Any], shooting_hand: str = 'right') -> Dict[str, Any]:
        """Calculate elbow angle at release (shoulder-elbow-wrist)"""
        try:
            # Normalize keypoint names to uppercase
            keypoints = self._normalize_keypoints(keypoints)
            # Get shooting arm keypoints
            hand = shooting_hand.upper()
            shoulder = np.array([keypoints[f'{hand}_SHOULDER']['x'], keypoints[f'{hand}_SHOULDER']['y']])
            elbow = np.array([keypoints[f'{hand}_ELBOW']['x'], keypoints[f'{hand}_ELBOW']['y']])
            wrist = np.array([keypoints[f'{hand}_WRIST']['x'], keypoints[f'{hand}_WRIST']['y']])
            
            # Calculate angle
            angle = self._calculate_angle(shoulder, elbow, wrist)
            
            # Quality score (0-10)
            quality = self._score_metric(angle, self.optimal_ranges['release_angle'])
            
            return {
                'angle_deg': round(angle, 1),
                'optimal_range': list(self.optimal_ranges['release_angle']),
                'in_range': self.optimal_ranges['release_angle'][0] <= angle <= self.optimal_ranges['release_angle'][1],
                'quality_score': round(quality, 1),
                'description': self._get_quality_description(quality)
            }
        except KeyError as e:
            logger.warning(f"Missing keypoint for release angle: {e}")
            return {
                'angle_deg': 0,
                'optimal_range': list(self.optimal_ranges['release_angle']),
                'in_range': False,
                'quality_score': 0,
                'description': 'Unknown',
                'error': f'Missing keypoint: {e}'
            }
    
    def _calculate_elbow_flare(self, keypoints: Dict[str, Any], shooting_hand: str = 'right') -> Dict[str, Any]:
        """Calculate elbow alignment (deviation from vertical plane)"""
        try:
            # Normalize keypoint names to uppercase
            keypoints = self._normalize_keypoints(keypoints)
            # Get shooting arm keypoints
            hand = shooting_hand.upper()
            shoulder = np.array([keypoints[f'{hand}_SHOULDER']['x'], keypoints[f'{hand}_SHOULDER']['y']])
            elbow = np.array([keypoints[f'{hand}_ELBOW']['x'], keypoints[f'{hand}_ELBOW']['y']])
            
            # Calculate horizontal deviation
            shoulder_to_elbow = elbow - shoulder
            flare_angle = abs(np.degrees(np.arctan2(shoulder_to_elbow[0], shoulder_to_elbow[1])))
            
            # Quality score
            quality = self._score_metric(flare_angle, self.optimal_ranges['elbow_flare'])
            
            return {
                'angle_deg': round(flare_angle, 1),
                'optimal_range': list(self.optimal_ranges['elbow_flare']),
                'in_range': flare_angle <= self.optimal_ranges['elbow_flare'][1],
                'quality_score': round(quality, 1),
                'description': self._get_quality_description(quality)
            }
        except KeyError as e:
            logger.warning(f"Missing keypoint for elbow flare: {e}")
            return {
                'angle_deg': 0,
                'optimal_range': list(self.optimal_ranges['elbow_flare']),
                'in_range': False,
                'quality_score': 0,
                'description': 'Unknown',
                'error': f'Missing keypoint: {e}'
            }
    
    def _calculate_knee_angle(self, keypoints: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate knee flexion angle at load (hip-knee-ankle)

        Note: We measure the interior angle at the knee.
        - Straight leg = ~180°
        - Fully bent (squat) = ~45°
        - Optimal load position = 120-150° (slight to moderate bend)
        """
        try:
            # Normalize keypoint names to uppercase
            keypoints = self._normalize_keypoints(keypoints)
            # Get right leg keypoints
            hip = np.array([keypoints['RIGHT_HIP']['x'], keypoints['RIGHT_HIP']['y']])
            knee = np.array([keypoints['RIGHT_KNEE']['x'], keypoints['RIGHT_KNEE']['y']])
            ankle = np.array([keypoints['RIGHT_ANKLE']['x'], keypoints['RIGHT_ANKLE']['y']])

            # Calculate interior angle at knee (straight leg = 180°, bent = lower)
            angle = self._calculate_angle(hip, knee, ankle)

            # Quality score - updated optimal range for interior angle
            # 120-150° represents a good athletic load position
            quality = self._score_metric(angle, self.optimal_ranges['knee_load'])
            
            return {
                'angle_deg': round(angle, 1),
                'optimal_range': list(self.optimal_ranges['knee_load']),
                'in_range': self.optimal_ranges['knee_load'][0] <= angle <= self.optimal_ranges['knee_load'][1],
                'quality_score': round(quality, 1),
                'description': self._get_quality_description(quality)
            }
        except KeyError as e:
            logger.warning(f"Missing keypoint for knee angle: {e}")
            return {
                'angle_deg': 0,
                'optimal_range': list(self.optimal_ranges['knee_load']),
                'in_range': False,
                'quality_score': 0,
                'description': 'Unknown',
                'error': f'Missing keypoint: {e}'
            }
    
    def _calculate_alignment(self, keypoints: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate hip-shoulder alignment (rotation)"""
        try:
            # Normalize keypoint names to uppercase
            keypoints = self._normalize_keypoints(keypoints)
            # Get shoulder and hip keypoints
            left_shoulder = np.array([keypoints['LEFT_SHOULDER']['x'], keypoints['LEFT_SHOULDER']['y']])
            right_shoulder = np.array([keypoints['RIGHT_SHOULDER']['x'], keypoints['RIGHT_SHOULDER']['y']])
            left_hip = np.array([keypoints['LEFT_HIP']['x'], keypoints['LEFT_HIP']['y']])
            right_hip = np.array([keypoints['RIGHT_HIP']['x'], keypoints['RIGHT_HIP']['y']])

            # Calculate shoulder and hip lines
            shoulder_angle = np.degrees(np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ))
            hip_angle = np.degrees(np.arctan2(
                right_hip[1] - left_hip[1],
                right_hip[0] - left_hip[0]
            ))

            # Rotation difference - handle angle wrapping
            rotation = abs(shoulder_angle - hip_angle)
            # Normalize to 0-180 range (angles can wrap around 360)
            if rotation > 180:
                rotation = 360 - rotation
            
            # Quality score
            quality = self._score_metric(rotation, self.optimal_ranges['hip_shoulder_alignment'])
            
            return {
                'angle_deg': round(rotation, 1),
                'optimal_range': list(self.optimal_ranges['hip_shoulder_alignment']),
                'in_range': rotation <= self.optimal_ranges['hip_shoulder_alignment'][1],
                'quality_score': round(quality, 1),
                'description': self._get_quality_description(quality)
            }
        except KeyError as e:
            logger.warning(f"Missing keypoint for alignment: {e}")
            return {
                'angle_deg': 0,
                'optimal_range': list(self.optimal_ranges['hip_shoulder_alignment']),
                'in_range': False,
                'quality_score': 0,
                'description': 'Unknown',
                'error': f'Missing keypoint: {e}'
            }
    
    def _calculate_base_width(self, keypoints: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate base width as ratio of body height"""
        try:
            # Normalize keypoint names to uppercase
            keypoints = self._normalize_keypoints(keypoints)
            # Get ankle positions
            left_ankle = np.array([keypoints['LEFT_ANKLE']['x'], keypoints['LEFT_ANKLE']['y']])
            right_ankle = np.array([keypoints['RIGHT_ANKLE']['x'], keypoints['RIGHT_ANKLE']['y']])
            
            # Get body height (nose to mid-ankle)
            nose = np.array([keypoints['NOSE']['x'], keypoints['NOSE']['y']])
            mid_ankle = (left_ankle + right_ankle) / 2
            body_height = np.linalg.norm(nose - mid_ankle)
            
            # Calculate base width
            base_width = np.linalg.norm(left_ankle - right_ankle)
            ratio = base_width / body_height if body_height > 0 else 0
            
            # Quality score
            quality = self._score_metric(ratio, self.optimal_ranges['base_width'])
            
            return {
                'ratio': round(ratio, 3),
                'optimal_range': list(self.optimal_ranges['base_width']),
                'in_range': self.optimal_ranges['base_width'][0] <= ratio <= self.optimal_ranges['base_width'][1],
                'quality_score': round(quality, 1),
                'description': self._get_quality_description(quality)
            }
        except (KeyError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating base width: {e}")
            return {
                'ratio': 0,
                'optimal_range': list(self.optimal_ranges['base_width']),
                'in_range': False,
                'quality_score': 0,
                'description': 'Unknown',
                'error': str(e)
            }
    
    def _calculate_lateral_sway(
        self, 
        keypoints_sequence: List[Dict[str, Any]], 
        phases: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate lateral movement during shot"""
        try:
            # Use dip_start if available, otherwise use load as starting point
            dip_start = phases.get('dip_start')
            load = phases.get('load')
            release = phases.get('release')

            if dip_start:
                start_frame = dip_start.get('frame', 0)
            elif load:
                start_frame = load.get('frame', 0)
            else:
                start_frame = 0

            release_frame = release.get('frame', len(keypoints_sequence) - 1) if release else len(keypoints_sequence) - 1

            # Track hip center movement
            hip_positions = []
            for i in range(start_frame, min(release_frame + 1, len(keypoints_sequence))):
                kp = self._normalize_keypoints(keypoints_sequence[i])
                left_hip = np.array([kp['LEFT_HIP']['x'], kp['LEFT_HIP']['y']])
                right_hip = np.array([kp['RIGHT_HIP']['x'], kp['RIGHT_HIP']['y']])
                mid_hip = (left_hip + right_hip) / 2
                hip_positions.append(mid_hip)

            if len(hip_positions) < 2:
                raise ValueError("Not enough frames to calculate sway")

            hip_positions = np.array(hip_positions)

            # Calculate maximum lateral deviation
            lateral_movement = np.std(hip_positions[:, 0])

            # Normalize by body height
            body_height = self._get_body_height(self._normalize_keypoints(keypoints_sequence[start_frame]))
            sway_ratio = lateral_movement / body_height if body_height > 0 else 0
            
            # Quality score (lower is better)
            quality = self._score_metric(sway_ratio, self.optimal_ranges['lateral_sway'], inverse=True)
            
            return {
                'ratio': round(sway_ratio, 4),
                'optimal_range': list(self.optimal_ranges['lateral_sway']),
                'in_range': sway_ratio <= self.optimal_ranges['lateral_sway'][1],
                'quality_score': round(quality, 1),
                'description': self._get_quality_description(quality)
            }
        except Exception as e:
            logger.warning(f"Error calculating lateral sway: {e}")
            return {
                'ratio': 0,
                'optimal_range': list(self.optimal_ranges['lateral_sway']),
                'in_range': True,
                'quality_score': 5,
                'description': 'Unknown',
                'error': str(e)
            }
    
    def _calculate_arc_trajectory(
        self,
        keypoints_sequence: List[Dict[str, Any]],
        phases: Dict[str, Any],
        shooting_hand: str = 'right'
    ) -> Dict[str, Any]:
        """Calculate shot arc angle from release"""
        try:
            release_frame = phases.get('release', {}).get('frame') if phases.get('release') else None
            follow_through_frame = phases.get('follow_through_end', {}).get('frame') if phases.get('follow_through_end') else None

            if release_frame is None or follow_through_frame is None:
                raise ValueError("Missing release or follow-through frames")

            # Get wrist positions using shooting hand
            hand = shooting_hand.upper()
            release_kp = self._normalize_keypoints(keypoints_sequence[min(release_frame, len(keypoints_sequence) - 1)])
            follow_kp = self._normalize_keypoints(keypoints_sequence[min(follow_through_frame, len(keypoints_sequence) - 1)])

            release_wrist = np.array([release_kp[f'{hand}_WRIST']['x'], release_kp[f'{hand}_WRIST']['y']])
            follow_wrist = np.array([follow_kp[f'{hand}_WRIST']['x'], follow_kp[f'{hand}_WRIST']['y']])
            
            # Calculate trajectory angle
            delta = follow_wrist - release_wrist
            arc_angle = abs(np.degrees(np.arctan2(delta[1], delta[0])))
            
            # Quality score
            quality = self._score_metric(arc_angle, self.optimal_ranges['arc_trajectory'])
            
            return {
                'angle_deg': round(arc_angle, 1),
                'optimal_range': list(self.optimal_ranges['arc_trajectory']),
                'in_range': self.optimal_ranges['arc_trajectory'][0] <= arc_angle <= self.optimal_ranges['arc_trajectory'][1],
                'quality_score': round(quality, 1),
                'description': self._get_quality_description(quality)
            }
        except Exception as e:
            logger.warning(f"Error calculating arc trajectory: {e}")
            return {
                'angle_deg': 0,
                'optimal_range': list(self.optimal_ranges['arc_trajectory']),
                'in_range': False,
                'quality_score': 5,
                'description': 'Unknown',
                'error': str(e)
            }
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _score_metric(self, value: float, optimal_range: Tuple[float, float], inverse: bool = False) -> float:
        """
        Score a metric on a 0-10 scale
        
        Args:
            value: Measured value
            optimal_range: (min, max) optimal range
            inverse: If True, lower values are better (for sway, flare, etc.)
        """
        min_val, max_val = optimal_range
        
        if inverse:
            # Lower is better (e.g., sway, flare)
            if value <= min_val:
                return 10.0
            elif value >= max_val:
                return 0.0
            else:
                # Linear decay from 10 to 0
                return 10.0 - (10.0 * (value - min_val) / (max_val - min_val))
        else:
            # Within range is better
            if min_val <= value <= max_val:
                # Perfect score in optimal range
                mid = (min_val + max_val) / 2
                deviation = abs(value - mid) / ((max_val - min_val) / 2)
                return 10.0 - (deviation * 2)  # Max 8-10 in range
            else:
                # Outside range - score decreases with distance
                if value < min_val:
                    distance = min_val - value
                    tolerance = min_val * 0.2  # 20% tolerance
                else:
                    distance = value - max_val
                    tolerance = max_val * 0.2
                
                score = 8.0 - (8.0 * distance / tolerance)
                return max(0.0, min(score, 8.0))
    
    def _get_quality_description(self, score: float) -> str:
        """Convert numeric score to quality description"""
        if score >= 9.0:
            return "Excellent"
        elif score >= 7.5:
            return "Very Good"
        elif score >= 6.0:
            return "Good"
        elif score >= 4.0:
            return "Fair"
        elif score >= 2.0:
            return "Needs Work"
        else:
            return "Poor"
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall score from all metrics"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics and 'quality_score' in metrics[metric_name]:
                score = metrics[metric_name]['quality_score']
                total_score += score * weight * 10  # Scale to 0-100
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def _get_body_height(self, keypoints: Dict[str, Any]) -> float:
        """Calculate body height from keypoints"""
        try:
            nose = np.array([keypoints['NOSE']['x'], keypoints['NOSE']['y']])
            left_ankle = np.array([keypoints['LEFT_ANKLE']['x'], keypoints['LEFT_ANKLE']['y']])
            right_ankle = np.array([keypoints['RIGHT_ANKLE']['x'], keypoints['RIGHT_ANKLE']['y']])
            mid_ankle = (left_ankle + right_ankle) / 2
            
            return np.linalg.norm(nose - mid_ankle)
        except KeyError:
            return 1.0  # Default fallback
