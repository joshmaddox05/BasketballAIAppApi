"""
Hand Processor - MediaPipe Hands Integration
Tracks 21 hand landmarks per hand for detailed wrist/finger mechanics
"""
import mediapipe as mp
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)


class HandProcessor:
    """Advanced hand tracking for shooting mechanics analysis"""

    # MediaPipe Hands landmark indices (21 landmarks per hand)
    HAND_LANDMARKS = {
        'wrist': 0,
        'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
        'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
        'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
        'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
        'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
    }

    MIN_DETECTION_CONFIDENCE = 0.3
    MIN_TRACKING_CONFIDENCE = 0.3

    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2):
        """Initialize hand processor"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        )
        self.enabled = True
        logger.info(f"✅ HandProcessor initialized (max_hands={max_num_hands})")

    def process_frame(self, rgb_frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """
        Process a single frame and extract hand landmarks

        Returns:
            Dict with 'left' and 'right' hand data, or None if hands not detected
        """
        if not self.enabled:
            return {'left': None, 'right': None, 'available': False}

        try:
            results = self.hands.process(rgb_frame)

            hand_data = {
                'left': None,
                'right': None,
                'available': True,
                'frame': frame_idx
            }

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Determine if left or right hand
                    hand_label = handedness.classification[0].label.lower()

                    # Extract landmarks
                    landmarks = self._extract_hand_landmarks(hand_landmarks)
                    hand_data[hand_label] = landmarks

            return hand_data

        except Exception as e:
            logger.warning(f"Hand processing error on frame {frame_idx}: {str(e)}")
            return {'left': None, 'right': None, 'available': False}

    def _extract_hand_landmarks(self, hand_landmarks) -> Dict[str, Any]:
        """Extract hand landmark coordinates with visibility"""
        landmarks = {}

        for name, idx in self.HAND_LANDMARKS.items():
            if idx < len(hand_landmarks.landmark):
                lm = hand_landmarks.landmark[idx]
                landmarks[name] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': getattr(lm, 'visibility', 1.0)
                }

        return landmarks

    def compute_palm_plane(self, hand_landmarks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute palm plane normal vector from hand landmarks
        Palm plane defined by: wrist, index_mcp, pinky_mcp

        Returns:
            Dict with 'normal' vector, 'center' point, and angles
        """
        if not all(k in hand_landmarks for k in ['wrist', 'index_mcp', 'pinky_mcp']):
            return None

        try:
            # Get 3D points
            wrist = np.array([hand_landmarks['wrist'][k] for k in ['x', 'y', 'z']])
            index_mcp = np.array([hand_landmarks['index_mcp'][k] for k in ['x', 'y', 'z']])
            pinky_mcp = np.array([hand_landmarks['pinky_mcp'][k] for k in ['x', 'y', 'z']])

            # Compute palm plane vectors
            v1 = index_mcp - wrist
            v2 = pinky_mcp - wrist

            # Normal vector (cross product)
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-6)

            # Center of palm
            center = (wrist + index_mcp + pinky_mcp) / 3.0

            return {
                'normal': normal,
                'center': center,
                'v1': v1,
                'v2': v2
            }

        except Exception as e:
            logger.warning(f"Palm plane computation error: {str(e)}")
            return None

    def compute_palm_angles(self, palm_plane: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute palm orientation angles
        - angle_to_vertical: angle between palm normal and vertical axis
        - angle_to_target: angle between palm normal and target direction (assumed +X)
        """
        normal = palm_plane['normal']

        # Vertical axis (Y-up in normalized coords)
        vertical = np.array([0, 1, 0])

        # Target axis (assume camera looking at basket = +X direction)
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
            'angle_to_target_deg': float(angle_to_target)
        }

    def compute_wrist_flick_angle(
        self,
        hand_landmarks: Dict[str, Any],
        elbow_pos: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        Compute wrist flick angle (hand pitch relative to forearm)
        If elbow available, use forearm-to-hand angle
        Otherwise use wrist-to-middle_mcp angle
        """
        if 'wrist' not in hand_landmarks or 'middle_mcp' not in hand_landmarks:
            return None

        try:
            wrist = np.array([hand_landmarks['wrist'][k] for k in ['x', 'y']])
            middle_mcp = np.array([hand_landmarks['middle_mcp'][k] for k in ['x', 'y']])

            if elbow_pos is not None:
                # Forearm vector
                forearm = wrist - elbow_pos[:2]
                # Hand vector
                hand = middle_mcp - wrist

                # Angle between forearm and hand
                angle = np.degrees(np.arccos(np.clip(
                    np.dot(forearm, hand) / (np.linalg.norm(forearm) * np.linalg.norm(hand) + 1e-6),
                    -1.0, 1.0
                )))
            else:
                # Just use wrist-to-knuckle angle vs horizontal
                hand_vec = middle_mcp - wrist
                horizontal = np.array([1, 0])
                angle = np.degrees(np.arccos(np.clip(
                    np.dot(hand_vec, horizontal) / (np.linalg.norm(hand_vec) + 1e-6),
                    -1.0, 1.0
                )))

            return float(angle)

        except Exception as e:
            logger.warning(f"Wrist flick angle error: {str(e)}")
            return None

    def close(self):
        """Release resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        logger.info("HandProcessor closed")

