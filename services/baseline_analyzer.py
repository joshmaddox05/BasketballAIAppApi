"""
Baseline Analyzer - Analyzes pro player shooting form to create comparison baselines
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class BaselineAnalyzer:
    """Analyzes pro player shooting form to create baseline comparisons"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key shooting form landmarks
        self.key_landmarks = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        # Pro player baselines directory
        self.baselines_dir = Path("baselines")
        self.baselines_dir.mkdir(exist_ok=True)
    
    def analyze_pro_video(self, video_path: str, player_name: str) -> Dict[str, Any]:
        """
        Analyze a pro player's shooting video to create a baseline
        
        Args:
            video_path: Path to the pro player's video
            player_name: Name of the pro player
            
        Returns:
            Baseline shooting form data
        """
        logger.info(f"🎯 Analyzing {player_name}'s shooting form...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {frame_count} frames at {fps:.2f} fps")
        
        all_keypoints = []
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract normalized keypoints
                keypoints = self._extract_keypoints(results.pose_landmarks)
                keypoints['frame'] = frame_number
                keypoints['timestamp'] = frame_number / fps if fps > 0 else frame_number / 30
                all_keypoints.append(keypoints)
            
            frame_number += 1
            
            if frame_number % 30 == 0:
                logger.info(f"Processed {frame_number}/{frame_count} frames...")
        
        cap.release()
        
        if not all_keypoints:
            raise ValueError(f"No pose detected in {player_name}'s video")
        
        logger.info(f"✅ Extracted pose data from {len(all_keypoints)} frames")
        
        # Analyze the shooting motion
        baseline_data = self._create_baseline(all_keypoints, player_name, fps if fps > 0 else 30)
        
        # Save baseline
        baseline_file = self.baselines_dir / f"{player_name.lower().replace(' ', '_')}.json"
        with open(baseline_file, 'w') as f:
            # Remove keypoints_sequence for storage efficiency (keep only summary metrics)
            save_data = baseline_data.copy()
            if 'keypoints_sequence' in save_data:
                # Keep only first, middle, and last frames
                full_sequence = save_data['keypoints_sequence']
                save_data['keypoints_sequence'] = [
                    full_sequence[0],
                    full_sequence[len(full_sequence)//2],
                    full_sequence[-1]
                ]
            json.dump(save_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"✅ Baseline created and saved for {player_name}")
        return baseline_data
    
    def _extract_keypoints(self, pose_landmarks) -> Dict[str, Dict[str, float]]:
        """Extract key landmarks from MediaPipe pose"""
        keypoints = {}
        
        for name, idx in self.key_landmarks.items():
            landmark = pose_landmarks.landmark[idx]
            keypoints[name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return keypoints
    
    def _create_baseline(self, keypoints_sequence: List[Dict], player_name: str, fps: float) -> Dict[str, Any]:
        """Create baseline measurements from keypoints sequence"""
        
        logger.info(f"Creating baseline metrics for {player_name}...")
        
        # Detect shooting phases
        shooting_phases = self._detect_shooting_phases(keypoints_sequence)
        
        # Calculate key metrics
        release_angle = self._calculate_release_angle(keypoints_sequence, shooting_phases)
        elbow_alignment = self._calculate_elbow_alignment(keypoints_sequence, shooting_phases)
        follow_through = self._analyze_follow_through(keypoints_sequence, shooting_phases)
        balance_metrics = self._analyze_balance(keypoints_sequence, shooting_phases)
        arc_trajectory = self._calculate_arc_trajectory(keypoints_sequence, shooting_phases)
        
        baseline = {
            'player_name': player_name,
            'total_frames': len(keypoints_sequence),
            'duration': keypoints_sequence[-1]['timestamp'],
            'fps': fps,
            'shooting_phases': shooting_phases,
            'metrics': {
                'release_angle': release_angle,
                'elbow_alignment': elbow_alignment,
                'follow_through': follow_through,
                'balance': balance_metrics,
                'arc_trajectory': arc_trajectory
            },
            'keypoints_sequence': keypoints_sequence,
            'created_at': str(np.datetime64('now'))
        }
        
        return baseline
    
    def _detect_shooting_phases(self, keypoints: List[Dict]) -> Dict[str, Any]:
        """Detect different phases of shooting motion"""
        
        # Track right wrist height (assuming right-handed shooter)
        wrist_heights = [kp['right_wrist']['y'] for kp in keypoints]
        
        # Find the release point (peak wrist height - lowest y value)
        peaks, _ = find_peaks([-h for h in wrist_heights], prominence=0.05)
        
        if len(peaks) == 0:
            # If no clear peak, use highest point
            release_frame = wrist_heights.index(min(wrist_heights))
        else:
            release_frame = peaks[0]
        
        # Define phases
        setup_end = max(0, release_frame - int(len(keypoints) * 0.3))
        release_start = release_frame
        follow_through_end = min(len(keypoints) - 1, release_frame + int(len(keypoints) * 0.4))
        
        return {
            'setup': {'start': 0, 'end': setup_end},
            'release': {'start': release_start, 'end': release_start},
            'follow_through': {'start': release_start, 'end': follow_through_end}
        }
    
    def _calculate_release_angle(self, keypoints: List[Dict], phases: Dict) -> Dict[str, Any]:
        """Calculate shooting release angle"""
        
        release_frame = phases['release']['start']
        kp = keypoints[release_frame]
        
        # Calculate angle between shoulder, elbow, and wrist
        shoulder = np.array([kp['right_shoulder']['x'], kp['right_shoulder']['y']])
        elbow = np.array([kp['right_elbow']['x'], kp['right_elbow']['y']])
        wrist = np.array([kp['right_wrist']['x'], kp['right_wrist']['y']])
        
        angle = self._calculate_angle(shoulder, elbow, wrist)
        
        # Calculate release trajectory angle (relative to horizontal)
        trajectory_angle = np.degrees(np.arctan2(
            -(wrist[1] - elbow[1]),  # Negative because y increases downward
            wrist[0] - elbow[0]
        ))
        
        # Ensure angle is in 0-90 range
        if trajectory_angle < 0:
            trajectory_angle += 180
        
        return {
            'elbow_angle': float(angle),
            'trajectory_angle': float(trajectory_angle),
            'optimal_range': [45, 50],
            'frame': release_frame,
            'quality_score': 10 - abs(trajectory_angle - 47.5) / 5  # Score based on proximity to 47.5 degrees
        }
    
    def _calculate_elbow_alignment(self, keypoints: List[Dict], phases: Dict) -> Dict[str, Any]:
        """Analyze elbow alignment throughout shot"""
        
        setup_frames = range(phases['setup']['start'], phases['release']['start'])
        
        alignments = []
        for frame in setup_frames:
            if frame >= len(keypoints):
                break
            kp = keypoints[frame]
            
            shoulder = np.array([kp['right_shoulder']['x'], kp['right_shoulder']['y']])
            elbow = np.array([kp['right_elbow']['x'], kp['right_elbow']['y']])
            wrist = np.array([kp['right_wrist']['x'], kp['right_wrist']['y']])
            
            # Check if elbow is under the ball (wrist)
            lateral_offset = abs(elbow[0] - wrist[0])
            alignments.append(lateral_offset)
        
        avg_alignment = np.mean(alignments) if alignments else 0.05
        consistency = 1 - np.std(alignments) if alignments else 0.8
        
        return {
            'average_offset': float(avg_alignment),
            'consistency': float(consistency),
            'optimal_threshold': 0.05,
            'frames_analyzed': len(alignments)
        }
    
    def _analyze_follow_through(self, keypoints: List[Dict], phases: Dict) -> Dict[str, Any]:
        """Analyze follow-through motion"""
        
        release_frame = phases['release']['start']
        follow_through_frames = range(
            phases['follow_through']['start'],
            min(phases['follow_through']['end'], len(keypoints))
        )
        
        wrist_positions = []
        for frame in follow_through_frames:
            if frame >= len(keypoints):
                break
            kp = keypoints[frame]
            wrist_positions.append([
                kp['right_wrist']['x'],
                kp['right_wrist']['y']
            ])
        
        # Calculate extension and snap
        if len(wrist_positions) > 1:
            extension = euclidean(wrist_positions[0], wrist_positions[-1])
            
            # Calculate wrist snap (change in y-direction)
            y_change = wrist_positions[-1][1] - wrist_positions[0][1]
        else:
            extension = 0.15
            y_change = 0.05
        
        return {
            'extension_distance': float(extension),
            'wrist_snap': float(y_change),
            'duration_frames': len(wrist_positions),
            'quality_score': min(10, extension * 20)
        }
    
    def _analyze_balance(self, keypoints: List[Dict], phases: Dict) -> Dict[str, Any]:
        """Analyze balance and stance throughout shot"""
        
        all_frames = range(len(keypoints))
        
        hip_widths = []
        ankle_widths = []
        center_of_masses = []
        
        for frame in all_frames:
            kp = keypoints[frame]
            
            # Calculate stance width
            hip_width = abs(kp['left_hip']['x'] - kp['right_hip']['x'])
            ankle_width = abs(kp['left_ankle']['x'] - kp['right_ankle']['x'])
            
            # Calculate center of mass (approximation)
            com_x = (kp['left_hip']['x'] + kp['right_hip']['x']) / 2
            com_y = (kp['left_hip']['y'] + kp['right_hip']['y']) / 2
            
            hip_widths.append(hip_width)
            ankle_widths.append(ankle_width)
            center_of_masses.append([com_x, com_y])
        
        # Calculate stability metrics
        avg_stance_width = np.mean(ankle_widths)
        stance_consistency = 1 - np.std(ankle_widths)
        
        # Calculate center of mass movement
        com_movement = np.sum([
            euclidean(center_of_masses[i], center_of_masses[i+1])
            for i in range(len(center_of_masses)-1)
        ])
        
        return {
            'average_stance_width': float(avg_stance_width),
            'stance_consistency': float(stance_consistency),
            'center_of_mass_movement': float(com_movement),
            'optimal_stance_range': [0.15, 0.25],
            'stability_score': float(stance_consistency * 10)
        }
    
    def _calculate_arc_trajectory(self, keypoints: List[Dict], phases: Dict) -> Dict[str, Any]:
        """Calculate ball trajectory arc"""
        
        release_to_follow = range(
            phases['release']['start'],
            min(phases['follow_through']['end'], len(keypoints))
        )
        
        wrist_trajectory = []
        for frame in release_to_follow:
            if frame >= len(keypoints):
                break
            kp = keypoints[frame]
            wrist_trajectory.append([
                kp['right_wrist']['x'],
                kp['right_wrist']['y']
            ])
        
        if len(wrist_trajectory) > 2:
            # Calculate arc characteristics
            x = np.array([p[0] for p in wrist_trajectory])
            y = np.array([p[1] for p in wrist_trajectory])
            
            arc_height = min(y) - max(y)
            arc_angle = np.degrees(np.arctan2(-(y[-1] - y[0]), x[-1] - x[0]))
            
            if arc_angle < 0:
                arc_angle += 180
        else:
            arc_height = 0.2
            arc_angle = 47
        
        return {
            'arc_height': float(arc_height),
            'arc_angle': float(arc_angle),
            'optimal_arc_angle_range': [45, 55],
            'trajectory_points': len(wrist_trajectory)
        }
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle between three points"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    
    def load_baseline(self, player_name: str) -> Dict[str, Any]:
        """Load a saved baseline"""
        baseline_file = self.baselines_dir / f"{player_name.lower().replace(' ', '_')}.json"
        
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline for {player_name} not found")
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
    
    def list_available_baselines(self) -> List[str]:
        """List all available player baselines"""
        baselines = []
        for file in self.baselines_dir.glob("*.json"):
            player_name = file.stem.replace('_', ' ').title()
            baselines.append(player_name)
        return baselines
