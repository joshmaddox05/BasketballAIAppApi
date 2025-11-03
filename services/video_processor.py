"""
Video Processor - Process videos and extract shooting form data
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process videos and extract shooting form data"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Use lite model (model_complexity=0) to save memory
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=Lite (150MB less memory)
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
    
    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        metadata = {
            'duration': frame_count / fps if fps > 0 else frame_count / 30,
            'fps': fps if fps > 0 else 30,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': frame_count
        }
        
        cap.release()
        return metadata
    
    def analyze_shooting_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze shooting video and extract all metrics"""
        from .baseline_analyzer import BaselineAnalyzer
        
        logger.info(f"🎥 Processing video: {video_path}")
        
        analyzer = BaselineAnalyzer()
        
        # Extract keypoints
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default FPS
        
        all_keypoints = []
        frame_number = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing {total_frames} frames at {fps:.2f} fps...")
        
        # Process frames with memory optimization (skip frames)
        frame_skip = 2  # Process every 2nd frame to save memory
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to reduce memory usage
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                keypoints = self._extract_keypoints(results.pose_landmarks)
                keypoints['frame'] = frame_number
                keypoints['timestamp'] = frame_number / fps
                all_keypoints.append(keypoints)
            
            frame_number += 1
            
            if frame_number % 30 == 0:
                logger.info(f"Processed {frame_number}/{total_frames} frames...")
        
        cap.release()
        
        if not all_keypoints:
            raise ValueError("No pose detected in video")
        
        logger.info(f"✅ Extracted pose from {len(all_keypoints)} frames")
        
        # Create analysis using baseline analyzer methods
        analysis = analyzer._create_baseline(all_keypoints, "User", fps)
        analysis['confidence'] = min(0.95, len(all_keypoints) / total_frames)
        
        return analysis
    
    def _extract_keypoints(self, pose_landmarks) -> Dict[str, Dict[str, float]]:
        """Extract keypoints from MediaPipe pose"""
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
