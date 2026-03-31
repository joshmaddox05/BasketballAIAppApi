"""
Unit tests for landmark extraction and extended metrics
"""
import pytest
import sys
sys.path.insert(0, '..')

from config import (
    LANDMARK_NAMES, LANDMARK_INDICES, CORE_LANDMARKS, EXTENDED_LANDMARKS,
    ALL_LANDMARKS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
)


class TestLandmarkConfiguration:
    """Test landmark configuration is correct"""

    def test_all_landmarks_defined(self):
        """Verify all expected landmarks are in the mapping"""
        expected_core = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ]

        expected_extended = [
            'left_pinky', 'right_pinky', 'left_thumb', 'right_thumb',
            'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
        ]

        for landmark in expected_core:
            assert landmark in CORE_LANDMARKS, f"Missing core landmark: {landmark}"
            assert landmark in LANDMARK_INDICES, f"Missing index for: {landmark}"

        for landmark in expected_extended:
            assert landmark in EXTENDED_LANDMARKS, f"Missing extended landmark: {landmark}"
            assert landmark in LANDMARK_INDICES, f"Missing index for: {landmark}"

    def test_landmark_indices_correct(self):
        """Verify MediaPipe landmark indices are correct"""
        expected_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }

        for name, expected_idx in expected_indices.items():
            assert LANDMARK_INDICES[name] == expected_idx, \
                f"Wrong index for {name}: expected {expected_idx}, got {LANDMARK_INDICES[name]}"

    def test_reverse_mapping_correct(self):
        """Verify LANDMARK_NAMES is correct reverse of LANDMARK_INDICES"""
        for name, idx in LANDMARK_INDICES.items():
            assert LANDMARK_NAMES[idx] == name, \
                f"Reverse mapping incorrect for index {idx}"

    def test_all_landmarks_combined(self):
        """Verify ALL_LANDMARKS contains core + extended"""
        for landmark in CORE_LANDMARKS:
            assert landmark in ALL_LANDMARKS
        for landmark in EXTENDED_LANDMARKS:
            assert landmark in ALL_LANDMARKS

    def test_confidence_thresholds_reasonable(self):
        """Verify confidence thresholds are in valid range"""
        assert 0.0 <= MIN_DETECTION_CONFIDENCE <= 1.0
        assert 0.0 <= MIN_TRACKING_CONFIDENCE <= 1.0


class TestMockKeypoints:
    """Test with mock keypoint data"""

    @pytest.fixture
    def mock_keypoints(self):
        """Create mock keypoints data similar to what PoseProcessor outputs"""
        return {
            'frame': 0,
            'timestamp': 0.0,
            'frame_confidence': 0.85,
            # Core landmarks
            'nose': {'x': 0.5, 'y': 0.2, 'z': 0.0, 'visibility': 0.95},
            'left_shoulder': {'x': 0.4, 'y': 0.35, 'z': 0.0, 'visibility': 0.9},
            'right_shoulder': {'x': 0.6, 'y': 0.35, 'z': 0.0, 'visibility': 0.9},
            'left_elbow': {'x': 0.35, 'y': 0.45, 'z': 0.0, 'visibility': 0.85},
            'right_elbow': {'x': 0.65, 'y': 0.45, 'z': 0.0, 'visibility': 0.85},
            'left_wrist': {'x': 0.3, 'y': 0.55, 'z': 0.0, 'visibility': 0.8},
            'right_wrist': {'x': 0.7, 'y': 0.55, 'z': 0.0, 'visibility': 0.8},
            'left_hip': {'x': 0.45, 'y': 0.55, 'z': 0.0, 'visibility': 0.9},
            'right_hip': {'x': 0.55, 'y': 0.55, 'z': 0.0, 'visibility': 0.9},
            'left_knee': {'x': 0.43, 'y': 0.75, 'z': 0.0, 'visibility': 0.85},
            'right_knee': {'x': 0.57, 'y': 0.75, 'z': 0.0, 'visibility': 0.85},
            'left_ankle': {'x': 0.42, 'y': 0.95, 'z': 0.0, 'visibility': 0.8},
            'right_ankle': {'x': 0.58, 'y': 0.95, 'z': 0.0, 'visibility': 0.8},
            # Extended landmarks
            'left_pinky': {'x': 0.28, 'y': 0.58, 'z': 0.0, 'visibility': 0.7},
            'right_pinky': {'x': 0.72, 'y': 0.58, 'z': 0.0, 'visibility': 0.7},
            'left_thumb': {'x': 0.32, 'y': 0.53, 'z': 0.0, 'visibility': 0.7},
            'right_thumb': {'x': 0.68, 'y': 0.53, 'z': 0.0, 'visibility': 0.7},
            'left_heel': {'x': 0.40, 'y': 0.98, 'z': 0.0, 'visibility': 0.75},
            'right_heel': {'x': 0.60, 'y': 0.98, 'z': 0.0, 'visibility': 0.75},
            'left_foot_index': {'x': 0.38, 'y': 0.99, 'z': 0.0, 'visibility': 0.7},
            'right_foot_index': {'x': 0.62, 'y': 0.99, 'z': 0.0, 'visibility': 0.7}
        }

    def test_mock_keypoints_have_core_landmarks(self, mock_keypoints):
        """Verify mock keypoints contain all core landmarks"""
        for landmark in CORE_LANDMARKS:
            assert landmark in mock_keypoints, f"Missing core landmark: {landmark}"
            assert 'x' in mock_keypoints[landmark]
            assert 'y' in mock_keypoints[landmark]
            assert 'visibility' in mock_keypoints[landmark]

    def test_mock_keypoints_have_extended_landmarks(self, mock_keypoints):
        """Verify mock keypoints contain all extended landmarks"""
        for landmark in EXTENDED_LANDMARKS:
            assert landmark in mock_keypoints, f"Missing extended landmark: {landmark}"
            assert 'x' in mock_keypoints[landmark]
            assert 'y' in mock_keypoints[landmark]
            assert 'visibility' in mock_keypoints[landmark]

    def test_metrics_calculator_with_mock_data(self, mock_keypoints):
        """Test MetricsCalculator can handle mock data with extended landmarks"""
        from services.metrics_calculator import MetricsCalculator

        calculator = MetricsCalculator()

        # Create a mock keypoints sequence
        keypoints_sequence = [mock_keypoints.copy() for _ in range(30)]
        for i, kp in enumerate(keypoints_sequence):
            kp['frame'] = i
            kp['timestamp'] = i / 30.0

        # Create mock phases
        phases = {
            'dip_start': {'frame': 5, 'timestamp': 5/30.0},
            'load': {'frame': 10, 'timestamp': 10/30.0, 'knee_angle': 140},
            'release': {'frame': 20, 'timestamp': 20/30.0},
            'follow_through_end': {'frame': 28, 'timestamp': 28/30.0},
            'shooting_hand': 'right',
            'valid': True,
            'confidence': 0.9
        }

        # Calculate metrics
        metrics = calculator.calculate_all_metrics(keypoints_sequence, phases)

        # Verify no error
        assert 'error' not in metrics, f"Got error: {metrics.get('error')}"

        # Verify core metrics present
        core_metrics = ['release_angle', 'elbow_flare', 'knee_load',
                       'hip_shoulder_alignment', 'base_width', 'lateral_sway', 'arc_trajectory']
        for metric in core_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert 'quality_score' in metrics[metric] or 'error' in metrics[metric]

        # Verify extended metrics present
        extended_metrics = ['hand_alignment', 'foot_alignment']
        for metric in extended_metrics:
            assert metric in metrics, f"Missing extended metric: {metric}"

        # Verify overall score
        assert 'overall_score' in metrics
        assert 0 <= metrics['overall_score'] <= 100


class TestConfigValues:
    """Test configuration values are sensible"""

    def test_concurrency_limits(self):
        """Verify concurrency limits are positive"""
        from config import MAX_CONCURRENT_REQUESTS, MAX_QUEUE_SIZE, QUEUE_TIMEOUT_SECONDS

        assert MAX_CONCURRENT_REQUESTS > 0
        assert MAX_QUEUE_SIZE >= 0
        assert QUEUE_TIMEOUT_SECONDS > 0

    def test_video_limits(self):
        """Verify video limits are reasonable"""
        from config import MAX_VIDEO_DURATION_SECONDS, MAX_FRAMES_TO_PROCESS, MAX_UPLOAD_SIZE_BYTES

        assert MAX_VIDEO_DURATION_SECONDS > 0
        assert MAX_FRAMES_TO_PROCESS > 0
        assert MAX_UPLOAD_SIZE_BYTES > 0

    def test_analysis_thresholds(self):
        """Verify analysis thresholds are in valid range"""
        from config import MIN_ANALYSIS_CONFIDENCE, MIN_FRAMES_WITH_POSE, MIN_LANDMARK_COVERAGE

        assert 0 <= MIN_ANALYSIS_CONFIDENCE <= 1
        assert MIN_FRAMES_WITH_POSE > 0
        assert 0 <= MIN_LANDMARK_COVERAGE <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
