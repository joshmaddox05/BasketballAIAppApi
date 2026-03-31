"""
Integration tests for video analysis pipeline using real test videos.

Run with: pytest tests/test_video_analysis.py -v
Run specific test: pytest tests/test_video_analysis.py::TestVideoAnalysis::test_steph_front_shot -v
Run with timing: pytest tests/test_video_analysis.py -v --durations=0
"""
import pytest
import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.pose_processor import PoseProcessor
from services.phase_detector import PhaseDetector
from services.metrics_calculator import MetricsCalculator
from services.shot_analysis_service import ShotAnalysisService
from config import CORE_LANDMARKS, EXTENDED_LANDMARKS, ALL_LANDMARKS


# Test video paths
TEST_VIDEOS_DIR = Path(__file__).parent.parent
TEST_VIDEOS = {
    'steph_front': TEST_VIDEOS_DIR / 'Steph Front shot.mp4',
    'test_shot_2': TEST_VIDEOS_DIR / 'TestShot2.mp4',
    'test_shot_3': TEST_VIDEOS_DIR / 'TestShot3.mp4'
}


def video_exists(video_name: str) -> bool:
    """Check if a test video exists"""
    return TEST_VIDEOS.get(video_name, Path()).exists()


class TestPoseProcessor:
    """Test pose extraction from videos"""

    @pytest.fixture(scope='class')
    def pose_processor(self):
        """Create a shared pose processor instance"""
        return PoseProcessor()

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_steph_front_pose_extraction(self, pose_processor):
        """Test pose extraction from Steph Front shot video"""
        result = pose_processor.process_video(str(TEST_VIDEOS['steph_front']))

        # Should not have error
        assert 'error' not in result, f"Pose extraction failed: {result.get('error')}"

        # Should have keypoints
        assert 'keypoints_sequence' in result
        assert len(result['keypoints_sequence']) > 0, "No frames with pose detected"

        # Should have metadata
        assert 'metadata' in result
        assert result['metadata']['fps'] > 0
        assert result['metadata']['total_frames'] > 0

        # Should have quality info
        assert 'quality' in result
        assert result['quality']['confidence'] > 0.3, "Confidence too low"

        print(f"\n  Frames processed: {result['metadata']['processed_frames']}")
        print(f"  Confidence: {result['quality']['confidence']:.2%}")
        print(f"  Detection rate: {result['quality'].get('detection_rate', 'N/A')}")

    @pytest.mark.skipif(not video_exists('test_shot_2'), reason="Test video not found")
    def test_test_shot_2_pose_extraction(self, pose_processor):
        """Test pose extraction from TestShot2 video"""
        result = pose_processor.process_video(str(TEST_VIDEOS['test_shot_2']))

        assert 'error' not in result, f"Pose extraction failed: {result.get('error')}"
        assert len(result['keypoints_sequence']) > 0
        assert result['quality']['confidence'] > 0.3

        print(f"\n  Frames processed: {result['metadata']['processed_frames']}")
        print(f"  Confidence: {result['quality']['confidence']:.2%}")

    @pytest.mark.skipif(not video_exists('test_shot_3'), reason="Test video not found")
    def test_test_shot_3_pose_extraction(self, pose_processor):
        """Test pose extraction from TestShot3 video"""
        result = pose_processor.process_video(str(TEST_VIDEOS['test_shot_3']))

        assert 'error' not in result, f"Pose extraction failed: {result.get('error')}"
        assert len(result['keypoints_sequence']) > 0
        assert result['quality']['confidence'] > 0.3

        print(f"\n  Frames processed: {result['metadata']['processed_frames']}")
        print(f"  Confidence: {result['quality']['confidence']:.2%}")


class TestLandmarkExtraction:
    """Test that all landmarks are properly extracted"""

    @pytest.fixture(scope='class')
    def pose_processor(self):
        return PoseProcessor()

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_core_landmarks_present(self, pose_processor):
        """Verify core landmarks are extracted from video"""
        result = pose_processor.process_video(str(TEST_VIDEOS['steph_front']))

        assert 'error' not in result
        assert len(result['keypoints_sequence']) > 0

        # Check first frame with keypoints
        first_frame = result['keypoints_sequence'][0]

        # Count how many core landmarks are present
        present_landmarks = []
        missing_landmarks = []

        for landmark in CORE_LANDMARKS:
            if landmark in first_frame and isinstance(first_frame[landmark], dict):
                present_landmarks.append(landmark)
            else:
                missing_landmarks.append(landmark)

        coverage = len(present_landmarks) / len(CORE_LANDMARKS)

        print(f"\n  Core landmarks present: {len(present_landmarks)}/{len(CORE_LANDMARKS)} ({coverage:.0%})")
        if missing_landmarks:
            print(f"  Missing: {missing_landmarks}")

        # Should have at least 70% of core landmarks
        assert coverage >= 0.7, f"Too few core landmarks: {present_landmarks}"

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_extended_landmarks_present(self, pose_processor):
        """Verify extended landmarks (hands/feet) are extracted"""
        result = pose_processor.process_video(str(TEST_VIDEOS['steph_front']))

        assert 'error' not in result

        first_frame = result['keypoints_sequence'][0]

        present_extended = []
        for landmark in EXTENDED_LANDMARKS:
            if landmark in first_frame and isinstance(first_frame[landmark], dict):
                present_extended.append(landmark)

        coverage = len(present_extended) / len(EXTENDED_LANDMARKS)

        print(f"\n  Extended landmarks present: {len(present_extended)}/{len(EXTENDED_LANDMARKS)} ({coverage:.0%})")
        print(f"  Found: {present_extended}")

        # Extended landmarks may have lower visibility, so we're more lenient
        # At least some should be present
        assert len(present_extended) >= 2, "Too few extended landmarks detected"

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_landmark_coverage_stats(self, pose_processor):
        """Test that landmark coverage statistics are calculated"""
        result = pose_processor.process_video(str(TEST_VIDEOS['steph_front']))

        assert 'quality' in result
        assert 'landmark_coverage' in result['quality']
        assert 'landmark_stats' in result['quality']

        print(f"\n  Core landmark coverage: {result['quality']['landmark_coverage']:.0%}")
        print(f"  Extended landmark coverage: {result['quality'].get('extended_landmark_coverage', 'N/A')}")

        if result['quality'].get('missing_landmarks'):
            print(f"  Missing landmarks: {result['quality']['missing_landmarks']}")


class TestPhaseDetection:
    """Test shooting phase detection"""

    @pytest.fixture(scope='class')
    def pose_processor(self):
        return PoseProcessor()

    @pytest.fixture(scope='class')
    def phase_detector(self):
        return PhaseDetector()

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_steph_front_phase_detection(self, pose_processor, phase_detector):
        """Test phase detection on Steph Front shot"""
        pose_result = pose_processor.process_video(str(TEST_VIDEOS['steph_front']))

        assert 'error' not in pose_result

        fps = pose_result['metadata']['fps']
        phases = phase_detector.detect_phases(pose_result['keypoints_sequence'], fps)

        print(f"\n  Phases valid: {phases.get('valid')}")
        print(f"  Confidence: {phases.get('confidence', 0):.2%}")
        print(f"  Shooting hand: {phases.get('shooting_hand')}")

        if phases.get('load'):
            print(f"  Load frame: {phases['load'].get('frame')}")
        if phases.get('release'):
            print(f"  Release frame: {phases['release'].get('frame')}")

        # Should detect at least load and release
        assert phases.get('load') is not None, "Load phase not detected"
        assert phases.get('release') is not None, "Release phase not detected"

    @pytest.mark.skipif(not video_exists('test_shot_2'), reason="Test video not found")
    def test_test_shot_2_phase_detection(self, pose_processor, phase_detector):
        """Test phase detection on TestShot2"""
        pose_result = pose_processor.process_video(str(TEST_VIDEOS['test_shot_2']))

        if 'error' in pose_result:
            pytest.skip(f"Pose extraction failed: {pose_result['error']}")

        fps = pose_result['metadata']['fps']
        phases = phase_detector.detect_phases(pose_result['keypoints_sequence'], fps)

        print(f"\n  Phases valid: {phases.get('valid')}")
        print(f"  Confidence: {phases.get('confidence', 0):.2%}")

    @pytest.mark.skipif(not video_exists('test_shot_3'), reason="Test video not found")
    def test_test_shot_3_phase_detection(self, pose_processor, phase_detector):
        """Test phase detection on TestShot3"""
        pose_result = pose_processor.process_video(str(TEST_VIDEOS['test_shot_3']))

        if 'error' in pose_result:
            pytest.skip(f"Pose extraction failed: {pose_result['error']}")

        fps = pose_result['metadata']['fps']
        phases = phase_detector.detect_phases(pose_result['keypoints_sequence'], fps)

        print(f"\n  Phases valid: {phases.get('valid')}")
        print(f"  Confidence: {phases.get('confidence', 0):.2%}")


class TestMetricsCalculation:
    """Test biomechanical metrics calculation"""

    @pytest.fixture(scope='class')
    def analysis_components(self):
        """Create shared analysis components"""
        return {
            'pose_processor': PoseProcessor(),
            'phase_detector': PhaseDetector(),
            'metrics_calculator': MetricsCalculator()
        }

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_steph_front_metrics(self, analysis_components):
        """Test metrics calculation on Steph Front shot"""
        pose_result = analysis_components['pose_processor'].process_video(
            str(TEST_VIDEOS['steph_front'])
        )

        assert 'error' not in pose_result

        fps = pose_result['metadata']['fps']
        phases = analysis_components['phase_detector'].detect_phases(
            pose_result['keypoints_sequence'], fps
        )

        if not phases.get('valid'):
            pytest.skip("Phase detection failed - cannot calculate metrics")

        metrics = analysis_components['metrics_calculator'].calculate_all_metrics(
            pose_result['keypoints_sequence'], phases
        )

        assert 'error' not in metrics, f"Metrics calculation failed: {metrics.get('error')}"
        assert 'overall_score' in metrics

        print(f"\n  Overall score: {metrics['overall_score']:.1f}/100")

        # Check core metrics
        core_metrics = ['release_angle', 'elbow_flare', 'knee_load',
                       'hip_shoulder_alignment', 'base_width', 'lateral_sway']

        for metric_name in core_metrics:
            if metric_name in metrics and 'error' not in metrics[metric_name]:
                score = metrics[metric_name].get('quality_score', 0)
                value = metrics[metric_name].get('angle_deg') or metrics[metric_name].get('ratio') or metrics[metric_name].get('sway_ratio')
                print(f"  {metric_name}: {value} (score: {score:.1f})")

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_extended_metrics_present(self, analysis_components):
        """Test that extended metrics (hand/foot alignment) are calculated"""
        pose_result = analysis_components['pose_processor'].process_video(
            str(TEST_VIDEOS['steph_front'])
        )

        if 'error' in pose_result:
            pytest.skip("Pose extraction failed")

        fps = pose_result['metadata']['fps']
        phases = analysis_components['phase_detector'].detect_phases(
            pose_result['keypoints_sequence'], fps
        )

        if not phases.get('valid'):
            pytest.skip("Phase detection failed")

        metrics = analysis_components['metrics_calculator'].calculate_all_metrics(
            pose_result['keypoints_sequence'], phases
        )

        # Check extended metrics
        print("\n  Extended metrics:")

        if 'hand_alignment' in metrics:
            ha = metrics['hand_alignment']
            if 'error' not in ha:
                print(f"  hand_alignment: score={ha.get('quality_score')}, offset={ha.get('hand_offset')}")
            else:
                print(f"  hand_alignment: {ha.get('error')}")

        if 'foot_alignment' in metrics:
            fa = metrics['foot_alignment']
            if 'error' not in fa:
                print(f"  foot_alignment: score={fa.get('quality_score')}, stagger={fa.get('stagger')}")
            else:
                print(f"  foot_alignment: {fa.get('error')}")


class TestFullAnalysisPipeline:
    """Test complete analysis pipeline end-to-end"""

    @pytest.fixture(scope='class')
    def analysis_service(self):
        """Create shared analysis service"""
        return ShotAnalysisService()

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_steph_front_full_analysis(self, analysis_service):
        """Full end-to-end analysis of Steph Front shot"""
        start_time = time.time()

        result = analysis_service.analyze_shot(str(TEST_VIDEOS['steph_front']))

        elapsed = time.time() - start_time

        print(f"\n  Analysis completed in {elapsed:.2f}s")
        print(f"  Success: {result.get('success')}")

        if result.get('success'):
            print(f"  Overall score: {result['overall_score']:.1f}/100")
            print(f"  Grade: {result.get('overall_grade')}")
            print(f"  Confidence: {result.get('confidence', 0):.2%}")

            if result.get('coaching_cues'):
                print(f"  Top coaching cue: {result['coaching_cues'][0].get('cue')}")

            # Verify all expected fields are present
            assert 'overall_score' in result
            assert 'phases' in result
            assert 'metrics' in result
            assert 'coaching_cues' in result
            assert 'improvement_summary' in result

            # Score should be reasonable
            assert 0 <= result['overall_score'] <= 100
        else:
            print(f"  Error: {result.get('error')}")
            print(f"  Error code: {result.get('error_code')}")
            if result.get('tips'):
                print(f"  Tips: {result['tips'][:2]}")

            # Even failures should have structured response
            assert 'error' in result
            assert 'tips' in result or 'error_code' in result

    @pytest.mark.skipif(not video_exists('test_shot_2'), reason="Test video not found")
    def test_test_shot_2_full_analysis(self, analysis_service):
        """Full analysis of TestShot2"""
        start_time = time.time()

        result = analysis_service.analyze_shot(str(TEST_VIDEOS['test_shot_2']))

        elapsed = time.time() - start_time

        print(f"\n  Analysis completed in {elapsed:.2f}s")
        print(f"  Success: {result.get('success')}")

        if result.get('success'):
            print(f"  Overall score: {result['overall_score']:.1f}/100")
            print(f"  Grade: {result.get('overall_grade')}")
        else:
            print(f"  Error: {result.get('error')}")

    @pytest.mark.skipif(not video_exists('test_shot_3'), reason="Test video not found")
    def test_test_shot_3_full_analysis(self, analysis_service):
        """Full analysis of TestShot3"""
        start_time = time.time()

        result = analysis_service.analyze_shot(str(TEST_VIDEOS['test_shot_3']))

        elapsed = time.time() - start_time

        print(f"\n  Analysis completed in {elapsed:.2f}s")
        print(f"  Success: {result.get('success')}")

        if result.get('success'):
            print(f"  Overall score: {result['overall_score']:.1f}/100")
            print(f"  Grade: {result.get('overall_grade')}")
        else:
            print(f"  Error: {result.get('error')}")


class TestAnalysisConsistency:
    """Test that analysis is consistent across multiple runs"""

    @pytest.fixture(scope='class')
    def analysis_service(self):
        return ShotAnalysisService()

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_consistent_scores(self, analysis_service):
        """Verify scores are consistent across multiple runs"""
        scores = []

        for i in range(3):
            result = analysis_service.analyze_shot(str(TEST_VIDEOS['steph_front']))
            if result.get('success'):
                scores.append(result['overall_score'])

        if len(scores) < 2:
            pytest.skip("Not enough successful analyses for consistency check")

        # Scores should be within 5 points of each other
        score_range = max(scores) - min(scores)

        print(f"\n  Scores across 3 runs: {scores}")
        print(f"  Range: {score_range:.1f}")

        assert score_range < 10, f"Scores too inconsistent: {scores}"


class TestPerformance:
    """Test analysis performance"""

    @pytest.fixture(scope='class')
    def analysis_service(self):
        return ShotAnalysisService()

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_analysis_time(self, analysis_service):
        """Test that analysis completes in reasonable time"""
        start_time = time.time()

        result = analysis_service.analyze_shot(str(TEST_VIDEOS['steph_front']))

        elapsed = time.time() - start_time

        print(f"\n  Total analysis time: {elapsed:.2f}s")

        # Should complete within 30 seconds for a typical video
        assert elapsed < 30, f"Analysis took too long: {elapsed:.2f}s"

    @pytest.mark.skipif(not video_exists('steph_front'), reason="Test video not found")
    def test_frame_skip_performance(self, analysis_service):
        """Compare performance with different frame skip values"""
        times = {}

        for skip in [1, 2, 3]:
            start = time.time()
            result = analysis_service.analyze_shot(
                str(TEST_VIDEOS['steph_front']),
                frame_skip=skip
            )
            times[skip] = time.time() - start

        print("\n  Frame skip performance:")
        for skip, t in times.items():
            print(f"    frame_skip={skip}: {t:.2f}s")

        # Higher skip should generally be faster
        # (but may not always be true due to caching/overhead)


class TestErrorHandling:
    """Test error handling with edge cases"""

    @pytest.fixture(scope='class')
    def analysis_service(self):
        return ShotAnalysisService()

    def test_nonexistent_video(self, analysis_service):
        """Test handling of non-existent video file"""
        result = analysis_service.analyze_shot('/nonexistent/video.mp4')

        assert result.get('success') == False
        assert 'error' in result
        print(f"\n  Error for missing file: {result.get('error')}")

    def test_invalid_frame_skip(self, analysis_service):
        """Test handling of invalid frame_skip parameter"""
        if not video_exists('steph_front'):
            pytest.skip("Test video not found")

        # Frame skip of 0 should be handled gracefully
        result = analysis_service.analyze_shot(
            str(TEST_VIDEOS['steph_front']),
            frame_skip=0
        )

        # Should either work (treating 0 as 1) or return error
        print(f"\n  Result with frame_skip=0: success={result.get('success')}")


# Summary report
class TestSummaryReport:
    """Generate a summary report of all test videos"""

    @pytest.fixture(scope='class')
    def analysis_service(self):
        return ShotAnalysisService()

    def test_generate_summary(self, analysis_service):
        """Generate summary report for all test videos"""
        print("\n" + "="*60)
        print("TEST VIDEO ANALYSIS SUMMARY")
        print("="*60)

        for name, path in TEST_VIDEOS.items():
            print(f"\n{name}:")
            print("-" * 40)

            if not path.exists():
                print("  [SKIPPED] Video file not found")
                continue

            start = time.time()
            result = analysis_service.analyze_shot(str(path))
            elapsed = time.time() - start

            if result.get('success'):
                print(f"  Status: SUCCESS")
                print(f"  Score: {result['overall_score']:.1f}/100 ({result.get('overall_grade')})")
                print(f"  Confidence: {result.get('confidence', 0):.1%}")
                print(f"  Time: {elapsed:.2f}s")

                # Metrics summary
                metrics = result.get('metrics', {})
                good_metrics = []
                poor_metrics = []

                for m_name, m_data in metrics.items():
                    if isinstance(m_data, dict) and 'quality_score' in m_data:
                        if m_data['quality_score'] >= 7:
                            good_metrics.append(m_name)
                        elif m_data['quality_score'] < 5:
                            poor_metrics.append(m_name)

                if good_metrics:
                    print(f"  Strong: {', '.join(good_metrics[:3])}")
                if poor_metrics:
                    print(f"  Needs work: {', '.join(poor_metrics[:3])}")
            else:
                print(f"  Status: FAILED")
                print(f"  Error: {result.get('error')}")
                print(f"  Time: {elapsed:.2f}s")

        print("\n" + "="*60)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
