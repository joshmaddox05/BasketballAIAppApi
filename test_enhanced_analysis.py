"""
Test Enhanced Analysis - Demo Head & Hands Features
Shows wrist flick, palm angles, and head stability on test videos
"""
import json
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor
from services.hand_processor import HandProcessor
from services.phase_detector import PhaseDetector
from services.enhanced_metrics_calculator import EnhancedMetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_analysis(video_path: str, enable_hands: bool = True):
    """Test the enhanced analysis pipeline with head and hands tracking"""

    print("=" * 80)
    print("🏀 TESTING ENHANCED SHOT ANALYSIS")
    print("=" * 80)
    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"👐 Hands tracking: {'ENABLED' if enable_hands else 'DISABLED'}")
    print()

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return

    # Initialize processors
    print("🔧 Initializing processors...")
    pose_processor = PoseProcessor(model_complexity=1)
    hand_processor = HandProcessor(max_num_hands=2) if enable_hands else None
    phase_detector = PhaseDetector()
    enhanced_calculator = EnhancedMetricsCalculator()

    # Step 1: Extract pose data
    print("\n📊 STEP 1: Extracting pose landmarks...")
    pose_data = pose_processor.process_video(video_path, frame_skip=1)

    metadata = pose_data['metadata']
    quality = pose_data['quality']

    print(f"   ✓ Processed {metadata['processed_frames']} frames")
    print(f"   ✓ Quality: {quality['confidence']:.1%}")
    print(f"   ✓ FPS: {metadata['fps']:.1f}")

    # Step 2: Extract hand data (if enabled)
    hand_data_sequence = None
    if enable_hands and hand_processor:
        print("\n👐 STEP 2: Extracting hand landmarks...")

        import cv2
        cap = cv2.VideoCapture(video_path)

        hand_data_sequence = []
        frame_idx = 0
        hands_detected = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_data = hand_processor.process_frame(rgb_frame, frame_idx)
            hand_data_sequence.append(hand_data)

            if hand_data.get('right') or hand_data.get('left'):
                hands_detected += 1

            frame_idx += 1

        cap.release()

        hands_coverage = hands_detected / len(hand_data_sequence) if hand_data_sequence else 0
        print(f"   ✓ Hands detected in {hands_detected}/{len(hand_data_sequence)} frames ({hands_coverage:.1%})")
    else:
        print("\n👐 STEP 2: Hand tracking disabled - using pose-only proxies")

    # Step 3: Detect phases
    print("\n🔍 STEP 3: Detecting shooting phases...")
    phases = phase_detector.detect_phases(pose_data['keypoints_sequence'])

    if phases.get('valid'):
        print(f"   ✓ Phases detected successfully")
        for phase_name in ['stance', 'load', 'release', 'follow_through']:
            if phase_name in phases:
                phase_info = phases.get(phase_name, {})
                if isinstance(phase_info, dict):
                    frame = phase_info.get('frame', 'N/A')
                    print(f"      {phase_name}: frame {frame}")
    else:
        print(f"   ⚠️  Phase detection confidence low: {phases.get('confidence', 0):.1%}")
        print(f"   ℹ️  Will compute metrics on available data")

    # Step 4: Calculate enhanced metrics
    print("\n📐 STEP 4: Computing enhanced metrics...")
    metrics = enhanced_calculator.calculate_all_metrics(
        pose_data['keypoints_sequence'],
        phases,
        metadata['fps'],
        hand_data_sequence
    )

    # Display results
    print("\n" + "=" * 80)
    print("📊 ENHANCED METRICS RESULTS")
    print("=" * 80)

    # Modalities
    print("\n🔧 Data Sources:")
    print(f"   Pose tracking: {'✓' if metrics.get('modalities', {}).get('pose') else '✗'}")
    print(f"   Hand tracking: {'✓' if metrics.get('modalities', {}).get('hands') else '✗'}")

    if 'debug' in metrics:
        debug = metrics['debug']
        if 'hands_coverage_pct' in debug:
            print(f"   Hand coverage: {debug['hands_coverage_pct']:.1f}%")

    # Key Metrics
    print("\n🎯 WRIST MECHANICS:")
    if metrics.get('wrist_flick_peak_deg_s'):
        wrist_flick = metrics['wrist_flick_peak_deg_s']
        target = "700-1100"
        status = "✓" if 700 <= wrist_flick <= 1100 else "⚠️"
        print(f"   {status} Wrist flick: {wrist_flick:.0f} deg/s (target: {target})")

    if metrics.get('wrist_followthrough_ms'):
        followthrough = metrics['wrist_followthrough_ms']
        target = "300-600"
        status = "✓" if 300 <= followthrough <= 600 else "⚠️"
        print(f"   {status} Follow-through: {followthrough:.0f} ms (target: {target})")

    print("\n🤚 PALM ORIENTATION:")
    if metrics.get('palm_angle_to_vertical_deg') is not None:
        palm_vert = metrics['palm_angle_to_vertical_deg']
        target = "10-30"
        status = "✓" if 10 <= palm_vert <= 30 else "⚠️"
        print(f"   {status} Palm to vertical: {palm_vert:.1f}° (target: {target})")
    else:
        print(f"   ⚠️  Palm metrics unavailable (no hand data)")

    if metrics.get('palm_toward_target_deg') is not None:
        palm_target = metrics['palm_toward_target_deg']
        target = "0-20"
        status = "✓" if palm_target <= 20 else "⚠️"
        print(f"   {status} Palm to target: {palm_target:.1f}° (target: {target})")

    print("\n🧠 HEAD STABILITY:")
    if metrics.get('head_tilt_deg') is not None:
        head_tilt = metrics['head_tilt_deg']
        target = "0-8"
        status = "✓" if head_tilt <= 8 else "⚠️"
        print(f"   {status} Head tilt: {head_tilt:.1f}° (target: {target})")

    if metrics.get('head_yaw_jitter_deg_s') is not None:
        head_jitter = metrics['head_yaw_jitter_deg_s']
        target = "<50"
        status = "✓" if head_jitter < 50 else "⚠️"
        print(f"   {status} Head jitter: {head_jitter:.1f} deg/s (target: {target})")

    if metrics.get('gaze_stability_cm') is not None:
        gaze = metrics['gaze_stability_cm']
        target = "<3"
        status = "✓" if gaze <= 3 else "⚠️"
        print(f"   {status} Gaze stability: {gaze:.1f} cm (target: {target})")

    # Overall Grade
    if 'overall_grade' in metrics:
        grade_info = metrics['overall_grade']
        print(f"\n🏆 OVERALL GRADE: {grade_info['grade']} ({grade_info['score']:.1%})")
        print(f"   {grade_info['description']}")

    # Coaching Cues
    if 'coaching_cues' in metrics and metrics['coaching_cues']:
        print("\n💡 COACHING CUES:")
        for i, cue in enumerate(metrics['coaching_cues'][:3], 1):  # Top 3
            print(f"\n   {i}. {cue['cue']}")
            print(f"      Why: {cue['why']}")
            print(f"      Drill: {cue['drill']}")
            if 'value' in cue:
                print(f"      Current: {cue['value']:.1f} | Target: {cue['target']}")

    # Save detailed results
    output_file = Path("output") / f"{Path(video_path).stem}_enhanced_analysis.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n💾 Full analysis saved to: {output_file}")
    print("\n" + "=" * 80)
    print("✅ ENHANCED ANALYSIS COMPLETE")
    print("=" * 80)
    print()

    return metrics


def main():
    """Main test function"""

    # Test on test video
    test_video = "tests/TestShot.mp4" if len(sys.argv) < 2 else sys.argv[1]
    enable_hands = True if len(sys.argv) < 3 else sys.argv[2].lower() != 'false'

    test_enhanced_analysis(test_video, enable_hands)


if __name__ == "__main__":
    main()

