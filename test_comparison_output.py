"""
Test Script - Show Complete API Output for Shot Comparison
Demonstrates the full analysis workflow and output format
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.shot_analysis_service import ShotAnalysisService
from services.pose_processor import PoseProcessor
from services.phase_detector import PhaseDetector
from services.metrics_calculator import MetricsCalculator
from services.shot_comparator import ShotComparator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_json_output(data: Dict[str, Any], indent: int = 2) -> str:
    """Pretty print JSON with custom formatting"""
    return json.dumps(data, indent=indent, default=str)


def print_section(title: str, char: str = "="):
    """Print a formatted section header"""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def analyze_video_detailed(video_path: str, baseline_player: str = "Stephen Curry"):
    """
    Perform detailed shot analysis and show all output data
    """
    print_section("🏀 BASKETBALL SHOT ANALYSIS TEST", "=")
    print(f"📹 Video: {video_path}")
    print(f"👤 Comparing to: {baseline_player}")
    print(f"⏱️  Starting analysis...\n")

    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found at {video_path}")
        return

    # Initialize services
    print("🔧 Initializing analysis services...")
    pose_processor = PoseProcessor(model_complexity=1)
    phase_detector = PhaseDetector()
    metrics_calculator = MetricsCalculator()
    shot_comparator = ShotComparator()

    # Step 1: Extract pose data
    print_section("STEP 1: POSE EXTRACTION", "-")
    print("🔍 Extracting pose keypoints from video frames...")

    pose_data = pose_processor.process_video(video_path, frame_skip=1)

    print(f"✅ Pose extraction complete!")
    print(f"   - Frames processed: {pose_data['metadata']['processed_frames']}")
    print(f"   - Video duration: {pose_data['metadata']['duration']:.2f}s")
    print(f"   - Average FPS: {pose_data['metadata']['fps']:.1f}")
    print(f"   - Overall confidence: {pose_data['quality']['confidence']:.1%}")
    print(f"   - Visibility ratio: {pose_data['quality']['visibility_ratio']:.1%}")

    print("\n📊 Pose Data Structure:")
    pose_summary = {
        "processed_frames": pose_data['metadata']['processed_frames'],
        "total_frames": pose_data['metadata']['total_frames'],
        "fps": pose_data['metadata']['fps'],
        "duration": pose_data['metadata']['duration'],
        "quality": pose_data['quality'],
        "keypoints_sequence_length": len(pose_data['keypoints_sequence'])
    }
    print(format_json_output(pose_summary))

    # Check quality
    if pose_data['quality']['confidence'] < 0.5:
        print(f"\n⚠️  WARNING: Low video quality detected!")
        print(f"   Confidence: {pose_data['quality']['confidence']:.1%}")
        if 'warning' in pose_data['quality']:
            print(f"   Issue: {pose_data['quality']['warning']}")
        return

    # Step 2: Phase detection
    print_section("STEP 2: PHASE DETECTION", "-")
    print("🔍 Detecting shooting phases (stance → preparation → release → follow-through)...")

    phases = phase_detector.detect_phases(pose_data['keypoints_sequence'])

    if phases.get('valid', False):
        print(f"✅ Phases detected successfully!")
        print(f"\n📊 Phase Information:")
        print(format_json_output(phases))

        print("\n📈 Phase Breakdown:")
        for phase_name in ['stance', 'preparation', 'release', 'follow_through']:
            if phase_name in phases:
                phase_info = phases[phase_name]
                start_frame = phase_info.get('start_frame', 0)
                end_frame = phase_info.get('end_frame', 0)
                duration = phase_info.get('duration', 0)
                print(f"   {phase_name.upper()}: frames {start_frame}-{end_frame} ({duration:.2f}s)")
    else:
        print(f"❌ Could not detect valid shooting phases")
        print(f"   Confidence: {phases.get('confidence', 0):.1%}")
        return

    # Step 3: Calculate metrics
    print_section("STEP 3: METRICS CALCULATION", "-")
    print("📐 Calculating shooting form metrics...")

    metrics = metrics_calculator.calculate_all_metrics(
        pose_data['keypoints_sequence'],
        phases
    )

    if 'error' in metrics:
        print(f"❌ Error calculating metrics: {metrics['error']}")
        return

    print(f"✅ Metrics calculated successfully!")
    print(f"\n📊 Complete Metrics Output:")
    print(format_json_output(metrics))

    print("\n🎯 Key Metrics Summary:")
    if 'release_angle' in metrics:
        print(f"   Release Angle: {metrics['release_angle'].get('angle', 0):.1f}°")
    if 'elbow_alignment' in metrics:
        print(f"   Elbow Alignment: {metrics['elbow_alignment'].get('score', 0):.1%}")
    if 'follow_through' in metrics:
        print(f"   Follow Through: {metrics['follow_through'].get('score', 0):.1%}")
    if 'balance' in metrics:
        print(f"   Balance Score: {metrics['balance'].get('score', 0):.1%}")
    if 'knee_bend' in metrics:
        print(f"   Knee Bend: {metrics['knee_bend'].get('angle', 0):.1f}°")

    # Step 4: Compare to baseline
    print_section("STEP 4: BASELINE COMPARISON", "-")
    print(f"⚖️  Comparing your shot to {baseline_player}'s form...")

    # Create user analysis structure
    user_analysis = {
        'metrics': metrics,
        'phases': phases,
        'quality': pose_data['quality']
    }

    comparison = shot_comparator.compare_to_baseline(user_analysis, baseline_player)

    if 'error' in comparison:
        print(f"❌ {comparison['error']}")
        print(f"   Available players: {comparison.get('available_players', [])}")
        return

    print(f"✅ Comparison complete!")
    print(f"\n📊 Complete Comparison Output:")
    print(format_json_output(comparison))

    # Step 5: Pretty print results
    print_section("FINAL RESULTS SUMMARY", "=")

    print(f"🎯 Overall Similarity: {comparison.get('overall_similarity', 0):.1f}%\n")

    print("📊 Metric-by-Metric Comparison:")
    print("-" * 80)
    for metric_name, metric_data in comparison.get('metric_comparisons', {}).items():
        similarity = metric_data.get('similarity', 0) * 100
        status = "✅" if similarity >= 80 else "⚠️" if similarity >= 60 else "❌"
        print(f"\n{status} {metric_name.replace('_', ' ').title()}:")
        print(f"   Similarity: {similarity:.1f}%")
        print(f"   Your value: {metric_data.get('user_value', 'N/A')}")
        print(f"   {baseline_player}'s value: {metric_data.get('baseline_value', 'N/A')}")
        if 'difference' in metric_data:
            print(f"   Difference: {metric_data['difference']}")
        if 'feedback' in metric_data:
            print(f"   💡 {metric_data['feedback']}")

    print("\n" + "-" * 80)
    print("\n💪 Your Strengths:")
    for strength in comparison.get('strengths', []):
        print(f"   ✓ {strength}")

    print("\n🎯 Areas for Improvement:")
    for improvement in comparison.get('areas_for_improvement', []):
        print(f"   → {improvement}")

    print("\n📝 Specific Feedback:")
    for feedback in comparison.get('specific_feedback', []):
        print(f"   • {feedback}")

    # Step 6: Save complete output to file
    print_section("SAVING OUTPUT", "-")
    output_file = Path("output") / "test_comparison_result.json"
    output_file.parent.mkdir(exist_ok=True)

    complete_output = {
        "video": str(video_path),
        "baseline_player": baseline_player,
        "pose_data_summary": pose_summary,
        "phases": phases,
        "metrics": metrics,
        "comparison": comparison,
        "analysis_metadata": {
            "frames_analyzed": pose_data['metadata']['processed_frames'],
            "video_quality": pose_data['quality']['confidence'],
            "analysis_complete": True
        }
    }

    with open(output_file, 'w') as f:
        json.dump(complete_output, f, indent=2, default=str)

    print(f"💾 Complete output saved to: {output_file}")
    print(f"📄 File size: {output_file.stat().st_size / 1024:.1f} KB")

    print_section("✅ ANALYSIS COMPLETE", "=")
    print("You can now see exactly what the API will return!")
    print(f"Check {output_file} for the full JSON output.\n")


if __name__ == "__main__":
    # Default to test video
    default_video = "tests/TestShot.mp4"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = default_video

    baseline_player = sys.argv[2] if len(sys.argv) > 2 else "Stephen Curry"

    analyze_video_detailed(video_path, baseline_player)
