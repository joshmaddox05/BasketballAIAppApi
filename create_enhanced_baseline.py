"""
Enhanced Baseline Creator - Process Multiple Camera Angles
Creates a robust baseline by analyzing Steph Curry's form from front, side, and back angles
Calculates statistical averages, ranges, and confidence intervals
"""
import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor
from services.phase_detector import PhaseDetector
from services.metrics_calculator import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAngleBaselineCreator:
    """Create enhanced baseline from multiple camera angles"""

    def __init__(self, baselines_dir: str = "baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.pose_processor = PoseProcessor(model_complexity=1)
        self.phase_detector = PhaseDetector()
        self.metrics_calculator = MetricsCalculator()

    def categorize_videos(self) -> Dict[str, List[str]]:
        """Categorize videos by camera angle based on filename"""
        video_files = list(self.baselines_dir.glob("*.mp4"))

        categories = {
            'front': [],
            'side': [],
            'back': [],
            'unknown': []
        }

        for video in video_files:
            name_lower = video.name.lower()
            if 'front' in name_lower:
                categories['front'].append(str(video))
            elif 'side' in name_lower:
                categories['side'].append(str(video))
            elif 'back' in name_lower:
                categories['back'].append(str(video))
            else:
                categories['unknown'].append(str(video))

        return categories

    def process_single_video(self, video_path: str, angle: str) -> Dict[str, Any]:
        """Process a single video and extract metrics"""
        logger.info(f"📹 Processing {Path(video_path).name} ({angle} angle)...")

        try:
            # Extract pose data
            pose_data = self.pose_processor.process_video(video_path, frame_skip=1)

            # Check quality
            if pose_data['quality']['confidence'] < 0.5:
                logger.warning(f"⚠️  Low quality: {pose_data['quality']['confidence']:.1%}")
                return None

            # Detect phases
            phases = self.phase_detector.detect_phases(pose_data['keypoints_sequence'])

            # Calculate metrics
            if phases.get('valid', False):
                metrics = self.metrics_calculator.calculate_all_metrics(
                    pose_data['keypoints_sequence'],
                    phases
                )

                if 'error' not in metrics:
                    return {
                        'video_name': Path(video_path).name,
                        'angle': angle,
                        'quality': pose_data['quality'],
                        'metadata': pose_data['metadata'],
                        'phases': phases,
                        'metrics': metrics,
                        'success': True
                    }

            logger.warning(f"⚠️  Could not detect valid phases for {Path(video_path).name}")
            return None

        except Exception as e:
            logger.error(f"❌ Error processing {video_path}: {str(e)}")
            return None

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple videos with statistical analysis"""

        # Extract all successful metrics
        all_metrics = [r['metrics'] for r in results if r and r.get('success')]

        if not all_metrics:
            raise ValueError("No valid metrics to aggregate")

        aggregated = {
            'player_name': 'Stephen Curry',
            'videos_analyzed': len(all_metrics),
            'angles_covered': list(set(r['angle'] for r in results if r)),
            'total_shots_analyzed': len(all_metrics),
            'metrics': {}
        }

        # Aggregate release angle
        if any('release_angle' in m for m in all_metrics):
            release_angles = []
            for m in all_metrics:
                if 'release_angle' in m and 'angle' in m['release_angle']:
                    release_angles.append(m['release_angle']['angle'])

            if release_angles:
                aggregated['metrics']['release_angle'] = {
                    'angle': float(np.mean(release_angles)),
                    'mean': float(np.mean(release_angles)),
                    'std': float(np.std(release_angles)),
                    'min': float(np.min(release_angles)),
                    'max': float(np.max(release_angles)),
                    'confidence_interval_95': [
                        float(np.mean(release_angles) - 1.96 * np.std(release_angles)),
                        float(np.mean(release_angles) + 1.96 * np.std(release_angles))
                    ],
                    'samples': len(release_angles),
                    'consistency_score': float(1.0 - (np.std(release_angles) / np.mean(release_angles))) if np.mean(release_angles) > 0 else 0.0
                }

        # Aggregate elbow alignment
        if any('elbow_alignment' in m for m in all_metrics):
            elbow_scores = []
            for m in all_metrics:
                if 'elbow_alignment' in m and 'score' in m['elbow_alignment']:
                    elbow_scores.append(m['elbow_alignment']['score'])

            if elbow_scores:
                aggregated['metrics']['elbow_alignment'] = {
                    'score': float(np.mean(elbow_scores)),
                    'mean': float(np.mean(elbow_scores)),
                    'std': float(np.std(elbow_scores)),
                    'min': float(np.min(elbow_scores)),
                    'max': float(np.max(elbow_scores)),
                    'confidence_interval_95': [
                        float(np.mean(elbow_scores) - 1.96 * np.std(elbow_scores)),
                        float(np.mean(elbow_scores) + 1.96 * np.std(elbow_scores))
                    ],
                    'samples': len(elbow_scores)
                }

        # Aggregate follow through
        if any('follow_through' in m for m in all_metrics):
            follow_through_scores = []
            for m in all_metrics:
                if 'follow_through' in m and 'score' in m['follow_through']:
                    follow_through_scores.append(m['follow_through']['score'])

            if follow_through_scores:
                aggregated['metrics']['follow_through'] = {
                    'score': float(np.mean(follow_through_scores)),
                    'mean': float(np.mean(follow_through_scores)),
                    'std': float(np.std(follow_through_scores)),
                    'min': float(np.min(follow_through_scores)),
                    'max': float(np.max(follow_through_scores)),
                    'confidence_interval_95': [
                        float(np.mean(follow_through_scores) - 1.96 * np.std(follow_through_scores)),
                        float(np.mean(follow_through_scores) + 1.96 * np.std(follow_through_scores))
                    ],
                    'samples': len(follow_through_scores)
                }

        # Aggregate balance
        if any('balance' in m for m in all_metrics):
            balance_scores = []
            for m in all_metrics:
                if 'balance' in m and 'score' in m['balance']:
                    balance_scores.append(m['balance']['score'])

            if balance_scores:
                aggregated['metrics']['balance'] = {
                    'score': float(np.mean(balance_scores)),
                    'mean': float(np.mean(balance_scores)),
                    'std': float(np.std(balance_scores)),
                    'min': float(np.min(balance_scores)),
                    'max': float(np.max(balance_scores)),
                    'confidence_interval_95': [
                        float(np.mean(balance_scores) - 1.96 * np.std(balance_scores)),
                        float(np.mean(balance_scores) + 1.96 * np.std(balance_scores))
                    ],
                    'samples': len(balance_scores)
                }

        # Aggregate knee bend
        if any('knee_bend' in m for m in all_metrics):
            knee_angles = []
            for m in all_metrics:
                if 'knee_bend' in m and 'angle' in m['knee_bend']:
                    knee_angles.append(m['knee_bend']['angle'])

            if knee_angles:
                aggregated['metrics']['knee_bend'] = {
                    'angle': float(np.mean(knee_angles)),
                    'mean': float(np.mean(knee_angles)),
                    'std': float(np.std(knee_angles)),
                    'min': float(np.min(knee_angles)),
                    'max': float(np.max(knee_angles)),
                    'confidence_interval_95': [
                        float(np.mean(knee_angles) - 1.96 * np.std(knee_angles)),
                        float(np.mean(knee_angles) + 1.96 * np.std(knee_angles))
                    ],
                    'samples': len(knee_angles)
                }

        # Add per-angle breakdown
        aggregated['metrics_by_angle'] = {}
        for angle in ['front', 'side', 'back']:
            angle_results = [r for r in results if r and r.get('angle') == angle]
            if angle_results:
                aggregated['metrics_by_angle'][angle] = {
                    'videos_count': len(angle_results),
                    'video_names': [r['video_name'] for r in angle_results]
                }

        return aggregated

    def create_enhanced_baseline(self) -> Dict[str, Any]:
        """Process all videos and create enhanced baseline"""

        print("=" * 80)
        print("🏀 CREATING ENHANCED STEPH CURRY BASELINE")
        print("=" * 80)
        print()

        # Categorize videos
        print("📂 Step 1: Categorizing videos by camera angle...")
        categories = self.categorize_videos()

        total_videos = sum(len(videos) for videos in categories.values() if videos)
        print(f"   Found {total_videos} videos:")
        for angle, videos in categories.items():
            if videos:
                print(f"   • {angle.upper()}: {len(videos)} videos")
        print()

        # Process all videos
        print("🔍 Step 2: Processing all videos...")
        all_results = []

        for angle, videos in categories.items():
            if not videos:
                continue

            print(f"\n   Processing {angle.upper()} angle videos:")
            for video in videos:
                result = self.process_single_video(video, angle)
                if result:
                    all_results.append(result)
                    print(f"      ✓ {Path(video).name} - Quality: {result['quality']['confidence']:.1%}")
                else:
                    print(f"      ✗ {Path(video).name} - Failed")

        print(f"\n   Successfully processed: {len(all_results)}/{total_videos} videos")

        if not all_results:
            raise ValueError("No videos could be processed successfully")

        # Aggregate metrics
        print("\n📊 Step 3: Aggregating metrics across all angles...")
        aggregated_baseline = self.aggregate_metrics(all_results)

        # Add individual video details
        aggregated_baseline['individual_videos'] = [
            {
                'name': r['video_name'],
                'angle': r['angle'],
                'quality': r['quality']['confidence'],
                'metrics_summary': {
                    'release_angle': r['metrics'].get('release_angle', {}).get('angle'),
                    'elbow_alignment': r['metrics'].get('elbow_alignment', {}).get('score'),
                    'follow_through': r['metrics'].get('follow_through', {}).get('score'),
                    'balance': r['metrics'].get('balance', {}).get('score')
                }
            }
            for r in all_results
        ]

        # Save baseline
        output_path = self.baselines_dir / "stephen_curry_enhanced.json"
        with open(output_path, 'w') as f:
            json.dump(aggregated_baseline, f, indent=2)

        print(f"\n✅ Enhanced baseline saved to: {output_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("📊 ENHANCED BASELINE SUMMARY")
        print("=" * 80)
        print(f"\nPlayer: {aggregated_baseline['player_name']}")
        print(f"Total Videos Analyzed: {aggregated_baseline['videos_analyzed']}")
        print(f"Camera Angles: {', '.join(aggregated_baseline['angles_covered'])}")

        print("\n🎯 KEY METRICS (Mean ± Std Dev):")
        print("-" * 80)

        metrics = aggregated_baseline['metrics']
        if 'release_angle' in metrics:
            m = metrics['release_angle']
            print(f"Release Angle: {m['mean']:.1f}° ± {m['std']:.1f}° (range: {m['min']:.1f}°-{m['max']:.1f}°)")

        if 'elbow_alignment' in metrics:
            m = metrics['elbow_alignment']
            print(f"Elbow Alignment: {m['mean']:.1%} ± {m['std']:.1%} (range: {m['min']:.1%}-{m['max']:.1%})")

        if 'follow_through' in metrics:
            m = metrics['follow_through']
            print(f"Follow Through: {m['mean']:.1%} ± {m['std']:.1%} (range: {m['min']:.1%}-{m['max']:.1%})")

        if 'balance' in metrics:
            m = metrics['balance']
            print(f"Balance Score: {m['mean']:.1%} ± {m['std']:.1%} (range: {m['min']:.1%}-{m['max']:.1%})")

        if 'knee_bend' in metrics:
            m = metrics['knee_bend']
            print(f"Knee Bend: {m['mean']:.1f}° ± {m['std']:.1f}° (range: {m['min']:.1f}°-{m['max']:.1f}°)")

        print("\n" + "=" * 80)
        print("🎉 COMPLETE! Enhanced baseline ready for comparison analysis.")
        print("=" * 80)
        print()

        return aggregated_baseline


def main():
    """Main function"""
    creator = MultiAngleBaselineCreator()

    try:
        baseline = creator.create_enhanced_baseline()

        print("\n💡 NEXT STEPS:")
        print("   1. Use 'stephen_curry_enhanced.json' for more accurate comparisons")
        print("   2. The baseline now includes statistical confidence intervals")
        print("   3. Comparison analysis will account for natural variation in form")
        print()

    except Exception as e:
        logger.error(f"❌ Failed to create baseline: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

