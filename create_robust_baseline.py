"""
Robust Multi-Angle Baseline Creator
Extracts metrics from all videos even without perfect phase detection
Uses statistical aggregation across multiple camera angles
"""
import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RobustBaselineCreator:
    """Extract shooting form metrics from multiple angles without strict phase detection"""

    def __init__(self, baselines_dir: str = "baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.pose_processor = PoseProcessor(model_complexity=1)

    def categorize_videos(self) -> Dict[str, List[str]]:
        """Categorize videos by camera angle"""
        video_files = list(self.baselines_dir.glob("*.mp4"))

        categories = {
            'front': [],
            'side': [],
            'back': [],
            'mixed': []
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
                categories['mixed'].append(str(video))

        return categories

    def extract_metrics_from_pose_data(self, keypoints_seq: List[Dict]) -> Dict[str, Any]:
        """Extract shooting metrics directly from pose keypoints"""

        metrics = {
            'elbow_angles': [],
            'wrist_heights': [],
            'knee_angles': [],
            'shoulder_alignments': [],
            'hip_alignments': []
        }

        for frame in keypoints_seq:
            # Skip frames with low visibility
            if not frame.get('visible', True):
                continue

            # Elbow angle (shooting arm)
            if all(k in frame for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                shoulder = frame['right_shoulder']
                elbow = frame['right_elbow']
                wrist = frame['right_wrist']

                if all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [shoulder, elbow, wrist]):
                    v1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
                    v2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])

                    angle = np.degrees(np.arccos(np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                        -1.0, 1.0
                    )))
                    metrics['elbow_angles'].append(angle)

            # Wrist height (release indicator)
            if 'right_wrist' in frame:
                wrist = frame['right_wrist']
                if isinstance(wrist, dict) and wrist.get('visibility', 0) > 0.5:
                    metrics['wrist_heights'].append(1 - wrist['y'])  # Normalized height

            # Knee angle (power generation)
            if all(k in frame for k in ['right_hip', 'right_knee', 'right_ankle']):
                hip = frame['right_hip']
                knee = frame['right_knee']
                ankle = frame['right_ankle']

                if all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [hip, knee, ankle]):
                    v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
                    v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])

                    angle = np.degrees(np.arccos(np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                        -1.0, 1.0
                    )))
                    metrics['knee_angles'].append(angle)

            # Shoulder alignment
            if all(k in frame for k in ['left_shoulder', 'right_shoulder']):
                left = frame['left_shoulder']
                right = frame['right_shoulder']

                if all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [left, right]):
                    shoulder_level = abs(left['y'] - right['y'])
                    metrics['shoulder_alignments'].append(shoulder_level)

            # Hip alignment
            if all(k in frame for k in ['left_hip', 'right_hip']):
                left = frame['left_hip']
                right = frame['right_hip']

                if all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [left, right]):
                    hip_level = abs(left['y'] - right['y'])
                    metrics['hip_alignments'].append(hip_level)

        return metrics

    def calculate_shooting_metrics(self, raw_metrics: Dict) -> Dict[str, Any]:
        """Calculate shooting form statistics from raw metrics"""

        result = {}

        # Elbow angle at release (use highest wrist position frames)
        if raw_metrics['elbow_angles'] and raw_metrics['wrist_heights']:
            # Find frames with highest wrist (likely release point)
            wrist_heights = np.array(raw_metrics['wrist_heights'])
            elbow_angles = np.array(raw_metrics['elbow_angles'][:len(wrist_heights)])

            # Get top 10% highest wrist positions
            top_percentile = np.percentile(wrist_heights, 90)
            release_frames = wrist_heights >= top_percentile

            if np.any(release_frames):
                release_angles = elbow_angles[:len(release_frames)][release_frames]
                result['release_angle'] = {
                    'angle': float(np.mean(release_angles)),
                    'mean': float(np.mean(release_angles)),
                    'std': float(np.std(release_angles)),
                    'min': float(np.min(release_angles)),
                    'max': float(np.max(release_angles)),
                    'samples': int(len(release_angles))
                }

        # Elbow alignment throughout shot
        if raw_metrics['elbow_angles']:
            angles = np.array(raw_metrics['elbow_angles'])
            result['elbow_alignment'] = {
                'score': float(1.0 - (np.std(angles) / 180.0)),  # Consistency score
                'mean_angle': float(np.mean(angles)),
                'std': float(np.std(angles)),
                'consistency': float(1.0 - (np.std(angles) / np.mean(angles))) if np.mean(angles) > 0 else 0.0
            }

        # Follow through (wrist extension)
        if raw_metrics['wrist_heights']:
            heights = np.array(raw_metrics['wrist_heights'])
            max_height = np.max(heights)
            result['follow_through'] = {
                'score': float(max_height),
                'max_extension': float(max_height),
                'mean_height': float(np.mean(heights)),
                'range': float(np.max(heights) - np.min(heights))
            }

        # Balance (shoulder and hip alignment)
        if raw_metrics['shoulder_alignments'] and raw_metrics['hip_alignments']:
            shoulder_align = np.array(raw_metrics['shoulder_alignments'])
            hip_align = np.array(raw_metrics['hip_alignments'])

            # Lower values = better alignment
            balance_score = 1.0 - (np.mean(shoulder_align) + np.mean(hip_align)) / 2
            result['balance'] = {
                'score': float(max(0.0, min(1.0, balance_score))),
                'shoulder_alignment': float(np.mean(shoulder_align)),
                'hip_alignment': float(np.mean(hip_align))
            }

        # Knee bend (power generation)
        if raw_metrics['knee_angles']:
            angles = np.array(raw_metrics['knee_angles'])
            # Find minimum angle (deepest bend)
            min_angle = np.min(angles)
            result['knee_bend'] = {
                'angle': float(min_angle),
                'min': float(min_angle),
                'mean': float(np.mean(angles)),
                'max': float(np.max(angles)),
                'range': float(np.max(angles) - np.min(angles))
            }

        return result

    def process_video(self, video_path: str, angle: str) -> Optional[Dict]:
        """Process a single video and extract metrics"""
        video_name = Path(video_path).name
        logger.info(f"📹 Processing {video_name} ({angle})...")

        try:
            # Extract pose data
            pose_data = self.pose_processor.process_video(video_path, frame_skip=1)

            quality = pose_data['quality']['confidence']
            logger.info(f"   Quality: {quality:.1%}")

            if quality < 0.4:
                logger.warning(f"   ⚠️ Skipping - quality too low")
                return None

            # Extract metrics from pose data
            raw_metrics = self.extract_metrics_from_pose_data(pose_data['keypoints_sequence'])

            # Calculate shooting metrics
            shooting_metrics = self.calculate_shooting_metrics(raw_metrics)

            if not shooting_metrics:
                logger.warning(f"   ⚠️ No metrics extracted")
                return None

            logger.info(f"   ✓ Extracted {len(shooting_metrics)} metric categories")

            return {
                'video_name': video_name,
                'angle': angle,
                'quality': quality,
                'frames': pose_data['metadata']['processed_frames'],
                'metrics': shooting_metrics,
                'success': True
            }

        except Exception as e:
            logger.error(f"   ❌ Error: {str(e)}")
            return None

    def aggregate_baseline(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics from all videos"""

        if not all_results:
            raise ValueError("No results to aggregate")

        baseline = {
            'player_name': 'Stephen Curry',
            'created_date': '2025-10-11',
            'total_videos': len(all_results),
            'angles_covered': sorted(list(set(r['angle'] for r in all_results))),
            'total_frames_analyzed': sum(r['frames'] for r in all_results),
            'average_quality': float(np.mean([r['quality'] for r in all_results])),
            'metrics': {}
        }

        # Aggregate each metric type
        metric_names = set()
        for r in all_results:
            metric_names.update(r['metrics'].keys())

        for metric_name in metric_names:
            values = []

            # Collect values from all videos that have this metric
            for result in all_results:
                if metric_name in result['metrics']:
                    metric = result['metrics'][metric_name]
                    # Use 'angle' for angles, 'score' for scores
                    if 'angle' in metric:
                        values.append(metric['angle'])
                    elif 'score' in metric:
                        values.append(metric['score'])
                    elif 'mean_angle' in metric:
                        values.append(metric['mean_angle'])

            if values:
                values = np.array(values)
                baseline['metrics'][metric_name] = {
                    'value': float(np.mean(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'confidence_interval_95': [
                        float(np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values))),
                        float(np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values)))
                    ],
                    'samples': len(values),
                    'consistency_score': float(1.0 - (np.std(values) / np.mean(values))) if np.mean(values) > 0 else 0.0
                }

        # Add per-angle breakdown
        baseline['by_angle'] = {}
        for angle in baseline['angles_covered']:
            angle_results = [r for r in all_results if r['angle'] == angle]
            baseline['by_angle'][angle] = {
                'video_count': len(angle_results),
                'videos': [r['video_name'] for r in angle_results],
                'average_quality': float(np.mean([r['quality'] for r in angle_results]))
            }

        # Add individual video details
        baseline['individual_videos'] = [
            {
                'name': r['video_name'],
                'angle': r['angle'],
                'quality': r['quality'],
                'frames': r['frames']
            }
            for r in all_results
        ]

        return baseline

    def create_baseline(self):
        """Main function to create enhanced baseline"""

        print("=" * 80)
        print("🏀 CREATING ROBUST MULTI-ANGLE BASELINE FOR STEPH CURRY")
        print("=" * 80)
        print()

        # Categorize videos
        categories = self.categorize_videos()
        total_videos = sum(len(v) for v in categories.values())

        print(f"📂 Found {total_videos} videos:")
        for angle, videos in categories.items():
            if videos:
                print(f"   • {angle.upper()}: {len(videos)} videos")
        print()

        # Process all videos
        print("🔍 Processing all videos...\n")
        all_results = []

        for angle, videos in categories.items():
            if not videos:
                continue

            for video in videos:
                result = self.process_video(video, angle)
                if result:
                    all_results.append(result)

        print(f"\n✅ Successfully processed: {len(all_results)}/{total_videos} videos\n")

        if not all_results:
            raise ValueError("No videos processed successfully")

        # Aggregate baseline
        print("📊 Aggregating metrics across all angles...\n")
        baseline = self.aggregate_baseline(all_results)

        # Save baseline
        output_path = self.baselines_dir / "stephen_curry_enhanced.json"
        with open(output_path, 'w') as f:
            json.dump(baseline, f, indent=2)

        print(f"💾 Saved enhanced baseline to: {output_path}\n")

        # Print summary
        print("=" * 80)
        print("📊 ENHANCED BASELINE SUMMARY")
        print("=" * 80)
        print(f"\nPlayer: {baseline['player_name']}")
        print(f"Videos Analyzed: {baseline['total_videos']}")
        print(f"Camera Angles: {', '.join(baseline['angles_covered'])}")
        print(f"Total Frames: {baseline['total_frames_analyzed']:,}")
        print(f"Average Quality: {baseline['average_quality']:.1%}")

        print("\n🎯 KEY METRICS (Mean ± 95% CI):")
        print("-" * 80)

        for metric_name, metric_data in baseline['metrics'].items():
            ci_low, ci_high = metric_data['confidence_interval_95']
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  Value: {metric_data['mean']:.3f}")
            print(f"  Range: [{metric_data['min']:.3f}, {metric_data['max']:.3f}]")
            print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"  Consistency: {metric_data['consistency_score']:.1%}")
            print(f"  Samples: {metric_data['samples']} videos")

        print("\n" + "=" * 80)
        print("🎉 COMPLETE! Enhanced baseline ready for comparison.")
        print("=" * 80)
        print()


def main():
    creator = RobustBaselineCreator()
    try:
        creator.create_baseline()
    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

