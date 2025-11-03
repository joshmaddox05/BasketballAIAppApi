"""
Enhanced Analysis with Fallbacks - Computes metrics even with partial data
"""
import json
import sys
from pathlib import Path
import logging
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor
from services.hand_processor import HandProcessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def analyze_with_fallbacks(video_path: str, enable_hands: bool = True):
    """Enhanced analysis that works with partial data"""

    print("=" * 80)
    print("🏀 ROBUST SHOT ANALYSIS (works with partial data)")
    print("=" * 80)
    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"👐 Hands tracking: {'ENABLED' if enable_hands else 'DISABLED'}\n")

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return

    # Initialize processors
    pose_processor = PoseProcessor(model_complexity=1)
    hand_processor = HandProcessor(max_num_hands=2) if enable_hands else None

    # Extract pose data
    print("📊 Step 1: Extracting pose data...")
    pose_data = pose_processor.process_video(video_path, frame_skip=1)

    metadata = pose_data['metadata']
    keypoints_seq = pose_data['keypoints_sequence']
    quality = pose_data['quality']

    print(f"   ✓ {metadata['processed_frames']} frames @ {metadata['fps']:.1f} fps")
    print(f"   ✓ Pose quality: {quality['confidence']:.1%}")

    # Extract hand data if enabled
    hand_data_seq = None
    if enable_hands and hand_processor:
        print("\n👐 Step 2: Extracting hand data...")
        cap = cv2.VideoCapture(video_path)

        hand_data_seq = []
        hands_detected = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_data = hand_processor.process_frame(rgb_frame, frame_idx)
            hand_data_seq.append(hand_data)

            if hand_data.get('right') or hand_data.get('left'):
                hands_detected += 1

            frame_idx += 1

        cap.release()

        hands_coverage = hands_detected / len(hand_data_seq) if hand_data_seq else 0
        print(f"   ✓ Hands: {hands_detected}/{len(hand_data_seq)} frames ({hands_coverage:.1%} coverage)")

    # Compute metrics with what we have
    print("\n📐 Step 3: Computing available metrics...")
    metrics = compute_robust_metrics(keypoints_seq, hand_data_seq, metadata['fps'])

    # Display results
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\n🔧 Data availability:")
    print(f"   Pose: {'✓ Available' if metrics['data_availability']['pose'] else '✗ Missing'}")
    print(f"   Hands: {'✓ Available' if metrics['data_availability']['hands'] else '✗ Missing'}")
    print(f"   Head: {'✓ Available' if metrics['data_availability']['head'] else '✗ Missing'}")

    print(f"\n🎯 Computed Metrics ({metrics['metrics_computed']}/{metrics['metrics_possible']}):")

    for category, category_metrics in metrics['metrics'].items():
        if category_metrics:
            print(f"\n   {category.upper()}:")
            for metric_name, value in category_metrics.items():
                if value is not None:
                    print(f"      ✓ {metric_name}: {value:.1f}")
                else:
                    print(f"      ✗ {metric_name}: unavailable")

    # Save results
    output_file = Path("output") / f"{Path(video_path).stem}_robust_analysis.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n💾 Saved to: {output_file}")
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    return metrics


def compute_robust_metrics(keypoints_seq, hand_data_seq, fps):
    """Compute all available metrics based on what data we have"""

    metrics = {
        'data_availability': {
            'pose': len(keypoints_seq) > 0,
            'hands': hand_data_seq is not None and any(h.get('right') for h in hand_data_seq),
            'head': any('nose' in frame for frame in keypoints_seq)
        },
        'metrics': {
            'wrist': {},
            'head': {},
            'body': {}
        },
        'metrics_computed': 0,
        'metrics_possible': 7
    }

    # Find release frame (highest wrist point)
    release_frame = find_release_frame(keypoints_seq)
    load_frame = max(0, release_frame - 12) if release_frame else len(keypoints_seq) // 3

    # 1. Wrist metrics (from pose)
    if release_frame:
        wrist_metrics = compute_wrist_metrics_simple(keypoints_seq, release_frame, load_frame, fps)
        metrics['metrics']['wrist'] = wrist_metrics
        metrics['metrics_computed'] += len([v for v in wrist_metrics.values() if v is not None])

    # 2. Head stability metrics
    head_metrics = compute_head_metrics_simple(keypoints_seq, load_frame, release_frame, fps)
    metrics['metrics']['head'] = head_metrics
    metrics['metrics_computed'] += len([v for v in head_metrics.values() if v is not None])

    # 3. Body alignment
    body_metrics = compute_body_metrics_simple(keypoints_seq)
    metrics['metrics']['body'] = body_metrics
    metrics['metrics_computed'] += len([v for v in body_metrics.values() if v is not None])

    return metrics


def find_release_frame(keypoints_seq):
    """Find frame with highest wrist position"""
    max_height = 0
    release_frame = None

    for i, frame in enumerate(keypoints_seq):
        if 'right_wrist' in frame and isinstance(frame['right_wrist'], dict):
            wrist = frame['right_wrist']
            if wrist.get('visibility', 0) > 0.5:
                height = 1 - wrist['y']
                if height > max_height:
                    max_height = height
                    release_frame = i

    return release_frame


def compute_wrist_metrics_simple(keypoints_seq, release_frame, load_frame, fps):
    """Compute basic wrist metrics from pose only"""
    metrics = {
        'release_height': None,
        'wrist_extension': None,
        'elbow_angle_at_release': None
    }

    if release_frame and release_frame < len(keypoints_seq):
        frame = keypoints_seq[release_frame]

        # Release height
        if 'right_wrist' in frame and isinstance(frame['right_wrist'], dict):
            wrist = frame['right_wrist']
            if wrist.get('visibility', 0) > 0.5:
                metrics['release_height'] = (1 - wrist['y']) * 100  # As percentage

        # Elbow angle at release
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
                metrics['elbow_angle_at_release'] = angle

    return metrics


def compute_head_metrics_simple(keypoints_seq, load_frame, release_frame, fps):
    """Compute basic head stability metrics"""
    metrics = {
        'head_tilt': None,
        'head_movement': None
    }

    if not release_frame:
        return metrics

    nose_positions = []

    for frame in keypoints_seq[load_frame:release_frame+1]:
        if 'nose' in frame and isinstance(frame['nose'], dict):
            nose = frame['nose']
            if nose.get('visibility', 0) > 0.5:
                nose_positions.append([nose['x'], nose['y']])

    if len(nose_positions) > 2:
        nose_positions = np.array(nose_positions)

        # Head movement (standard deviation)
        movement = np.std(nose_positions[:, 0]) * 100  # Lateral movement as percentage
        metrics['head_movement'] = movement

        # Head tilt from first frame
        first_frame = keypoints_seq[load_frame]
        if all(k in first_frame for k in ['left_eye', 'right_eye']):
            left_eye = first_frame['left_eye']
            right_eye = first_frame['right_eye']

            if all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [left_eye, right_eye]):
                eye_vec = np.array([right_eye['x'] - left_eye['x'], right_eye['y'] - left_eye['y']])
                tilt = abs(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
                metrics['head_tilt'] = tilt

    return metrics


def compute_body_metrics_simple(keypoints_seq):
    """Compute basic body alignment metrics"""
    metrics = {
        'shoulder_level': None,
        'hip_level': None
    }

    if not keypoints_seq:
        return metrics

    # Average across all frames
    shoulder_tilts = []
    hip_tilts = []

    for frame in keypoints_seq:
        # Shoulder level
        if all(k in frame for k in ['left_shoulder', 'right_shoulder']):
            left = frame['left_shoulder']
            right = frame['right_shoulder']

            if all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [left, right]):
                tilt = abs(left['y'] - right['y']) * 100
                shoulder_tilts.append(tilt)

        # Hip level
        if all(k in frame for k in ['left_hip', 'right_hip']):
            left = frame['left_hip']
            right = frame['right_hip']

            if all(isinstance(x, dict) and x.get('visibility', 0) > 0.5 for x in [left, right]):
                tilt = abs(left['y'] - right['y']) * 100
                hip_tilts.append(tilt)

    if shoulder_tilts:
        metrics['shoulder_level'] = np.mean(shoulder_tilts)

    if hip_tilts:
        metrics['hip_level'] = np.mean(hip_tilts)

    return metrics


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "tests/TestShot.mp4"
    enable_hands = True if len(sys.argv) < 3 else sys.argv[2].lower() != 'false'

    analyze_with_fallbacks(video, enable_hands)

