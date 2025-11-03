"""
Data Coverage Visualizer - Shows exactly where pose/hand tracking is missing
Creates annotated video with visibility heatmap and missing data indicators
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from services.pose_processor import PoseProcessor
from services.hand_processor import HandProcessor

logging.basicConfig(level=logging.WARNING)


class DataCoverageVisualizer:
    """Visualize data availability and missing points"""

    COLORS = {
        'good': (0, 255, 0),      # Green - good visibility
        'medium': (0, 255, 255),  # Yellow - medium visibility
        'poor': (0, 165, 255),    # Orange - poor visibility
        'missing': (0, 0, 255),   # Red - missing
        'text': (255, 255, 255),  # White
        'bg': (0, 0, 0)           # Black
    }

    CRITICAL_POSE_LANDMARKS = [
        'nose', 'left_eye', 'right_eye',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]

    def __init__(self):
        self.pose_processor = PoseProcessor(model_complexity=1)
        self.hand_processor = HandProcessor(max_num_hands=2)

    def create_coverage_visualization(self, video_path: str, output_path: str):
        """Create video showing data coverage"""

        print("=" * 80)
        print("🔍 DATA COVERAGE ANALYSIS")
        print("=" * 80)
        print(f"\n📹 Input: {Path(video_path).name}")
        print(f"💾 Output: {output_path}\n")

        # Extract all data
        print("📊 Step 1: Analyzing pose coverage...")
        pose_data = self.pose_processor.process_video(video_path, frame_skip=1)
        keypoints_seq = pose_data['keypoints_sequence']
        metadata = pose_data['metadata']

        print(f"   ✓ {metadata['processed_frames']} frames analyzed")

        print("\n👐 Step 2: Analyzing hand coverage...")
        cap = cv2.VideoCapture(video_path)

        hand_data_seq = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_data = self.hand_processor.process_frame(rgb_frame, frame_idx)
            hand_data_seq.append(hand_data)
            frame_idx += 1

        cap.release()

        print(f"   ✓ {len(hand_data_seq)} frames analyzed")

        # Compute coverage statistics
        print("\n📈 Step 3: Computing coverage statistics...")
        coverage_stats = self._compute_coverage_stats(keypoints_seq, hand_data_seq)

        # Create visualization
        print("\n🎨 Step 4: Creating visualization...")
        self._create_annotated_video(
            video_path,
            output_path,
            keypoints_seq,
            hand_data_seq,
            coverage_stats,
            metadata
        )

        # Print summary
        print("\n" + "=" * 80)
        print("📊 COVERAGE SUMMARY")
        print("=" * 80)

        print(f"\n🎯 POSE LANDMARKS:")
        for landmark, stats in coverage_stats['pose'].items():
            pct = stats['coverage'] * 100
            status = "✓" if pct > 80 else "⚠️" if pct > 50 else "✗"
            print(f"   {status} {landmark:20s}: {pct:5.1f}% ({stats['frames']:3d}/{stats['total']:3d} frames)")

        print(f"\n👐 HAND TRACKING:")
        print(f"   Right hand: {coverage_stats['hands']['right']['coverage']*100:.1f}%")
        print(f"   Left hand:  {coverage_stats['hands']['left']['coverage']*100:.1f}%")

        print(f"\n🏆 OVERALL QUALITY:")
        print(f"   Pose:  {coverage_stats['overall']['pose']*100:.1f}%")
        print(f"   Hands: {coverage_stats['overall']['hands']*100:.1f}%")

        print(f"\n💾 Visualization saved to: {output_path}")
        print("=" * 80 + "\n")

    def _compute_coverage_stats(self, keypoints_seq, hand_data_seq):
        """Compute coverage statistics for all landmarks"""

        stats = {
            'pose': {},
            'hands': {
                'right': {'frames': 0, 'total': len(hand_data_seq), 'coverage': 0},
                'left': {'frames': 0, 'total': len(hand_data_seq), 'coverage': 0}
            },
            'overall': {}
        }

        # Pose landmark coverage
        for landmark in self.CRITICAL_POSE_LANDMARKS:
            visible_frames = 0
            for frame in keypoints_seq:
                if landmark in frame and isinstance(frame[landmark], dict):
                    if frame[landmark].get('visibility', 0) > 0.5:
                        visible_frames += 1

            stats['pose'][landmark] = {
                'frames': visible_frames,
                'total': len(keypoints_seq),
                'coverage': visible_frames / len(keypoints_seq) if keypoints_seq else 0
            }

        # Hand coverage
        for hand_data in hand_data_seq:
            if hand_data.get('right'):
                stats['hands']['right']['frames'] += 1
            if hand_data.get('left'):
                stats['hands']['left']['frames'] += 1

        stats['hands']['right']['coverage'] = stats['hands']['right']['frames'] / len(hand_data_seq) if hand_data_seq else 0
        stats['hands']['left']['coverage'] = stats['hands']['left']['frames'] / len(hand_data_seq) if hand_data_seq else 0

        # Overall scores
        pose_coverages = [s['coverage'] for s in stats['pose'].values()]
        stats['overall']['pose'] = np.mean(pose_coverages) if pose_coverages else 0
        stats['overall']['hands'] = (stats['hands']['right']['coverage'] + stats['hands']['left']['coverage']) / 2

        return stats

    def _create_annotated_video(self, video_path, output_path, keypoints_seq, hand_data_seq, stats, metadata):
        """Create annotated video showing coverage"""

        cap = cv2.VideoCapture(video_path)

        fps = metadata['fps']
        width = metadata['width']
        height = metadata['height']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_frames = len(keypoints_seq)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < len(keypoints_seq):
                keypoints = keypoints_seq[frame_idx]
                hand_data = hand_data_seq[frame_idx] if frame_idx < len(hand_data_seq) else None

                # Draw coverage indicators
                frame = self._draw_coverage_indicators(
                    frame, keypoints, hand_data, stats, frame_idx, total_frames, width, height
                )

            out.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"   Progress: {progress:.1f}%", end='\r')

        cap.release()
        out.release()
        print(f"   Progress: 100.0%")

    def _draw_coverage_indicators(self, frame, keypoints, hand_data, stats, frame_idx, total_frames, width, height):
        """Draw coverage indicators on frame"""

        # Draw pose landmarks with visibility coloring
        for landmark in self.CRITICAL_POSE_LANDMARKS:
            if landmark in keypoints and isinstance(keypoints[landmark], dict):
                lm = keypoints[landmark]
                vis = lm.get('visibility', 0)

                x = int(lm['x'] * width)
                y = int(lm['y'] * height)

                # Color based on visibility
                if vis > 0.8:
                    color = self.COLORS['good']
                elif vis > 0.5:
                    color = self.COLORS['medium']
                elif vis > 0.3:
                    color = self.COLORS['poor']
                else:
                    continue

                # Draw circle
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)
            else:
                # Mark missing with X
                overall_coverage = stats['pose'].get(landmark, {}).get('coverage', 0)
                if overall_coverage > 0.3:  # Only show if usually visible
                    # Estimate position from neighboring landmarks
                    pos = self._estimate_position(landmark, keypoints, width, height)
                    if pos:
                        x, y = pos
                        cv2.drawMarker(frame, (x, y), self.COLORS['missing'],
                                     cv2.MARKER_TILTED_CROSS, 12, 2)

        # Draw hand indicators
        hand_y = height - 150
        if hand_data:
            if hand_data.get('right'):
                cv2.circle(frame, (width - 100, hand_y), 20, self.COLORS['good'], -1)
                cv2.putText(frame, "R", (width - 110, hand_y + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 2)
            else:
                cv2.circle(frame, (width - 100, hand_y), 20, self.COLORS['missing'], 2)
                cv2.putText(frame, "R", (width - 110, hand_y + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['missing'], 2)

            if hand_data.get('left'):
                cv2.circle(frame, (width - 150, hand_y), 20, self.COLORS['good'], -1)
                cv2.putText(frame, "L", (width - 160, hand_y + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 2)
            else:
                cv2.circle(frame, (width - 150, hand_y), 20, self.COLORS['missing'], 2)
                cv2.putText(frame, "L", (width - 160, hand_y + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['missing'], 2)

        # Draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), self.COLORS['bg'], -1)
        cv2.rectangle(overlay, (0, height - 100), (width, height), self.COLORS['bg'], -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        cv2.putText(frame, "DATA COVERAGE ANALYSIS", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, self.COLORS['text'], 2)

        # Frame info
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)

        # Legend
        legend_x = 20
        legend_y = height - 60

        cv2.circle(frame, (legend_x, legend_y), 8, self.COLORS['good'], -1)
        cv2.putText(frame, "Good (>80%)", (legend_x + 15, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

        cv2.circle(frame, (legend_x + 150, legend_y), 8, self.COLORS['medium'], -1)
        cv2.putText(frame, "Medium (50-80%)", (legend_x + 165, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

        cv2.drawMarker(frame, (legend_x + 340, legend_y), self.COLORS['missing'],
                      cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(frame, "Missing", (legend_x + 355, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

        # Coverage scores
        pose_score = stats['overall']['pose']
        hands_score = stats['overall']['hands']

        cv2.putText(frame, f"Pose: {pose_score*100:.0f}%", (width - 180, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.COLORS['good'] if pose_score > 0.8 else self.COLORS['medium'], 2)

        cv2.putText(frame, f"Hands: {hands_score*100:.0f}%", (width - 180, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.COLORS['good'] if hands_score > 0.5 else self.COLORS['poor'], 2)

        return frame

    def _estimate_position(self, landmark, keypoints, width, height):
        """Estimate position of missing landmark from neighbors"""
        # Simple averaging of nearby landmarks
        estimates = {
            'right_wrist': ['right_elbow'],
            'right_elbow': ['right_shoulder', 'right_wrist'],
            'nose': ['left_eye', 'right_eye']
        }

        if landmark not in estimates:
            return None

        positions = []
        for neighbor in estimates[landmark]:
            if neighbor in keypoints and isinstance(keypoints[neighbor], dict):
                lm = keypoints[neighbor]
                if lm.get('visibility', 0) > 0.5:
                    positions.append([lm['x'] * width, lm['y'] * height])

        if positions:
            return tuple(map(int, np.mean(positions, axis=0)))

        return None


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "tests/TestShot.mp4"
    output = sys.argv[2] if len(sys.argv) > 2 else f"output/{Path(video).stem}_coverage.mp4"

    Path(output).parent.mkdir(exist_ok=True)

    visualizer = DataCoverageVisualizer()
    visualizer.create_coverage_visualization(video, output)


if __name__ == "__main__":
    main()

