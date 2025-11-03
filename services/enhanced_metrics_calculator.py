"""
Enhanced Metrics Calculator - Integrates Head, Hands, and Pose
Computes comprehensive shooting mechanics including wrist flick, palm angles, and head stability
"""
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .wrist_mechanics_calculator import WristMechanicsCalculator
from .head_motion_analyzer import HeadMotionAnalyzer
from .hand_processor import HandProcessor

logger = logging.getLogger(__name__)


class EnhancedMetricsCalculator:
    """
    Enhanced shooting metrics with head and hand tracking
    Maintains backward compatibility with existing MetricsCalculator
    """

    def __init__(self):
        self.wrist_calculator = WristMechanicsCalculator()
        self.head_analyzer = HeadMotionAnalyzer()
        logger.info("✅ EnhancedMetricsCalculator initialized")

    def calculate_all_metrics(
        self,
        keypoints_sequence: List[Dict],
        phases: Dict[str, Any],
        fps: float = 30.0,
        hand_data_sequence: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive shooting metrics including new head & hand features

        Args:
            keypoints_sequence: Pose landmarks over time
            phases: Detected shooting phases
            fps: Video frame rate
            hand_data_sequence: Optional hand landmarks for enhanced analysis

        Returns:
            Dict with all metrics and modality info
        """
        if not phases.get('valid'):
            return {'error': 'Invalid shooting phases detected'}

        # Check data availability
        has_hands = hand_data_sequence is not None and len(hand_data_sequence) > 0
        hands_available_count = 0
        if has_hands:
            hands_available_count = sum(1 for h in hand_data_sequence if h.get('right') is not None)

        hands_coverage = hands_available_count / len(keypoints_sequence) if keypoints_sequence else 0

        # Initialize results
        metrics = {
            'modalities': {
                'pose': True,
                'hands': has_hands and hands_coverage > 0.5
            },
            'debug': {
                'hands_frames_available': hands_available_count,
                'hands_coverage_pct': float(hands_coverage * 100),
                'total_frames': len(keypoints_sequence)
            }
        }

        try:
            # 1. Wrist Flick & Release Dynamics
            logger.info("📐 Computing wrist flick metrics...")
            wrist_metrics = self.wrist_calculator.compute_wrist_flick_velocity(
                keypoints_sequence,
                phases,
                fps,
                hand_data_sequence
            )

            if 'error' not in wrist_metrics:
                metrics['wrist_flick_peak_deg_s'] = wrist_metrics.get('peak_flick_deg_s', 0)
                metrics['wrist_followthrough_ms'] = wrist_metrics.get('followthrough_ms', 0)
                metrics['wrist_angle_range_deg'] = wrist_metrics.get('angle_range_deg', 0)
                metrics['wrist_details'] = wrist_metrics

            # 2. Palm Orientation (only if hands available)
            if has_hands and hands_coverage > 0.5:
                logger.info("📐 Computing palm orientation metrics...")
                palm_metrics = self.wrist_calculator.compute_palm_metrics(
                    hand_data_sequence,
                    phases
                )

                if 'error' not in palm_metrics and 'average' in palm_metrics:
                    metrics['palm_angle_to_vertical_deg'] = palm_metrics['average']['palm_angle_to_vertical_deg']
                    metrics['palm_toward_target_deg'] = palm_metrics['average']['palm_toward_target_deg']
                    metrics['palm_details'] = palm_metrics
            else:
                metrics['palm_angle_to_vertical_deg'] = None
                metrics['palm_toward_target_deg'] = None
                logger.info("⚠️  Palm metrics unavailable (no hand data)")

            # 3. Head Motion & Stability
            logger.info("📐 Computing head stability metrics...")
            head_metrics = self.head_analyzer.analyze_head_motion_sequence(
                keypoints_sequence,
                phases,
                fps
            )

            if 'error' not in head_metrics and 'summary' in head_metrics:
                summary = head_metrics['summary']
                metrics['head_tilt_deg'] = summary['head_tilt_deg']
                metrics['head_yaw_jitter_deg_s'] = summary['head_yaw_jitter_deg_s']
                metrics['gaze_stability_cm'] = summary['gaze_stability_cm']

                if 'stability_score' in head_metrics:
                    metrics['head_stability_score'] = head_metrics['stability_score']

                metrics['head_details'] = head_metrics

            # 4. Generate Coaching Cues
            metrics['coaching_cues'] = self._generate_coaching_cues(metrics)

            # 5. Overall Assessment
            metrics['overall_grade'] = self._compute_overall_grade(metrics)

            logger.info(f"✅ Enhanced metrics calculated successfully")
            return metrics

        except Exception as e:
            logger.error(f"❌ Error calculating enhanced metrics: {str(e)}")
            return {
                'error': str(e),
                'modalities': metrics['modalities'],
                'debug': metrics['debug']
            }

    def _generate_coaching_cues(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate actionable coaching cues based on metrics

        Returns list of cues with: issue, cue, drill, priority
        """
        cues = []

        # Wrist flick cues
        wrist_flick = metrics.get('wrist_flick_peak_deg_s')
        if wrist_flick is not None:
            if wrist_flick < 600:
                cues.append({
                    'issue': 'weak_wrist_flick',
                    'cue': 'Snap wrist through the ball.',
                    'why': 'Creates consistent backspin and arc.',
                    'drill': 'flick-wall-taps',
                    'priority': 'high',
                    'value': wrist_flick,
                    'target': '700-1100 deg/s'
                })
            elif wrist_flick > 1200:
                cues.append({
                    'issue': 'excessive_wrist_flick',
                    'cue': 'Smooth, controlled release - less wrist snap.',
                    'why': 'Excessive flick reduces accuracy.',
                    'drill': 'soft-touch-shooting',
                    'priority': 'medium',
                    'value': wrist_flick,
                    'target': '700-1100 deg/s'
                })

        # Follow-through cues
        followthrough = metrics.get('wrist_followthrough_ms')
        if followthrough is not None and followthrough < 300:
            cues.append({
                'issue': 'short_followthrough',
                'cue': 'Hold your follow-through longer.',
                'why': 'Ensures complete energy transfer to ball.',
                'drill': 'freeze-form-holds',
                'priority': 'medium',
                'value': followthrough,
                'target': '300-600 ms'
            })

        # Palm orientation cues
        palm_target = metrics.get('palm_toward_target_deg')
        if palm_target is not None and palm_target > 25:
            cues.append({
                'issue': 'palm_misalignment',
                'cue': 'Square your palm to the rim.',
                'why': 'Aligns force straight to target.',
                'drill': 'palm-square-reps',
                'priority': 'high',
                'value': palm_target,
                'target': '0-20 deg'
            })

        palm_vertical = metrics.get('palm_angle_to_vertical_deg')
        if palm_vertical is not None:
            if palm_vertical < 10 or palm_vertical > 30:
                cues.append({
                    'issue': 'palm_angle_extreme',
                    'cue': 'Adjust palm angle - avoid extreme tilt.',
                    'why': 'Optimal angle provides best arc and control.',
                    'drill': 'palm-angle-practice',
                    'priority': 'medium',
                    'value': palm_vertical,
                    'target': '10-30 deg'
                })

        # Head stability cues
        head_jitter = metrics.get('head_yaw_jitter_deg_s')
        gaze_stability = metrics.get('gaze_stability_cm')

        if head_jitter is not None and head_jitter > 50:
            cues.append({
                'issue': 'head_movement',
                'cue': 'Keep eyes still on target.',
                'why': 'Head stability improves accuracy.',
                'drill': 'focus-spot-holds',
                'priority': 'high',
                'value': head_jitter,
                'target': '<50 deg/s'
            })

        if gaze_stability is not None and gaze_stability > 3:
            cues.append({
                'issue': 'gaze_drift',
                'cue': 'Lock your eyes on the rim throughout.',
                'why': 'Reduces spatial inconsistency.',
                'drill': 'target-focus-drills',
                'priority': 'high',
                'value': gaze_stability,
                'target': '<3 cm'
            })

        head_tilt = metrics.get('head_tilt_deg')
        if head_tilt is not None and head_tilt > 8:
            cues.append({
                'issue': 'head_tilt',
                'cue': 'Keep your head level.',
                'why': 'Maintains balanced spatial awareness.',
                'drill': 'head-level-practice',
                'priority': 'low',
                'value': head_tilt,
                'target': '0-8 deg'
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        cues.sort(key=lambda x: priority_order.get(x['priority'], 3))

        return cues

    def _compute_overall_grade(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall shooting grade based on all metrics"""
        scores = []

        # Wrist flick score
        wrist_flick = metrics.get('wrist_flick_peak_deg_s')
        if wrist_flick is not None:
            if 700 <= wrist_flick <= 1100:
                scores.append(1.0)
            elif 550 <= wrist_flick < 700 or 1100 < wrist_flick <= 1300:
                scores.append(0.7)
            else:
                scores.append(0.4)

        # Follow-through score
        followthrough = metrics.get('wrist_followthrough_ms')
        if followthrough is not None:
            if 300 <= followthrough <= 600:
                scores.append(1.0)
            elif 200 <= followthrough < 300 or 600 < followthrough <= 800:
                scores.append(0.7)
            else:
                scores.append(0.4)

        # Palm alignment score
        palm_target = metrics.get('palm_toward_target_deg')
        if palm_target is not None:
            if palm_target <= 20:
                scores.append(1.0)
            elif palm_target <= 30:
                scores.append(0.7)
            else:
                scores.append(0.4)

        # Head stability score
        if 'head_stability_score' in metrics:
            scores.append(metrics['head_stability_score']['score'])

        # Compute overall
        if scores:
            overall_score = np.mean(scores)

            if overall_score >= 0.9:
                grade = 'A'
                description = 'Elite form'
            elif overall_score >= 0.75:
                grade = 'B'
                description = 'Good form with minor adjustments needed'
            elif overall_score >= 0.6:
                grade = 'C'
                description = 'Developing - focus on key areas'
            else:
                grade = 'D'
                description = 'Needs significant work'

            return {
                'score': float(overall_score),
                'grade': grade,
                'description': description,
                'components_evaluated': len(scores)
            }
        else:
            return {
                'score': 0.0,
                'grade': 'N/A',
                'description': 'Insufficient data',
                'components_evaluated': 0
            }

