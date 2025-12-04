"""
Shot Form Analysis Service
Analyzes basketball shooting form and provides actionable feedback for improvement
"""
import logging
from typing import Dict, List, Any, Optional

from .pose_processor import PoseProcessor
from .phase_detector import PhaseDetector
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class ShotAnalysisService:
    """Complete shot analysis pipeline with detailed coaching feedback"""

    def __init__(self, baselines_dir: str = None):
        """
        Initialize the shot analysis service

        Args:
            baselines_dir: Deprecated, kept for backwards compatibility
        """
        self.pose_processor = PoseProcessor(model_complexity=1)
        self.phase_detector = PhaseDetector()
        self.metrics_calculator = MetricsCalculator()

        logger.info("✅ ShotAnalysisService initialized for form analysis")

    def analyze_shot(
        self,
        video_path: str,
        frame_skip: int = 1
    ) -> Dict[str, Any]:
        """
        Complete shot analysis pipeline

        Analyzes user's shooting form against optimal biomechanical ranges
        and provides actionable coaching feedback.

        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame (1 = all frames)

        Returns:
            Comprehensive analysis results with metrics and coaching cues
        """
        logger.info(f"🏀 Starting shot form analysis for: {video_path}")

        try:
            # Step 1: Extract pose keypoints from video
            pose_data = self.pose_processor.process_video(video_path, frame_skip=frame_skip)

            if pose_data['quality']['confidence'] < 0.5:
                return {
                    'success': False,
                    'error': 'Low video quality or poor visibility of shooter',
                    'warning': pose_data['quality'].get('warning'),
                    'confidence': pose_data['quality']['confidence'],
                    'tips': [
                        'Ensure good lighting in your recording area',
                        'Record from the side angle for best results',
                        'Make sure your full body is visible in the frame',
                        'Avoid baggy clothing that obscures body position'
                    ]
                }

            # Step 2: Detect shooting phases (dip, load, release, follow-through)
            phases = self.phase_detector.detect_phases(pose_data['keypoints_sequence'])

            if not phases.get('valid', False):
                return {
                    'success': False,
                    'error': 'Could not detect complete shooting motion',
                    'phases': phases,
                    'confidence': phases.get('confidence', 0.0),
                    'tips': [
                        'Make sure to record a complete shot from start to finish',
                        'Include the catch/dip, load, release, and follow-through',
                        'Try recording at a slower pace if motion blur is an issue'
                    ]
                }

            # Step 3: Calculate biomechanical metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                pose_data['keypoints_sequence'],
                phases
            )

            if 'error' in metrics:
                return {
                    'success': False,
                    'error': metrics['error'],
                    'confidence': metrics.get('confidence', 0.0)
                }

            # Step 4: Generate detailed coaching feedback
            coaching_cues = self._generate_coaching_cues(metrics)

            # Step 5: Generate improvement summary
            improvement_summary = self._generate_improvement_summary(metrics)

            # Compile final results
            results = {
                'success': True,
                'analysis_mode': 'form_analysis',
                'overall_score': metrics['overall_score'],
                'overall_grade': self._get_grade(metrics['overall_score']),
                'confidence': min(pose_data['quality']['confidence'], phases['confidence']),
                'phases': {
                    'dip_start': phases['dip_start'],
                    'load': phases['load'],
                    'release': phases['release'],
                    'follow_through_end': phases['follow_through_end'],
                    'timing': phases.get('timing', {})
                },
                'metrics': metrics,
                'coaching_cues': coaching_cues,
                'improvement_summary': improvement_summary,
                'quality_info': pose_data['quality'],
                'metadata': pose_data['metadata']
            }

            logger.info(f"✅ Analysis complete! Score: {metrics['overall_score']:.1f}/100")
            return results

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }

    def _generate_coaching_cues(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate detailed coaching cues based on metric analysis

        Returns prioritized list of actionable coaching feedback with:
        - Clear cue to focus on
        - Why it matters
        - Specific drill to practice
        - Common mistakes to avoid
        """
        cues = []

        # Define comprehensive coaching rules
        rules = self._get_coaching_rules()

        # Evaluate each rule against metrics
        for rule in rules:
            metric_name = rule['metric']
            if metric_name not in metrics:
                continue

            metric_data = metrics[metric_name]
            if not isinstance(metric_data, dict) or 'quality_score' not in metric_data:
                continue

            # Check if metric needs improvement
            if self._evaluate_condition(metric_data, rule['condition']):
                # Calculate priority based on how far from optimal and metric importance
                priority = (10.0 - metric_data.get('quality_score', 5.0)) * rule['weight']

                cues.append({
                    'cue': rule['cue'],
                    'why': rule['why'],
                    'drill': rule['drill'],
                    'drill_description': rule['drill_description'],
                    'common_mistakes': rule['common_mistakes'],
                    'visual_cue': rule['visual_cue'],
                    'priority': priority,
                    'metric': metric_name,
                    'current_value': self._get_metric_value(metric_data),
                    'optimal_range': metric_data.get('optimal_range', [])
                })

        # Sort by priority and return top 3
        cues.sort(key=lambda x: x['priority'], reverse=True)

        # Clean up response (remove internal priority field)
        for cue in cues[:3]:
            del cue['priority']

        return cues[:3]

    def _get_coaching_rules(self) -> List[Dict[str, Any]]:
        """Define comprehensive coaching rules for each metric"""
        return [
            {
                'metric': 'elbow_flare',
                'condition': 'angle_deg > 12',
                'weight': 0.25,
                'cue': 'Keep your elbow tucked under the ball',
                'why': 'Elbow flare causes the ball to spin sideways, leading to inconsistent left/right misses. A tucked elbow creates backspin for a softer touch.',
                'drill': 'Wall Elbow Slides',
                'drill_description': 'Stand sideways against a wall with your shooting elbow touching it. Practice your shooting motion keeping elbow contact with the wall throughout. Do 3 sets of 20 reps.',
                'common_mistakes': [
                    'Chicken wing - elbow pointing out to the side',
                    'Starting with ball on the wrong side of your head',
                    'Gripping the ball too wide'
                ],
                'visual_cue': 'From behind, your elbow should form a straight line with your wrist and the basket'
            },
            {
                'metric': 'release_angle',
                'condition': 'angle_deg < 55 or angle_deg > 62',
                'weight': 0.25,
                'cue': 'Adjust your release angle for optimal arc',
                'why': 'A 55-62 degree release angle creates the ideal arc for the ball to enter the basket at an angle that maximizes the effective target area.',
                'drill': 'Arc Training',
                'drill_description': 'Use a visual target about 3 feet above the rim (like tape on a wall or backboard). Practice shooting over this target from close range, then gradually move back. Focus on consistent high arc.',
                'common_mistakes': [
                    'Flat shot - pushing the ball forward instead of up',
                    'Releasing too early in the motion',
                    'Not extending fully through the shot'
                ],
                'visual_cue': 'The ball should reach its peak height about 2/3 of the way to the basket'
            },
            {
                'metric': 'knee_load',
                'condition': 'angle_deg < 70 or angle_deg > 90',
                'weight': 0.20,
                'cue': 'Bend your knees to 70-90 degrees at your load point',
                'why': 'Proper knee bend stores elastic energy that transfers up through your body into the shot. Too little bend means arm-only shooting; too much wastes energy.',
                'drill': 'Chair Shooting',
                'drill_description': 'Place a chair behind you. Practice sitting back until you lightly touch the chair (this is your load point), then shoot. This builds muscle memory for proper depth.',
                'common_mistakes': [
                    'Barely bending knees (shooting with arms only)',
                    'Squatting too deep and losing balance',
                    'Bending at the waist instead of the knees'
                ],
                'visual_cue': 'At your lowest point, your thighs should be roughly parallel to the ground'
            },
            {
                'metric': 'hip_shoulder_alignment',
                'condition': 'angle_deg > 15',
                'weight': 0.15,
                'cue': 'Square your shoulders and hips to the basket',
                'why': 'Misaligned shoulders and hips create rotation in your shot, making it inconsistent. Proper alignment creates a straight power line to the basket.',
                'drill': 'Line Shooting',
                'drill_description': 'Stand on a court line pointing at the basket. Align both feet, hips, and shoulders parallel to this line. Shoot while maintaining this alignment. Start close and move back.',
                'common_mistakes': [
                    'Opening up hips too early',
                    'Rotating shoulders during the shot',
                    'Inconsistent foot placement'
                ],
                'visual_cue': 'Both shoulders should face the basket throughout your shot'
            },
            {
                'metric': 'lateral_sway',
                'condition': 'sway_ratio > 0.05',
                'weight': 0.15,
                'cue': 'Shoot straight up and down with minimal drift',
                'why': 'Lateral movement during your shot adds variables that make it harder to be consistent. The best shooters go straight up and land in nearly the same spot.',
                'drill': 'Spot Landing',
                'drill_description': 'Mark your starting foot position with tape. Shoot and focus on landing within inches of your starting spot. Any significant drift means you\'re adding lateral movement.',
                'common_mistakes': [
                    'Fading away unnecessarily',
                    'Jumping forward into the shot',
                    'Drifting to your shooting side'
                ],
                'visual_cue': 'You should land within 6 inches of where you took off'
            },
            {
                'metric': 'base_width',
                'condition': 'ratio < 0.15 or ratio > 0.25',
                'weight': 0.10,
                'cue': 'Set your feet shoulder-width apart for a stable base',
                'why': 'Your stance is your foundation. Too narrow and you lack stability; too wide and you can\'t generate power efficiently through your legs.',
                'drill': 'Stance Check',
                'drill_description': 'Before each shot in practice, pause and check: are your feet shoulder-width apart? Shooting foot slightly forward? Weight balanced? Make this a pre-shot habit.',
                'common_mistakes': [
                    'Feet too close together (unstable)',
                    'Feet too wide (can\'t generate power)',
                    'Weight on heels instead of balls of feet'
                ],
                'visual_cue': 'Your feet should be roughly under your shoulders, with shooting foot slightly forward'
            },
            {
                'metric': 'arc_trajectory',
                'condition': 'arc_angle_deg < 45 or arc_angle_deg > 55',
                'weight': 0.10,
                'cue': 'Focus on a 45-55 degree ball trajectory',
                'why': 'The ball\'s flight path determines how much of the rim is "open" for the ball to go through. A 45-55 degree entry angle maximizes your margin for error.',
                'drill': 'Swish Shooting',
                'drill_description': 'From close range, focus on making nothing-but-net shots. A swish requires the right arc. Count how many swishes you can make in a row.',
                'common_mistakes': [
                    'Line drive shots that hit the back of the rim',
                    'Rainbow shots that are hard to control',
                    'Inconsistent arc from shot to shot'
                ],
                'visual_cue': 'The ball should drop into the basket from above, not fly in horizontally'
            }
        ]

    def _evaluate_condition(self, metric_data: Dict[str, Any], condition: str) -> bool:
        """Evaluate a condition string against metric data"""
        try:
            if 'or' in condition:
                parts = condition.split('or')
                return any(self._evaluate_condition(metric_data, part.strip()) for part in parts)

            if '>' in condition:
                var, threshold = condition.split('>')
                var = var.strip()
                threshold = float(threshold.strip())
                return metric_data.get(var, 0) > threshold

            if '<' in condition:
                var, threshold = condition.split('<')
                var = var.strip()
                threshold = float(threshold.strip())
                return metric_data.get(var, 0) < threshold

            return False

        except Exception as e:
            logger.warning(f"⚠️ Failed to evaluate condition '{condition}': {e}")
            return False

    def _get_metric_value(self, metric_data: Dict[str, Any]) -> str:
        """Extract the primary value from a metric for display"""
        if 'angle_deg' in metric_data:
            return f"{metric_data['angle_deg']}°"
        elif 'ratio' in metric_data:
            return f"{metric_data['ratio']:.3f}"
        elif 'sway_ratio' in metric_data:
            return f"{metric_data['sway_ratio']:.3f}"
        elif 'arc_angle_deg' in metric_data:
            return f"{metric_data['arc_angle_deg']}°"
        return "N/A"

    def _generate_improvement_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall improvement summary"""
        overall_score = metrics.get('overall_score', 0)

        # Categorize metrics by performance
        strengths = []
        needs_work = []

        metric_names = {
            'release_angle': 'Release Angle',
            'elbow_flare': 'Elbow Alignment',
            'knee_load': 'Knee Bend',
            'hip_shoulder_alignment': 'Body Alignment',
            'base_width': 'Stance Width',
            'lateral_sway': 'Balance/Stability',
            'arc_trajectory': 'Shot Arc'
        }

        for metric_key, display_name in metric_names.items():
            if metric_key in metrics and isinstance(metrics[metric_key], dict):
                score = metrics[metric_key].get('quality_score', 0)
                if score >= 7.5:
                    strengths.append(display_name)
                elif score < 6.0:
                    needs_work.append(display_name)

        # Generate summary message
        if overall_score >= 85:
            summary_message = "Excellent shooting form! Focus on consistency and minor refinements."
        elif overall_score >= 70:
            summary_message = "Good foundation! Work on the specific areas below to take your shot to the next level."
        elif overall_score >= 55:
            summary_message = "Solid basics with room for improvement. Focus on one area at a time."
        else:
            summary_message = "Let's build a stronger foundation. Start with the top recommendation and master it before moving on."

        return {
            'message': summary_message,
            'strengths': strengths,
            'areas_to_improve': needs_work,
            'next_step': needs_work[0] if needs_work else 'Maintain your form with consistent practice'
        }

    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
