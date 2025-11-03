"""
Comprehensive Shot Analysis Service
Integrates pose processing, phase detection, metrics calculation, and baseline comparison
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .pose_processor import PoseProcessor
from .phase_detector import PhaseDetector
from .metrics_calculator import MetricsCalculator
from .video_comparator import VideoComparator

logger = logging.getLogger(__name__)

class ShotAnalysisService:
    """Complete shot analysis pipeline with NBA baseline comparison"""
    
    def __init__(self, baselines_dir: str = "baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.pose_processor = PoseProcessor(model_complexity=1)
        self.phase_detector = PhaseDetector()
        self.metrics_calculator = MetricsCalculator()
        self.video_comparator = VideoComparator()
        
        # Load NBA baselines
        self.baselines = self._load_baselines()
        logger.info(f"✅ ShotAnalysisService initialized with {len(self.baselines)} baselines")
    
    def analyze_shot(
        self, 
        video_path: str,
        baseline_player: Optional[str] = "Stephen Curry",
        frame_skip: int = 1
    ) -> Dict[str, Any]:
        """
        Complete shot analysis pipeline
        
        Args:
            video_path: Path to video file
            baseline_player: NBA player to compare against
            frame_skip: Process every Nth frame
            
        Returns:
            Comprehensive analysis results with comparison
        """
        logger.info(f"🏀 Starting shot analysis for: {video_path}")
        logger.info(f"📊 Comparing to: {baseline_player}")
        
        try:
            # Step 1: Extract pose keypoints
            pose_data = self.pose_processor.process_video(video_path, frame_skip=frame_skip)
            
            if pose_data['quality']['confidence'] < 0.5:
                return {
                    'success': False,
                    'error': 'Low video quality or visibility',
                    'warning': pose_data['quality'].get('warning'),
                    'confidence': pose_data['quality']['confidence']
                }
            
            # Step 2: Detect shooting phases
            phases = self.phase_detector.detect_phases(pose_data['keypoints_sequence'])
            
            if not phases.get('valid', False):
                return {
                    'success': False,
                    'error': 'Could not detect shot phases',
                    'phases': phases,
                    'confidence': phases.get('confidence', 0.0)
                }
            
            # Step 3: Calculate metrics
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
            
            # Step 4: Compare to baseline
            baseline_data = self.baselines.get(baseline_player)
            if baseline_data:
                comparison = self._compare_to_baseline(metrics, baseline_data, baseline_player)
            else:
                comparison = {
                    'warning': f'Baseline not found for {baseline_player}',
                    'available_baselines': list(self.baselines.keys())
                }
            
            # Step 5: Generate coaching cues
            coaching_cues = self._generate_coaching_cues(metrics, comparison)
            
            # Compile final results
            results = {
                'success': True,
                'analysis_mode': 'comprehensive',
                'player': baseline_player,
                'overall_score': metrics['overall_score'],
                'confidence': min(pose_data['quality']['confidence'], phases['confidence']),
                'phases': {
                    'dip_start': phases['dip_start'],
                    'load': phases['load'],
                    'release': phases['release'],
                    'follow_through_end': phases['follow_through_end'],
                    'timing': phases.get('timing', {})
                },
                'metrics': metrics,
                'comparison': comparison,
                'coaching_cues': coaching_cues,
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
    
    def analyze_with_comparison_video(
        self,
        video_path: str,
        baseline_player: Optional[str] = "Stephen Curry",
        comparison_mode: str = 'split',
        generate_video: bool = True
    ) -> Dict[str, Any]:
        """
        Complete analysis with optional comparison video generation
        
        Args:
            video_path: Path to user's video
            baseline_player: NBA player to compare against
            comparison_mode: 'split', 'overlay', or 'ghost'
            generate_video: Whether to generate comparison video
            
        Returns:
            Analysis results with comparison_video_path if generated
        """
        logger.info(f"🎬 Analysis with comparison video: {comparison_mode} mode")
        
        # Run standard analysis
        results = self.analyze_shot(video_path, baseline_player)
        
        if not results.get('success'):
            return results
        
        # Generate comparison video if requested
        if generate_video:
            try:
                # Get baseline data
                baseline_data = self.baselines.get(baseline_player)
                baseline_video_path = None
                baseline_keypoints = None
                baseline_phases = None
                
                if baseline_data:
                    baseline_video_path = baseline_data.get('video_path')
                    baseline_keypoints = baseline_data.get('keypoints_sequence')
                    baseline_phases = baseline_data.get('phases')
                
                # Get user keypoints from pose processor cache
                pose_data = self.pose_processor.process_video(video_path, frame_skip=1)
                
                # Generate comparison video
                comparison_video_path = self.video_comparator.create_comparison_video(
                    user_video_path=video_path,
                    user_keypoints=pose_data['keypoints_sequence'],
                    user_phases=results['phases'],
                    baseline_video_path=baseline_video_path,
                    baseline_keypoints=baseline_keypoints,
                    baseline_phases=baseline_phases,
                    mode=comparison_mode
                )
                
                results['comparison_video_path'] = comparison_video_path
                logger.info(f"✅ Comparison video generated: {comparison_video_path}")
                
            except Exception as e:
                logger.error(f"❌ Comparison video generation failed: {e}", exc_info=True)
                results['comparison_video_error'] = str(e)
        
        return results
    
    def _load_baselines(self) -> Dict[str, Dict]:
        """Load NBA baseline data from JSON files"""
        baselines = {}
        
        if not self.baselines_dir.exists():
            logger.warning(f"⚠️ Baselines directory not found: {self.baselines_dir}")
            return baselines
        
        for json_file in self.baselines_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract player name
                    player_name = data.get('player_name')
                    if player_name:
                        baselines[player_name] = data
                        logger.info(f"📁 Loaded baseline: {player_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {json_file}: {e}")
        
        return baselines
    
    def _compare_to_baseline(
        self, 
        user_metrics: Dict[str, Any],
        baseline_data: Dict[str, Any],
        player_name: str
    ) -> Dict[str, Any]:
        """
        Compare user metrics to NBA baseline
        
        Returns similarity score and metric-by-metric comparison
        """
        baseline_metrics = baseline_data.get('metrics', {})
        
        comparison = {
            'player': player_name,
            'overall_similarity': 0.0,
            'metric_comparisons': {},
            'strengths': [],
            'areas_for_improvement': []
        }
        
        # Define comparison weights
        weights = {
            'release_angle': 0.25,
            'elbow_flare': 0.20,
            'knee_load': 0.15,
            'timing': 0.15,
            'base_width': 0.10,
            'lateral_sway': 0.10,
            'arc_trajectory': 0.05
        }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        # Compare each metric
        for metric_name, weight in weights.items():
            if metric_name in user_metrics and metric_name in baseline_metrics:
                similarity = self._compare_metric(
                    user_metrics[metric_name],
                    baseline_metrics[metric_name],
                    metric_name
                )
                
                comparison['metric_comparisons'][metric_name] = similarity
                
                if similarity['similarity_score'] > 0:
                    total_similarity += similarity['similarity_score'] * weight
                    total_weight += weight
                
                # Categorize strengths and weaknesses
                if similarity['similarity_score'] >= 0.8:
                    comparison['strengths'].append(metric_name.replace('_', ' ').title())
                elif similarity['similarity_score'] < 0.6:
                    comparison['areas_for_improvement'].append(metric_name.replace('_', ' ').title())
        
        if total_weight > 0:
            comparison['overall_similarity'] = (total_similarity / total_weight) * 100
        
        logger.info(f"📊 Similarity to {player_name}: {comparison['overall_similarity']:.1f}%")
        
        return comparison
    
    def _compare_metric(
        self,
        user_metric: Dict[str, Any],
        baseline_metric: Dict[str, Any],
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Compare individual metric to baseline
        Returns similarity score (0-1) and delta
        """
        comparison = {
            'similarity_score': 0.0,
            'delta': None,
            'user_value': None,
            'baseline_value': None
        }
        
        # Handle different metric structures
        if metric_name == 'timing':
            # Compare timing metrics
            user_load_to_release = user_metric.get('load_to_release_ms', 0)
            baseline_load_to_release = baseline_metric.get('load_to_release_ms', 300)
            
            if user_load_to_release > 0:
                delta_ms = abs(user_load_to_release - baseline_load_to_release)
                # 100ms difference = 0.5 similarity, 0ms = 1.0 similarity
                similarity = max(0.0, 1.0 - (delta_ms / 200))
                
                comparison['similarity_score'] = similarity
                comparison['delta'] = user_load_to_release - baseline_load_to_release
                comparison['user_value'] = user_load_to_release
                comparison['baseline_value'] = baseline_load_to_release
        
        elif 'angle_deg' in user_metric:
            # Angle-based metrics
            user_angle = user_metric['angle_deg']
            
            # Extract baseline angle (depends on metric structure)
            if 'angle_deg' in baseline_metric:
                baseline_angle = baseline_metric['angle_deg']
            elif metric_name == 'release_angle' and 'elbow_angle' in baseline_metric:
                baseline_angle = baseline_metric['elbow_angle']
            else:
                return comparison
            
            # Calculate similarity (smaller angle difference = higher similarity)
            angle_diff = abs(user_angle - baseline_angle)
            similarity = max(0.0, 1.0 - (angle_diff / 30))  # 30° diff = 0 similarity
            
            comparison['similarity_score'] = similarity
            comparison['delta'] = user_angle - baseline_angle
            comparison['user_value'] = user_angle
            comparison['baseline_value'] = baseline_angle
        
        elif 'ratio' in user_metric:
            # Ratio-based metrics
            user_ratio = user_metric['ratio']
            baseline_ratio = baseline_metric.get('ratio', 0.2)
            
            ratio_diff = abs(user_ratio - baseline_ratio)
            similarity = max(0.0, 1.0 - (ratio_diff / 0.15))
            
            comparison['similarity_score'] = similarity
            comparison['delta'] = user_ratio - baseline_ratio
            comparison['user_value'] = user_ratio
            comparison['baseline_value'] = baseline_ratio
        
        return comparison
    
    def _generate_coaching_cues(
        self,
        metrics: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate top 3 coaching cues based on metrics and comparison
        Ranked by (distance from optimal) × (metric weight)
        """
        cues = []
        
        # Define coaching rules
        rules = self._get_coaching_rules()
        
        # Evaluate each rule
        for rule in rules:
            metric_name = rule['metric']
            if metric_name not in metrics:
                continue
            
            metric_data = metrics[metric_name]
            if isinstance(metric_data, dict) and 'quality_score' not in metric_data:
                continue
            
            # Check condition
            if self._evaluate_condition(metric_data, rule['condition']):
                priority = (10.0 - metric_data.get('quality_score', 5.0)) * rule['weight']
                
                cues.append({
                    'cue': rule['cue'],
                    'why': rule['why'],
                    'drill_id': rule.get('drill_id', ''),
                    'priority': priority,
                    'metric': metric_name
                })
        
        # Sort by priority and return top 3
        cues.sort(key=lambda x: x['priority'], reverse=True)
        return cues[:3]
    
    def _get_coaching_rules(self) -> List[Dict[str, Any]]:
        """Define coaching rules for each metric"""
        return [
            {
                'metric': 'elbow_flare',
                'condition': 'angle_deg > 12',
                'cue': 'Tuck your elbow; keep forearm aligned under the ball.',
                'why': 'Reduces side-spin and left/right misses.',
                'drill_id': 'wall-elbow-slides',
                'weight': 0.25
            },
            {
                'metric': 'release_angle',
                'condition': 'angle_deg < 55 or angle_deg > 62',
                'cue': 'Adjust your release angle to 55-62 degrees for optimal arc.',
                'why': 'Higher arc gives better basket entry angle and room for error.',
                'drill_id': 'form-shooting',
                'weight': 0.25
            },
            {
                'metric': 'knee_load',
                'condition': 'angle_deg < 70 or angle_deg > 90',
                'cue': 'Maintain 70-90 degree knee bend at your load point.',
                'why': 'Optimal power generation and balance.',
                'drill_id': 'chair-shooting',
                'weight': 0.20
            },
            {
                'metric': 'hip_shoulder_alignment',
                'condition': 'angle_deg > 15',
                'cue': 'Square your shoulders and hips to the basket.',
                'why': 'Improves accuracy and power transfer.',
                'drill_id': 'balance-shooting',
                'weight': 0.15
            },
            {
                'metric': 'lateral_sway',
                'condition': 'sway_ratio > 0.05',
                'cue': 'Minimize lateral movement; shoot straight up and down.',
                'why': 'Reduces inconsistency and improves balance.',
                'drill_id': 'line-shooting',
                'weight': 0.15
            },
            {
                'metric': 'base_width',
                'condition': 'ratio < 0.15 or ratio > 0.25',
                'cue': 'Adjust your stance to shoulder-width for better balance.',
                'why': 'Optimal base provides stability and power.',
                'drill_id': 'stance-drills',
                'weight': 0.10
            },
            {
                'metric': 'arc_trajectory',
                'condition': 'arc_angle_deg < 45 or arc_angle_deg > 55',
                'cue': 'Focus on shooting with a 45-55 degree arc.',
                'why': 'Optimal arc maximizes basket entry angle.',
                'drill_id': 'arc-training',
                'weight': 0.10
            }
        ]
    
    def _evaluate_condition(self, metric_data: Dict[str, Any], condition: str) -> bool:
        """Evaluate a condition string against metric data"""
        try:
            # Extract variable name and comparison
            if '>' in condition:
                var, threshold = condition.split('>')
                var = var.strip()
                threshold = float(threshold.strip())
                return metric_data.get(var, 0) > threshold
            
            elif '<' in condition:
                var, threshold = condition.split('<')
                var = var.strip()
                threshold = float(threshold.strip())
                return metric_data.get(var, 0) < threshold
            
            elif 'or' in condition:
                # Handle compound conditions
                parts = condition.split('or')
                return any(self._evaluate_condition(metric_data, part.strip()) for part in parts)
            
            return False
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to evaluate condition '{condition}': {e}")
            return False
    
    def get_available_baselines(self) -> List[str]:
        """Get list of available NBA baseline players"""
        return list(self.baselines.keys())
    
    def get_baseline_data(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Get baseline data for a specific player"""
        return self.baselines.get(player_name)
