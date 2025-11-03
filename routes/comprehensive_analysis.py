"""
API Routes for Comprehensive Shot Analysis
"""
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import uuid
import shutil
import gc
import logging

logger = logging.getLogger(__name__)

def setup_comprehensive_analysis_routes(app, upload_dir: Path, get_shot_analysis_service_func):
    """Setup comprehensive analysis API routes"""
    
    @app.post("/analyze/comprehensive")
    async def analyze_shot_comprehensive(
        video: UploadFile = File(...),
        baseline_player: str = Form(default="Stephen Curry"),
        frame_skip: int = Form(default=1)
    ):
        """
        Comprehensive shot analysis with MediaPipe Pose Landmarker
        
        Features:
        - Advanced pose estimation with visibility filtering
        - Phase detection (dip, load, release, follow-through, landing)
        - Biomechanical metrics (knee load, elbow flare, release angle, etc.)
        - NBA baseline comparison
        - Top 3 coaching cues with drill recommendations
        
        Args:
            video: Video file (mp4, mov, avi, mkv)
            baseline_player: NBA player to compare against (default: Stephen Curry)
            frame_skip: Process every Nth frame (default: 1 for all frames)
            
        Returns:
            Comprehensive analysis with metrics, comparison, and coaching cues
        """
        file_path = None
        
        try:
            # Get analysis service
            analysis_service = get_shot_analysis_service_func()
            
            # Generate unique video ID
            video_id = str(uuid.uuid4())
            
            # Validate file type
            if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Supported: mp4, mov, avi, mkv"
                )
            
            # Save video file
            file_path = upload_dir / f"{video_id}.mp4"
            logger.info(f"💾 Saving video to: {file_path}")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            logger.info(f"✅ Video saved, starting comprehensive analysis...")
            
            # Perform comprehensive analysis
            results = analysis_service.analyze_shot(
                video_path=str(file_path),
                baseline_player=baseline_player,
                frame_skip=frame_skip
            )
            
            if not results.get('success', False):
                # Analysis failed - return error with details
                error_response = {
                    "video_id": video_id,
                    "success": False,
                    "error": results.get('error', 'Unknown error'),
                    "confidence": results.get('confidence', 0.0),
                    "warning": results.get('warning'),
                    "analyzed_at": datetime.now().isoformat()
                }
                
                # Clean up video file
                if file_path and file_path.exists():
                    file_path.unlink()
                
                logger.warning(f"⚠️ Analysis failed: {error_response['error']}")
                return JSONResponse(status_code=200, content=error_response)
            
            # Format successful response
            response = _format_comprehensive_response(video_id, results, baseline_player)
            
            logger.info(f"✅ Comprehensive analysis complete - Score: {response['overall_score']}/100")
            
            # Clean up video file to save storage
            if file_path and file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Cleaned up video file")
            
            # Force garbage collection
            gc.collect()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis error: {str(e)}", exc_info=True)
            
            # Clean up on error
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
            
            gc.collect()
            
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
    
    @app.post("/analyze/comparison-video")
    async def generate_comparison_video(
        video: UploadFile = File(...),
        baseline_player: str = Form(default="Stephen Curry"),
        comparison_mode: str = Form(default="split"),
        frame_skip: int = Form(default=1)
    ):
        """
        Generate side-by-side comparison video with pose overlays
        
        Modes:
        - split: Side-by-side user (left) vs baseline (right)
        - overlay: Both skeletons on user video
        - ghost: Transparent baseline skeleton over user video
        
        Args:
            video: Video file (mp4, mov, avi, mkv)
            baseline_player: NBA player to compare against
            comparison_mode: 'split', 'overlay', or 'ghost'
            frame_skip: Process every Nth frame (default: 1)
            
        Returns:
            Analysis results + comparison video download URL
        """
        file_path = None
        
        try:
            # Get analysis service
            analysis_service = get_shot_analysis_service_func()
            
            # Validate inputs
            if comparison_mode not in ['split', 'overlay', 'ghost']:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid mode. Use 'split', 'overlay', or 'ghost'"
                )
            
            if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Supported: mp4, mov, avi, mkv"
                )
            
            # Generate unique video ID
            video_id = str(uuid.uuid4())
            
            # Save video file
            file_path = upload_dir / f"{video_id}.mp4"
            logger.info(f"💾 Saving video to: {file_path}")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            logger.info(f"✅ Video saved, generating comparison video...")
            
            # Perform analysis with comparison video
            results = analysis_service.analyze_with_comparison_video(
                video_path=str(file_path),
                baseline_player=baseline_player,
                comparison_mode=comparison_mode,
                generate_video=True
            )
            
            if not results.get('success', False):
                # Analysis failed
                error_response = {
                    "video_id": video_id,
                    "success": False,
                    "error": results.get('error', 'Unknown error'),
                    "confidence": results.get('confidence', 0.0),
                    "analyzed_at": datetime.now().isoformat()
                }
                
                # Clean up
                if file_path and file_path.exists():
                    file_path.unlink()
                
                logger.warning(f"⚠️ Comparison video generation failed: {error_response['error']}")
                return JSONResponse(status_code=200, content=error_response)
            
            # Format response with comparison video
            response = _format_comprehensive_response(video_id, results, baseline_player)
            
            # Add comparison video info
            if 'comparison_video_path' in results:
                comparison_path = Path(results['comparison_video_path'])
                response['comparison_video'] = {
                    'path': str(comparison_path),
                    'filename': comparison_path.name,
                    'mode': comparison_mode,
                    'download_url': f"/download/comparison/{comparison_path.name}"
                }
                logger.info(f"✅ Comparison video ready: {comparison_path.name}")
            
            if 'comparison_video_error' in results:
                response['comparison_video_error'] = results['comparison_video_error']
            
            # Clean up original video
            if file_path and file_path.exists():
                file_path.unlink()
            
            logger.info(f"✅ Comparison video generation complete")
            
            # Force garbage collection
            gc.collect()
            
            return JSONResponse(status_code=200, content=response)
            
        except Exception as e:
            logger.error(f"❌ Comparison video endpoint failed: {e}", exc_info=True)
            
            # Clean up on error
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
            
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/baselines/available")
    async def get_available_baselines():
        """Get list of available NBA baseline players"""
        analysis_service = get_shot_analysis_service_func()
        return {
            "available_baselines": analysis_service.get_available_baselines()
        }
    
    @app.get("/baselines/{player_name}")
    async def get_baseline_info(player_name: str):
        """Get baseline data for a specific NBA player"""
        analysis_service = get_shot_analysis_service_func()
        baseline_data = analysis_service.get_baseline_data(player_name)
        
        if not baseline_data:
            raise HTTPException(
                status_code=404,
                detail=f"Baseline not found for {player_name}"
            )
        
        return {
            "player_name": player_name,
            "baseline_data": baseline_data
        }


def _format_comprehensive_response(video_id: str, results: Dict[str, Any], baseline_player: str) -> Dict[str, Any]:
    """Format comprehensive analysis results for API response"""
    
    return {
        "video_id": video_id,
        "success": True,
        "analysis_mode": "comprehensive",
        "player": baseline_player,
        "overall_score": round(results['overall_score'], 1),
        "confidence": round(results['confidence'], 2),
        
        # Shooting phases with timestamps
        "phases": {
            "dip_start": {
                "timestamp": results['phases']['dip_start']['timestamp'] if results['phases'].get('dip_start') else None,
                "frame": results['phases']['dip_start']['frame'] if results['phases'].get('dip_start') else None
            },
            "load": {
                "timestamp": results['phases']['load']['timestamp'] if results['phases'].get('load') else None,
                "frame": results['phases']['load']['frame'] if results['phases'].get('load') else None,
                "knee_angle": results['phases']['load'].get('knee_angle')
            },
            "release": {
                "timestamp": results['phases']['release']['timestamp'] if results['phases'].get('release') else None,
                "frame": results['phases']['release']['frame'] if results['phases'].get('release') else None
            },
            "follow_through_end": {
                "timestamp": results['phases']['follow_through_end']['timestamp'] if results['phases'].get('follow_through_end') else None,
                "frame": results['phases']['follow_through_end']['frame'] if results['phases'].get('follow_through_end') else None
            },
            "timing": results['phases'].get('timing', {})
        },
        
        # Biomechanical metrics
        "metrics": {
            "knee_load": _format_metric(results['metrics'].get('knee_load')),
            "hip_shoulder_alignment": _format_metric(results['metrics'].get('hip_shoulder_alignment')),
            "elbow_flare": _format_metric(results['metrics'].get('elbow_flare')),
            "release_angle": _format_metric(results['metrics'].get('release_angle')),
            "base_width": _format_metric(results['metrics'].get('base_width')),
            "lateral_sway": _format_metric(results['metrics'].get('lateral_sway')),
            "arc_trajectory": _format_metric(results['metrics'].get('arc_trajectory'))
        },
        
        # Comparison to NBA baseline
        "comparison": {
            "player": results['comparison'].get('player', baseline_player),
            "overall_similarity": round(results['comparison'].get('overall_similarity', 0), 1),
            "strengths": results['comparison'].get('strengths', []),
            "areas_for_improvement": results['comparison'].get('areas_for_improvement', []),
            "metric_comparisons": results['comparison'].get('metric_comparisons', {})
        },
        
        # Top 3 coaching cues
        "coaching_cues": [
            {
                "cue": cue['cue'],
                "why": cue['why'],
                "drill_id": cue.get('drill_id', ''),
                "metric": cue.get('metric', '')
            }
            for cue in results.get('coaching_cues', [])
        ],
        
        # Quality information
        "quality": {
            "visibility_ratio": round(results['quality_info']['visibility_ratio'], 2),
            "confidence": round(results['quality_info']['confidence'], 2),
            "warning": results['quality_info'].get('warning')
        },
        
        # Video metadata
        "metadata": {
            "duration": round(results['metadata']['duration'], 2),
            "fps": round(results['metadata']['fps'], 1),
            "total_frames": results['metadata']['total_frames'],
            "processed_frames": results['metadata']['processed_frames']
        },
        
        "analyzed_at": datetime.now().isoformat()
    }


def _format_metric(metric_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Format metric data for API response"""
    if not metric_data or 'error' in metric_data:
        return {
            "error": metric_data.get('error', 'Not available') if metric_data else 'Not available',
            "quality_score": 0.0
        }
    
    formatted = {
        "quality_score": round(metric_data.get('quality_score', 0), 1)
    }
    
    # Add metric-specific fields
    if 'angle_deg' in metric_data:
        formatted['angle_deg'] = round(metric_data['angle_deg'], 1)
    if 'ratio' in metric_data:
        formatted['ratio'] = round(metric_data['ratio'], 3)
    if 'sway_ratio' in metric_data:
        formatted['sway_ratio'] = round(metric_data['sway_ratio'], 3)
    if 'arc_angle_deg' in metric_data:
        formatted['arc_angle_deg'] = round(metric_data['arc_angle_deg'], 1)
    if 'optimal_range' in metric_data:
        formatted['optimal_range'] = metric_data['optimal_range']
    if 'in_range' in metric_data:
        formatted['in_range'] = metric_data['in_range']
    if 'description' in metric_data:
        formatted['description'] = metric_data['description']
    
    return formatted
