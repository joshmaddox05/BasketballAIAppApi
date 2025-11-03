# FastAPI Backend for Basketball AI Analysis with Pro Player Baselines
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uuid
import os
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import shutil
from pathlib import Path
import numpy as np
import logging
import gc  # For garbage collection
import json

from services.baseline_analyzer import BaselineAnalyzer, NumpyEncoder
from services.video_processor import VideoProcessor
from services.shot_analysis_service import ShotAnalysisService
from routes.comprehensive_analysis import setup_comprehensive_analysis_routes
from services.shot_analysis_service import ShotAnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    import numpy as np
    
    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        logger.debug(f"Converting numpy integer {type(obj)} to int")
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        logger.debug(f"Converting numpy float {type(obj)} to float")
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        logger.debug(f"Converting numpy bool {type(obj)} to bool")
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        logger.debug(f"Converting numpy array {type(obj)} to list")
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects with __dict__
        try:
            return {key: convert_numpy_types(value) for key, value in obj.__dict__.items()}
        except:
            return str(obj)
    else:
        return obj

def ensure_json_serializable(obj):
    """Ensure object is JSON serializable by using NumpyEncoder"""
    try:
        # Try to serialize and deserialize to catch any remaining numpy types
        json_str = json.dumps(obj, cls=NumpyEncoder)
        return json.loads(json_str)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed, converting to string: {e}")
        logger.warning(f"Problematic object type: {type(obj)}")
        return str(obj)

app = FastAPI(
    title="Basketball AI Analysis API",
    version="2.0.0",
    description="AI-powered basketball shooting form analysis with pro player comparisons"
)

# Enable CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory management: Limit concurrent video processing
active_requests = 0
MAX_CONCURRENT_REQUESTS = 1  # Only process 1 video at a time to save memory

@app.middleware("http")
async def limit_concurrent_processing(request, call_next):
    """Limit concurrent video processing to prevent OOM"""
    global active_requests
    
    # Only limit video analysis endpoints
    if "/analyze" in request.url.path:
        if active_requests >= MAX_CONCURRENT_REQUESTS:
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is busy processing another video. Please try again in a moment."},
                cls=NumpyEncoder
            )
        active_requests += 1
        try:
            response = await call_next(request)
            return response
        finally:
            active_requests -= 1
    else:
        return await call_next(request)

# Initialize services lazily to save memory
baseline_analyzer = None
shot_comparator = None
video_processor = None
shot_analysis_service = None

def get_baseline_analyzer():
    global baseline_analyzer
    if baseline_analyzer is None:
        baseline_analyzer = BaselineAnalyzer()
    return baseline_analyzer


def get_video_processor():
    global video_processor
    if video_processor is None:
        video_processor = VideoProcessor()
    return video_processor

def get_shot_analysis_service():
    global shot_analysis_service
    if shot_analysis_service is None:
        shot_analysis_service = ShotAnalysisService(baselines_dir=str(BASELINES_DIR))
    return shot_analysis_service

# Storage directories
UPLOAD_DIR = Path("uploads")
BASELINES_DIR = Path("baselines")
OUTPUT_DIR = Path("output")
PROCESSED_DIR = Path("processed")

# Create directories
for dir_path in [UPLOAD_DIR, BASELINES_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# In-memory storage (use database in production)
video_storage = {}
analysis_cache = {}

# Setup comprehensive analysis routes
setup_comprehensive_analysis_routes(app, UPLOAD_DIR, get_shot_analysis_service)

@app.get("/")
async def root():
    analyzer = get_baseline_analyzer()
    return {
        "message": "Basketball AI Analysis API",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Real-time pose detection with MediaPipe",
            "Pro player baseline comparisons",
            "Detailed shooting form analysis",
            "Personalized feedback generation",
            "Comprehensive shot analysis with phase detection"
        ],
        "available_baselines": analyzer.list_available_baselines(),
        "endpoints": {
            "comprehensive_analysis": "/analyze/comprehensive",
            "baselines": "/baselines/available",
            "legacy_curry_comparison": "/analyze/compare-to-curry"
        }
    }

@app.post("/upload/video")
async def upload_video(video: UploadFile = File(...)):
    """Upload video file for analysis"""
    try:
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Supported: mp4, mov, avi, mkv"
            )
        
        # Save video file
        file_path = UPLOAD_DIR / f"{video_id}.mp4"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Get video metadata
        try:
            metadata = video_processor.get_video_metadata(str(file_path))
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
            metadata = {"error": "Could not extract metadata"}
        
        # Store video metadata
        video_storage[video_id] = {
            "id": video_id,
            "filename": video.filename,
            "filepath": str(file_path),
            "metadata": metadata,
            "uploaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Video uploaded: {video_id}")
        
        return {
            "video_id": video_id,
            "message": "Video uploaded successfully",
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/analyze/shooting")
async def analyze_shooting_form(request: Dict[str, Any]):
    """Analyze shooting form from uploaded video with optional pro player comparison"""
    try:
        # Get services lazily
        processor = get_video_processor()
        
        video_id = request.get("video_id")
        analysis_mode = request.get("analysis_mode", "shooting")
        camera_type = request.get("camera_type", "back")
        duration = request.get("duration", 5.0)
        # Removed comparison_player parameter - no longer comparing to specific players
        
        if video_id not in video_storage:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_info = video_storage[video_id]
        video_path = video_info['filepath']
        
        logger.info(f"🎯 Analyzing video: {video_id}")
        
        # Process video and extract pose data
        user_analysis = processor.analyze_shooting_video(video_path)
        
        # Removed comparison logic - no longer comparing to specific players
        
        # Generate comprehensive results
        results = {
            "video_id": video_id,
            "analysis_mode": analysis_mode,
            "overall_score": _calculate_overall_score(user_analysis),
            "confidence": user_analysis.get('confidence', 0.85),
            "metrics": _format_metrics_for_app(user_analysis['metrics']),
            "keypoints": user_analysis.get('keypoints_sequence', [])[:10],  # First 10 frames
            "recommendations": _generate_recommendations(user_analysis),
            "biomechanics": _format_biomechanics(user_analysis),
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Cache results
        analysis_cache[video_id] = results
        
        logger.info(f"✅ Analysis complete - Score: {results['overall_score']}")
        
        # Force garbage collection
        gc.collect()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {str(e)}")
        gc.collect()  # Clean up memory even on error
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Removed /analyze/compare-to-curry endpoint - no longer comparing to specific players

@app.post("/analyze/form-analysis")
async def analyze_form_analysis(video: UploadFile = File(...)):
    """
    Pure form analysis without player comparison.
    Analyzes shooting technique against optimal biomechanical ranges.
    """
    file_path = None
    try:
        # Get services lazily
        processor = get_video_processor()
        
        # Step 1: Upload and save video
        video_id = str(uuid.uuid4())
        
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Supported: mp4, mov, avi, mkv"
            )
        
        # Save video file
        file_path = UPLOAD_DIR / f"{video_id}.mp4"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"📹 Video uploaded for form analysis: {video_id}")
        
        # Store video metadata
        video_storage[video_id] = {
            "id": video_id,
            "filename": video.filename,
            "filepath": str(file_path),
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Step 2: Analyze user's shooting form
        logger.info(f"🎯 Analyzing shooting form...")
        user_analysis = processor.analyze_shooting_video(str(file_path))
        
        # Step 3: Generate form analysis results
        overall_score = _calculate_overall_score(user_analysis)
        
        results = {
            "video_id": video_id,
            "overall_score": overall_score,
            "overall_grade": _get_grade_from_score(overall_score),
            "confidence": user_analysis.get('confidence', 0.85),
            "orientation": "side",  # Could be detected from video
            
            # Metrics by category with optimal ranges
            "wrist_metrics": _format_wrist_metrics(user_analysis),
            "head_metrics": _format_head_metrics(user_analysis),
            "body_metrics": _format_body_metrics(user_analysis),
            
            # Top recommendations
            "top_recommendations": _generate_recommendations(user_analysis),
            
            # Coaching cues
            "coaching_cues": _generate_coaching_cues(user_analysis),
            
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Cache results
        analysis_cache[video_id] = results
        
        logger.info(f"✅ Form analysis complete - Score: {overall_score}")
        
        # Force garbage collection to free memory
        gc.collect()
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Form analysis error: {str(e)}")
        # Clean up file if it was created
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    finally:
        # Always try to clean up memory
        gc.collect()

@app.post("/analyze/form-analysis")
async def analyze_form(
    video: UploadFile = File(...),
    frame_skip: int = Form(default=1)
):
    """
    Analyze shooting form for pure technique assessment without player comparison.
    
    Features:
    - Advanced pose estimation with visibility filtering
    - Phase detection (dip, load, release, follow-through, landing)
    - Biomechanical metrics with optimal ranges
    - Pure technique scoring (no player comparison)
    - Top 3 coaching cues with drill recommendations
    
    Args:
        video: Video file (mp4, mov, avi, mkv)
        frame_skip: Process every Nth frame (default: 1 for all frames)
        
    Returns:
        Pure form analysis with technique scores, optimal ranges, and coaching cues
    """
    file_path = None
    
    try:
        # Get analysis service
        analysis_service = get_shot_analysis_service()
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Supported: mp4, mov, avi, mkv"
            )
        
        # Save video file
        file_path = UPLOAD_DIR / f"{video_id}.mp4"
        logger.info(f"💾 Saving video to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"✅ Video saved, starting form analysis...")
        
        # Perform pure form analysis (no baseline comparison)
        results = analysis_service.analyze_shot(
            video_path=str(file_path),
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
            
            logger.warning(f"⚠️ Form analysis failed: {error_response['error']}")
            return JSONResponse(status_code=200, content=error_response, cls=NumpyEncoder)
        
        # Convert numpy types to native Python types
        results = convert_numpy_types(results)
        
        # Format successful response for pure form analysis
        response = {
            "video_id": video_id,
            "success": True,
            "analysis_mode": "form_analysis",
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
            
            # Biomechanical metrics with optimal ranges
            "metrics": {
                "knee_load": _format_form_metric(results['metrics'].get('knee_load')),
                "hip_shoulder_alignment": _format_form_metric(results['metrics'].get('hip_shoulder_alignment')),
                "elbow_flare": _format_form_metric(results['metrics'].get('elbow_flare')),
                "release_angle": _format_form_metric(results['metrics'].get('release_angle')),
                "base_width": _format_form_metric(results['metrics'].get('base_width')),
                "lateral_sway": _format_form_metric(results['metrics'].get('lateral_sway')),
                "arc_trajectory": _format_form_metric(results['metrics'].get('arc_trajectory'))
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
        
        # Convert response to native Python types and ensure JSON serializable
        try:
            response = convert_numpy_types(response)
            response = ensure_json_serializable(response)
        except Exception as e:
            logger.error(f"Error converting response types: {e}")
            # Fallback: convert everything to strings
            response = str(response)
        
        # Cache results
        analysis_cache[video_id] = response
        
        logger.info(f"✅ Form analysis complete - Score: {response.get('overall_score', 'N/A')}/100")
        
        # Clean up video file to save storage
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️ Cleaned up video file")
        
        # Force garbage collection
        gc.collect()
        
        return JSONResponse(content=response, cls=NumpyEncoder)
        
    except Exception as e:
        logger.error(f"❌ Form analysis error: {str(e)}", exc_info=True)
        
        # Clean up on error
        if file_path and file_path.exists():
            file_path.unlink()
        
        gc.collect()
        
        raise HTTPException(
            status_code=500,
            detail=f"Form analysis failed: {str(e)}"
        )

@app.post("/analyze/comprehensive")
async def analyze_shot_comprehensive(
    video: UploadFile = File(...),
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
        analysis_service = get_shot_analysis_service()
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Supported: mp4, mov, avi, mkv"
            )
        
        # Save video file
        file_path = UPLOAD_DIR / f"{video_id}.mp4"
        logger.info(f"💾 Saving video to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"✅ Video saved, starting analysis...")
        
        # Perform comprehensive analysis
        results = analysis_service.analyze_shot(
            video_path=str(file_path),
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
            return JSONResponse(status_code=200, content=error_response, cls=NumpyEncoder)
        
        # Convert numpy types to native Python types
        results = convert_numpy_types(results)
        
        # Format successful response
        response = {
            "video_id": video_id,
            "success": True,
            "analysis_mode": "comprehensive",
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
        
        # Convert response to native Python types and ensure JSON serializable
        try:
            response = convert_numpy_types(response)
            response = ensure_json_serializable(response)
        except Exception as e:
            logger.error(f"Error converting response types: {e}")
            # Fallback: convert everything to strings
            response = str(response)
        
        # Cache results
        analysis_cache[video_id] = response
        
        logger.info(f"✅ Comprehensive analysis complete - Score: {response.get('overall_score', 'N/A')}/100")
        
        # Clean up video file to save storage
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️ Cleaned up video file")
        
        # Force garbage collection
        gc.collect()
        
        return JSONResponse(content=response, cls=NumpyEncoder)
        
    except Exception as e:
        logger.error(f"❌ Comprehensive analysis error: {str(e)}", exc_info=True)
        
        # Clean up on error
        if file_path and file_path.exists():
            file_path.unlink()
        
        gc.collect()
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

def _format_metric(metric_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Format metric data for API response"""
    if not metric_data or 'error' in metric_data:
        return {
            "error": metric_data.get('error', 'Not available') if metric_data else 'Not available',
            "quality_score": 0.0
        }
    
    # Convert numpy types to native Python types
    metric_data = convert_numpy_types(metric_data)
    
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

def _format_form_metric(metric_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Format metric data for pure form analysis (no player comparison)"""
    if not metric_data or 'error' in metric_data:
        return {
            "error": metric_data.get('error', 'Not available') if metric_data else 'Not available',
            "quality_score": 0.0,
            "optimal_range": "N/A",
            "status": "error"
        }
    
    # Convert numpy types to native Python types
    metric_data = convert_numpy_types(metric_data)
    
    formatted = {
        "quality_score": round(metric_data.get('quality_score', 0), 1),
        "optimal_range": metric_data.get('optimal_range', 'N/A'),
        "status": "good" if metric_data.get('quality_score', 0) >= 0.7 else "improve"
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
    if 'in_range' in metric_data:
        formatted['in_range'] = metric_data['in_range']
    if 'description' in metric_data:
        formatted['description'] = metric_data['description']
    
    return formatted

@app.get("/analysis/{video_id}")
async def get_analysis(video_id: str):
    """Get cached analysis results"""
    if video_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_cache[video_id]

@app.get("/health")
async def health_check():
    """Health check endpoint that doesn't initialize heavy services"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "baseline_analyzer": "ready",
                "shot_comparator": "ready",
                "video_processor": "ready"
            },
            "storage": {
                "videos": len(video_storage),
                "analyses": len(analysis_cache),
                "baselines": "available"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/baseline/create")
async def create_baseline(
    video: UploadFile = File(...),
    player_name: str = Form(...),
    position: str = Form("Guard"),
    team: str = Form("")
):
    """Create a new pro player baseline from video"""
    try:
        # Save baseline video
        baseline_id = str(uuid.uuid4())
        video_path = BASELINES_DIR / f"{baseline_id}.mp4"
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"📹 Creating baseline for {player_name}...")
        
        # Analyze and create baseline
        baseline_data = baseline_analyzer.analyze_pro_video(
            str(video_path),
            player_name
        )
        
        # Add metadata
        baseline_data['position'] = position
        baseline_data['team'] = team
        baseline_data['video_id'] = baseline_id
        
        logger.info(f"✅ Baseline created for {player_name}")
        
        return {
            "message": f"Baseline created for {player_name}",
            "player_name": player_name,
            "position": position,
            "team": team,
            "metrics_summary": {
                "release_angle": baseline_data['metrics']['release_angle']['trajectory_angle'],
                "follow_through_score": baseline_data['metrics']['follow_through']['quality_score'],
                "balance_score": baseline_data['metrics']['balance']['stability_score']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Baseline creation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Baseline creation failed: {str(e)}"
        )

@app.get("/baselines/list")
async def list_baselines():
    """List all available pro player baselines"""
    try:
        analyzer = get_baseline_analyzer()
        baselines = analyzer.list_available_baselines()
        
        detailed_baselines = []
        for player in baselines:
            try:
                baseline = analyzer.load_baseline(player)
                detailed_baselines.append({
                    "name": player,
                    "position": baseline.get('position', 'Unknown'),
                    "team": baseline.get('team', 'Unknown'),
                    "created_at": baseline.get('created_at', 'Unknown'),
                    "metrics": {
                        "release_angle": baseline['metrics']['release_angle']['trajectory_angle'],
                        "follow_through": baseline['metrics']['follow_through']['quality_score'],
                        "balance": baseline['metrics']['balance']['stability_score']
                    }
                })
            except Exception as e:
                logger.warning(f"Could not load baseline for {player}: {e}")
                detailed_baselines.append({"name": player, "error": "Could not load"})
        
        return {
            "count": len(detailed_baselines),
            "baselines": detailed_baselines
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for formatting responses
def _calculate_overall_score(analysis: Dict) -> float:
    """Calculate overall shooting form score"""
    metrics = analysis.get('metrics', {})
    
    weights = {
        'release_angle': 0.25,
        'elbow_alignment': 0.20,
        'follow_through': 0.20,
        'balance': 0.20,
        'arc_trajectory': 0.15
    }
    
    score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in metrics and 'quality_score' in metrics[metric]:
            score += metrics[metric]['quality_score'] * weight * 10
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 70.0


def _format_metrics_for_app(metrics: Dict) -> List[Dict]:
    """Format metrics for mobile app display"""
    formatted = []
    
    metric_configs = {
        'release_angle': {
            'name': 'Release Angle',
            'id': 'release_angle',
            'ideal': '45-50°',
            'importance': 'high'
        },
        'elbow_alignment': {
            'name': 'Elbow Alignment',
            'id': 'elbow_alignment',
            'ideal': 'Under ball',
            'importance': 'high'
        },
        'follow_through': {
            'name': 'Follow Through',
            'id': 'follow_through',
            'ideal': 'Full extension',
            'importance': 'medium'
        },
        'balance': {
            'name': 'Balance & Stance',
            'id': 'balance',
            'ideal': 'Stable base',
            'importance': 'medium'
        },
        'arc_trajectory': {
            'name': 'Shot Arc',
            'id': 'arc_trajectory',
            'ideal': '45-50°',
            'importance': 'medium'
        }
    }
    
    for metric_id, config in metric_configs.items():
        if metric_id in metrics:
            metric_data = metrics[metric_id]
            formatted.append({
                'id': config['id'],
                'name': config['name'],
                'value': str(metric_data.get('value', 'N/A')),
                'score': round(metric_data.get('quality_score', 0) * 10, 1),
                'ideal': config['ideal'],
                'status': 'good' if metric_data.get('quality_score', 0) >= 0.8 else 'improve',
                'feedback': metric_data.get('feedback', '')
            })
    
    return formatted


def _generate_recommendations(analysis: Dict) -> List[str]:
    """Generate coaching recommendations based on technique analysis"""
    recommendations = []
    metrics = analysis.get('metrics', {})
    
    # Release angle
    if 'release_angle' in metrics:
        angle = metrics['release_angle'].get('quality_score', 1.0)
        if angle < 0.7:
            recommendations.append("Practice your release angle - aim for 45-50° for optimal arc")
    
    # Elbow alignment
    if 'elbow_flare' in metrics:
        elbow = metrics['elbow_flare'].get('quality_score', 1.0)
        if elbow < 0.7:
            recommendations.append("Work on keeping your elbow aligned under the ball")
    
    # Knee load
    if 'knee_load' in metrics:
        knee = metrics['knee_load'].get('quality_score', 1.0)
        if knee < 0.7:
            recommendations.append("Maintain proper knee bend (70-90°) for power generation")
    
    # Balance
    if 'lateral_sway' in metrics:
        sway = metrics['lateral_sway'].get('quality_score', 1.0)
        if sway < 0.7:
            recommendations.append("Minimize lateral movement - shoot straight up and down")
    
    return recommendations[:5]  # Top 5 recommendations

def _get_grade_from_score(score: float) -> str:
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

def _format_wrist_metrics(analysis: Dict) -> Dict:
    """Format wrist-related metrics"""
    metrics = analysis.get('metrics', {})
    return {
        'release_angle': metrics.get('release_angle', {}),
        'elbow_flare': metrics.get('elbow_flare', {})
    }

def _format_head_metrics(analysis: Dict) -> Dict:
    """Format head-related metrics"""
    metrics = analysis.get('metrics', {})
    return {
        'hip_shoulder_alignment': metrics.get('hip_shoulder_alignment', {})
    }

def _format_body_metrics(analysis: Dict) -> Dict:
    """Format body-related metrics"""
    metrics = analysis.get('metrics', {})
    return {
        'knee_load': metrics.get('knee_load', {}),
        'base_width': metrics.get('base_width', {}),
        'lateral_sway': metrics.get('lateral_sway', {}),
        'arc_trajectory': metrics.get('arc_trajectory', {})
    }

def _generate_coaching_cues(analysis: Dict) -> List[Dict]:
    """Generate coaching cues based on analysis"""
    metrics = analysis.get('metrics', {})
    cues = []
    
    # Check each metric and generate cues for poor performance
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict) and metric_data.get('quality_score', 10) < 6.0:
            cue = {
                'metric': metric_name,
                'score': metric_data.get('quality_score', 0),
                'message': f"Improve your {metric_name.replace('_', ' ')}"
            }
            cues.append(cue)
    
    return cues[:3]  # Top 3 cues


def _format_biomechanics(analysis: Dict) -> Dict:
    """Format biomechanical data"""
    metrics = analysis.get('metrics', {})
    
    return {
        'release_angle': metrics.get('release_angle', {}).get('trajectory_angle', 0),
        'elbow_angle': metrics.get('release_angle', {}).get('elbow_angle', 0),
        'follow_through_distance': metrics.get('follow_through', {}).get('extension_distance', 0),
        'stance_width': metrics.get('balance', {}).get('average_stance_width', 0),
        'arc_angle': metrics.get('arc_trajectory', {}).get('arc_angle', 0)
    }


if __name__ == "__main__":
    import uvicorn
    # Increased timeout for long-running AI model processing
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=600,  # 10 minutes keep-alive
        timeout_notify=600,       # 10 minutes before timeout notification
        limit_concurrency=10,
        limit_max_requests=1000
    )
