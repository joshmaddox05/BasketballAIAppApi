"""
Basketball Shot Form Analysis API
AI-powered shooting form analysis with personalized coaching feedback
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uuid
import os
from typing import Dict, Any, Optional
import json
from datetime import datetime
import shutil
from pathlib import Path
import numpy as np
import logging
import gc

from services.shot_analysis_service import ShotAnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    return obj


def ensure_json_serializable(obj):
    """Ensure object is JSON serializable"""
    try:
        json_str = json.dumps(obj, cls=NumpyEncoder)
        return json.loads(json_str)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed: {e}")
        return str(obj)


# Initialize FastAPI app
app = FastAPI(
    title="Basketball Shot Form Analysis API",
    version="3.0.0",
    description="AI-powered basketball shooting form analysis with personalized coaching feedback. "
                "Upload a video of your shot and receive detailed analysis with actionable improvement tips."
)

# Enable CORS for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory management: Limit concurrent video processing
active_requests = 0
MAX_CONCURRENT_REQUESTS = 1


@app.middleware("http")
async def limit_concurrent_processing(request, call_next):
    """Limit concurrent video processing to prevent OOM"""
    global active_requests

    if "/analyze" in request.url.path:
        if active_requests >= MAX_CONCURRENT_REQUESTS:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Server is busy processing another video. Please try again in a moment.",
                    "retry_after_seconds": 30
                }
            )
        active_requests += 1
        try:
            response = await call_next(request)
            return response
        finally:
            active_requests -= 1
    else:
        return await call_next(request)


# Lazy service initialization
shot_analysis_service = None


def get_shot_analysis_service():
    global shot_analysis_service
    if shot_analysis_service is None:
        shot_analysis_service = ShotAnalysisService()
    return shot_analysis_service


# Storage directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory storage (use database in production)
analysis_cache = {}


@app.get("/")
async def root():
    """API information and available endpoints"""
    return {
        "name": "Basketball Shot Form Analysis API",
        "version": "3.0.0",
        "description": "Upload a video of your basketball shot and receive AI-powered form analysis with personalized coaching feedback.",
        "features": [
            "Real-time pose detection using MediaPipe",
            "Shooting phase detection (dip, load, release, follow-through)",
            "7 biomechanical metrics with optimal ranges",
            "Personalized coaching cues with specific drills",
            "Quality scoring and improvement tracking"
        ],
        "how_to_use": {
            "1": "Record a video of your basketball shot (side angle recommended)",
            "2": "POST the video to /analyze/shot",
            "3": "Receive detailed analysis with scores and coaching tips",
            "4": "Practice the recommended drills and re-analyze to track progress"
        },
        "endpoints": {
            "analyze_shot": "POST /analyze/shot - Primary analysis endpoint",
            "get_analysis": "GET /analysis/{video_id} - Retrieve cached results",
            "health": "GET /health - Service health check"
        },
        "video_requirements": {
            "formats": ["mp4", "mov", "avi", "mkv"],
            "max_size_mb": 100,
            "recommended_angle": "Side view (perpendicular to shooting direction)",
            "tips": [
                "Ensure good lighting",
                "Keep full body in frame",
                "Record complete shot motion",
                "Avoid baggy clothing"
            ]
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "services": {
            "shot_analysis": "ready",
            "pose_detection": "ready",
            "phase_detection": "ready"
        },
        "storage": {
            "cached_analyses": len(analysis_cache)
        }
    }


@app.post("/analyze/shot")
async def analyze_shot(
    video: UploadFile = File(...),
    frame_skip: int = Form(default=1)
):
    """
    Analyze basketball shooting form from video

    Upload a video of your shot and receive:
    - Overall form score (0-100) with letter grade
    - 7 biomechanical metrics compared to optimal ranges
    - Shooting phase timing analysis
    - Top 3 personalized coaching cues with drills
    - Improvement summary with strengths and areas to work on

    Args:
        video: Video file (mp4, mov, avi, mkv)
        frame_skip: Process every Nth frame (default: 1 for best accuracy)

    Returns:
        Comprehensive shot analysis with actionable coaching feedback
    """
    file_path = None

    try:
        analysis_service = get_shot_analysis_service()
        video_id = str(uuid.uuid4())

        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid file type",
                    "supported_formats": ["mp4", "mov", "avi", "mkv"],
                    "received": video.filename.split('.')[-1] if '.' in video.filename else "unknown"
                }
            )

        # Save video file
        file_path = UPLOAD_DIR / f"{video_id}.mp4"
        logger.info(f"Saving video: {file_path}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        logger.info(f"Starting analysis for video: {video_id}")

        # Perform analysis
        results = analysis_service.analyze_shot(
            video_path=str(file_path),
            frame_skip=frame_skip
        )

        # Handle analysis failure
        if not results.get('success', False):
            error_response = {
                "video_id": video_id,
                "success": False,
                "error": results.get('error', 'Unknown error'),
                "confidence": results.get('confidence', 0.0),
                "tips": results.get('tips', []),
                "analyzed_at": datetime.now().isoformat()
            }

            if file_path and file_path.exists():
                file_path.unlink()

            logger.warning(f"Analysis failed: {error_response['error']}")
            return JSONResponse(status_code=200, content=error_response)

        # Convert numpy types
        results = convert_numpy_types(results)

        # Format successful response
        response = {
            "video_id": video_id,
            "success": True,

            # Overall assessment
            "overall_score": round(results['overall_score'], 1),
            "overall_grade": results.get('overall_grade', 'N/A'),
            "confidence": round(results['confidence'], 2),

            # Shooting phases with timing
            "phases": _format_phases(results.get('phases', {})),

            # Biomechanical metrics
            "metrics": _format_metrics(results.get('metrics', {})),

            # Coaching feedback
            "coaching_cues": results.get('coaching_cues', []),

            # Improvement summary
            "improvement_summary": results.get('improvement_summary', {}),

            # Quality information
            "quality": {
                "visibility_ratio": round(results['quality_info']['visibility_ratio'], 2),
                "confidence": round(results['quality_info']['confidence'], 2),
                "warning": results['quality_info'].get('warning')
            },

            # Video metadata
            "metadata": {
                "duration_seconds": round(results['metadata']['duration'], 2),
                "fps": round(results['metadata']['fps'], 1),
                "total_frames": results['metadata']['total_frames'],
                "processed_frames": results['metadata']['processed_frames']
            },

            "analyzed_at": datetime.now().isoformat()
        }

        # Ensure JSON serializable
        response = ensure_json_serializable(response)

        # Cache results
        analysis_cache[video_id] = response

        logger.info(f"Analysis complete - Score: {response['overall_score']}/100, Grade: {response['overall_grade']}")

        # Clean up video file
        if file_path and file_path.exists():
            file_path.unlink()

        gc.collect()

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)

        if file_path and file_path.exists():
            file_path.unlink()

        gc.collect()

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/analysis/{video_id}")
async def get_analysis(video_id: str):
    """Retrieve cached analysis results by video ID"""
    if video_id not in analysis_cache:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Analysis not found",
                "video_id": video_id,
                "suggestion": "Analysis results are cached temporarily. Please re-upload your video for a new analysis."
            }
        )

    return analysis_cache[video_id]


def _format_phases(phases: Dict[str, Any]) -> Dict[str, Any]:
    """Format shooting phases for API response"""
    formatted = {}

    phase_names = ['dip_start', 'load', 'release', 'follow_through_end']

    for phase_name in phase_names:
        phase_data = phases.get(phase_name)
        if phase_data:
            formatted[phase_name] = {
                "timestamp": phase_data.get('timestamp'),
                "frame": phase_data.get('frame')
            }
            # Include knee angle for load phase
            if phase_name == 'load' and 'knee_angle' in phase_data:
                formatted[phase_name]['knee_angle'] = phase_data['knee_angle']
        else:
            formatted[phase_name] = None

    # Include timing metrics
    formatted['timing'] = phases.get('timing', {})

    return formatted


def _format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Format biomechanical metrics for API response"""
    formatted = {}

    metric_configs = {
        'release_angle': {
            'display_name': 'Release Angle',
            'unit': 'degrees',
            'description': 'Arm angle at ball release'
        },
        'elbow_flare': {
            'display_name': 'Elbow Alignment',
            'unit': 'degrees',
            'description': 'Lateral elbow deviation from body'
        },
        'knee_load': {
            'display_name': 'Knee Bend',
            'unit': 'degrees',
            'description': 'Knee flexion at load phase'
        },
        'hip_shoulder_alignment': {
            'display_name': 'Body Alignment',
            'unit': 'degrees',
            'description': 'Hip-shoulder rotation difference'
        },
        'base_width': {
            'display_name': 'Stance Width',
            'unit': 'ratio',
            'description': 'Stance width relative to body height'
        },
        'lateral_sway': {
            'display_name': 'Balance/Stability',
            'unit': 'ratio',
            'description': 'Lateral movement during shot'
        },
        'arc_trajectory': {
            'display_name': 'Shot Arc',
            'unit': 'degrees',
            'description': 'Ball trajectory angle'
        }
    }

    for metric_key, config in metric_configs.items():
        metric_data = metrics.get(metric_key, {})

        if not metric_data or 'error' in metric_data:
            formatted[metric_key] = {
                'display_name': config['display_name'],
                'error': metric_data.get('error', 'Not available') if metric_data else 'Not available',
                'quality_score': 0
            }
            continue

        formatted[metric_key] = {
            'display_name': config['display_name'],
            'description': config['description'],
            'quality_score': round(metric_data.get('quality_score', 0), 1),
            'quality_description': metric_data.get('description', 'Unknown'),
            'optimal_range': metric_data.get('optimal_range', []),
            'in_optimal_range': metric_data.get('in_range', False),
            'status': 'good' if metric_data.get('quality_score', 0) >= 7 else 'needs_work'
        }

        # Add metric-specific values
        if 'angle_deg' in metric_data:
            formatted[metric_key]['value'] = round(metric_data['angle_deg'], 1)
            formatted[metric_key]['unit'] = 'degrees'
        elif 'ratio' in metric_data:
            formatted[metric_key]['value'] = round(metric_data['ratio'], 3)
            formatted[metric_key]['unit'] = 'ratio'
        elif 'sway_ratio' in metric_data:
            formatted[metric_key]['value'] = round(metric_data['sway_ratio'], 4)
            formatted[metric_key]['unit'] = 'ratio'
        elif 'arc_angle_deg' in metric_data:
            formatted[metric_key]['value'] = round(metric_data['arc_angle_deg'], 1)
            formatted[metric_key]['unit'] = 'degrees'

    return formatted


@app.post("/analyze/with-overlay")
async def analyze_with_overlay_visualization(
    video: UploadFile = File(...),
    frame_skip: int = Form(default=1)
):
    """
    Analyze shot and generate color-coded overlay video

    Features:
    - Full shot analysis with metrics
    - Color-coded skeleton overlay (green=good, red=poor)
    - Real-time metrics display
    - Phase indicators

    Args:
        video: Video file (mp4, mov, avi, mkv)
        frame_skip: Process every Nth frame (default: 1)

    Returns:
        Analysis results + visualization video download URL
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
        logger.info(f"💾 Saving video: {file_path}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        logger.info(f"✅ Video saved, starting analysis with overlay...")

        # Perform analysis with overlay
        results = analysis_service.analyze_with_overlay(
            video_path=str(file_path),
            frame_skip=frame_skip
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

            logger.warning(f"⚠️ Analysis with overlay failed: {error_response['error']}")
            return JSONResponse(status_code=200, content=error_response, cls=NumpyEncoder)

        # Format response
        response = {
            "video_id": video_id,
            "success": True,
            "analysis_mode": "overlay_visualization",
            "overall_score": round(results['overall_score'], 1),
            "confidence": round(results['confidence'], 2),

            # Overlay video info
            "visualization_video": {
                "path": results.get('overlay_video_path', ''),
                "download_url": f"/download/overlay/{Path(results.get('overlay_video_path', '')).name}" if results.get('overlay_video_path') else None
            },

            # Phases
            "phases": results.get('phases', {}),

            # Metrics summary
            "metrics": _format_metrics(results.get('metrics', {})),

            # Coaching cues
            "coaching_cues": results.get('coaching_cues', []),

            "analyzed_at": datetime.now().isoformat()
        }

        # Clean up original video
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️ Cleaned up original video")

        logger.info(f"✅ Overlay video analysis complete!")

        # Force garbage collection
        gc.collect()

        return JSONResponse(content=response, cls=NumpyEncoder)

    except Exception as e:
        logger.error(f"❌ Overlay analysis error: {str(e)}", exc_info=True)

        # Clean up on error
        if file_path and file_path.exists():
            file_path.unlink()

        gc.collect()

        raise HTTPException(
            status_code=500,
            detail=f"Overlay analysis failed: {str(e)}"
        )


@app.get("/download/overlay/{filename}")
async def download_overlay(filename: str):
    """Download generated overlay video"""
    from pathlib import Path as PathLib
    file_path = PathLib("output/visualizations") / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Overlay video not found")

    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=600,
        limit_concurrency=10,
        limit_max_requests=1000
    )
