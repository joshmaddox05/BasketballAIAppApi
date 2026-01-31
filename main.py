"""
Basketball Shot Form Analysis API
AI-powered shooting form analysis with personalized coaching feedback
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
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
import session_store
import object_store
import queueing
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models for Async Session Pipeline
# =============================================================================

class UploadUrlRequest(BaseModel):
    """Request body for generating a presigned upload URL."""
    filename: str
    content_type: str = "video/mp4"


class StartRequest(BaseModel):
    """Request body for starting analysis on an uploaded video."""
    analysis_mode: str = "shot"  # "shot" or "overlay" (phase 2)
    frame_skip: int = 1


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


# =============================================================================
# Async Session Pipeline Endpoints
# =============================================================================

@app.post("/analysis-sessions")
async def create_analysis_session():
    """
    Create a new analysis session.

    This is step 1 of the async flow:
    1. Create session -> returns session_id
    2. Get upload URL -> returns presigned PUT URL
    3. PUT video to upload URL
    4. Start analysis -> enqueues job
    5. Poll session for results

    Returns:
        Session creation response with session_id and status
    """
    try:
        session = session_store.create_session()
        return {
            "success": True,
            "session_id": session["id"],
            "status": session["status"]
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/analysis-sessions/{session_id}/upload-url")
async def get_upload_url(session_id: str, request: UploadUrlRequest):
    """
    Generate a presigned URL for uploading a video.

    Step 2 of the async flow. Client uses the returned URL to PUT the video directly
    to object storage, bypassing the API server.

    Args:
        session_id: The session UUID
        request: Contains filename and content_type

    Returns:
        Presigned upload URL and required headers
    """
    # Validate session exists
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate content type
    if not request.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Must be a video format (e.g., video/mp4)"
        )

    try:
        # Generate object key and presigned URL
        object_key = object_store.make_object_key(session_id, request.filename)
        presigned = object_store.presign_put(object_key, request.content_type)

        # Update session with video info
        session_store.update_session(session_id, {
            "status": session_store.STATUS_UPLOAD_URL_ISSUED,
            "video_object_key": object_key,
            "video_content_type": request.content_type
        })

        return {
            "success": True,
            "session_id": session_id,
            "status": session_store.STATUS_UPLOAD_URL_ISSUED,
            **presigned
        }

    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/analysis-sessions/{session_id}/start")
async def start_analysis(session_id: str, request: StartRequest):
    """
    Start video analysis for a session.

    Step 4 of the async flow. This enqueues the analysis job for background
    processing. Poll GET /analysis-sessions/{session_id} for results.

    Args:
        session_id: The session UUID
        request: Analysis configuration (mode, frame_skip)

    Returns:
        Confirmation that analysis was queued
    """
    # Load session
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate video was uploaded
    video_object_key = session.get("video_object_key")
    if not video_object_key:
        raise HTTPException(
            status_code=400,
            detail="No video uploaded. Please upload a video first using the presigned URL."
        )

    # Verify video exists in object store
    try:
        head_result = object_store.head_object(video_object_key)
        if not head_result.get("exists"):
            raise HTTPException(
                status_code=400,
                detail="Video file not found. Please upload the video using the presigned URL."
            )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Update session and enqueue job
    try:
        session_store.update_session(session_id, {
            "status": session_store.STATUS_QUEUED,
            "payload": request.dict()
        })

        queueing.enqueue_analysis(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "status": session_store.STATUS_QUEUED
        }

    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/analysis-sessions/{session_id}")
async def get_analysis_session(session_id: str):
    """
    Get the current state of an analysis session.

    Poll this endpoint to check analysis progress and retrieve results.
    Status progression: CREATED -> UPLOAD_URL_ISSUED -> QUEUED ->
                        PROCESSING_EXTRACT -> PROCESSING_ANALYSIS -> DONE/FAILED

    Args:
        session_id: The session UUID

    Returns:
        Full session state including status, results (if done), or error (if failed)
    """
    try:
        session = session_store.get_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "success": True,
        "session": session
    }


# =============================================================================
# Synchronous Analysis Endpoints (with async_mode option)
# =============================================================================


@app.post("/analyze/shot")
async def analyze_shot(
    video: UploadFile = File(...),
    frame_skip: int = Form(default=1),
    async_mode: bool = Form(default=False)
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
        async_mode: If True, enqueue for background processing and return session_id (HTTP 202)

    Returns:
        Comprehensive shot analysis with actionable coaching feedback,
        or session_id if async_mode=True
    """
    file_path = None

    try:
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

        # ASYNC MODE: Upload to object store and enqueue job
        if async_mode:
            try:
                # Create session
                session = session_store.create_session(payload={
                    "analysis_mode": "shot",
                    "frame_skip": frame_skip
                })
                session_id = session["id"]

                # Determine content type
                content_type = video.content_type or "video/mp4"

                # Generate object key and upload
                object_key = object_store.make_object_key(session_id, video.filename)
                object_store.upload_file(str(file_path), object_key, content_type)

                # Update session with video info
                session_store.update_session(session_id, {
                    "status": session_store.STATUS_QUEUED,
                    "video_object_key": object_key,
                    "video_content_type": content_type
                })

                # Enqueue the job
                queueing.enqueue_analysis(session_id)

                # Clean up local file
                if file_path and file_path.exists():
                    file_path.unlink()

                logger.info(f"Async analysis queued: session_id={session_id}")

                return JSONResponse(
                    status_code=202,
                    content={
                        "success": True,
                        "session_id": session_id,
                        "status": session_store.STATUS_QUEUED,
                        "message": "Analysis queued. Poll GET /analysis-sessions/{session_id} for results."
                    }
                )

            except ValueError as e:
                # Object store or Redis not configured
                logger.warning(f"Async mode unavailable: {e}. Falling back to sync.")
                # Fall through to synchronous processing

        # SYNCHRONOUS MODE: Process immediately
        analysis_service = get_shot_analysis_service()

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
                "confidence": round(results['quality_info'].get('confidence', 0.0), 2),
                "low_confidence_frames": results['quality_info'].get('low_confidence_frames', 0),
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
    frame_skip: int = Form(default=1),
    async_mode: bool = Form(default=False)
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
        async_mode: If True, enqueue for background processing and return session_id (HTTP 202)

    Returns:
        Analysis results + visualization video download URL,
        or session_id if async_mode=True
    """
    file_path = None

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
        logger.info(f"Saving video: {file_path}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # ASYNC MODE: Upload to object store and enqueue job
        if async_mode:
            try:
                # Create session
                session = session_store.create_session(payload={
                    "analysis_mode": "overlay",
                    "frame_skip": frame_skip
                })
                session_id = session["id"]

                # Determine content type
                content_type = video.content_type or "video/mp4"

                # Generate object key and upload
                object_key = object_store.make_object_key(session_id, video.filename)
                object_store.upload_file(str(file_path), object_key, content_type)

                # Update session with video info
                session_store.update_session(session_id, {
                    "status": session_store.STATUS_QUEUED,
                    "video_object_key": object_key,
                    "video_content_type": content_type
                })

                # Enqueue the job
                queueing.enqueue_analysis(session_id)

                # Clean up local file
                if file_path and file_path.exists():
                    file_path.unlink()

                logger.info(f"Async overlay analysis queued: session_id={session_id}")

                return JSONResponse(
                    status_code=202,
                    content={
                        "success": True,
                        "session_id": session_id,
                        "status": session_store.STATUS_QUEUED,
                        "message": "Overlay analysis queued. Poll GET /analysis-sessions/{session_id} for results."
                    }
                )

            except ValueError as e:
                # Object store or Redis not configured
                logger.warning(f"Async mode unavailable: {e}. Falling back to sync.")
                # Fall through to synchronous processing

        # SYNCHRONOUS MODE: Process immediately
        # Get analysis service
        analysis_service = get_shot_analysis_service()

        logger.info(f"Video saved, starting analysis with overlay...")

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

            logger.warning(f"Analysis with overlay failed: {error_response['error']}")
            return JSONResponse(status_code=200, content=error_response)

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
            logger.info(f"Cleaned up original video")

        logger.info(f"Overlay video analysis complete!")

        # Force garbage collection
        gc.collect()

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Overlay analysis error: {str(e)}", exc_info=True)

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
