"""
Worker Tasks for Async Video Analysis

Contains the heavy processing logic that runs on worker nodes.
"""
import gc
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np

import config
import object_store
import session_store
from services.shot_analysis_service import ShotAnalysisService

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
            if phase_name == 'load' and 'knee_angle' in phase_data:
                formatted[phase_name]['knee_angle'] = phase_data['knee_angle']
        else:
            formatted[phase_name] = None

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

    for metric_key, cfg in metric_configs.items():
        metric_data = metrics.get(metric_key, {})

        if not metric_data or 'error' in metric_data:
            formatted[metric_key] = {
                'display_name': cfg['display_name'],
                'error': metric_data.get('error', 'Not available') if metric_data else 'Not available',
                'quality_score': 0
            }
            continue

        formatted[metric_key] = {
            'display_name': cfg['display_name'],
            'description': cfg['description'],
            'quality_score': round(metric_data.get('quality_score', 0), 1),
            'quality_description': metric_data.get('description', 'Unknown'),
            'optimal_range': metric_data.get('optimal_range', []),
            'in_optimal_range': metric_data.get('in_range', False),
            'status': 'good' if metric_data.get('quality_score', 0) >= 7 else 'needs_work'
        }

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


def process_session(session_id: str) -> None:
    """
    Process a video analysis session.

    This is the main worker task that:
    1. Downloads the video from object storage
    2. Runs the analysis pipeline
    3. Stores results back in the session

    Args:
        session_id: The session UUID to process
    """
    temp_path = None
    overlay_temp_path = None
    timings_ms = {}
    start_time = time.time()

    try:
        # Load session
        session = session_store.get_session(session_id)
        if session is None:
            logger.warning(f"Session not found: {session_id}")
            return

        logger.info(f"Processing session: {session_id}")

        # Update status to extracting
        session_store.update_session(session_id, {"status": session_store.STATUS_PROCESSING_EXTRACT})

        # Download video from object store
        video_object_key = session.get("video_object_key")
        if not video_object_key:
            session_store.set_failed(
                session_id,
                code="BAD_VIDEO",
                message="No video file associated with session",
                tips=["Please upload a video before starting analysis"]
            )
            return

        download_start = time.time()
        try:
            temp_path = object_store.download_to_temp(video_object_key)
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            session_store.set_failed(
                session_id,
                code="BAD_VIDEO",
                message=f"Failed to download video: {str(e)}",
                tips=["Please try uploading the video again"]
            )
            return

        timings_ms["download"] = int((time.time() - download_start) * 1000)

        # Update status to analyzing
        session_store.update_session(session_id, {"status": session_store.STATUS_PROCESSING_ANALYSIS})

        # Get payload settings
        payload = session.get("payload") or {}
        frame_skip = payload.get("frame_skip", config.DEFAULT_FRAME_SKIP)
        analysis_mode = payload.get("analysis_mode", "shot")

        # Run analysis
        analysis_start = time.time()
        analysis_service = ShotAnalysisService()

        if analysis_mode == "overlay":
            results = analysis_service.analyze_with_overlay(
                video_path=temp_path,
                frame_skip=frame_skip
            )
        else:
            results = analysis_service.analyze_shot(
                video_path=temp_path,
                frame_skip=frame_skip
            )

        timings_ms["analysis"] = int((time.time() - analysis_start) * 1000)
        timings_ms["total"] = int((time.time() - start_time) * 1000)

        # Handle analysis failure
        if not results.get('success', False):
            error_code = "LOW_QUALITY"
            if "pose" in results.get('error', '').lower():
                error_code = "NO_POSE"
            elif "video" in results.get('error', '').lower():
                error_code = "BAD_VIDEO"

            session_store.set_failed(
                session_id,
                code=error_code,
                message=results.get('error', 'Unknown analysis error'),
                tips=results.get('tips', [])
            )
            return

        # Convert numpy types
        results = convert_numpy_types(results)

        # Build quality stats
        quality_info = results.get('quality_info', {})
        quality = {
            "confidence": round(quality_info.get('confidence', 0.0), 2),
            "low_confidence_frames": quality_info.get('low_confidence_frames', 0),
            "warning": quality_info.get('warning')
        }

        # Compute landmark coverage if available
        if 'landmark_coverage' in quality_info:
            quality['landmark_coverage'] = quality_info['landmark_coverage']

        # Build response in same format as /analyze/shot
        response = {
            "session_id": session_id,
            "video_id": session_id,  # For backwards compatibility
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
            "quality": quality,

            # Video metadata
            "metadata": {
                "duration_seconds": round(results['metadata']['duration'], 2),
                "fps": round(results['metadata']['fps'], 1),
                "total_frames": results['metadata']['total_frames'],
                "processed_frames": results['metadata']['processed_frames']
            },

            "analyzed_at": datetime.utcnow().isoformat() + "Z"
        }

        # Handle overlay video if present
        overlay_temp_path = None
        if 'overlay_video_path' in results and results['overlay_video_path']:
            overlay_temp_path = results['overlay_video_path']

            # Upload overlay video to object storage
            try:
                overlay_object_key = object_store.make_overlay_key(session_id)
                object_store.upload_file(overlay_temp_path, overlay_object_key, "video/mp4")

                # Generate presigned download URL (valid for 7 days)
                overlay_download_url = object_store.presign_get(overlay_object_key, expires_in=604800)

                response['overlay_video'] = {
                    "object_key": overlay_object_key,
                    "download_url": overlay_download_url
                }

                logger.info(f"Overlay video uploaded: {overlay_object_key}")

            except Exception as e:
                logger.error(f"Failed to upload overlay video: {e}")
                response['overlay_video_error'] = str(e)

        # Ensure JSON serializable
        response = ensure_json_serializable(response)

        # Store results
        session_store.set_done(
            session_id,
            result=response,
            quality=quality,
            timings_ms=timings_ms
        )

        logger.info(f"Session {session_id} completed: score={response['overall_score']}")

    except Exception as e:
        logger.error(f"Session {session_id} failed with exception: {e}", exc_info=True)
        session_store.set_failed(
            session_id,
            code="INTERNAL",
            message=f"Internal processing error: {str(e)}",
            tips=["Please try again. If the problem persists, contact support."]
        )

    finally:
        # Clean up temp files
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

        if overlay_temp_path and os.path.exists(overlay_temp_path):
            try:
                os.unlink(overlay_temp_path)
                logger.info(f"Cleaned up overlay temp file: {overlay_temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up overlay temp file {overlay_temp_path}: {e}")

        # Force garbage collection
        gc.collect()
