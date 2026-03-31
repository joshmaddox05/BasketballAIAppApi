"""
Redis-backed Session Storage for Async Analysis Pipeline

Manages session state for video analysis jobs with TTL-based expiration.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import redis

import config

logger = logging.getLogger(__name__)

# Session status constants
STATUS_CREATED = "CREATED"
STATUS_UPLOAD_URL_ISSUED = "UPLOAD_URL_ISSUED"
STATUS_QUEUED = "QUEUED"
STATUS_PROCESSING_EXTRACT = "PROCESSING_EXTRACT"
STATUS_PROCESSING_ANALYSIS = "PROCESSING_ANALYSIS"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"


def get_redis() -> redis.Redis:
    """
    Get a Redis connection from the configured REDIS_URL.

    Raises:
        ValueError: If REDIS_URL is not configured
    """
    if not config.REDIS_URL:
        raise ValueError("Redis not configured: REDIS_URL environment variable is required")

    return redis.from_url(config.REDIS_URL, decode_responses=True)


def _session_key(session_id: str) -> str:
    """Generate the Redis key for a session."""
    return f"session:{session_id}"


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def create_session(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a new session in Redis.

    Args:
        payload: Optional initial payload data

    Returns:
        The created session dict
    """
    r = get_redis()
    session_id = str(uuid4())
    now = _now_iso()

    session = {
        "id": session_id,
        "status": STATUS_CREATED,
        "created_at": now,
        "updated_at": now,
        "video_object_key": None,
        "video_content_type": None,
        "payload": payload,
        "result": None,
        "error": None,
        "timings_ms": None,
        "quality": None
    }

    key = _session_key(session_id)
    r.set(key, json.dumps(session), ex=config.SESSION_TTL_SECONDS)

    logger.info(f"Created session: {session_id}")
    return session


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a session from Redis.

    Args:
        session_id: The session UUID

    Returns:
        The session dict, or None if not found
    """
    r = get_redis()
    key = _session_key(session_id)
    data = r.get(key)

    if data is None:
        return None

    return json.loads(data)


def update_session(session_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update a session with the provided patch data.

    Args:
        session_id: The session UUID
        patch: Dict of fields to update

    Returns:
        The updated session dict, or None if session not found
    """
    r = get_redis()
    key = _session_key(session_id)

    data = r.get(key)
    if data is None:
        logger.warning(f"Session not found for update: {session_id}")
        return None

    session = json.loads(data)
    session.update(patch)
    session["updated_at"] = _now_iso()

    # Preserve remaining TTL
    ttl = r.ttl(key)
    if ttl < 0:
        ttl = config.SESSION_TTL_SECONDS

    r.set(key, json.dumps(session), ex=ttl)

    logger.info(f"Updated session {session_id}: status={session.get('status')}")
    return session


def set_failed(
    session_id: str,
    code: str,
    message: str,
    tips: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Mark a session as failed with error details.

    Args:
        session_id: The session UUID
        code: Error code (e.g., "NO_POSE", "LOW_QUALITY", "BAD_VIDEO", "INTERNAL")
        message: Human-readable error message
        tips: Optional list of tips for the user
        extra: Optional additional error data

    Returns:
        The updated session dict, or None if session not found
    """
    error = {
        "code": code,
        "message": message,
        "tips": tips or []
    }
    if extra:
        error.update(extra)

    patch = {
        "status": STATUS_FAILED,
        "error": error
    }

    logger.error(f"Session {session_id} failed: {code} - {message}")
    return update_session(session_id, patch)


def set_done(
    session_id: str,
    result: Dict[str, Any],
    quality: Optional[Dict[str, Any]] = None,
    timings_ms: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Mark a session as done with analysis results.

    Args:
        session_id: The session UUID
        result: The analysis result dict
        quality: Optional quality metrics
        timings_ms: Optional timing information

    Returns:
        The updated session dict, or None if session not found
    """
    patch = {
        "status": STATUS_DONE,
        "result": result
    }

    if quality is not None:
        patch["quality"] = quality

    if timings_ms is not None:
        patch["timings_ms"] = timings_ms

    logger.info(f"Session {session_id} completed successfully")
    return update_session(session_id, patch)