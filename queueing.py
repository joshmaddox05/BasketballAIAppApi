"""
RQ (Redis Queue) Task Queueing Module

Manages job queuing for async video analysis processing.
"""
import logging
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job

import config

logger = logging.getLogger(__name__)


def get_queue() -> Queue:
    """
    Get the RQ queue for analysis jobs.

    Returns:
        Configured RQ Queue instance

    Raises:
        ValueError: If REDIS_URL is not configured
    """
    if not config.REDIS_URL:
        raise ValueError("Queue not configured: REDIS_URL environment variable is required")

    connection = Redis.from_url(config.REDIS_URL)
    return Queue(name=config.RQ_QUEUE_NAME, connection=connection)


def enqueue_analysis(session_id: str) -> Job:
    """
    Enqueue a video analysis job for processing.

    Args:
        session_id: The session UUID to process

    Returns:
        The enqueued RQ Job
    """
    queue = get_queue()

    # Import here to avoid circular imports
    import worker_tasks

    job = queue.enqueue(
        worker_tasks.process_session,
        session_id,
        job_timeout="10m",  # 10 minute timeout for video processing
        result_ttl=3600,    # Keep result for 1 hour
        failure_ttl=86400   # Keep failed job info for 24 hours
    )

    logger.info(f"Enqueued analysis job for session {session_id}: job_id={job.id}")
    return job