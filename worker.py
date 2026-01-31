#!/usr/bin/env python3
"""
RQ Worker Process

Run this script on Render Worker service or any dedicated worker node.
Connects to Redis and processes analysis jobs from the queue.

Usage:
    python worker.py
"""
import logging
import sys

from redis import Redis
from rq import Worker

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Start the RQ worker process."""
    if not config.REDIS_URL:
        logger.error("REDIS_URL environment variable is not set")
        sys.exit(1)

    logger.info(f"Connecting to Redis...")
    logger.info(f"Queue name: {config.RQ_QUEUE_NAME}")

    try:
        connection = Redis.from_url(config.REDIS_URL)

        # Test connection
        connection.ping()
        logger.info("Connected to Redis successfully")

        # Create and start worker
        worker = Worker(
            queues=[config.RQ_QUEUE_NAME],
            connection=connection,
            name=None,  # Auto-generate worker name
            default_worker_ttl=420,  # 7 minutes
            job_monitoring_interval=5
        )

        logger.info(f"Starting worker for queue: {config.RQ_QUEUE_NAME}")
        worker.work(with_scheduler=False)

    except Exception as e:
        logger.error(f"Worker failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
