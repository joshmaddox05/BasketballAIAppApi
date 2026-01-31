"""
Object Storage Module (S3/R2 Compatible)

Provides presigned URL generation and file operations for video storage.
Supports AWS S3, Cloudflare R2, and other S3-compatible storage providers.
"""
import logging
import os
import re
import tempfile
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig

import config

logger = logging.getLogger(__name__)


def _get_client():
    """
    Get a configured boto3 S3 client.

    Raises:
        ValueError: If required object store config is missing
    """
    if not config.OBJECT_STORE_BUCKET:
        raise ValueError("Object store not configured: OBJECT_STORE_BUCKET is required")

    if not config.OBJECT_STORE_ACCESS_KEY_ID or not config.OBJECT_STORE_SECRET_ACCESS_KEY:
        raise ValueError("Object store not configured: credentials are required")

    client_kwargs = {
        "service_name": "s3",
        "region_name": config.OBJECT_STORE_REGION,
        "aws_access_key_id": config.OBJECT_STORE_ACCESS_KEY_ID,
        "aws_secret_access_key": config.OBJECT_STORE_SECRET_ACCESS_KEY,
    }

    # For R2 and other S3-compatible providers
    if config.OBJECT_STORE_ENDPOINT_URL:
        client_kwargs["endpoint_url"] = config.OBJECT_STORE_ENDPOINT_URL

    # Force path style for providers that require it (e.g., MinIO)
    if config.OBJECT_STORE_FORCE_PATH_STYLE:
        client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

    return boto3.client(**client_kwargs)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename and extract a safe extension.

    Args:
        filename: Original filename from upload

    Returns:
        Safe extension (e.g., ".mp4")
    """
    # Get extension, default to .mp4
    if "." in filename:
        ext = os.path.splitext(filename)[1].lower()
        # Only allow known video extensions
        if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            return ext
    return ".mp4"


def make_object_key(session_id: str, filename: str) -> str:
    """
    Generate an object key for the video file.

    Args:
        session_id: The session UUID
        filename: Original filename

    Returns:
        Object key in format "uploads/{session_id}{ext}"
    """
    ext = sanitize_filename(filename)
    return f"uploads/{session_id}{ext}"


def presign_put(object_key: str, content_type: str) -> Dict[str, Any]:
    """
    Generate a presigned PUT URL for uploading a video.

    Args:
        object_key: The S3 object key
        content_type: MIME type of the video

    Returns:
        Dict with upload_url, headers, and object_key
    """
    client = _get_client()

    presigned_url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": config.OBJECT_STORE_BUCKET,
            "Key": object_key,
            "ContentType": content_type
        },
        ExpiresIn=config.PRESIGN_EXPIRE_SECONDS
    )

    logger.info(f"Generated presigned PUT URL for: {object_key}")

    return {
        "upload_url": presigned_url,
        "headers": {
            "Content-Type": content_type
        },
        "object_key": object_key
    }


def head_object(object_key: str) -> Dict[str, Any]:
    """
    Check if an object exists and get its metadata.

    Args:
        object_key: The S3 object key

    Returns:
        Dict with exists (bool) and size (int, if exists)
    """
    client = _get_client()

    try:
        response = client.head_object(
            Bucket=config.OBJECT_STORE_BUCKET,
            Key=object_key
        )
        return {
            "exists": True,
            "size": response.get("ContentLength", 0)
        }
    except client.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404" or error_code == "NoSuchKey":
            return {"exists": False, "size": 0}
        raise


def download_to_temp(object_key: str) -> str:
    """
    Download an object to a temporary file.

    Args:
        object_key: The S3 object key

    Returns:
        Path to the temporary file (caller must clean up)
    """
    client = _get_client()

    # Determine extension from object key
    ext = os.path.splitext(object_key)[1] or ".mp4"

    # Create temp file (delete=False so worker can manage cleanup)
    temp_file = tempfile.NamedTemporaryFile(
        suffix=ext,
        prefix="video_",
        delete=False
    )
    temp_path = temp_file.name
    temp_file.close()

    logger.info(f"Downloading {object_key} to {temp_path}")

    client.download_file(
        Bucket=config.OBJECT_STORE_BUCKET,
        Key=object_key,
        Filename=temp_path
    )

    logger.info(f"Downloaded {object_key} successfully")
    return temp_path


def upload_file(file_path: str, object_key: str, content_type: str = "video/mp4") -> None:
    """
    Upload a file to object storage.

    Args:
        file_path: Local path to the file
        object_key: Target S3 object key
        content_type: MIME type of the file
    """
    client = _get_client()

    logger.info(f"Uploading {file_path} to {object_key}")

    client.upload_file(
        Filename=file_path,
        Bucket=config.OBJECT_STORE_BUCKET,
        Key=object_key,
        ExtraArgs={"ContentType": content_type}
    )

    logger.info(f"Uploaded {object_key} successfully")