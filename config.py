"""
Centralized Configuration for Basketball Shot Analysis API
All tunable parameters in one place for easy adjustment.
"""
import os

# =============================================================================
# CONCURRENCY & RESOURCE LIMITS
# =============================================================================

# Maximum concurrent video analyses (increase for more throughput, decrease for memory safety)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

# Maximum queue size for waiting requests (0 = reject immediately when at capacity)
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "10"))

# Queue timeout in seconds (how long a request waits before 503)
QUEUE_TIMEOUT_SECONDS = float(os.getenv("QUEUE_TIMEOUT_SECONDS", "30.0"))

# =============================================================================
# VIDEO CONSTRAINTS
# =============================================================================

# Maximum video duration in seconds (longer videos are truncated/sampled)
MAX_VIDEO_DURATION_SECONDS = float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "15.0"))

# Maximum frames to process (safety limit)
MAX_FRAMES_TO_PROCESS = int(os.getenv("MAX_FRAMES_TO_PROCESS", "500"))

# Maximum upload size in bytes (100MB default)
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024)))

# =============================================================================
# MEDIAPIPE POSE DETECTION
# =============================================================================

# Model complexity: 0=Lite (fastest), 1=Full (balanced), 2=Heavy (most accurate)
MODEL_COMPLEXITY = int(os.getenv("MODEL_COMPLEXITY", "1"))

# Initial detection confidence (0.0-1.0) - lower = more detections but more noise
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5"))

# Initial tracking confidence (0.0-1.0) - lower = better tracking but more drift
MIN_TRACKING_CONFIDENCE = float(os.getenv("MIN_TRACKING_CONFIDENCE", "0.5"))

# Retry thresholds (used when initial pass fails)
RETRY_DETECTION_CONFIDENCE = float(os.getenv("RETRY_DETECTION_CONFIDENCE", "0.3"))
RETRY_TRACKING_CONFIDENCE = float(os.getenv("RETRY_TRACKING_CONFIDENCE", "0.3"))

# Minimum visibility to include a keypoint (0.0-1.0)
MIN_KEYPOINT_VISIBILITY = float(os.getenv("MIN_KEYPOINT_VISIBILITY", "0.3"))

# Frame-level confidence threshold for "low confidence" count
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.4"))

# =============================================================================
# ANALYSIS THRESHOLDS
# =============================================================================

# Minimum overall confidence to proceed with analysis (below = failure)
MIN_ANALYSIS_CONFIDENCE = float(os.getenv("MIN_ANALYSIS_CONFIDENCE", "0.30"))

# Minimum frames with pose detection to proceed
MIN_FRAMES_WITH_POSE = int(os.getenv("MIN_FRAMES_WITH_POSE", "10"))

# Minimum landmark coverage to proceed (fraction of expected landmarks present)
MIN_LANDMARK_COVERAGE = float(os.getenv("MIN_LANDMARK_COVERAGE", "0.5"))

# =============================================================================
# FRAME SKIP SETTINGS (for performance)
# =============================================================================

# Default frame skip (1 = process every frame)
DEFAULT_FRAME_SKIP = int(os.getenv("DEFAULT_FRAME_SKIP", "1"))

# Adaptive frame skip thresholds
HIGH_FPS_THRESHOLD = float(os.getenv("HIGH_FPS_THRESHOLD", "60.0"))  # FPS above this uses higher skip
LONG_VIDEO_THRESHOLD = float(os.getenv("LONG_VIDEO_THRESHOLD", "10.0"))  # Seconds above this uses higher skip
ADAPTIVE_FRAME_SKIP_HIGH_FPS = int(os.getenv("ADAPTIVE_FRAME_SKIP_HIGH_FPS", "2"))
ADAPTIVE_FRAME_SKIP_LONG_VIDEO = int(os.getenv("ADAPTIVE_FRAME_SKIP_LONG_VIDEO", "2"))

# =============================================================================
# DEBUG & LOGGING
# =============================================================================

# Enable debug timing in responses
DEBUG_TIMING = os.getenv("DEBUG_TIMING", "false").lower() == "true"

# Log every Nth frame during processing (0 = disabled)
LOG_FRAME_INTERVAL = int(os.getenv("LOG_FRAME_INTERVAL", "30"))

# =============================================================================
# LANDMARK DEFINITIONS
# =============================================================================

# Core landmarks required for basic analysis
CORE_LANDMARKS = [
    'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    'right_knee', 'left_ankle', 'right_ankle'
]

# Extended landmarks for enhanced analysis
EXTENDED_LANDMARKS = [
    'left_pinky', 'right_pinky', 'left_thumb', 'right_thumb',
    'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]

# All tracked landmarks
ALL_LANDMARKS = CORE_LANDMARKS + EXTENDED_LANDMARKS

# MediaPipe landmark index mapping
LANDMARK_INDICES = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# Reverse mapping: index -> name
LANDMARK_NAMES = {v: k for k, v in LANDMARK_INDICES.items()}

# =============================================================================
# ASYNC SESSION + QUEUE
# =============================================================================

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "604800"))
RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "analysis")
SYNC_WAIT_SECONDS = float(os.getenv("SYNC_WAIT_SECONDS", "0"))  # for legacy endpoints if we want to wait briefly

# =============================================================================
# OBJECT STORAGE (S3/R2)
# =============================================================================

OBJECT_STORE_BUCKET = os.getenv("OBJECT_STORE_BUCKET", "")
OBJECT_STORE_REGION = os.getenv("OBJECT_STORE_REGION", "us-east-1")
OBJECT_STORE_ENDPOINT_URL = os.getenv("OBJECT_STORE_ENDPOINT_URL", "")
OBJECT_STORE_ACCESS_KEY_ID = os.getenv("OBJECT_STORE_ACCESS_KEY_ID", "")
OBJECT_STORE_SECRET_ACCESS_KEY = os.getenv("OBJECT_STORE_SECRET_ACCESS_KEY", "")
OBJECT_STORE_FORCE_PATH_STYLE = os.getenv("OBJECT_STORE_FORCE_PATH_STYLE", "false").lower() == "true"
PRESIGN_EXPIRE_SECONDS = int(os.getenv("PRESIGN_EXPIRE_SECONDS", "900"))

# =============================================================================
# REDIS
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "")
