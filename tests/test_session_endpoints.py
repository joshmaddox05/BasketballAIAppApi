"""
Tests for the async session pipeline endpoints.

These tests mock Redis and object storage to avoid requiring real infrastructure.

Run with: pytest tests/test_session_endpoints.py -v
"""
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


class MockRedis:
    """Mock Redis client for testing"""

    def __init__(self):
        self._store = {}
        self._ttls = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, ex=None):
        self._store[key] = value
        if ex:
            self._ttls[key] = ex

    def ttl(self, key):
        return self._ttls.get(key, -1)

    def delete(self, key):
        if key in self._store:
            del self._store[key]
        if key in self._ttls:
            del self._ttls[key]

    def ping(self):
        return True


@pytest.fixture
def mock_redis():
    """Create a mock Redis instance"""
    return MockRedis()


@pytest.fixture
def mock_get_redis(mock_redis):
    """Patch session_store.get_redis to return mock"""
    with patch('session_store.get_redis', return_value=mock_redis):
        yield mock_redis


@pytest.fixture
def mock_object_store():
    """Patch object store functions"""
    with patch('object_store._get_client') as mock_client, \
         patch('object_store.presign_put') as mock_presign, \
         patch('object_store.head_object') as mock_head, \
         patch('object_store.download_to_temp') as mock_download, \
         patch('object_store.upload_file') as mock_upload:

        mock_presign.return_value = {
            "upload_url": "https://mock-bucket.s3.amazonaws.com/uploads/test.mp4?presigned",
            "headers": {"Content-Type": "video/mp4"},
            "object_key": "uploads/test.mp4"
        }

        mock_head.return_value = {"exists": True, "size": 1024000}

        yield {
            "client": mock_client,
            "presign_put": mock_presign,
            "head_object": mock_head,
            "download_to_temp": mock_download,
            "upload_file": mock_upload
        }


@pytest.fixture
def mock_queueing():
    """Patch queueing functions"""
    with patch('queueing.get_queue') as mock_queue, \
         patch('queueing.enqueue_analysis') as mock_enqueue:

        mock_job = MagicMock()
        mock_job.id = "test-job-id"
        mock_enqueue.return_value = mock_job

        yield {
            "get_queue": mock_queue,
            "enqueue_analysis": mock_enqueue
        }


@pytest.fixture
def client(mock_get_redis, mock_object_store, mock_queueing):
    """Create test client with all mocks"""
    # Need to import main after patches are applied
    import main
    return TestClient(main.app)


class TestCreateSession:
    """Test POST /analysis-sessions"""

    def test_create_session_success(self, client, mock_get_redis):
        """Should create a new session and return session_id"""
        response = client.post("/analysis-sessions")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "session_id" in data
        assert data["status"] == "CREATED"

        # Session should be stored in Redis
        session_key = f"session:{data['session_id']}"
        stored = mock_get_redis.get(session_key)
        assert stored is not None

        stored_data = json.loads(stored)
        assert stored_data["id"] == data["session_id"]
        assert stored_data["status"] == "CREATED"


class TestUploadUrl:
    """Test POST /analysis-sessions/{session_id}/upload-url"""

    def test_get_upload_url_success(self, client, mock_get_redis, mock_object_store):
        """Should return a presigned upload URL"""
        # First create a session
        create_resp = client.post("/analysis-sessions")
        session_id = create_resp.json()["session_id"]

        # Request upload URL
        response = client.post(
            f"/analysis-sessions/{session_id}/upload-url",
            json={"filename": "my_shot.mp4", "content_type": "video/mp4"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["session_id"] == session_id
        assert data["status"] == "UPLOAD_URL_ISSUED"
        assert "upload_url" in data
        assert "headers" in data
        assert "object_key" in data

    def test_get_upload_url_session_not_found(self, client):
        """Should return 404 for non-existent session"""
        response = client.post(
            "/analysis-sessions/non-existent-session/upload-url",
            json={"filename": "shot.mp4", "content_type": "video/mp4"}
        )

        assert response.status_code == 404

    def test_get_upload_url_invalid_content_type(self, client, mock_get_redis):
        """Should reject non-video content types"""
        # First create a session
        create_resp = client.post("/analysis-sessions")
        session_id = create_resp.json()["session_id"]

        response = client.post(
            f"/analysis-sessions/{session_id}/upload-url",
            json={"filename": "file.txt", "content_type": "text/plain"}
        )

        assert response.status_code == 400
        assert "video" in response.json()["detail"].lower()


class TestStartAnalysis:
    """Test POST /analysis-sessions/{session_id}/start"""

    def test_start_analysis_success(self, client, mock_get_redis, mock_object_store, mock_queueing):
        """Should enqueue analysis job and return QUEUED status"""
        # Create session
        create_resp = client.post("/analysis-sessions")
        session_id = create_resp.json()["session_id"]

        # Get upload URL (this updates session with video_object_key)
        client.post(
            f"/analysis-sessions/{session_id}/upload-url",
            json={"filename": "shot.mp4", "content_type": "video/mp4"}
        )

        # Start analysis
        response = client.post(
            f"/analysis-sessions/{session_id}/start",
            json={"analysis_mode": "shot", "frame_skip": 1}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["session_id"] == session_id
        assert data["status"] == "QUEUED"

        # Verify job was enqueued
        mock_queueing["enqueue_analysis"].assert_called_once_with(session_id)

    def test_start_analysis_no_video(self, client, mock_get_redis):
        """Should return 400 if no video uploaded"""
        # Create session but don't upload
        create_resp = client.post("/analysis-sessions")
        session_id = create_resp.json()["session_id"]

        response = client.post(
            f"/analysis-sessions/{session_id}/start",
            json={"analysis_mode": "shot", "frame_skip": 1}
        )

        assert response.status_code == 400
        assert "upload" in response.json()["detail"].lower()

    def test_start_analysis_session_not_found(self, client):
        """Should return 404 for non-existent session"""
        response = client.post(
            "/analysis-sessions/non-existent-session/start",
            json={"analysis_mode": "shot", "frame_skip": 1}
        )

        assert response.status_code == 404


class TestGetSession:
    """Test GET /analysis-sessions/{session_id}"""

    def test_get_session_success(self, client, mock_get_redis):
        """Should return session data"""
        # Create session
        create_resp = client.post("/analysis-sessions")
        session_id = create_resp.json()["session_id"]

        response = client.get(f"/analysis-sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "session" in data
        assert data["session"]["id"] == session_id
        assert data["session"]["status"] == "CREATED"

    def test_get_session_not_found(self, client):
        """Should return 404 for non-existent session"""
        response = client.get("/analysis-sessions/non-existent-session")

        assert response.status_code == 404


class TestSessionStatusTransitions:
    """Test that session status transitions correctly"""

    def test_full_status_flow(self, client, mock_get_redis, mock_object_store, mock_queueing):
        """Test complete session lifecycle"""
        # Step 1: Create session
        resp1 = client.post("/analysis-sessions")
        session_id = resp1.json()["session_id"]

        get_resp = client.get(f"/analysis-sessions/{session_id}")
        assert get_resp.json()["session"]["status"] == "CREATED"

        # Step 2: Get upload URL
        resp2 = client.post(
            f"/analysis-sessions/{session_id}/upload-url",
            json={"filename": "shot.mp4", "content_type": "video/mp4"}
        )

        get_resp = client.get(f"/analysis-sessions/{session_id}")
        assert get_resp.json()["session"]["status"] == "UPLOAD_URL_ISSUED"
        assert get_resp.json()["session"]["video_object_key"] is not None

        # Step 3: Start analysis (simulates video upload completed)
        resp3 = client.post(
            f"/analysis-sessions/{session_id}/start",
            json={"analysis_mode": "shot", "frame_skip": 1}
        )

        get_resp = client.get(f"/analysis-sessions/{session_id}")
        assert get_resp.json()["session"]["status"] == "QUEUED"
        assert get_resp.json()["session"]["payload"] is not None


class TestSessionStore:
    """Unit tests for session_store module"""

    def test_create_session_generates_uuid(self, mock_get_redis):
        """Session ID should be a valid UUID"""
        import session_store

        session = session_store.create_session()

        assert "id" in session
        assert len(session["id"]) == 36  # UUID format
        assert session["status"] == "CREATED"
        assert session["created_at"] is not None

    def test_create_session_with_payload(self, mock_get_redis):
        """Payload should be stored in session"""
        import session_store

        payload = {"analysis_mode": "overlay", "frame_skip": 2}
        session = session_store.create_session(payload=payload)

        assert session["payload"] == payload

    def test_update_session(self, mock_get_redis):
        """Session updates should be persisted"""
        import session_store

        session = session_store.create_session()
        session_id = session["id"]

        session_store.update_session(session_id, {"status": "PROCESSING_ANALYSIS"})

        updated = session_store.get_session(session_id)
        assert updated["status"] == "PROCESSING_ANALYSIS"
        assert updated["updated_at"] != session["created_at"]

    def test_set_done(self, mock_get_redis):
        """set_done should store result and update status"""
        import session_store

        session = session_store.create_session()
        session_id = session["id"]

        result = {"overall_score": 85.5, "success": True}
        quality = {"confidence": 0.92}
        timings = {"download": 500, "analysis": 3000}

        session_store.set_done(session_id, result, quality, timings)

        final = session_store.get_session(session_id)
        assert final["status"] == "DONE"
        assert final["result"] == result
        assert final["quality"] == quality
        assert final["timings_ms"] == timings

    def test_set_failed(self, mock_get_redis):
        """set_failed should store error and update status"""
        import session_store

        session = session_store.create_session()
        session_id = session["id"]

        session_store.set_failed(
            session_id,
            code="NO_POSE",
            message="Could not detect pose in video",
            tips=["Use better lighting", "Stand closer to camera"]
        )

        final = session_store.get_session(session_id)
        assert final["status"] == "FAILED"
        assert final["error"]["code"] == "NO_POSE"
        assert final["error"]["message"] == "Could not detect pose in video"
        assert len(final["error"]["tips"]) == 2

    def test_get_nonexistent_session(self, mock_get_redis):
        """Getting non-existent session should return None"""
        import session_store

        result = session_store.get_session("nonexistent-id")
        assert result is None


class TestObjectStore:
    """Unit tests for object_store module"""

    def test_sanitize_filename_valid_extensions(self):
        """Should accept valid video extensions"""
        import object_store

        assert object_store.sanitize_filename("video.mp4") == ".mp4"
        assert object_store.sanitize_filename("video.mov") == ".mov"
        assert object_store.sanitize_filename("video.avi") == ".avi"
        assert object_store.sanitize_filename("video.mkv") == ".mkv"
        assert object_store.sanitize_filename("video.webm") == ".webm"

    def test_sanitize_filename_invalid_extensions(self):
        """Should default to .mp4 for invalid extensions"""
        import object_store

        assert object_store.sanitize_filename("file.txt") == ".mp4"
        assert object_store.sanitize_filename("file.pdf") == ".mp4"
        assert object_store.sanitize_filename("noextension") == ".mp4"

    def test_make_object_key(self):
        """Should generate correct object key format"""
        import object_store

        key = object_store.make_object_key("abc-123", "shot.mp4")
        assert key == "uploads/abc-123.mp4"

        key = object_store.make_object_key("xyz-789", "video.mov")
        assert key == "uploads/xyz-789.mov"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
