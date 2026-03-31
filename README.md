# Basketball AI Backend

FastAPI backend service providing AI-powered basketball shot analysis using MediaPipe and OpenCV.

## Features

- Shot analysis using MediaPipe pose detection
- Video processing with OpenCV
- Stephen Curry shooting form comparison
- RESTful API endpoints
- Real-time pose tracking and feedback
- Comprehensive shooting metrics

## Tech Stack

- **FastAPI** 0.104.1 - Modern web framework
- **MediaPipe** 0.10.8 - Pose detection and tracking
- **OpenCV** 4.8.1.78 - Video processing
- **NumPy** 1.24.3 - Numerical computations
- **Python** 3.9+

## Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/joshmaddox05/BasketballAIAppApi.git
   cd BasketballAIAppApi
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

1. **Start the server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API:**
   - API Base: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## API Endpoints

### Health Check
```http
GET /health
```
Returns server status.

### Shot Analysis (Synchronous)
```http
POST /analyze/shot
Content-Type: multipart/form-data

Body:
  video: Video file (MP4, MOV, AVI, MKV)
  frame_skip: int (optional, default: 1)
  async_mode: bool (optional, default: false)
```

Returns comprehensive shot analysis with metrics and coaching feedback.

### Shot Analysis with Overlay
```http
POST /analyze/with-overlay
Content-Type: multipart/form-data

Body:
  video: Video file (MP4, MOV, AVI, MKV)
  frame_skip: int (optional, default: 1)
  async_mode: bool (optional, default: false)
```

Returns analysis plus a color-coded overlay video visualization.

---

## Async Session Flow (Recommended for Mobile)

The async session pipeline allows clients to upload videos directly to object storage and poll for results, reducing API server load and improving reliability.

### Step 1: Create Session
```http
POST /analysis-sessions

Response:
{
  "success": true,
  "session_id": "uuid",
  "status": "CREATED"
}
```

### Step 2: Get Upload URL
```http
POST /analysis-sessions/{session_id}/upload-url
Content-Type: application/json

{
  "filename": "shot.mp4",
  "content_type": "video/mp4"
}

Response:
{
  "success": true,
  "session_id": "uuid",
  "status": "UPLOAD_URL_ISSUED",
  "upload_url": "https://...",
  "headers": {"Content-Type": "video/mp4"},
  "object_key": "uploads/uuid.mp4"
}
```

### Step 3: Upload Video
```http
PUT {upload_url}
Content-Type: video/mp4

Body: raw video bytes
```

Upload directly to object storage using the presigned URL.

### Step 4: Start Analysis
```http
POST /analysis-sessions/{session_id}/start
Content-Type: application/json

{
  "analysis_mode": "shot",
  "frame_skip": 1
}

Response:
{
  "success": true,
  "session_id": "uuid",
  "status": "QUEUED"
}
```

### Step 5: Poll for Results
```http
GET /analysis-sessions/{session_id}

Response (processing):
{
  "success": true,
  "session": {
    "id": "uuid",
    "status": "PROCESSING_ANALYSIS",
    ...
  }
}

Response (complete):
{
  "success": true,
  "session": {
    "id": "uuid",
    "status": "DONE",
    "result": { ...full analysis results... },
    "quality": {...},
    "timings_ms": {...}
  }
}
```

### Session Status Values
- `CREATED` - Session created, waiting for upload URL request
- `UPLOAD_URL_ISSUED` - Presigned URL generated, waiting for video upload
- `QUEUED` - Analysis job queued for processing
- `PROCESSING_EXTRACT` - Downloading video from object storage
- `PROCESSING_ANALYSIS` - Running pose detection and analysis
- `DONE` - Analysis complete, results available
- `FAILED` - Analysis failed, error details available

---

### Legacy Endpoints (Deprecated)

These endpoints are maintained for backward compatibility:

### Video Upload
```http
POST /upload/video
Content-Type: multipart/form-data

Body: video file (MP4, MOV, AVI)
```

### Get Curry Baseline
```http
GET /baselines/curry
```
Returns Stephen Curry baseline shooting form data.

### Compare with Curry
```http
POST /compare/curry
Content-Type: application/json

{
  "user_shot_data": {...}
}
```

## Deployment

### Production Deployment (Render)

This backend is deployed on Render: https://basketballaiapp.onrender.com

The deployment configuration is defined in the root directory (if using Render.yaml) or through Render dashboard settings:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables

Set these in your deployment platform:

- `PORT` - Server port (set automatically by Render)
- `PYTHON_VERSION` - 3.9.16 (optional, for Render)

### Deploying to Other Platforms

**Railway:**
```bash
railway up
```

**Heroku:**
```bash
heroku create basketball-ai-backend
git push heroku main
```

**Docker:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Project Structure

```
BasketballAIAppApi/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version for deployment
├── services/              # Core analysis services
│   ├── pose_analyzer.py
│   ├── shot_analyzer.py
│   └── comparison_service.py
├── routes/                # API route handlers
├── baselines/             # Curry baseline data
├── tests/                 # Test files
└── venv/                  # Virtual environment (gitignored)
```

## Related Repositories

- **Mobile App**: [BasketballAIApp](https://github.com/YOUR_USERNAME/BasketballAIApp) - React Native/Expo frontend

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .
```

### Adding New Features

1. Create new service in `services/`
2. Add route handler in `routes/`
3. Register route in `main.py`
4. Update tests in `tests/`

## CORS Configuration

The backend allows requests from:
- `http://localhost:8081` (Expo dev server)
- `exp://192.168.*` (Expo Go on local network)
- All origins (for mobile app access)

Update CORS settings in `main.py` if needed:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance

- Average analysis time: 2-4 seconds per video
- Supports videos up to 100MB
- MediaPipe pose detection: ~30 FPS

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### MediaPipe Installation Issues
```bash
pip install --upgrade mediapipe
```

### Video Processing Errors
- Ensure video codec is supported (H.264 recommended)
- Check video file size (max 100MB)
- Verify video contains clear human figure

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- Create an issue in this repository
- Contact: jmaddox0503@example.com

## Acknowledgments

- MediaPipe by Google for pose detection
- OpenCV for video processing
- FastAPI for the web framework
