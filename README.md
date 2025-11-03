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

### Video Upload
```http
POST /upload/video
Content-Type: multipart/form-data

Body: video file (MP4, MOV, AVI)
```

### Comprehensive Analysis
```http
POST /analyze/comprehensive
Content-Type: application/json

{
  "video_id": "uuid",
  "analysis_mode": "shooting",
  "camera_type": "back"
}
```

### Shooting Form Analysis
```http
POST /analyze/shooting
Content-Type: application/json

{
  "video_id": "uuid",
  "user_id": "string"
}
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
