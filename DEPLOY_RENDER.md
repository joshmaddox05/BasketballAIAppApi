# Deploying to Render

This guide covers deploying the Basketball AI API with the async session pipeline to Render.

## Architecture Overview

The async pipeline uses:
- **Web Service**: FastAPI app handling HTTP requests (creates sessions, generates presigned URLs)
- **Worker Service**: RQ worker processing video analysis jobs in the background
- **Redis**: Session state and job queue storage
- **Object Storage (S3/R2)**: Video file storage with presigned URLs for direct uploads

## Prerequisites

- Render account
- S3-compatible object storage (AWS S3 or Cloudflare R2)
- Your repository connected to Render

## Important: FFmpeg Requirement

The overlay video feature requires ffmpeg for web-compatible video encoding. Update your **Build Command** for both Web and Worker services:

```bash
apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt
```

Or if you don't have root access, the overlay will still work but videos may not play in some browsers.

## Step 1: Create Redis Instance

1. In Render Dashboard, click **New > Redis**
2. Choose a name (e.g., `basketball-ai-redis`)
3. Select region (same as your services for low latency)
4. Choose plan (Free tier works for development)
5. Click **Create Redis**
6. Copy the **Internal URL** (starts with `redis://`)

## Step 2: Configure Object Storage

### Option A: AWS S3

1. Create an S3 bucket in your preferred region
2. Create an IAM user with S3 permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:HeadObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        }
    ]
}
```
3. Generate access keys for the IAM user

### Option B: Cloudflare R2

1. In Cloudflare Dashboard, go to **R2 > Create bucket**
2. Name your bucket (e.g., `basketball-ai-videos`)
3. Go to **R2 > Manage R2 API Tokens**
4. Create a token with Object Read & Write permissions
5. Note the Account ID, Access Key ID, and Secret Access Key
6. Your endpoint URL will be: `https://<account-id>.r2.cloudflarestorage.com`

## Step 3: Create Web Service

1. Click **New > Web Service**
2. Connect your repository
3. Configure:
   - **Name**: `basketball-ai-api`
   - **Region**: Same as Redis
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. Add Environment Variables:

| Variable | Value |
|----------|-------|
| `REDIS_URL` | Your Redis Internal URL |
| `OBJECT_STORE_BUCKET` | Your bucket name |
| `OBJECT_STORE_REGION` | `us-east-1` (or your region) |
| `OBJECT_STORE_ENDPOINT_URL` | R2: `https://<account-id>.r2.cloudflarestorage.com` / S3: leave empty |
| `OBJECT_STORE_ACCESS_KEY_ID` | Your access key |
| `OBJECT_STORE_SECRET_ACCESS_KEY` | Your secret key |
| `OBJECT_STORE_FORCE_PATH_STYLE` | `true` for R2, `false` for S3 |
| `SESSION_TTL_SECONDS` | `604800` (7 days) |
| `PRESIGN_EXPIRE_SECONDS` | `900` (15 minutes) |

5. Click **Create Web Service**

## Step 4: Create Worker Service

1. Click **New > Background Worker**
2. Connect the same repository
3. Configure:
   - **Name**: `basketball-ai-worker`
   - **Region**: Same as Redis and Web Service
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python worker.py`

4. Add the **same environment variables** as the Web Service

5. Click **Create Background Worker**

## Step 5: Configure CORS (Optional)

If your mobile app needs to upload directly to object storage, configure CORS on your bucket:

### S3 CORS Configuration
```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["PUT", "GET", "HEAD"],
        "AllowedOrigins": ["*"],
        "ExposeHeaders": ["ETag"],
        "MaxAgeSeconds": 3600
    }
]
```

### R2 CORS Configuration
In Cloudflare Dashboard > R2 > Your Bucket > Settings > CORS Policy:
```json
[
    {
        "AllowedOrigins": ["*"],
        "AllowedMethods": ["PUT", "GET", "HEAD"],
        "AllowedHeaders": ["*"],
        "ExposeHeaders": ["ETag"],
        "MaxAgeSeconds": 3600
    }
]
```

## Scaling Recommendations

### Web Service
- Start with 1 instance
- Scale horizontally for more concurrent session creation/polling
- Minimal memory needed (256MB-512MB)

### Worker Service
- Start with 1 instance
- Scale based on analysis queue depth
- Needs more memory for video processing (1GB+ recommended)
- Consider multiple workers for higher throughput

### Redis
- Free tier: 25MB, good for development
- Starter tier: 100MB, good for production
- Monitor memory usage for session storage

## Monitoring

### Health Check
The web service exposes `/health` for Render's health checks.

### Logs
- Web Service: Check Render logs for API request errors
- Worker: Check Render logs for processing errors

### Queue Monitoring
Connect to Redis and use RQ dashboard or CLI:
```bash
rq info --url $REDIS_URL
```

## Troubleshooting

### Worker not processing jobs
1. Verify `REDIS_URL` is correct on worker
2. Check worker logs for connection errors
3. Verify `RQ_QUEUE_NAME` matches between web and worker

### Presigned URL upload fails
1. Check CORS configuration on bucket
2. Verify `Content-Type` header matches when uploading
3. Check bucket permissions for the IAM user

### Session not found errors
1. Check `SESSION_TTL_SECONDS` isn't too short
2. Verify Redis is accessible from both services
3. Check for Redis memory limits

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | Yes | - | Redis connection URL |
| `OBJECT_STORE_BUCKET` | Yes | - | S3/R2 bucket name |
| `OBJECT_STORE_REGION` | No | `us-east-1` | AWS region |
| `OBJECT_STORE_ENDPOINT_URL` | R2 only | - | R2 endpoint URL |
| `OBJECT_STORE_ACCESS_KEY_ID` | Yes | - | Access key |
| `OBJECT_STORE_SECRET_ACCESS_KEY` | Yes | - | Secret key |
| `OBJECT_STORE_FORCE_PATH_STYLE` | No | `false` | Use path-style URLs |
| `SESSION_TTL_SECONDS` | No | `604800` | Session expiration (7 days) |
| `PRESIGN_EXPIRE_SECONDS` | No | `900` | Upload URL expiration (15 min) |
| `RQ_QUEUE_NAME` | No | `analysis` | RQ queue name |
