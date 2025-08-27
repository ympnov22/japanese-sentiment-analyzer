# Frontend Deployment Guide

## Production Frontend
- **URL**: https://japanese-sentiment-frontend.fly.dev/
- **App Name**: japanese-sentiment-frontend
- **Region**: nrt (Tokyo)
- **Memory**: 512MB
- **Machines**: 1 (no autoscaling)

## API Configuration
- **Backend URL**: https://japanese-sentiment-analyzer.fly.dev/
- **Health Endpoint**: /health
- **Prediction Endpoint**: /predict

## Verification Steps
1. Frontend loads successfully
2. API health check passes (check browser console)
3. Prediction functionality works
4. No CORS errors in browser console

## Deployment Commands
```bash
cd frontend/
export PATH="/home/ubuntu/.fly/bin:$PATH"
flyctl deploy --app japanese-sentiment-frontend
```

## Architecture
The frontend is a static HTML/CSS/JavaScript application served by nginx:
- Static files served from `/usr/share/nginx/html/`
- nginx configuration includes gzip compression and security headers
- API calls made directly from browser JavaScript to backend
- CORS handled by backend FastAPI configuration

## Environment Configuration
- **Development**: API URL automatically set to `http://localhost:8000`
- **Production**: API URL configured to `https://japanese-sentiment-analyzer.fly.dev`
- Configuration loaded dynamically based on hostname detection

## Troubleshooting
- Check browser console for API connection errors
- Verify backend health endpoint is accessible
- Ensure CORS headers are properly configured on backend
- Check network tab for failed API requests
