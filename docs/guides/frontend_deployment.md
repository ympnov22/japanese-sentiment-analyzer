# Frontend Deployment Guide

## Production Frontend
- **URL**: https://japanese-sentiment-frontend.fly.dev/
- **App Name**: japanese-sentiment-frontend
- **Region**: nrt (Tokyo)
- **Memory**: 512MB
- **Machines**: 1 (no autoscaling)

## Staging Frontend
- **URL**: https://japanese-sentiment-frontend-staging.fly.dev/
- **App Name**: japanese-sentiment-frontend-staging
- **Region**: nrt (Tokyo)
- **Memory**: 256MB
- **Machines**: 1 (no autoscaling)
- **Features**: Modern TailwindCSS UI, dark mode toggle, responsive design, history functionality

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

### Production Deployment
```bash
cd frontend/
export PATH="/home/ubuntu/.fly/bin:$PATH"
flyctl deploy --app japanese-sentiment-frontend
```

### Staging Deployment
```bash
cd frontend/
export PATH="/home/ubuntu/.fly/bin:$PATH"
flyctl deploy --app japanese-sentiment-frontend-staging
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

### Known Issues
- **Staging CORS**: The staging URL `https://japanese-sentiment-frontend-staging.fly.dev` needs to be added to the backend's ALLOWED_ORIGINS environment variable for API connectivity to work
- **Dark Mode**: Theme preference is stored in localStorage and persists across sessions
- **History**: Analysis history is stored locally and limited to 5 most recent items

## UI Features (Modern Version)
- **TailwindCSS**: Modern utility-first CSS framework via CDN
- **Dark Mode**: Manual toggle with auto-detection of system preference
- **Responsive Design**: Mobile-first approach with proper breakpoints
- **Accessibility**: WCAG AA compliant with keyboard navigation and ARIA labels
- **History**: Last 5 analysis results with click-to-restore functionality
- **Color System**: Slate backgrounds, emerald (positive) / rose (negative) sentiment indicators
