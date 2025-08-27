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

## Environment Mapping

| Environment | Frontend URL | Backend URL | CORS Policy |
|-------------|-------------|-------------|-------------|
| **Staging** | https://japanese-sentiment-frontend-staging.fly.dev | https://japanese-sentiment-analyzer-staging.fly.dev | Staging FE → Staging BE only |
| **Production** | https://japanese-sentiment-frontend.fly.dev | https://japanese-sentiment-analyzer.fly.dev | Production FE → Production BE only |

### Environment Variables
- **Staging**: `.env.staging` with `VITE_API_BASE_URL=https://japanese-sentiment-analyzer-staging.fly.dev`
- **Production**: `.env.production` with `VITE_API_BASE_URL=https://japanese-sentiment-analyzer.fly.dev`
- **Build System**: Docker build process injects environment-specific API URLs during deployment

### CORS Configuration
- **Explicit Origins**: No wildcard (*) allowed, exact frontend URLs only
- **Staging Backend**: `ALLOWED_ORIGINS=https://japanese-sentiment-frontend-staging.fly.dev`
- **Production Backend**: `ALLOWED_ORIGINS=https://japanese-sentiment-frontend.fly.dev`
- **Methods**: `["GET", "POST", "OPTIONS"]`
- **Headers**: `["*"]`
- **Credentials**: `False`

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
- **Frontend**: Static HTML/CSS/JavaScript served by nginx with environment-specific build system
- **Backend**: FastAPI Python application with explicit CORS configuration
- **Deployment**: Fly.io containers in Tokyo (nrt) region with environment isolation
- **CORS**: Backend configured with exact frontend origins, no cross-environment calls

## Environment Configuration
- **Development**: API URL automatically set to `http://localhost:8000`
- **Staging**: API URL injected from `.env.staging` during Docker build
- **Production**: API URL injected from `.env.production` during Docker build
- **Build System**: Replaces `__VITE_API_BASE_URL__` placeholder with actual URL

## Troubleshooting
- Check browser console for API connection errors
- Verify backend health endpoint is accessible
- Ensure CORS headers are properly configured on backend
- Check network tab for failed API requests
- Verify staging frontend logs show correct API base URL

### Configuration Notes
- **Environment Isolation**: Complete separation between staging and production environments
- **API URL Injection**: Build system replaces `__VITE_API_BASE_URL__` placeholder during Docker build
- **Staging Verification**: Staging frontend logs API base URL to console for verification
- **Dark Mode**: Theme preference is stored in localStorage and persists across sessions
- **History**: Analysis history is stored locally and limited to 5 most recent items

## UI Features (Modern Version)
- **TailwindCSS**: Modern utility-first CSS framework via CDN
- **Dark Mode**: Manual toggle with auto-detection of system preference
- **Responsive Design**: Mobile-first approach with proper breakpoints
- **Accessibility**: WCAG AA compliant with keyboard navigation and ARIA labels
- **History**: Last 5 analysis results with click-to-restore functionality
- **Color System**: Slate backgrounds, emerald (positive) / rose (negative) sentiment indicators
