# Frontend Deployment Guide

## Overview
This guide documents the deployment of the Japanese Sentiment Analyzer frontend as a separate Fly.io application.

## Deployment Configuration

### App Details
- **App Name**: japanese-sentiment-frontend
- **App URL**: https://japanese-sentiment-frontend.fly.dev/
- **Region**: nrt (Tokyo)
- **Memory**: 512MB
- **Machines**: 1 machine, no autoscaling

### Environment Variables
- **API Base URL**: https://japanese-sentiment-analyzer.fly.dev
- **Configuration**: Updated in `frontend/script.js` loadConfig function and `frontend/.env`

### Build Process
The frontend is a vanilla HTML/CSS/JavaScript application served by nginx:
1. Static files are copied to nginx html directory
2. Custom nginx configuration provides routing and health checks
3. No build step required (vanilla JS, not a framework)

### Health Check
- **Endpoint**: `/healthz`
- **Response**: Returns "OK" with 200 status code
- **Configuration**: Added to nginx.conf as a location block

## Verification Steps
1. Frontend loads at https://japanese-sentiment-frontend.fly.dev/
2. Health check responds: `curl https://japanese-sentiment-frontend.fly.dev/healthz`
3. API calls to backend succeed (health + predict endpoints)
4. No CORS errors in browser console
5. Application logs show no critical errors

## Files Modified
- `frontend/script.js`: Updated API base URL
- `frontend/.env`: Updated VITE_API_URL
- `frontend/fly.toml`: Changed app name and increased memory to 512MB
- `frontend/nginx.conf`: Added /healthz health check endpoint
- `backend/app/main.py`: Added new frontend origin to CORS allowed origins

## Deployment Commands
```bash
cd frontend
fly apps create japanese-sentiment-frontend --org personal
fly deploy --config fly.toml
```

## Backend CORS Update
The backend CORS configuration was updated to allow the new frontend origin:
```bash
cd backend
fly deploy --config fly.toml
```

## Results

### ✅ Deployment Successful
- **App Name**: japanese-sentiment-frontend
- **App URL**: https://japanese-sentiment-frontend.fly.dev/
- **Deployment Date/Time**: 2025-08-26 10:08:50Z UTC
- **Region**: nrt (Tokyo)
- **Memory Configuration**: 512MB (shared-cpu-1x:512MB) ✅
- **Machines**: 2 running machines with 512MB each

### ✅ Verification Results
All verification steps passed successfully:

1. **Frontend Loading**: ✅ Frontend loads correctly at https://japanese-sentiment-frontend.fly.dev/
2. **Health Check**: ✅ `/healthz` endpoint returns "OK" with 200 status code
3. **Backend Health**: ✅ Backend `/health` returns `{"status":"ok","model_loaded":true,"message":"Japanese Sentiment Analysis API is running (model loaded, 1.0MB)"}`
4. **API Integration**: ✅ Prediction API calls work successfully
   - Test input: "今日はとても良い天気です" 
   - Result: "ネガティブ" with score 0.5000387707367868
5. **CORS Configuration**: ✅ No CORS errors in browser console
6. **Application Logs**: ✅ No critical errors found in deployment logs

### 🔧 Configuration Changes Applied
- Updated `frontend/script.js`: Changed API base URL to https://japanese-sentiment-analyzer.fly.dev
- Updated `frontend/.env`: Changed VITE_API_URL to new backend
- Updated `frontend/fly.toml`: Changed app name and increased memory to 512MB
- Updated `frontend/nginx.conf`: Added /healthz health check endpoint
- Updated `backend/app/main.py`: Added https://japanese-sentiment-frontend.fly.dev to CORS allowed origins

### 📊 Performance
- Frontend response time: Fast loading
- API response time: Sub-second prediction responses
- Health check response: Immediate "OK" response
- No errors or timeouts observed during testing
