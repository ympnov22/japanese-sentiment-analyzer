# Branch Status Report

Last updated: 2025-08-26 11:00:00 UTC

## Active Branches

| Branch | Last Activity | Status | Open PRs |
|--------|---------------|--------|----------|
| `main` | 2025-08-26 | ✅ Protected | N/A |

## Cleanup Policy

- Branches older than 30 days with no open PRs are automatically deleted
- This report is updated weekly via GitHub Actions
- Protected branches (main) are never deleted

## Legend

- ✅ Active branch with open PR(s)
- ⏳ Recent branch (< 30 days old)
- 🔄 Stale branch (> 30 days old, candidate for deletion)
- ❌ No open PRs

*This report will be automatically updated by the cleanup workflow once it's deployed.*

## Production Promotion Confirmation

### Deployment Details
- **Promotion Time**: 2025-08-27 01:44:27 UTC
- **Commit SHA**: 54583cddc975e0992c0a054cc3e96cf88122112c
- **Model Version**: 1.0.0
- **Model SHA256**: 
  - Classifier: a58de37ca8f13d929dbec4dfd2dc87368663b4e4e8db3638cf9d8d6e0344fc4e
  - Vectorizer: 19d6cbf11529de5e4be61f0d3e5d1cfa572c9443d15b96f16d3adb02bd8e7eb0
- **Accuracy Baseline**: 0.8744 (87.44%)
- **Machine Count**: 1 (rolling deployment strategy)
- **Memory Configuration**: 4GB maintained

### Verification Results
- ✅ **Health Endpoint**: All metadata matches staging exactly
- ✅ **Model Verification**: SHA256 checksums verified successfully
- ✅ **Prediction Tests**: 8-sample sanity check passed with identical results to staging
- ✅ **Startup Logs**: Clean startup with no critical errors
- ✅ **Performance**: Sub-second response times, comparable to staging

### Production URLs
- **Production**: https://japanese-sentiment-analyzer.fly.dev/
- **Staging**: https://japanese-sentiment-analyzer-staging.fly.dev/

### Model Performance Summary
- Simple positive cases: 84-91% confidence (working well)
- Simple negative cases: 69-97% confidence (working well)  
- Complex expressions: 50-61% confidence (expected limitations)
- Edge cases: Some misclassifications (e.g., "だめ" → positive) - tracked in follow-up issues

## Frontend Production Deployment

### Deployment Details
- **Frontend URL**: https://japanese-sentiment-frontend.fly.dev/
- **Deployment Time**: 2025-08-27 02:09:36 UTC
- **App Configuration**: 512MB memory, 1 machine, nrt region
- **Backend Integration**: https://japanese-sentiment-analyzer.fly.dev/

### Verification Results
- ✅ **Frontend Loading**: Successfully loads at production URL
- ✅ **API Connection**: Connects to production backend /health endpoint
- ✅ **Prediction Functionality**: Japanese text prediction working
- ✅ **CORS Configuration**: No cross-origin issues detected
