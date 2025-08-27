# Release and Branching Guide

This document describes the release process and branching strategy for the Japanese Sentiment Analyzer project.

## Overview

The project uses a structured approach to prevent model regressions and maintain code quality through:

- **Model Registry**: Version-pinned artifacts with SHA256 verification
- **Environment Separation**: Development → Staging → Production pipeline
- **Branch Protection**: Required reviews and automated quality checks
- **Automated Cleanup**: Stale branch removal and status reporting

## Model Registry System

### Model Versioning

Models are registered with version numbers and SHA256 checksums for integrity verification:

```python
model_registry = {
    "ultra": {
        "classifier_file": "japanese_sentiment_model_ultra_classifier.pkl",
        "vectorizer_file": "japanese_sentiment_model_ultra_vectorizer.pkl", 
        "metadata_file": "japanese_sentiment_model_ultra_metadata.json",
        "classifier_sha256": "a58de37ca8f13d929dbec4dfd2dc87368663b4e4e8db3638cf9d8d6e0344fc4e",
        "vectorizer_sha256": "19d6cbf11529de5e4be61f0d3e5d1cfa572c9443d15b96f16d3adb02bd8e7eb0",
        "version": "1.0.0",
        "accuracy_baseline": 0.8744
    }
}
```

### SHA256 Verification

- Boot-time checksum validation prevents deployment of corrupted models
- Models fail to load if SHA256 mismatches are detected
- Verification status is exposed in `/health` endpoint

### Accuracy Baselines

- Minimum accuracy threshold: **85%** (based on ultra model's 87.4% performance)
- CI tests validate model performance on curated dataset
- Prevents deployment of regressed models

## Environment Strategy

### Three-Tier Architecture

1. **Development** (`japanese-sentiment-analyzer-dev`)
   - Auto-scaling (0-1 machines)
   - Used for feature development and testing
   - Minimal resource allocation

2. **Staging** (`japanese-sentiment-analyzer-staging`)
   - Always-on (1+ machines)
   - Pre-production testing environment
   - Production-like configuration

3. **Production** (`japanese-sentiment-analyzer-prod`)
   - High availability (2+ machines)
   - Only tagged releases deployed
   - Enhanced monitoring and resources

### Deployment Configuration

Each environment has its own Fly.io configuration:

- `backend/fly.dev.toml` - Development environment
- `backend/fly.staging.toml` - Staging environment  
- `backend/fly.prod.toml` - Production environment

## Branching Strategy

### Branch Protection Rules

**Main Branch Protection:**
- Require pull request reviews (1+ reviewers)
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Squash merging only (clean history)
- Automatically delete head branches after merge

### Branch Naming Convention

Use descriptive branch names with timestamps:
```
devin/{timestamp}-{feature-description}
```

Examples:
- `devin/1756205734-model-registry-and-ci`
- `devin/1756205800-branch-hygiene-and-docs`

### Required Status Checks

All PRs must pass:
- **Linting**: Code style and formatting checks
- **Type Checking**: MyPy static analysis
- **Unit Tests**: Core functionality tests
- **Model Accuracy Tests**: Minimum 85% accuracy validation
- **Security Scan**: Dependency vulnerability checks

## Release Process

### 1. Development Phase

```bash
# Create feature branch
git checkout -b devin/$(date +%s)-feature-name

# Make changes and commit
git add .
git commit -m "feat: implement feature"

# Push and create PR
git push origin devin/$(date +%s)-feature-name
```

### 2. Pull Request Review

- Automated CI checks must pass
- Code review by repository owners
- Model accuracy validation (if applicable)
- Documentation updates reviewed

### 3. Staging Deployment

```bash
# Deploy to staging
fly deploy --config backend/fly.staging.toml

# Verify deployment
curl https://japanese-sentiment-analyzer-staging.fly.dev/health
```

### 4. Production Release

**Only after staging validation:**

```bash
# Create release tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Deploy to production
fly deploy --config backend/fly.prod.toml
```

## Quality Gates

### CI Pipeline

The CI pipeline enforces quality standards:

1. **Code Quality**
   - Flake8 linting (max line length: 120)
   - MyPy type checking
   - Security vulnerability scanning

2. **Testing**
   - Unit test coverage
   - Model accuracy validation
   - Edge case handling

3. **Model Validation**
   - SHA256 integrity verification
   - Accuracy threshold compliance (>85%)
   - Prediction format validation

### Model Accuracy Testing

The CI includes comprehensive model testing:

```python
# Curated test dataset with expected labels
test_cases = [
    ("この商品は本当に素晴らしいです！", "ポジティブ"),
    ("最悪の商品でした。二度と買いません。", "ネガティブ"),
    # ... more test cases
]

# Minimum accuracy threshold
min_accuracy_threshold = 0.85
```

## Branch Maintenance

### Automated Cleanup

**Weekly Cleanup Schedule:**
- Runs every Sunday at 2 AM UTC
- Deletes branches >30 days old with no open PRs
- Preserves main branch and active development branches
- Updates branch status report

### Branch Status Monitoring

The system maintains a live branch status report at `docs/reports/branch_status.md`:

- Lists all active branches
- Shows last activity dates
- Indicates PR status
- Identifies stale branches

## Health Monitoring

### Enhanced Health Endpoint

The `/health` endpoint provides comprehensive system status:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory_usage_mb": 107.3,
  "model_components_mb": 3.0,
  "build_metadata": {
    "git_commit": "abc123...",
    "model_version": "1.0.0",
    "model_sha256": "classifier:a58de37c...,vectorizer:19d6cbf1...",
    "model_verified": true,
    "accuracy_baseline": 0.8744
  }
}
```

### Key Metrics

- **Model Verification**: SHA256 checksum validation status
- **Accuracy Baseline**: Expected model performance
- **Memory Usage**: Resource consumption monitoring
- **Build Metadata**: Deployment traceability

## Troubleshooting

### Model Loading Issues

**SHA256 Mismatch:**
```
WARNING: Classifier SHA256 mismatch. Expected: a58de37c..., Got: different_hash...
```

**Solution:** Re-download or retrain model with correct checksums.

**Low Accuracy:**
```
Model accuracy 0.75 is below minimum threshold 0.85
```

**Solution:** Investigate model regression, retrain if necessary.

### Deployment Issues

**Environment Configuration:**
- Verify correct Fly.io app names
- Check environment variables
- Validate resource allocation

**CI Failures:**
- Review failed checks in GitHub Actions
- Address linting/testing issues
- Ensure model files are available

## Best Practices

### Model Development

1. **Always validate accuracy** before committing model changes
2. **Update SHA256 checksums** when retraining models
3. **Test in staging** before production deployment
4. **Document model changes** in PR descriptions

### Code Development

1. **Follow branch naming conventions**
2. **Write comprehensive tests** for new features
3. **Update documentation** for significant changes
4. **Request reviews** from code owners

### Deployment

1. **Test locally** before creating PRs
2. **Verify CI passes** before requesting review
3. **Deploy to staging first** for validation
4. **Monitor health endpoints** after deployment

## Support

For questions or issues with the release process:

1. Check this documentation first
2. Review CI logs for specific errors
3. Contact repository owners (@ympnov22)
4. Create an issue for process improvements

---

*This document is maintained as part of the automated branch cleanup process and updated with each release.*
