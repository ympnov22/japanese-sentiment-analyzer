from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional
import os
from app.model_loader import LightweightSentimentModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Japanese Sentiment Analysis API",
    description="API for Japanese text sentiment classification (ポジティブ/ネガティブ)",
    version="1.0.0"
)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173,"
    "https://jpn-sentiment-web-nrt.fly.dev,https://japanese-sentiment-analyzer-app-50t1mfcr.devinapps.com,"
    "https://japanese-sentiment-frontend.fly.dev,https://japanese-sentiment-frontend-staging.fly.dev"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Japanese text to analyze")


class PredictResponse(BaseModel):
    result: str = Field(..., description="Sentiment classification result")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str
    build_metadata: Optional[Dict[str, Any]] = None


sentiment_service = LightweightSentimentModel()


@app.on_event("startup")
async def startup_event():
    """Initialize API and preload model"""
    logger.info("Starting Japanese Sentiment Analysis API with model preloading...")

    try:
        logger.info("Loading sentiment model during startup...")
        if sentiment_service.load_model():
            logger.info("Model preloaded successfully during startup")
        else:
            logger.warning("Model preloading failed, will fall back to lazy loading")
    except Exception as e:
        logger.error("Error during model preloading: %s", str(e))
        logger.warning("Continuing with lazy loading as fallback")

    logger.info("API ready for requests")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with build metadata"""
    memory_info = sentiment_service.get_memory_info()

    git_commit = os.getenv("GIT_COMMIT", "unknown")
    model_version = memory_info.get("model_version", "unknown")

    model_sha256 = "unknown"
    if sentiment_service.metadata and sentiment_service.metadata.get("verified_sha256"):
        ultra_config = sentiment_service.model_registry.get("ultra", {})
        classifier_hash = ultra_config.get('classifier_sha256', 'unknown')[:8]
        vectorizer_hash = ultra_config.get('vectorizer_sha256', 'unknown')[:8]
        model_sha256 = f"classifier:{classifier_hash}...,vectorizer:{vectorizer_hash}..."

    message = "Japanese Sentiment Analysis API is running"
    if sentiment_service.is_loaded:
        message += f" (model loaded, {memory_info.get('total_model_size_mb', 0):.1f}MB)"
        if memory_info.get("model_verified"):
            message += " [VERIFIED]"
        else:
            message += " [UNVERIFIED]"
    else:
        message += " (lazy loading)"

    return HealthResponse(
        status="ok",
        model_loaded=sentiment_service.is_loaded,
        message=message,
        build_metadata={
            "git_commit": git_commit,
            "model_version": model_version,
            "model_sha256": model_sha256,
            "model_verified": memory_info.get("model_verified", False),
            "accuracy_baseline": memory_info.get("accuracy_baseline", "unknown")
        }
    )


@app.get("/healthz")
async def healthz():
    """Simple health check for compatibility"""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    """Predict sentiment for Japanese text (with lazy loading)"""
    try:
        logger.info("Received prediction request for text: %s...", request.text[:50])

        result = sentiment_service.predict(request.text)

        return PredictResponse(
            result=result["result"],
            score=result["score"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in predict endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
