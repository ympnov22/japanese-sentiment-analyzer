from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from pathlib import Path
from typing import Dict, Any, List
import os
from app.model_loader import LightweightSentimentModel
# from app.models.batch_request import (
#     BatchPredictRequest, BatchPredictResponse, AnalysisHistoryRequest, AnalysisHistoryResponse
# )
# from app.services.batch_service import BatchSentimentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Japanese Sentiment Analysis API",
    description="API for Japanese text sentiment classification (ポジティブ/ネガティブ)",
    version="1.0.0"
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173,https://jpn-sentiment-web-nrt.fly.dev,https://japanese-sentiment-analyzer-app-50t1mfcr.devinapps.com").split(",")

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

sentiment_service = LightweightSentimentModel()
# batch_service = BatchSentimentService(sentiment_service)  # Temporarily disabled

@app.on_event("startup")
async def startup_event():
    """Initialize API with lazy loading strategy"""
    logger.info("Starting Japanese Sentiment Analysis API with lazy loading strategy...")
    logger.info("Models will be loaded on first prediction request to reduce startup memory usage")
    logger.info("API ready for requests")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory_info = sentiment_service.get_memory_info()
    return HealthResponse(
        status="ok",
        model_loaded=sentiment_service.is_loaded,
        message=f"Japanese Sentiment Analysis API is running" + 
                (f" (model loaded, {memory_info.get('total_model_size_mb', 0):.1f}MB)" if sentiment_service.is_loaded else " (lazy loading)")
    )

@app.get("/healthz")
async def healthz():
    """Simple health check for compatibility"""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    """Predict sentiment for Japanese text (with lazy loading)"""
    try:
        logger.info(f"Received prediction request for text: {request.text[:50]}...")
        
        result = sentiment_service.predict(request.text)
        
        return PredictResponse(
            result=result["result"],
            score=result["score"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# @app.post("/predict/batch", response_model=BatchPredictResponse)
# async def predict_batch_sentiment(request: BatchPredictRequest):
#     """Predict sentiment for multiple Japanese texts"""
#     try:
#         logger.info(f"Received batch prediction request for {len(request.texts)} texts")
#         
#         if len(request.texts) > 1000:
#             raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 1000 texts")
#         
#         result = await batch_service.process_batch(request)
#         
#         logger.info(f"Batch processing completed: {result.summary['successful_predictions']}/{result.summary['total_texts']} successful")
#         
#         return result
#         
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in batch predict endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analyze/stats")
async def get_analysis_stats():
    """Get analysis statistics and model information"""
    try:
        memory_info = sentiment_service.get_memory_info()
        
        stats = {
            "model_info": {
                "is_loaded": sentiment_service.is_loaded,
                "model_type": "Binary Classification (ポジティブ/ネガティブ)",
                "memory_usage_mb": memory_info.get('total_model_size_mb', 0),
                "classes": ["ネガティブ", "ポジティブ"]
            },
            "api_info": {
                "version": "1.0.0",
                "supported_features": [
                    "single_prediction",
                    "batch_prediction", 
                    "detailed_analysis",
                    "custom_thresholds"
                ],
                "batch_limits": {
                    "max_texts_per_batch": 1000,
                    "max_text_length": 1000
                }
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting analysis stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
