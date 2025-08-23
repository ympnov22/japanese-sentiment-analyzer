from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Japanese Sentiment Analysis API",
    description="API for Japanese text sentiment classification (ポジティブ/ネガティブ)",
    version="1.0.0"
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

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

vectorizer = None
classifier = None
model_metadata = None
model_loaded = False

class SentimentAnalysisService:
    """Service class for Japanese sentiment analysis"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.vectorizer = None
        self.classifier = None
        self.metadata = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the trained model components"""
        try:
            logger.info("Loading sentiment analysis model...")
            
            vectorizer_path = self.model_dir / "japanese_sentiment_model_vectorizer.pkl"
            if not vectorizer_path.exists():
                logger.error(f"Vectorizer file not found: {vectorizer_path}")
                return False
            self.vectorizer = joblib.load(vectorizer_path)
            
            classifier_path = self.model_dir / "japanese_sentiment_model_classifier.pkl"
            if not classifier_path.exists():
                logger.error(f"Classifier file not found: {classifier_path}")
                return False
            self.classifier = joblib.load(classifier_path)
            
            metadata_path = self.model_dir / "japanese_sentiment_model_metadata.json"
            if not metadata_path.exists():
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            logger.info(f"Model supports labels: {self.metadata.get('sentiment_labels', [])}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for given text"""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        try:
            text_vector = self.vectorizer.transform([text.strip()])
            
            prediction = self.classifier.predict(text_vector)[0]
            probabilities = self.classifier.predict_proba(text_vector)[0]
            
            index_to_label = {int(k): v for k, v in self.metadata['index_to_label'].items()}
            sentiment_label = index_to_label[prediction]
            
            confidence_score = float(probabilities[prediction])
            
            logger.info(f"Prediction: {sentiment_label}, Confidence: {confidence_score:.3f}")
            
            return {
                "result": sentiment_label,
                "score": confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail="Prediction failed")

sentiment_service = SentimentAnalysisService()

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    global model_loaded
    logger.info("Starting Japanese Sentiment Analysis API...")
    model_loaded = sentiment_service.load_model()
    if model_loaded:
        logger.info("API ready for sentiment analysis")
    else:
        logger.warning("API started but model loading failed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        message="Japanese Sentiment Analysis API is running" if model_loaded else "API running but model not loaded"
    )

@app.get("/healthz")
async def healthz():
    """Simple health check for compatibility"""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    """Predict sentiment for Japanese text"""
    try:
        logger.info(f"Received prediction request for text: {request.text[:50]}...")
        
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
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
