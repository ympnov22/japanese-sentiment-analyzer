#!/usr/bin/env python3
"""
Batch processing models for Japanese sentiment analysis API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class BatchPredictRequest(BaseModel):
    """Request model for batch sentiment analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="List of Japanese texts to analyze")
    include_details: bool = Field(default=False, description="Include detailed analysis (feature contributions, etc.)")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Custom confidence threshold")

class SentimentResult(BaseModel):
    """Individual sentiment analysis result"""
    text: str = Field(..., description="Original input text")
    result: str = Field(..., description="Sentiment classification result")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    all_scores: Dict[str, float] = Field(..., description="Scores for all classes")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")

class DetailedSentimentResult(SentimentResult):
    """Extended sentiment result with detailed analysis"""
    text_length: int = Field(..., description="Length of input text")
    word_count: Optional[int] = Field(default=None, description="Estimated word count")
    feature_contributions: Optional[Dict[str, float]] = Field(default=None, description="Top feature contributions")
    statistical_features: Optional[Dict[str, Any]] = Field(default=None, description="Statistical text features")

class BatchPredictResponse(BaseModel):
    """Response model for batch sentiment analysis"""
    results: List[SentimentResult] = Field(..., description="List of sentiment analysis results")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")

class BatchSummary(BaseModel):
    """Summary statistics for batch processing"""
    total_texts: int = Field(..., description="Total number of texts processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    sentiment_distribution: Dict[str, int] = Field(..., description="Distribution of sentiment classifications")
    average_confidence: float = Field(..., description="Average confidence score")
    processing_stats: Dict[str, float] = Field(..., description="Processing time statistics")

class AnalysisHistoryRequest(BaseModel):
    """Request model for analysis history"""
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results to return")
    sentiment_filter: Optional[str] = Field(default=None, description="Filter by sentiment (ポジティブ/ネガティブ)")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    start_date: Optional[datetime] = Field(default=None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(default=None, description="End date for filtering")

class AnalysisHistoryResponse(BaseModel):
    """Response model for analysis history"""
    results: List[SentimentResult] = Field(..., description="Historical analysis results")
    total_count: int = Field(..., description="Total number of matching results")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
