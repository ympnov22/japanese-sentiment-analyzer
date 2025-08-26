#!/usr/bin/env python3
"""
Batch processing service for Japanese sentiment analysis
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import statistics

from app.model_loader import LightweightSentimentModel
from app.models.batch_request import (
    BatchPredictRequest, BatchPredictResponse, SentimentResult, 
    DetailedSentimentResult, BatchSummary
)
from app.services.feature_service import JapaneseFeatureExtractor

logger = logging.getLogger(__name__)

class BatchSentimentService:
    """Service for batch sentiment analysis processing"""
    
    def __init__(self, model_service: LightweightSentimentModel, max_workers: int = 4):
        self.model_service = model_service
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.feature_extractor = JapaneseFeatureExtractor()
        
    async def process_batch(self, request: BatchPredictRequest) -> BatchPredictResponse:
        """Process batch sentiment analysis request"""
        start_time = time.time()
        
        logger.info(f"Processing batch of {len(request.texts)} texts")
        
        tasks = []
        for i, text in enumerate(request.texts):
            task = asyncio.create_task(
                self._process_single_text(
                    text, 
                    include_details=request.include_details,
                    confidence_threshold=request.confidence_threshold,
                    index=i
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing text: {result}")
                failed_count += 1
            else:
                successful_results.append(result)
        
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        summary = self._generate_summary(successful_results, failed_count, total_time)
        
        return BatchPredictResponse(
            results=successful_results,
            summary=summary,
            total_processing_time_ms=total_time,
            timestamp=datetime.now()
        )
    
    async def _process_single_text(
        self, 
        text: str, 
        include_details: bool = False,
        confidence_threshold: Optional[float] = None,
        index: int = 0
    ) -> SentimentResult:
        """Process a single text for sentiment analysis"""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                self.executor, 
                self.model_service.predict, 
                text
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if confidence_threshold is not None:
                all_scores = prediction.get('all_scores', {})
                if len(all_scores) == 2:
                    positive_score = list(all_scores.values())[1]  # Assuming second is positive
                    if positive_score > confidence_threshold:
                        prediction['result'] = 'ポジティブ'
                        prediction['score'] = positive_score
                    else:
                        prediction['result'] = 'ネガティブ'
                        prediction['score'] = 1 - positive_score
            
            if include_details:
                return DetailedSentimentResult(
                    text=text,
                    result=prediction['result'],
                    score=prediction['score'],
                    all_scores=prediction.get('all_scores', {}),
                    processing_time_ms=processing_time,
                    text_length=len(text),
                    word_count=self.feature_extractor.estimate_word_count(text),
                    feature_contributions=self._get_feature_contributions(text),
                    statistical_features=self.feature_extractor.extract_all_features(text)
                )
            else:
                return SentimentResult(
                    text=text,
                    result=prediction['result'],
                    score=prediction['score'],
                    all_scores=prediction.get('all_scores', {}),
                    processing_time_ms=processing_time
                )
                
        except Exception as e:
            logger.error(f"Error processing text at index {index}: {e}")
            raise
    
    def _generate_summary(
        self, 
        results: List[SentimentResult], 
        failed_count: int, 
        total_time: float
    ) -> Dict[str, Any]:
        """Generate summary statistics for batch processing"""
        
        if not results:
            return {
                'total_texts': failed_count,
                'successful_predictions': 0,
                'failed_predictions': failed_count,
                'sentiment_distribution': {},
                'average_confidence': 0.0,
                'processing_stats': {
                    'total_time_ms': total_time,
                    'average_time_per_text_ms': 0.0,
                    'texts_per_second': 0.0
                }
            }
        
        sentiment_counts = {}
        confidence_scores = []
        processing_times = []
        
        for result in results:
            sentiment = result.result
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            confidence_scores.append(result.score)
            if result.processing_time_ms:
                processing_times.append(result.processing_time_ms)
        
        total_texts = len(results) + failed_count
        avg_time_per_text = total_time / total_texts if total_texts > 0 else 0
        texts_per_second = (total_texts / total_time * 1000) if total_time > 0 else 0
        
        processing_stats = {
            'total_time_ms': total_time,
            'average_time_per_text_ms': avg_time_per_text,
            'texts_per_second': texts_per_second
        }
        
        if processing_times:
            processing_stats.update({
                'min_time_ms': min(processing_times),
                'max_time_ms': max(processing_times),
                'median_time_ms': statistics.median(processing_times),
                'std_time_ms': statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0
            })
        
        return {
            'total_texts': total_texts,
            'successful_predictions': len(results),
            'failed_predictions': failed_count,
            'sentiment_distribution': sentiment_counts,
            'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0.0,
            'processing_stats': processing_stats
        }
    
    def _get_feature_contributions(self, text: str) -> Optional[Dict[str, float]]:
        """Get top feature contributions for the prediction"""
        try:
            return None
        except Exception as e:
            logger.warning(f"Could not extract feature contributions: {e}")
            return None
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
