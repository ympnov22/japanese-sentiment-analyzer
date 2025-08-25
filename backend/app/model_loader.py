#!/usr/bin/env python3
"""
Custom lightweight model loader for Japanese sentiment analysis
Uses numpy-based serialization to avoid joblib memory spikes
Optimized for Fly.io deployment with <300MB RSS target
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
import json

logger = logging.getLogger(__name__)

class LightweightSentimentModel:
    """
    Custom lightweight sentiment model with numpy-based serialization
    Avoids joblib memory spikes during loading
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.vectorizer = None
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.metadata = None
        self.is_loaded = False
        self._loading_lock = False
        
    def _create_vectorizer(self, vectorizer_params: Dict[str, Any]):
        """Create vectorizer with specified parameters (HashingVectorizer or TfidfVectorizer)"""
        if vectorizer_params.get('analyzer') == 'char':
            ngram_range_str = vectorizer_params.get('ngram_range', '(1, 2)')
            if isinstance(ngram_range_str, str):
                ngram_range = tuple(map(int, ngram_range_str.strip('()').split(', ')))
            else:
                ngram_range = tuple(ngram_range_str)
            
            return TfidfVectorizer(
                analyzer=vectorizer_params.get('analyzer', 'word'),
                ngram_range=ngram_range,
                min_df=vectorizer_params.get('min_df', 1),
                max_df=vectorizer_params.get('max_df', 1.0),
                max_features=vectorizer_params.get('max_features', None),
                sublinear_tf=vectorizer_params.get('sublinear_tf', False),
                norm=vectorizer_params.get('norm', 'l2'),
                lowercase=vectorizer_params.get('lowercase', True)
            )
        else:
            return HashingVectorizer(
                n_features=vectorizer_params.get('n_features', 2**18),
                alternate_sign=vectorizer_params.get('alternate_sign', False),
                ngram_range=tuple(vectorizer_params.get('ngram_range', [1, 2])),
                lowercase=True,
                norm='l2',
                binary=False
            )
    
    def load_model(self) -> bool:
        """
        Load model weights from numpy files (memory-efficient)
        Uses mmap_mode='r' to avoid memory copying
        """
        if self.is_loaded:
            return True
            
        if self._loading_lock:
            logger.info("Model loading already in progress...")
            return False
            
        self._loading_lock = True
        
        try:
            logger.info("Loading lightweight sentiment model with numpy...")
            
            weights_path = self.model_dir / "weights.npz"
            metadata_path = self.model_dir / "model_metadata.json"
            
            if not weights_path.exists() or not metadata_path.exists():
                logger.warning("Numpy weights not found, falling back to joblib models...")
                logger.info(f"Checking paths: weights={weights_path.exists()}, metadata={metadata_path.exists()}")
                return self._load_joblib_fallback()
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loading model weights from {weights_path}")
            
            weights_data = np.load(weights_path, mmap_mode='r')
            
            self.coef_ = weights_data['coef'].astype(np.float32)
            self.intercept_ = weights_data['intercept'].astype(np.float32)
            self.classes_ = np.array(self.metadata['sentiment_labels'])
            
            del weights_data
            import gc
            gc.collect()
            
            vectorizer_params = self.metadata.get('vectorizer_params', {})
            self.vectorizer = self._create_vectorizer(vectorizer_params)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully. Coef shape: {self.coef_.shape}")
            logger.info(f"Classes: {self.classes_}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading numpy model: {str(e)}")
            logger.info("Attempting joblib fallback...")
            return self._load_joblib_fallback()
        finally:
            self._loading_lock = False
    
    def _load_joblib_fallback(self) -> bool:
        """Fallback to joblib loading if numpy weights not available"""
        try:
            import joblib
            
            model_names = ["japanese_sentiment_model", "japanese_sentiment_model_ultra", "japanese_sentiment_model_lite"]
            logger.info(f"Trying to load joblib models in order (ensemble models temporarily disabled): {model_names}")
            
            for model_name in model_names:
                vectorizer_path = self.model_dir / f"{model_name}_vectorizer.pkl"
                classifier_path = self.model_dir / f"{model_name}_classifier.pkl"
                metadata_path = self.model_dir / f"{model_name}_metadata.json"
                
                logger.info(f"Checking {model_name}: vec={vectorizer_path.exists()}, clf={classifier_path.exists()}, meta={metadata_path.exists()}")
                
                if all(path.exists() for path in [vectorizer_path, classifier_path, metadata_path]):
                    logger.info(f"Loading joblib model: {model_name}")
                    
                    self.vectorizer = joblib.load(vectorizer_path)
                    classifier = joblib.load(classifier_path)
                    
                    self.coef_ = classifier.coef_.astype(np.float32)
                    self.intercept_ = classifier.intercept_.astype(np.float32)
                    self.classes_ = classifier.classes_
                    
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                    
                    self.is_loaded = True
                    logger.info(f"Joblib fallback successful for {model_name}")
                    logger.info(f"Loaded vectorizer type: {type(self.vectorizer)}")
                    logger.info(f"Model classes: {self.classes_}")
                    return True
            
            logger.error("No valid model files found")
            return False
            
        except Exception as e:
            logger.error(f"Joblib fallback failed: {str(e)}")
            return False
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid function with numerical stability"""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment with custom inference (no sklearn predict)
        """
        if not self.is_loaded:
            logger.info("Model not loaded, attempting lazy loading...")
            if not self.load_model():
                raise RuntimeError("Model loading failed")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            text_vector = self.vectorizer.transform([text.strip()])
            
            logits = text_vector @ self.coef_.T + self.intercept_
            
            if len(self.classes_) == 2:
                prob_positive = self._sigmoid(logits[0, 0])
                probabilities = np.array([1 - prob_positive, prob_positive])
                prediction = 1 if prob_positive > 0.5 else 0
            else:
                exp_logits = np.exp(logits[0] - np.max(logits[0]))
                probabilities = exp_logits / np.sum(exp_logits)
                prediction = np.argmax(probabilities)
            
            if hasattr(self, 'metadata') and 'index_to_label' in self.metadata:
                index_to_label = {int(k): v for k, v in self.metadata['index_to_label'].items()}
                sentiment_label = index_to_label[prediction]
            else:
                sentiment_label = self.classes_[prediction]
            
            confidence_score = float(probabilities[prediction])
            
            logger.info(f"Prediction: {sentiment_label}, Confidence: {confidence_score:.3f}")
            
            if hasattr(self, 'metadata') and 'index_to_label' in self.metadata:
                index_to_label = {int(k): v for k, v in self.metadata['index_to_label'].items()}
                all_scores = {
                    index_to_label[i]: float(probabilities[i]) 
                    for i in range(len(self.classes_))
                }
            else:
                all_scores = {
                    str(self.classes_[i]): float(probabilities[i]) 
                    for i in range(len(self.classes_))
                }
            
            return {
                "result": sentiment_label,
                "score": confidence_score,
                "all_scores": all_scores
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        info = {
            "is_loaded": self.is_loaded,
            "model_components": {}
        }
        
        if self.is_loaded:
            if self.coef_ is not None:
                info["model_components"]["coef_size_mb"] = self.coef_.nbytes / (1024 * 1024)
            if self.intercept_ is not None:
                info["model_components"]["intercept_size_mb"] = self.intercept_.nbytes / (1024 * 1024)
            
            total_size = sum(info["model_components"].values())
            info["total_model_size_mb"] = total_size
            
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                info["process_rss_mb"] = process.memory_info().rss / (1024 * 1024)
            except ImportError:
                info["process_rss_mb"] = "psutil not available"
        
        return info
