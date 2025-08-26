#!/usr/bin/env python3
"""
Custom lightweight model loader for Japanese sentiment analysis
Uses numpy-based serialization to avoid joblib memory spikes
Optimized for Fly.io deployment with <300MB RSS target
Enhanced with SHA256 verification and model registry
"""

import numpy as np
import logging
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import HashingVectorizer
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
        self.model_registry = {
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
        
    def _calculate_file_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _verify_model_integrity(self, model_config: Dict[str, Any]) -> bool:
        """Verify model file integrity using SHA256 checksums"""
        try:
            classifier_path = self.model_dir / model_config["classifier_file"]
            vectorizer_path = self.model_dir / model_config["vectorizer_file"]
            
            logger.info(f"Checking classifier path: {classifier_path}")
            logger.info(f"Checking vectorizer path: {vectorizer_path}")
            logger.info(f"Classifier exists: {classifier_path.exists()}")
            logger.info(f"Vectorizer exists: {vectorizer_path.exists()}")
            
            if not classifier_path.exists() or not vectorizer_path.exists():
                logger.error(f"Model files not found. Classifier: {classifier_path.exists()}, Vectorizer: {vectorizer_path.exists()}")
                return False
                
            classifier_hash = self._calculate_file_sha256(classifier_path)
            vectorizer_hash = self._calculate_file_sha256(vectorizer_path)
            
            logger.info(f"Calculated classifier SHA256: {classifier_hash}")
            logger.info(f"Expected classifier SHA256: {model_config['classifier_sha256']}")
            logger.info(f"Calculated vectorizer SHA256: {vectorizer_hash}")
            logger.info(f"Expected vectorizer SHA256: {model_config['vectorizer_sha256']}")
            
            classifier_valid = classifier_hash == model_config["classifier_sha256"]
            vectorizer_valid = vectorizer_hash == model_config["vectorizer_sha256"]
            
            if not classifier_valid:
                logger.warning(f"Classifier SHA256 mismatch. Expected: {model_config['classifier_sha256']}, Got: {classifier_hash}")
            if not vectorizer_valid:
                logger.warning(f"Vectorizer SHA256 mismatch. Expected: {model_config['vectorizer_sha256']}, Got: {vectorizer_hash}")
                
            return classifier_valid and vectorizer_valid
            
        except Exception as e:
            logger.error(f"Error verifying model integrity: {e}")
            return False

    def _create_vectorizer(self, vectorizer_params: Dict[str, Any]) -> HashingVectorizer:
        """Create HashingVectorizer with specified parameters"""
        ngram_range = vectorizer_params.get('ngram_range', [1, 2])
        if isinstance(ngram_range, list) and len(ngram_range) == 2:
            ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))
        else:
            ngram_tuple = (1, 2)
        return HashingVectorizer(
            n_features=vectorizer_params.get('n_features', 2**18),
            alternate_sign=vectorizer_params.get('alternate_sign', False),
            ngram_range=ngram_tuple,
            lowercase=True,
            norm='l2',
            binary=False
        )
    
    def load_model(self) -> bool:
        """
        Load model with SHA256 verification and memory optimization
        Prioritizes ultra model, falls back to other models if needed
        """
        if self.is_loaded:
            return True
            
        if self._loading_lock:
            logger.info("Model loading already in progress...")
            return False
            
        self._loading_lock = True
        
        try:
            logger.info("Loading lightweight sentiment model with SHA256 verification...")
            
            ultra_config = self.model_registry["ultra"]
            verification_result = self._verify_model_integrity(ultra_config)
            logger.info(f"Ultra model verification result: {verification_result}")
            if verification_result:
                logger.info("Loading verified ultra model...")
                return self._load_ultra_model(ultra_config)
            else:
                logger.warning("Ultra model verification failed, trying fallback models...")
            
            weights_path = self.model_dir / "weights.npz"
            metadata_path = self.model_dir / "model_metadata.json"
            
            if weights_path.exists() and metadata_path.exists():
                logger.info("Loading numpy-based lightweight model...")
                return self._load_numpy_model(weights_path, metadata_path)
            else:
                logger.warning("Numpy weights not found, falling back to joblib models...")
                return self._load_joblib_fallback()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
        finally:
            self._loading_lock = False
    
    def _load_ultra_model(self, model_config: Dict[str, Any]) -> bool:
        """Load the verified ultra model"""
        try:
            import joblib
            
            classifier_path = self.model_dir / model_config["classifier_file"]
            vectorizer_path = self.model_dir / model_config["vectorizer_file"]
            metadata_path = self.model_dir / model_config["metadata_file"]
            
            logger.info("Loading verified ultra-lightweight model components...")
            classifier = joblib.load(classifier_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            self.coef_ = classifier.coef_.astype(np.float32)
            self.intercept_ = classifier.intercept_.astype(np.float32)
            self.classes_ = classifier.classes_
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.metadata["registry_version"] = model_config["version"]
            self.metadata["accuracy_baseline"] = model_config["accuracy_baseline"]
            self.metadata["verified_sha256"] = True
            
            self.is_loaded = True
            logger.info("Ultra-lightweight model loaded and verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ultra model: {e}")
            return False
    
    def _load_numpy_model(self, weights_path: Path, metadata_path: Path) -> bool:
        """Load model weights from numpy files (memory-efficient)"""
        try:
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
            
            self.metadata["verified_sha256"] = False
            
            self.is_loaded = True
            logger.info(f"Numpy model loaded successfully. Coef shape: {self.coef_.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading numpy model: {str(e)}")
            return False
    
    def _load_joblib_fallback(self) -> bool:
        """Fallback to joblib loading if numpy weights not available"""
        try:
            import joblib
            
            model_names = ["japanese_sentiment_model_ultra", "japanese_sentiment_model_lite"]
            
            for model_name in model_names:
                vectorizer_path = self.model_dir / f"{model_name}_vectorizer.pkl"
                classifier_path = self.model_dir / f"{model_name}_classifier.pkl"
                metadata_path = self.model_dir / f"{model_name}_metadata.json"
                
                if all(path.exists() for path in [vectorizer_path, classifier_path, metadata_path]):
                    logger.info(f"Loading joblib model: {model_name}")
                    
                    self.vectorizer = joblib.load(vectorizer_path)
                    classifier = joblib.load(classifier_path)
                    
                    self.coef_ = classifier.coef_.astype(np.float32)
                    self.intercept_ = classifier.intercept_.astype(np.float32)
                    self.classes_ = classifier.classes_
                    
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                    
                    self.metadata["verified_sha256"] = False
                    
                    self.is_loaded = True
                    logger.info(f"Joblib fallback successful for {model_name} (unverified)")
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
            if self.vectorizer is None or self.coef_ is None or self.classes_ is None:
                raise RuntimeError("Model components not properly loaded")
                
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
            
            if self.metadata and 'index_to_label' in self.metadata:
                index_to_label = {int(k): v for k, v in self.metadata['index_to_label'].items()}
                sentiment_label = index_to_label[int(prediction)]
            else:
                sentiment_label = self.classes_[prediction]
            
            confidence_score = float(probabilities[prediction])
            
            logger.info(f"Prediction: {sentiment_label}, Confidence: {confidence_score:.3f}")
            
            return {
                "result": sentiment_label,
                "score": confidence_score,
                "all_scores": {
                    self.classes_[i]: float(probabilities[i]) 
                    for i in range(len(self.classes_))
                }
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information with registry details"""
        info = {
            "is_loaded": self.is_loaded,
            "model_components": {},
            "model_verified": self.metadata.get("verified_sha256", False) if self.metadata else False,
            "model_version": self.metadata.get("registry_version", "unknown") if self.metadata else "unknown",
            "accuracy_baseline": self.metadata.get("accuracy_baseline", "unknown") if self.metadata else "unknown"
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
