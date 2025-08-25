#!/usr/bin/env python3
"""
Test suite for ensemble models
"""

import pytest
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

class TestEnsembleModels:
    @classmethod
    def setup_class(cls):
        cls.data_dir = Path("data")
        cls.models_dir = Path("models")
        if (cls.data_dir / "test.csv").exists():
            cls.test_data = pd.read_csv(cls.data_dir / "test.csv")
        else:
            cls.test_data = None
        
    def test_ensemble_models_exist(self):
        """Test that ensemble models are saved and loadable"""
        voting_path = self.models_dir / "ensemble_sentiment_model_voting.pkl"
        stacking_path = self.models_dir / "ensemble_sentiment_model_stacking.pkl"
        
        ensemble_exists = voting_path.exists() or stacking_path.exists()
        assert ensemble_exists, "No ensemble models found. Expected voting or stacking ensemble."
        
        if voting_path.exists():
            voting_model = joblib.load(voting_path)
            assert hasattr(voting_model, 'predict'), "Voting model missing predict method"
            print(f"Voting ensemble model loaded successfully from {voting_path}")
        
        if stacking_path.exists():
            stacking_model = joblib.load(stacking_path)
            assert hasattr(stacking_model, 'predict'), "Stacking model missing predict method"
            print(f"Stacking ensemble model loaded successfully from {stacking_path}")
    
    def test_ensemble_performance_improvement(self):
        """Test that ensemble models exceed baseline performance"""
        if self.test_data is None:
            pytest.skip("Test data not available")
        
        baseline_f1 = 0.903  # Current Phase 2 baseline
        target_f1 = 0.953    # +5% improvement target
        
        voting_path = self.models_dir / "ensemble_sentiment_model_voting.pkl"
        stacking_path = self.models_dir / "ensemble_sentiment_model_stacking.pkl"
        
        ensemble_results = []
        
        if voting_path.exists():
            voting_model = joblib.load(voting_path)
            
            X_test = self.test_data['text'].values
            y_test = self.test_data['sentiment'].map({'ネガティブ': 0, 'ポジティブ': 1}).values
            
            y_pred = voting_model.predict(X_test)
            voting_f1 = f1_score(y_test, y_pred, average='macro')
            voting_accuracy = accuracy_score(y_test, y_pred)
            
            ensemble_results.append(('Voting', voting_f1, voting_accuracy))
            
            print(f"Voting Ensemble - F1: {voting_f1:.4f}, Accuracy: {voting_accuracy:.4f}")
        
        if stacking_path.exists():
            stacking_model = joblib.load(stacking_path)
            
            X_test = self.test_data['text'].values
            y_test = self.test_data['sentiment'].map({'ネガティブ': 0, 'ポジティブ': 1}).values
            
            y_pred = stacking_model.predict(X_test)
            stacking_f1 = f1_score(y_test, y_pred, average='macro')
            stacking_accuracy = accuracy_score(y_test, y_pred)
            
            ensemble_results.append(('Stacking', stacking_f1, stacking_accuracy))
            
            print(f"Stacking Ensemble - F1: {stacking_f1:.4f}, Accuracy: {stacking_accuracy:.4f}")
        
        assert len(ensemble_results) > 0, "No ensemble models available for testing"
        
        best_f1 = max(result[1] for result in ensemble_results)
        best_model = max(ensemble_results, key=lambda x: x[1])
        
        print(f"Baseline F1: {baseline_f1:.4f}")
        print(f"Best Ensemble ({best_model[0]}) F1: {best_f1:.4f}")
        print(f"Improvement: +{best_f1 - baseline_f1:.4f}")
        
        assert best_f1 > baseline_f1, f"Best ensemble F1 ({best_f1:.4f}) should exceed baseline ({baseline_f1:.4f})"
        
        if best_f1 > target_f1:
            print(f"✅ Target F1 ({target_f1:.4f}) achieved!")
        else:
            print(f"⚠️ Target F1 ({target_f1:.4f}) not yet achieved, but baseline exceeded")
    
    def test_ensemble_prediction_consistency(self):
        """Test that ensemble models produce consistent predictions"""
        voting_path = self.models_dir / "ensemble_sentiment_model_voting.pkl"
        stacking_path = self.models_dir / "ensemble_sentiment_model_stacking.pkl"
        
        test_texts = [
            "この映画は本当に素晴らしかった！",
            "最悪の商品でした。",
            "普通の品質だと思います。"
        ]
        
        if voting_path.exists():
            voting_model = joblib.load(voting_path)
            
            for text in test_texts:
                prediction = voting_model.predict([text])[0]
                probabilities = voting_model.predict_proba([text])[0]
                
                assert prediction in [0, 1], f"Invalid prediction: {prediction}"
                assert len(probabilities) == 2, f"Invalid probabilities shape: {probabilities.shape}"
                assert abs(sum(probabilities) - 1.0) < 1e-6, f"Probabilities don't sum to 1: {probabilities}"
                
                print(f"Voting - Text: '{text[:30]}...' -> {prediction} (prob: {probabilities})")
        
        if stacking_path.exists():
            stacking_model = joblib.load(stacking_path)
            
            for text in test_texts:
                prediction = stacking_model.predict([text])[0]
                probabilities = stacking_model.predict_proba([text])[0]
                
                assert prediction in [0, 1], f"Invalid prediction: {prediction}"
                assert len(probabilities) == 2, f"Invalid probabilities shape: {probabilities.shape}"
                assert abs(sum(probabilities) - 1.0) < 1e-6, f"Probabilities don't sum to 1: {probabilities}"
                
                print(f"Stacking - Text: '{text[:30]}...' -> {prediction} (prob: {probabilities})")
    
    def test_ensemble_metadata_exists(self):
        """Test that ensemble metadata files exist"""
        voting_meta_path = self.models_dir / "ensemble_sentiment_model_voting_metadata.json"
        stacking_meta_path = self.models_dir / "ensemble_sentiment_model_stacking_metadata.json"
        
        meta_exists = voting_meta_path.exists() or stacking_meta_path.exists()
        
        if meta_exists:
            print("✅ Ensemble metadata files found")
        else:
            print("⚠️ No ensemble metadata files found (optional)")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
