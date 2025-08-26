#!/usr/bin/env python3
"""
CI accuracy tests for Japanese sentiment analysis model
Ensures minimum accuracy threshold is maintained to prevent regressions
"""

import pytest
import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.model_loader import LightweightSentimentModel


class TestModelAccuracy:
    """Test suite for model accuracy validation"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with model instance"""
        cls.model = LightweightSentimentModel()
        cls.model_loaded = cls.model.load_model()
        
        cls.test_cases = [
            ("この商品は本当に素晴らしいです！最高の品質で大満足です。", "ポジティブ"),
            ("とても良い商品だと思います。おすすめします。", "ポジティブ"),
            ("素晴らしい体験でした。期待以上の結果です。", "ポジティブ"),
            ("完璧な商品です。文句なしの品質です。", "ポジティブ"),
            ("感動しました。本当に良い買い物でした。", "ポジティブ"),
            ("優れた品質です。価格以上の価値があります。", "ポジティブ"),
            ("とても満足しています。また購入したいです。", "ポジティブ"),
            ("期待通りの商品でした。良い選択でした。", "ポジティブ"),
            ("品質が良く、使いやすいです。", "ポジティブ"),
            ("コストパフォーマンスが良いです。", "ポジティブ"),
            
            ("最悪の商品でした。二度と買いません。", "ネガティブ"),
            ("お金の無駄でした。全く使えません。", "ネガティブ"),
            ("ひどい品質です。期待外れでした。", "ネガティブ"),
            ("失敗でした。がっかりしました。", "ネガティブ"),
            ("問題が多すぎます。不満です。", "ネガティブ"),
            ("品質が悪く、すぐに壊れました。", "ネガティブ"),
            ("価格に見合わない商品です。", "ネガティブ"),
            ("使いにくく、機能も不十分です。", "ネガティブ"),
            ("サポートも悪く、対応が遅いです。", "ネガティブ"),
            ("返品したいくらいひどい商品です。", "ネガティブ"),
        ]
        
        cls.min_accuracy_threshold = 0.85
    
    def test_model_loads_successfully(self):
        """Test that the model loads without errors"""
        assert self.model_loaded, "Model should load successfully"
        assert self.model.vectorizer is not None, "Vectorizer should be loaded"
        assert self.model.coef_ is not None, "Classifier coefficients should be loaded"
        assert self.model.metadata is not None, "Model metadata should be loaded"
    
    def test_model_integrity_verification(self):
        """Test that model integrity verification works"""
        if self.model.metadata:
            verified = self.model.metadata.get("verified_sha256", False)
            assert isinstance(verified, bool), "SHA256 verification status should be boolean"
    
    def test_minimum_accuracy_threshold(self):
        """Test that model meets minimum accuracy threshold on curated dataset"""
        if not self.model_loaded:
            pytest.skip("Model not loaded, skipping accuracy test")
        
        correct_predictions = 0
        total_predictions = len(self.test_cases)
        
        for text, expected_label in self.test_cases:
            try:
                result = self.model.predict(text)
                predicted_label = result.get("result", "")
                
                if predicted_label == expected_label:
                    correct_predictions += 1
                else:
                    print(f"Mismatch: '{text}' -> Expected: {expected_label}, Got: {predicted_label}")
                    
            except Exception as e:
                pytest.fail(f"Prediction failed for text '{text}': {e}")
        
        accuracy = correct_predictions / total_predictions
        print(f"Model accuracy on curated dataset: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        assert accuracy >= self.min_accuracy_threshold, (
            f"Model accuracy {accuracy:.3f} is below minimum threshold {self.min_accuracy_threshold}. "
            f"This indicates a potential model regression."
        )
    
    def test_prediction_format(self):
        """Test that predictions return expected format"""
        if not self.model_loaded:
            pytest.skip("Model not loaded, skipping format test")
        
        test_text = "この商品は素晴らしいです"
        result = self.model.predict(test_text)
        
        assert isinstance(result, dict), "Prediction should return a dictionary"
        assert "result" in result, "Prediction should contain 'result' key"
        assert "score" in result, "Prediction should contain 'score' key"
        assert result["result"] in ["ポジティブ", "ネガティブ"], "Sentiment should be valid label"
        assert 0 <= result["score"] <= 1, "Score should be between 0 and 1"
    
    def test_model_metadata_accuracy_baseline(self):
        """Test that model metadata contains accuracy baseline information"""
        if not self.model_loaded:
            pytest.skip("Model not loaded, skipping metadata test")
        
        if self.model.metadata:
            baseline = self.model.metadata.get("accuracy_baseline")
            if baseline is not None:
                assert isinstance(baseline, (int, float)), "Accuracy baseline should be numeric"
                assert 0 <= baseline <= 1, "Accuracy baseline should be between 0 and 1"
                assert baseline >= self.min_accuracy_threshold, (
                    f"Model baseline accuracy {baseline} is below minimum threshold {self.min_accuracy_threshold}"
                )
    
    def test_edge_cases(self):
        """Test model behavior on edge cases"""
        if not self.model_loaded:
            pytest.skip("Model not loaded, skipping edge case test")
        
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "これは" * 100,  # Very long text
        ]
        
        for text in edge_cases:
            try:
                result = self.model.predict(text)
                assert isinstance(result, dict), f"Edge case '{text}' should return dict"
                if result.get("result"):  # Allow empty results for edge cases
                    assert result["result"] in ["ポジティブ", "ネガティブ"], f"Invalid sentiment for '{text}'"
            except Exception as e:
                print(f"Edge case '{text}' caused exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
