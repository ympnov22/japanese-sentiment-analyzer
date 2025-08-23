import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.main import SentimentAnalysisService

class TestSentimentAnalysisService:
    """Test cases for the SentimentAnalysisService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SentimentAnalysisService()
    
    def test_service_initialization(self):
        """Test service initializes correctly"""
        assert self.service is not None
        assert hasattr(self.service, 'vectorizer')
        assert hasattr(self.service, 'classifier')
        assert hasattr(self.service, 'is_loaded')
    
    def test_model_loading(self):
        """Test model loading functionality"""
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        vectorizer_path = os.path.join(models_dir, 'japanese_sentiment_vectorizer.pkl')
        classifier_path = os.path.join(models_dir, 'japanese_sentiment_classifier.pkl')
        
        if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
            assert self.service.is_loaded == True
            assert self.service.vectorizer is not None
            assert self.service.classifier is not None
        else:
            assert self.service.is_loaded == False
    
    def test_predict_with_trained_model(self):
        """Test prediction with trained model"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping prediction test")
        
        test_texts = [
            "この映画は素晴らしかった",
            "最悪の商品でした",
            "普通の品質です",
            "とても良い体験でした",
            "ひどいサービスだった"
        ]
        
        for text in test_texts:
            result = self.service.predict(text)
            
            assert isinstance(result, dict)
            assert "result" in result
            assert "score" in result
            assert result["result"] in ["ポジティブ", "ネガティブ"]
            assert isinstance(result["score"], (int, float))
            assert 0 <= result["score"] <= 1
    
    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping prediction test")
        
        with pytest.raises(ValueError):
            self.service.predict("")
    
    def test_predict_whitespace_only(self):
        """Test prediction with whitespace-only text"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping prediction test")
        
        with pytest.raises(ValueError):
            self.service.predict("   \n\t   ")
    
    def test_predict_very_long_text(self):
        """Test prediction with very long text"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping prediction test")
        
        long_text = "この映画は素晴らしかった。" * 100  # Repeat 100 times
        result = self.service.predict(long_text)
        
        assert isinstance(result, dict)
        assert "result" in result
        assert "score" in result
    
    def test_predict_special_characters(self):
        """Test prediction with special characters"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping prediction test")
        
        special_texts = [
            "これは！？素晴らしい映画です。",
            "価格は￥1,000でした。",
            "評価: ★★★★☆",
            "メール: test@example.com",
            "URL: https://example.com"
        ]
        
        for text in special_texts:
            result = self.service.predict(text)
            assert isinstance(result, dict)
            assert "result" in result
            assert "score" in result
    
    @patch('joblib.load')
    def test_model_loading_error(self, mock_joblib_load):
        """Test model loading error handling"""
        mock_joblib_load.side_effect = FileNotFoundError("Model file not found")
        
        service = SentimentAnalysisService()
        assert service.is_loaded == False
        assert service.vectorizer is None
        assert service.classifier is None
    
    def test_predict_without_trained_model(self):
        """Test prediction without trained model"""
        service = SentimentAnalysisService()
        service.is_loaded = False
        service.vectorizer = None
        service.classifier = None
        
        with pytest.raises(Exception):
            service.predict("テストテキスト")
    
    @patch('app.main.SentimentAnalysisService.load_model')
    def test_service_initialization_with_load_error(self, mock_load_models):
        """Test service initialization when model loading fails"""
        mock_load_models.side_effect = Exception("Loading error")
        
        service = SentimentAnalysisService()
        assert service.is_loaded == False
    
    def test_text_preprocessing(self):
        """Test text preprocessing functionality"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping preprocessing test")
        
        test_cases = [
            ("  テストテキスト  ", "should handle leading/trailing spaces"),
            ("テスト\nテキスト", "should handle newlines"),
            ("テスト\tテキスト", "should handle tabs"),
            ("テスト　テキスト", "should handle full-width spaces"),
        ]
        
        for text, description in test_cases:
            try:
                result = self.service.predict(text)
                assert isinstance(result, dict), f"Failed: {description}"
            except Exception as e:
                pytest.fail(f"Failed {description}: {str(e)}")

class TestModelConsistency:
    """Test model prediction consistency"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.service = SentimentAnalysisService()
    
    def test_prediction_consistency(self):
        """Test that same input produces same output"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping consistency test")
        
        test_text = "この映画は素晴らしかった"
        
        results = []
        for _ in range(5):
            result = self.service.predict(test_text)
            results.append(result)
        
        first_result = results[0]
        for result in results[1:]:
            assert result["result"] == first_result["result"]
            assert abs(result["score"] - first_result["score"]) < 1e-10
    
    def test_score_range_validation(self):
        """Test that all prediction scores are in valid range"""
        if not self.service.is_loaded:
            pytest.skip("Model not trained, skipping score validation test")
        
        test_texts = [
            "最高の映画でした",
            "普通の品質です",
            "最悪の商品でした",
            "とても良い",
            "ひどい"
        ]
        
        for text in test_texts:
            result = self.service.predict(text)
            score = result["score"]
            
            assert isinstance(score, (int, float)), f"Score should be numeric for text: {text}"
            assert 0 <= score <= 1, f"Score {score} out of range [0,1] for text: {text}"

if __name__ == "__main__":
    pytest.main([__file__])
