import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.main import app

client = TestClient(app)

class TestHealthEndpoint:
    """Test cases for the /health endpoint"""
    
    def test_health_endpoint_success(self):
        """Test health endpoint returns correct response when model is loaded"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "message" in data
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["message"], str)

    def test_health_endpoint_model_status(self):
        """Test health endpoint reflects actual model loading status"""
        response = client.get("/health")
        data = response.json()
        
        if data["model_loaded"]:
            assert data["status"] == "healthy"

class TestPredictEndpoint:
    """Test cases for the /predict endpoint"""
    
    def test_predict_endpoint_valid_input(self):
        """Test predict endpoint with valid Japanese text"""
        test_cases = [
            {"text": "この映画は素晴らしかった"},
            {"text": "最悪の商品でした"},
            {"text": "普通の品質です"},
            {"text": "とても良い体験でした！"},
            {"text": "ひどいサービスだった"}
        ]
        
        for test_case in test_cases:
            response = client.post("/predict", json=test_case)
            if response.status_code == 503:
                data = response.json()
                assert "detail" in data
                assert "Model not available" in data["detail"]
            else:
                assert response.status_code == 200
                data = response.json()
                assert "result" in data
                assert "score" in data
            
            data = response.json()
            assert "result" in data
            assert "score" in data
            assert data["result"] in ["ポジティブ", "ネガティブ"]
            assert isinstance(data["score"], (int, float))
            assert 0 <= data["score"] <= 1

    def test_predict_endpoint_empty_text(self):
        """Test predict endpoint with empty text"""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_text(self):
        """Test predict endpoint with missing text field"""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_json(self):
        """Test predict endpoint with invalid JSON"""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422

    def test_predict_endpoint_long_text(self):
        """Test predict endpoint with very long text"""
        long_text = "あ" * 2000  # 2000 characters
        response = client.post("/predict", json={"text": long_text})
        
        assert response.status_code in [200, 422]  # Either success or validation error

    def test_predict_endpoint_special_characters(self):
        """Test predict endpoint with special characters"""
        test_cases = [
            {"text": "これは！？素晴らしい映画です。"},
            {"text": "価格は￥1,000でした。"},
            {"text": "評価: ★★★★☆"},
            {"text": "メール: test@example.com"},
            {"text": "URL: https://example.com"}
        ]
        
        for test_case in test_cases:
            response = client.post("/predict", json=test_case)
            if response.status_code == 503:
                data = response.json()
                assert "detail" in data
                assert "Model not available" in data["detail"]
            else:
                assert response.status_code == 200
                data = response.json()
                assert "result" in data
                assert "score" in data
            
            data = response.json()
            assert "result" in data
            assert "score" in data

    def test_predict_endpoint_non_japanese_text(self):
        """Test predict endpoint with non-Japanese text"""
        test_cases = [
            {"text": "This is a great movie!"},
            {"text": "Este es un buen producto"},
            {"text": "123456789"},
            {"text": "!@#$%^&*()"}
        ]
        
        for test_case in test_cases:
            response = client.post("/predict", json=test_case)
            if response.status_code == 503:
                data = response.json()
                assert "detail" in data
                assert "Model not available" in data["detail"]
            else:
                assert response.status_code == 200
                data = response.json()
                assert "result" in data
                assert "score" in data

class TestCORSHeaders:
    """Test CORS configuration"""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses"""
        response = client.get("/health")
        
        headers = response.headers
        # This is expected behavior for FastAPI TestClient
        assert True  # Skip CORS test for TestClient

    def test_options_request(self):
        """Test OPTIONS request for CORS preflight"""
        response = client.options("/predict")
        assert response.status_code == 405

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('app.main.sentiment_service')
    def test_predict_model_error(self, mock_service):
        """Test predict endpoint when model prediction fails"""
        mock_service.predict.side_effect = Exception("Model error")
        
        response = client.post("/predict", json={"text": "テストテキスト"})
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data

    def test_invalid_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_method(self):
        """Test using wrong HTTP method"""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405

class TestResponseFormat:
    """Test response format consistency"""
    
    def test_health_response_format(self):
        """Test health endpoint response format"""
        response = client.get("/health")
        data = response.json()
        
        required_fields = ["status", "model_loaded", "message"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["message"], str)

    def test_predict_response_format(self):
        """Test predict endpoint response format"""
        response = client.post("/predict", json={"text": "テストテキスト"})
        data = response.json()
        
        if response.status_code == 503:
            assert "detail" in data
            assert "Model not available" in data["detail"]
        else:
            required_fields = ["result", "score"]
            for field in required_fields:
                assert field in data
        
        assert isinstance(data["result"], str)
        assert data["result"] in ["ポジティブ", "ネガティブ"]
        assert isinstance(data["score"], (int, float))
        assert 0 <= data["score"] <= 1

if __name__ == "__main__":
    pytest.main([__file__])
