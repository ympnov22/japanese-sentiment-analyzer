import pytest
import requests
import time
from unittest.mock import patch, MagicMock
from typing import Dict, Any

class TestErrorHandling:
    """Comprehensive error handling tests for the Japanese sentiment analysis API"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_empty_text_validation(self):
        """Test API handling of empty text input"""
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json={"text": ""},
            timeout=10
        )
        
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        
        print("Empty text validation test passed")
    
    def test_whitespace_only_text(self):
        """Test API handling of whitespace-only text"""
        whitespace_inputs = [
            "   ",
            "\n\n\n",
            "\t\t\t",
            "   \n\t   ",
            "　　　"  # Full-width spaces
        ]
        
        for whitespace_text in whitespace_inputs:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": whitespace_text},
                timeout=10
            )
            
            assert response.status_code in [400, 422], f"Whitespace text '{repr(whitespace_text)}' not properly rejected"
        
        print("Whitespace-only text validation test passed")
    
    def test_missing_text_field(self):
        """Test API handling of missing text field"""
        invalid_payloads = [
            {},
            {"message": "この映画は素晴らしかった"},
            {"content": "この映画は素晴らしかった"},
            {"input": "この映画は素晴らしかった"},
            {"data": "この映画は素晴らしかった"}
        ]
        
        for payload in invalid_payloads:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json=payload,
                timeout=10
            )
            
            assert response.status_code == 422, f"Invalid payload {payload} not properly rejected"
        
        print("Missing text field validation test passed")
    
    def test_invalid_data_types(self):
        """Test API handling of invalid data types"""
        invalid_types = [
            {"text": None},
            {"text": 12345},
            {"text": 123.45},
            {"text": True},
            {"text": False},
            {"text": []},
            {"text": {}},
            {"text": ["array", "of", "strings"]}
        ]
        
        for invalid_payload in invalid_types:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json=invalid_payload,
                timeout=10
            )
            
            assert response.status_code == 422, f"Invalid data type {type(invalid_payload['text'])} not properly rejected"
        
        print("Invalid data types validation test passed")
    
    def test_extremely_long_text(self):
        """Test API handling of extremely long text"""
        long_text_sizes = [5000, 10000, 50000, 100000]
        
        for size in long_text_sizes:
            long_text = "あ" * size
            
            try:
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json={"text": long_text},
                    timeout=30
                )
                
                assert response.status_code in [200, 413, 422], f"Unexpected response for {size} character text: {response.status_code}"
                
                if response.status_code == 200:
                    data = response.json()
                    assert "result" in data
                    assert "score" in data
                    print(f"Successfully processed {size} character text")
                else:
                    print(f"Properly rejected {size} character text with status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"Request timeout for {size} character text - acceptable behavior")
            except requests.exceptions.RequestException as e:
                print(f"Request failed for {size} character text: {e}")
        
        print("Extremely long text handling test completed")
    
    def test_unicode_edge_cases(self):
        """Test API handling of various Unicode edge cases"""
        unicode_test_cases = [
            "🎬🎭🎪",  # Emojis
            "𝕋𝕙𝕚𝕤 𝕚𝕤 𝕞𝕒𝕥𝕙 𝕗𝕠𝕟𝕥",  # Mathematical symbols
            "ＴＨＩＳｉｓｆｕｌｌｗｉｄｔｈ",  # Full-width characters
            "Ｔｈｉｓ　ｉｓ　ｍｉｘｅｄ　text",  # Mixed width
            "‌‍‎‏",  # Zero-width characters
            "﷽",  # Arabic ligature
            "👨‍👩‍👧‍👦",  # Complex emoji sequence
            "\u200B\u200C\u200D\uFEFF",  # Various zero-width spaces
        ]
        
        for test_text in unicode_test_cases:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": test_text},
                timeout=10
            )
            
            assert response.status_code in [200, 422], f"Unexpected response for Unicode text: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                assert "result" in data
                assert "score" in data
        
        print("Unicode edge cases test passed")
    
    def test_malformed_json_requests(self):
        """Test API handling of malformed JSON"""
        malformed_json_cases = [
            '{"text": "test"',  # Missing closing brace
            '{"text": test"}',  # Missing quotes around value
            '{"text": "test",}',  # Trailing comma
            '{text: "test"}',  # Missing quotes around key
            '{"text": "test"} extra content',  # Extra content
            '{"text": "test", "text": "duplicate"}',  # Duplicate keys
            '{"": "empty key"}',  # Empty key
            '{"text": }',  # Missing value
            'not json at all',  # Not JSON
            '',  # Empty string
        ]
        
        for malformed_json in malformed_json_cases:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                data=malformed_json,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            assert response.status_code in [400, 422], f"Malformed JSON not properly rejected: {response.status_code}"
        
        print("Malformed JSON handling test passed")
    
    def test_network_timeout_simulation(self):
        """Test client-side timeout handling"""
        short_timeouts = [0.001, 0.01, 0.1]
        
        for timeout in short_timeouts:
            try:
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json={"text": "この映画は素晴らしかった"},
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    print(f"Request completed within {timeout}s timeout")
                    
            except requests.exceptions.Timeout:
                print(f"Request properly timed out after {timeout}s")
            except requests.exceptions.RequestException as e:
                print(f"Request failed with timeout {timeout}s: {e}")
        
        print("Network timeout simulation test completed")
    
    def test_concurrent_error_scenarios(self):
        """Test error handling under concurrent load"""
        import concurrent.futures
        
        def make_invalid_request():
            try:
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json={"text": ""},  # Invalid empty text
                    timeout=5
                )
                return response.status_code
            except requests.exceptions.RequestException:
                return 0
        
        num_concurrent = 20
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_invalid_request) for _ in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        error_responses = sum(1 for status in results if status == 422)
        successful_error_handling = error_responses / len(results)
        
        print(f"Concurrent error handling: {error_responses}/{len(results)} properly handled")
        assert successful_error_handling >= 0.8, f"Error handling success rate {successful_error_handling:.2%} below 80%"
        
        print("Concurrent error scenarios test passed")
    
    def test_health_endpoint_error_scenarios(self):
        """Test health endpoint error scenarios"""
        response = requests.get(f"{self.BASE_URL}/health", timeout=5)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "message" in data
        
        invalid_methods = ["POST", "PUT", "DELETE", "PATCH"]
        for method in invalid_methods:
            response = requests.request(method, f"{self.BASE_URL}/health", timeout=5)
            assert response.status_code == 405, f"Method {method} not properly rejected for health endpoint"
        
        print("Health endpoint error scenarios test passed")
    
    def test_content_type_errors(self):
        """Test content type validation errors"""
        invalid_content_types = [
            "text/plain",
            "application/xml",
            "multipart/form-data",
            "application/x-www-form-urlencoded",
            "text/html",
            "image/jpeg",
            "application/octet-stream"
        ]
        
        for content_type in invalid_content_types:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                data='{"text": "test"}',
                headers={"Content-Type": content_type},
                timeout=10
            )
            
            assert response.status_code in [415, 422], f"Invalid content type {content_type} not properly rejected"
        
        print("Content type validation errors test passed")
    
    def test_large_payload_errors(self):
        """Test large payload handling"""
        large_payloads = [
            {"text": "a" * 1000000},  # 1MB text
            {"text": "あ" * 500000},   # Large Japanese text
        ]
        
        for payload in large_payloads:
            try:
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json=payload,
                    timeout=30
                )
                
                assert response.status_code in [200, 413, 422], f"Unexpected response to large payload: {response.status_code}"
                
                if response.status_code == 413:
                    print("Large payload properly rejected with 413 Payload Too Large")
                elif response.status_code == 422:
                    print("Large payload properly rejected with 422 Validation Error")
                else:
                    print("Large payload processed successfully")
                    
            except requests.exceptions.RequestException as e:
                print(f"Large payload request failed: {e}")
        
        print("Large payload errors test completed")
    
    def test_api_error_response_format(self):
        """Test that error responses follow consistent format"""
        error_scenarios = [
            {"json": {"text": ""}, "expected_status": 422},
            {"json": {}, "expected_status": 422},
            {"json": {"text": None}, "expected_status": 422},
        ]
        
        for scenario in error_scenarios:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json=scenario["json"],
                timeout=10
            )
            
            assert response.status_code == scenario["expected_status"]
            
            try:
                data = response.json()
                assert isinstance(data, dict), "Error response should be JSON object"
                assert "detail" in data, "Error response should contain 'detail' field"
            except ValueError:
                pytest.fail("Error response should be valid JSON")
        
        print("API error response format test passed")

class TestModelErrorHandling:
    """Test error handling related to model operations"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_model_prediction_edge_cases(self):
        """Test model prediction with edge case inputs"""
        edge_cases = [
            ".",  # Single punctuation
            "123",  # Numbers only
            "!@#$%^&*()",  # Special characters only
            "ａｂｃ",  # Full-width ASCII
            "　",  # Full-width space
            "〜",  # Wave dash
            "・",  # Middle dot
        ]
        
        for text in edge_cases:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": text},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "result" in data
                assert "score" in data
                assert data["result"] in ["ポジティブ", "ネガティブ"]
                assert 0 <= data["score"] <= 1
                print(f"Edge case '{text}' processed successfully")
            else:
                print(f"Edge case '{text}' rejected with status {response.status_code}")
        
        print("Model prediction edge cases test completed")
    
    def test_model_consistency_under_load(self):
        """Test model prediction consistency under load"""
        test_text = "この映画は素晴らしかった"
        num_requests = 50
        
        results = []
        for i in range(num_requests):
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": test_text},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results.append((data["result"], data["score"]))
            else:
                print(f"Request {i} failed with status {response.status_code}")
        
        if results:
            first_result = results[0]
            consistent_results = sum(1 for result in results if result == first_result)
            consistency_rate = consistent_results / len(results)
            
            print(f"Model consistency: {consistent_results}/{len(results)} ({consistency_rate:.2%})")
            assert consistency_rate >= 0.95, f"Model consistency {consistency_rate:.2%} below 95%"
        
        print("Model consistency under load test completed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
