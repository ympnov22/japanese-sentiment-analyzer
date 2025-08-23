import pytest
import requests
import json
from typing import Dict, Any

class TestSecurity:
    """Security testing for the Japanese sentiment analysis API"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_cors_headers(self):
        """Test CORS headers are properly configured"""
        response = requests.get(f"{self.BASE_URL}/health")
        
        assert response.status_code == 200
        
        headers = response.headers
        cors_headers = [
            "access-control-allow-origin",
            "Access-Control-Allow-Origin"
        ]
        
        has_cors = any(header.lower() in [h.lower() for h in headers.keys()] for header in cors_headers)
        assert has_cors, "CORS headers not found in response"
        
        print("CORS headers found in response")
    
    def test_options_preflight_request(self):
        """Test OPTIONS preflight request handling"""
        response = requests.options(
            f"{self.BASE_URL}/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code in [200, 204], f"OPTIONS request failed with status {response.status_code}"
        print(f"OPTIONS preflight request handled correctly: {response.status_code}")
    
    def test_sql_injection_attempts(self):
        """Test API resilience against SQL injection attempts"""
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; SELECT * FROM users; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "admin'/*",
            "' OR 1=1#",
            "' OR 1=1--",
            "' OR 1=1/*"
        ]
        
        for payload in sql_injection_payloads:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": payload},
                timeout=10
            )
            
            assert response.status_code in [200, 422], f"Unexpected response to SQL injection: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                assert "result" in data
                assert "score" in data
                assert data["result"] in ["ポジティブ", "ネガティブ"]
        
        print("SQL injection tests passed - API handles malicious SQL payloads safely")
    
    def test_xss_injection_attempts(self):
        """Test API resilience against XSS injection attempts"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "';alert('XSS');//",
            "<script>document.location='http://evil.com'</script>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>"
        ]
        
        for payload in xss_payloads:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": payload},
                timeout=10
            )
            
            assert response.status_code in [200, 422], f"Unexpected response to XSS injection: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                assert "result" in data
                assert "score" in data
                
                response_text = json.dumps(data)
                assert "<script>" not in response_text, "XSS payload found in response"
                assert "javascript:" not in response_text, "JavaScript protocol found in response"
        
        print("XSS injection tests passed - API handles malicious scripts safely")
    
    def test_command_injection_attempts(self):
        """Test API resilience against command injection attempts"""
        command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "; cat /etc/shadow",
            "| whoami",
            "; id",
            "&& cat /proc/version",
            "; uname -a",
            "| ps aux",
            "; netstat -an"
        ]
        
        for payload in command_injection_payloads:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": payload},
                timeout=10
            )
            
            assert response.status_code in [200, 422], f"Unexpected response to command injection: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                assert "result" in data
                assert "score" in data
                assert data["result"] in ["ポジティブ", "ネガティブ"]
        
        print("Command injection tests passed - API handles malicious commands safely")
    
    def test_path_traversal_attempts(self):
        """Test API resilience against path traversal attempts"""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "../../../proc/self/environ",
            "..\\..\\..\\boot.ini",
            "....\\\\....\\\\....\\\\windows\\\\system32\\\\drivers\\\\etc\\\\hosts"
        ]
        
        for payload in path_traversal_payloads:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": payload},
                timeout=10
            )
            
            assert response.status_code in [200, 422], f"Unexpected response to path traversal: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                assert "result" in data
                assert "score" in data
        
        print("Path traversal tests passed - API handles directory traversal attempts safely")
    
    def test_large_payload_handling(self):
        """Test API handling of extremely large payloads"""
        large_text = "あ" * 100000  # 100KB of Japanese characters
        
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json={"text": large_text},
            timeout=30
        )
        
        assert response.status_code in [200, 413, 422], f"Unexpected response to large payload: {response.status_code}"
        
        if response.status_code == 413:
            print("Large payload correctly rejected with 413 Payload Too Large")
        elif response.status_code == 422:
            print("Large payload correctly rejected with 422 Validation Error")
        else:
            print("Large payload processed successfully")
    
    def test_malformed_json_handling(self):
        """Test API handling of malformed JSON"""
        malformed_payloads = [
            '{"text": "test"',  # Missing closing brace
            '{"text": test"}',  # Missing quotes
            '{"text": "test", }',  # Trailing comma
            '{text: "test"}',  # Missing quotes on key
            '{"text": "test"} extra',  # Extra content
            '{"text": "test", "text": "duplicate"}',  # Duplicate keys
            '{"": "empty key"}',  # Empty key
            '{"text": }',  # Missing value
        ]
        
        for payload in malformed_payloads:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            assert response.status_code in [400, 422], f"Malformed JSON not properly rejected: {response.status_code}"
        
        print("Malformed JSON tests passed - API properly rejects invalid JSON")
    
    def test_http_method_security(self):
        """Test that only allowed HTTP methods are accepted"""
        methods_to_test = ["PUT", "DELETE", "PATCH", "TRACE", "CONNECT"]
        
        for method in methods_to_test:
            response = requests.request(
                method,
                f"{self.BASE_URL}/predict",
                json={"text": "test"},
                timeout=10
            )
            
            assert response.status_code in [405, 501], f"Method {method} not properly rejected: {response.status_code}"
        
        print("HTTP method security tests passed - Only allowed methods accepted")
    
    def test_content_type_validation(self):
        """Test content type validation"""
        invalid_content_types = [
            "text/plain",
            "application/xml",
            "multipart/form-data",
            "application/x-www-form-urlencoded"
        ]
        
        for content_type in invalid_content_types:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                data='{"text": "test"}',
                headers={"Content-Type": content_type},
                timeout=10
            )
            
            assert response.status_code in [415, 422], f"Invalid content type {content_type} not rejected: {response.status_code}"
        
        print("Content type validation tests passed")
    
    def test_rate_limiting_behavior(self):
        """Test if there's any rate limiting in place"""
        rapid_requests = 50
        text = "テスト"
        
        responses = []
        for i in range(rapid_requests):
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": text},
                timeout=5
            )
            responses.append(response.status_code)
        
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        print(f"Rapid requests: {success_count}/{rapid_requests} successful, {rate_limited_count} rate limited")
        
        if rate_limited_count > 0:
            print("Rate limiting detected - good security practice")
        else:
            print("No rate limiting detected - consider implementing for production")
    
    def test_information_disclosure(self):
        """Test for information disclosure in error responses"""
        invalid_requests = [
            {"invalid": "field"},
            {"text": None},
            {"text": 12345},
            {"text": []},
            {"text": {}}
        ]
        
        for invalid_request in invalid_requests:
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json=invalid_request,
                timeout=10
            )
            
            if response.status_code >= 400:
                response_text = response.text.lower()
                
                sensitive_info = [
                    "traceback",
                    "stack trace",
                    "file path",
                    "/home/",
                    "/usr/",
                    "python",
                    "fastapi",
                    "uvicorn"
                ]
                
                for info in sensitive_info:
                    assert info not in response_text, f"Sensitive information '{info}' found in error response"
        
        print("Information disclosure tests passed - No sensitive data in error responses")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
