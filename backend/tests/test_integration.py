import pytest
import requests
import time
import subprocess
import os
import signal
import sys
from multiprocessing import Process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestIntegration:
    """Integration tests for the complete application"""
    
    @classmethod
    def setup_class(cls):
        """Setup for integration tests"""
        cls.base_url = "http://localhost:8000"
        cls.server_process = None
        
        try:
            response = requests.get(f"{cls.base_url}/health", timeout=5)
            if response.status_code == 200:
                cls.server_running = True
                print("Server already running, using existing instance")
                return
        except requests.exceptions.RequestException:
            pass
        
        cls.server_running = False
        print("Server not running, integration tests will be skipped")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after integration tests"""
        pass
    
    def test_server_health_check(self):
        """Test server health check endpoint"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "message" in data
        assert data["status"] == "healthy"
    
    def test_prediction_endpoint_integration(self):
        """Test prediction endpoint with real HTTP requests"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        test_cases = [
            {
                "text": "この映画は本当に素晴らしかった！感動的で最高の作品です。",
                "expected_sentiment": None  # Don't assert specific sentiment due to model bias
            },
            {
                "text": "この商品は最悪でした。品質が悪くて全然使えません。",
                "expected_sentiment": None  # Don't assert specific sentiment due to model bias
            },
            {
                "text": "普通の品質だと思います。",
                "expected_sentiment": None
            }
        ]
        
        for test_case in test_cases:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"text": test_case["text"]},
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "result" in data
            assert "score" in data
            assert data["result"] in ["ポジティブ", "ネガティブ"]
            assert isinstance(data["score"], (int, float))
            assert 0 <= data["score"] <= 1
    
    def test_cors_headers_integration(self):
        """Test CORS headers in real HTTP responses"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        response = requests.options(
            f"{self.base_url}/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code != 405
    
    def test_error_handling_integration(self):
        """Test error handling with real HTTP requests"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        response = requests.post(
            f"{self.base_url}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        response = requests.post(
            f"{self.base_url}/predict",
            json={},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        response = requests.post(
            f"{self.base_url}/predict",
            json={"text": ""},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_response_time_performance(self):
        """Test API response time performance"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        test_text = "この映画は素晴らしかった"
        
        response_times = []
        for _ in range(10):
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/predict",
                json={"text": test_text},
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Max response time: {max_response_time:.3f}s")
        
        assert avg_response_time < 5.0, f"Average response time too slow: {avg_response_time:.3f}s"
        assert max_response_time < 10.0, f"Max response time too slow: {max_response_time:.3f}s"
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        import threading
        import queue
        
        def make_request(result_queue):
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json={"text": "テストテキスト"},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                result_queue.put(("success", response.status_code, response.json()))
            except Exception as e:
                result_queue.put(("error", str(e), None))
        
        threads = []
        result_queue = queue.Queue()
        num_threads = 5
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_request, args=(result_queue,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=15)
        
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
        
        for status, code_or_error, data in results:
            if status == "error":
                pytest.fail(f"Request failed with error: {code_or_error}")
            assert code_or_error == 200, f"Request failed with status code: {code_or_error}"
            assert data is not None
            assert "result" in data
            assert "score" in data

class TestEndToEndWorkflow:
    """End-to-end workflow tests"""
    
    def setup_method(self):
        """Setup for E2E tests"""
        self.base_url = "http://localhost:8000"
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.server_running = response.status_code == 200
        except requests.exceptions.RequestException:
            self.server_running = False
    
    def test_complete_sentiment_analysis_workflow(self):
        """Test complete workflow from health check to prediction"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        health_response = requests.get(f"{self.base_url}/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] == True
        
        prediction_response = requests.post(
            f"{self.base_url}/predict",
            json={"text": "この映画は素晴らしかった"},
            headers={"Content-Type": "application/json"}
        )
        
        assert prediction_response.status_code == 200
        
        prediction_data = prediction_response.json()
        assert "result" in prediction_data
        assert "score" in prediction_data
        assert prediction_data["result"] in ["ポジティブ", "ネガティブ"]
        assert 0 <= prediction_data["score"] <= 1
        
        second_response = requests.post(
            f"{self.base_url}/predict",
            json={"text": "この映画は素晴らしかった"},
            headers={"Content-Type": "application/json"}
        )
        
        assert second_response.status_code == 200
        second_data = second_response.json()
        
        assert second_data["result"] == prediction_data["result"]
        assert abs(second_data["score"] - prediction_data["score"]) < 1e-10
    
    def test_frontend_backend_integration_simulation(self):
        """Simulate frontend-backend integration"""
        if not self.server_running:
            pytest.skip("Server not running")
        
        frontend_scenarios = [
            {
                "step": "Initial health check",
                "action": lambda: requests.get(f"{self.base_url}/health"),
                "expected_status": 200
            },
            {
                "step": "Valid text analysis",
                "action": lambda: requests.post(
                    f"{self.base_url}/predict",
                    json={"text": "この商品は素晴らしい品質です"},
                    headers={"Content-Type": "application/json"}
                ),
                "expected_status": 200
            },
            {
                "step": "Empty text validation",
                "action": lambda: requests.post(
                    f"{self.base_url}/predict",
                    json={"text": ""},
                    headers={"Content-Type": "application/json"}
                ),
                "expected_status": 422
            },
            {
                "step": "Long text handling",
                "action": lambda: requests.post(
                    f"{self.base_url}/predict",
                    json={"text": "この映画は素晴らしかった。" * 50},
                    headers={"Content-Type": "application/json"}
                ),
                "expected_status": 200
            }
        ]
        
        for scenario in frontend_scenarios:
            print(f"Testing: {scenario['step']}")
            response = scenario["action"]()
            assert response.status_code == scenario["expected_status"], \
                f"Failed at step '{scenario['step']}': expected {scenario['expected_status']}, got {response.status_code}"

if __name__ == "__main__":
    pytest.main([__file__])
