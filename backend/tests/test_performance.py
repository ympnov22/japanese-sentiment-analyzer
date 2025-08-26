import pytest
import time
import asyncio
import concurrent.futures
import requests
import statistics
from typing import List, Dict, Any

class TestPerformance:
    """Performance testing for the Japanese sentiment analysis API"""
    
    BASE_URL = "http://localhost:8000"
    
    def setup_method(self):
        """Setup test fixtures"""
        self.test_texts = [
            "この映画は素晴らしかった",
            "最悪の商品でした",
            "普通の品質です",
            "とても良い体験でした",
            "ひどいサービスだった",
            "価格に見合った品質だと思います",
            "期待していたほどではありませんでした",
            "非常に満足しています",
            "改善の余地があります",
            "完璧な商品です"
        ]
    
    def test_api_response_time_single_request(self):
        """Test single API request response time"""
        text = "この映画は素晴らしかった"
        
        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json={"text": text},
            timeout=10
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0, f"Response time {response_time:.2f}s exceeds 5s threshold"
        
        print(f"Single request response time: {response_time:.3f}s")
    
    def test_api_response_time_multiple_texts(self):
        """Test API response time with different text lengths"""
        response_times = []
        
        for text in self.test_texts:
            start_time = time.time()
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": text},
                timeout=10
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert response.status_code == 200
            assert response_time < 5.0, f"Response time {response_time:.2f}s exceeds 5s threshold for text: {text[:50]}..."
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Max response time: {max_response_time:.3f}s")
        print(f"Min response time: {min_response_time:.3f}s")
        
        assert avg_response_time < 3.0, f"Average response time {avg_response_time:.2f}s exceeds 3s threshold"
    
    def test_concurrent_requests_performance(self):
        """Test API performance under concurrent load"""
        num_concurrent_requests = 10
        text = "この映画は素晴らしかった"
        
        def make_request():
            start_time = time.time()
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": text},
                timeout=15
            )
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_requests = sum(1 for result in results if result["success"])
        response_times = [result["response_time"] for result in results if result["success"]]
        
        assert successful_requests == num_concurrent_requests, f"Only {successful_requests}/{num_concurrent_requests} requests succeeded"
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        print(f"Concurrent requests: {num_concurrent_requests}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Max response time: {max_response_time:.3f}s")
        print(f"Requests per second: {num_concurrent_requests / total_time:.2f}")
        
        assert avg_response_time < 8.0, f"Average response time under load {avg_response_time:.2f}s exceeds 8s threshold"
        assert max_response_time < 15.0, f"Max response time under load {max_response_time:.2f}s exceeds 15s threshold"
    
    def test_long_text_performance(self):
        """Test API performance with long text inputs"""
        base_text = "この映画は本当に素晴らしかった。"
        
        text_lengths = [100, 500, 1000]
        
        for length in text_lengths:
            long_text = (base_text * (length // len(base_text) + 1))[:length]
            
            start_time = time.time()
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": long_text},
                timeout=15
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < 10.0, f"Response time {response_time:.2f}s exceeds 10s threshold for {length} chars"
            
            print(f"Text length {length} chars: {response_time:.3f}s")
    
    def test_health_endpoint_performance(self):
        """Test health endpoint response time"""
        response_times = []
        
        for _ in range(10):
            start_time = time.time()
            response = requests.get(f"{self.BASE_URL}/health", timeout=5)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert response.status_code == 200
            assert response_time < 1.0, f"Health endpoint response time {response_time:.2f}s exceeds 1s threshold"
        
        avg_response_time = statistics.mean(response_times)
        print(f"Health endpoint average response time: {avg_response_time:.3f}s")
        
        assert avg_response_time < 0.5, f"Health endpoint average response time {avg_response_time:.2f}s exceeds 0.5s threshold"
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        text = "この映画は素晴らしかった"
        
        for i in range(50):
            response = requests.post(
                f"{self.BASE_URL}/predict",
                json={"text": text},
                timeout=10
            )
            assert response.status_code == 200
            
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                print(f"Request {i}: Memory usage: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
                
                assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB, possible memory leak"

class TestStressTest:
    """Stress testing for the API"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_rapid_sequential_requests(self):
        """Test rapid sequential requests"""
        text = "テストテキスト"
        num_requests = 100
        
        start_time = time.time()
        successful_requests = 0
        
        for i in range(num_requests):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json={"text": text},
                    timeout=5
                )
                if response.status_code == 200:
                    successful_requests += 1
            except requests.exceptions.RequestException:
                pass
        
        end_time = time.time()
        total_time = end_time - start_time
        success_rate = successful_requests / num_requests
        
        print(f"Rapid sequential requests: {successful_requests}/{num_requests} successful")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests per second: {num_requests / total_time:.2f}")
        
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95% threshold"
    
    def test_burst_load(self):
        """Test burst load handling"""
        text = "この映画は素晴らしかった"
        burst_size = 20
        
        def make_burst_request():
            try:
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json={"text": text},
                    timeout=10
                )
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = [executor.submit(make_burst_request) for _ in range(burst_size)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()
        
        successful_requests = sum(results)
        success_rate = successful_requests / burst_size
        total_time = end_time - start_time
        
        print(f"Burst load test: {successful_requests}/{burst_size} successful")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Burst completion time: {total_time:.2f}s")
        
        assert success_rate >= 0.90, f"Burst load success rate {success_rate:.2%} below 90% threshold"
        assert total_time < 30.0, f"Burst completion time {total_time:.2f}s exceeds 30s threshold"

    def test_batch_processing_performance(self):
        """Test batch processing meets performance targets"""
        texts = ["テストテキスト"] * 100
        
        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/predict/batch",
            json={"texts": texts},
            timeout=30
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        total_time = end_time - start_time
        texts_per_second = len(texts) / total_time
        
        print(f"Batch processing: {len(texts)} texts in {total_time:.2f}s")
        print(f"Throughput: {texts_per_second:.2f} texts/second")
        
        min_throughput = 50  # Reasonable minimum for 100 texts
        assert texts_per_second > min_throughput, f"Batch throughput {texts_per_second:.2f} below {min_throughput} texts/second"
        
        data = response.json()
        assert "results" in data
        assert "summary" in data
        assert "total_processing_time_ms" in data
        assert len(data["results"]) == len(texts)
        
        # Check summary statistics
        summary = data["summary"]
        assert summary["total_texts"] == len(texts)
        assert summary["successful_predictions"] > 0
        assert "sentiment_distribution" in summary
        assert "processing_stats" in summary
    
    def test_batch_processing_detailed_analysis(self):
        """Test batch processing with detailed analysis"""
        texts = [
            "この映画は本当に素晴らしかった！",
            "最悪の商品でした。",
            "普通の品質だと思います。"
        ]
        
        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/predict/batch",
            json={
                "texts": texts,
                "include_details": True,
                "confidence_threshold": 0.7
            },
            timeout=15
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["results"]) == len(texts)
        
        for result in data["results"]:
            assert "text" in result
            assert "result" in result
            assert "score" in result
            assert "all_scores" in result
            assert "processing_time_ms" in result
            
            if "text_length" in result:  # Detailed result
                assert "word_count" in result or result["word_count"] is None
        
        print(f"Detailed batch processing: {len(texts)} texts in {end_time - start_time:.2f}s")
    
    def test_api_stats_endpoint(self):
        """Test the new analysis stats endpoint"""
        response = requests.get(f"{self.BASE_URL}/analyze/stats", timeout=10)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "model_info" in data
        assert "api_info" in data
        
        model_info = data["model_info"]
        assert "is_loaded" in model_info
        assert "model_type" in model_info
        assert "classes" in model_info
        
        api_info = data["api_info"]
        assert "version" in api_info
        assert "supported_features" in api_info
        assert "batch_limits" in api_info
        
        print(f"API Stats - Model loaded: {model_info['is_loaded']}")
        print(f"Supported features: {api_info['supported_features']}")
        print(f"Batch limits: {api_info['batch_limits']}")
    
    def test_large_batch_processing(self):
        """Test processing larger batches efficiently"""
        texts = [f"テストテキスト{i}" for i in range(500)]
        
        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/predict/batch",
            json={"texts": texts},
            timeout=60  # Allow more time for larger batch
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        total_time = end_time - start_time
        texts_per_second = len(texts) / total_time
        
        print(f"Large batch processing: {len(texts)} texts in {total_time:.2f}s")
        print(f"Throughput: {texts_per_second:.2f} texts/second")
        
        min_throughput = 100
        assert texts_per_second > min_throughput, f"Large batch throughput {texts_per_second:.2f} below {min_throughput} texts/second"
        
        data = response.json()
        assert len(data["results"]) == len(texts)
        assert data["summary"]["total_texts"] == len(texts)
        assert data["summary"]["successful_predictions"] == len(texts)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
