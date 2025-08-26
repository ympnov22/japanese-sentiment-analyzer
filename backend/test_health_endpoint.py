#!/usr/bin/env python3
"""
Test script to verify health endpoint returns correct metadata
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

os.environ['GIT_COMMIT'] = 'test-commit-hash'

from app.main import app
from fastapi.testclient import TestClient

def test_health_endpoint():
    print('=== Health Endpoint Test ===')
    
    client = TestClient(app)
    response = client.get('/health')
    
    print(f'Health endpoint status: {response.status_code}')
    
    if response.status_code == 200:
        data = response.json()
        print(f'Status: {data.get("status")}')
        print(f'Model loaded: {data.get("model_loaded")}')
        print(f'Message: {data.get("message")}')
        
        build_metadata = data.get("build_metadata", {})
        print(f'Git commit: {build_metadata.get("git_commit")}')
        print(f'Model version: {build_metadata.get("model_version")}')
        print(f'Model SHA256: {build_metadata.get("model_sha256")}')
        print(f'Model verified: {build_metadata.get("model_verified")}')
        print(f'Accuracy baseline: {build_metadata.get("accuracy_baseline")}')
        
        return True
    else:
        print(f'Health endpoint failed with status {response.status_code}')
        print(f'Response: {response.text}')
        return False

if __name__ == "__main__":
    test_health_endpoint()
