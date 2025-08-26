#!/usr/bin/env python3
"""
Test script to verify model loading and SHA256 verification works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.model_loader import LightweightSentimentModel

def test_model_loading():
    print('=== Model Loading Test ===')
    model = LightweightSentimentModel()
    success = model.load_model()
    print(f'Model loaded: {success}')

    if success:
        memory_info = model.get_memory_info()
        print(f'Model verified: {memory_info.get("model_verified", False)}')
        print(f'Model version: {memory_info.get("model_version", "unknown")}')
        print(f'Accuracy baseline: {memory_info.get("accuracy_baseline", "unknown")}')
        
        result = model.predict('この商品は素晴らしいです')
        print(f'Test prediction: {result}')
        
        ultra_config = model.model_registry['ultra']
        print(f'Expected classifier SHA256: {ultra_config["classifier_sha256"][:16]}...')
        print(f'Expected vectorizer SHA256: {ultra_config["vectorizer_sha256"][:16]}...')
        
        return True
    else:
        print('Model loading failed')
        return False

if __name__ == "__main__":
    test_model_loading()
