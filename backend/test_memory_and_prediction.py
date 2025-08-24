#!/usr/bin/env python3
"""
Test script to analyze current memory usage and prediction bias issues
"""

import psutil
import os
from app.model_loader import LightweightSentimentModel
import json

def test_memory_and_prediction():
    """Test memory usage and prediction functionality"""
    print("=== Memory and Prediction Analysis ===")
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory:.1f}MB")
    
    model = LightweightSentimentModel()
    after_create_memory = process.memory_info().rss / 1024 / 1024
    print(f"After model creation: {after_create_memory:.1f}MB (+{after_create_memory - initial_memory:.1f}MB)")
    
    print("\n--- Loading Model ---")
    success = model.load_model()
    after_load_memory = process.memory_info().rss / 1024 / 1024
    print(f"Model loaded: {success}")
    print(f"After model loading: {after_load_memory:.1f}MB (+{after_load_memory - initial_memory:.1f}MB)")
    
    if success:
        memory_info = model.get_memory_info()
        print(f"Model memory info: {json.dumps(memory_info, indent=2, ensure_ascii=False)}")
        
        print("\n--- Testing Predictions ---")
        test_texts = [
            "この商品は本当に素晴らしいです！最高の品質で大満足です。",  # Should be positive
            "普通の商品だと思います。特に良くも悪くもありません。",  # Should be neutral
            "最悪の商品でした。二度と買いません。お金の無駄でした。",  # Should be negative
            "まあまあです。",  # Should be neutral
            "とても良い！",  # Should be positive
            "ひどい。",  # Should be negative
        ]
        
        for i, text in enumerate(test_texts, 1):
            try:
                result = model.predict(text)
                print(f"{i}. Text: {text}")
                print(f"   Result: {result['result']} (confidence: {result['score']:.3f})")
                
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"   Memory: {current_memory:.1f}MB")
                print()
                
            except Exception as e:
                print(f"{i}. Error predicting '{text[:30]}...': {e}")
                print()
    
    else:
        print("Model loading failed - cannot test predictions")
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory: {final_memory:.1f}MB")
    print(f"Total memory increase: {final_memory - initial_memory:.1f}MB")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    test_memory_and_prediction()
