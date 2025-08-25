#!/usr/bin/env python3
"""
Test script to verify the new improved model works correctly
"""

from app.model_loader import LightweightSentimentModel

def test_new_model():
    print("=== Testing New Improved Model ===")
    
    model = LightweightSentimentModel('models')
    model.load_model()
    print("Model loaded successfully")
    
    positive_text = "最高に嬉しい！"
    negative_text = "最悪で腹が立つ。"
    
    result1 = model.predict(positive_text)
    result2 = model.predict(negative_text)
    
    print(f"\nSanity Test Results:")
    print(f"Positive text: '{positive_text}' -> {result1['result']} (score: {result1['score']:.3f})")
    print(f"Negative text: '{negative_text}' -> {result2['result']} (score: {result2['score']:.3f})")
    
    if result1['result'] == result2['result']:
        print("❌ BIAS DETECTED: Both texts have same classification")
        return False
    else:
        print("✅ NO BIAS: Texts have different classifications")
        return True

if __name__ == "__main__":
    test_new_model()
