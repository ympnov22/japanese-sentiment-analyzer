#!/usr/bin/env python3
"""
Debug script to understand which model is being loaded
"""

import logging
from app.model_loader import LightweightSentimentModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def debug_model_loading():
    print("=== Debug Model Loading ===")
    
    model = LightweightSentimentModel('models')
    success = model.load_model()
    
    print(f"Model loaded successfully: {success}")
    print(f"Model is_loaded: {model.is_loaded}")
    print(f"Vectorizer type: {type(model.vectorizer)}")
    print(f"Classes: {model.classes_}")
    print(f"Metadata: {model.metadata}")
    
    if hasattr(model, 'coef_') and model.coef_ is not None:
        print(f"Coef shape: {model.coef_.shape}")
    
    positive_text = "最高に嬉しい！"
    negative_text = "最悪で腹が立つ。"
    
    result1 = model.predict(positive_text)
    result2 = model.predict(negative_text)
    
    print(f"\nPrediction Results:")
    print(f"Positive: '{positive_text}' -> {result1}")
    print(f"Negative: '{negative_text}' -> {result2}")

if __name__ == "__main__":
    debug_model_loading()
