#!/usr/bin/env python3
"""
Lightweight Model Training Script for Japanese Sentiment Analysis
Optimized for Fly.io deployment with memory constraints (256-512MB)
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.model_training import JapaneseSentimentModel, load_processed_data
import pandas as pd
import numpy as np
import json
from datetime import datetime

def train_lightweight_model():
    """
    Train a lightweight model optimized for low memory deployment
    """
    print("=== Training Lightweight Japanese Sentiment Model ===")
    print("Target: Fly.io deployment with 256-512MB memory")
    
    try:
        train_df, val_df, test_df = load_processed_data()
        
        model = JapaneseSentimentModel()
        
        print("\n--- Classic-lite Configuration ---")
        print("TF-IDF: max_features=30000, ngram_range=(1,2), sublinear_tf=True, min_df=2")
        print("Classifier: LogisticRegression with liblinear solver")
        print("Compression: gzip level 3")
        
        model.create_tfidf_vectorizer(
            max_features=30000,
            ngram_range=(1, 2)
        )
        
        model.create_classifier()
        
        training_results = model.train(train_df, val_df)
        
        test_results = model.evaluate(test_df, dataset_name="Test")
        
        cm_plot_path = model.create_confusion_matrix_plot(
            np.array(test_results['confusion_matrix'])
        )
        
        model_paths = model.save_model("japanese_sentiment_model_lite")
        
        lightweight_report = {
            'model_type': 'Classic-lite',
            'optimization_target': 'Fly.io 256-512MB deployment',
            'vectorizer_config': {
                'max_features': 30000,
                'ngram_range': '(1, 2)',
                'sublinear_tf': True,
                'min_df': 2,
                'solver': 'liblinear'
            },
            'compression': 'gzip level 3',
            'training_results': training_results,
            'test_results': test_results,
            'model_paths': model_paths,
            'confusion_matrix_plot': cm_plot_path,
            'evaluation_time': datetime.now().isoformat()
        }
        
        report_path = Path("models") / "lightweight_model_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(lightweight_report, f, ensure_ascii=False, indent=2)
        
        print(f"\nLightweight model report saved to: {report_path}")
        
        print("\n=== Testing Lightweight Model Predictions ===")
        
        test_texts = [
            "この商品は本当に素晴らしいです！",
            "最悪の商品でした。二度と買いません。",
            "普通の商品だと思います。"
        ]
        
        for text in test_texts:
            result = model.predict(text)
            print(f"Text: {text}")
            print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            print()
        
        print("=== Lightweight Model Training Complete ===")
        
        return lightweight_report
        
    except Exception as e:
        print(f"Error during lightweight model training: {e}")
        raise

if __name__ == "__main__":
    train_lightweight_model()
