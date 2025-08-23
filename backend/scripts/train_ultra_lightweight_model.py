#!/usr/bin/env python3
"""
Ultra-Lightweight Model Training Script for Japanese Sentiment Analysis
Optimized for extreme memory constraints on Fly.io (256MB)
Uses HashingVectorizer + SGDClassifier for minimal memory footprint
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data():
    """Load the processed dataset"""
    data_dir = Path("data")
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv") 
    test_df = pd.read_csv(data_dir / "test.csv")
    
    return train_df, val_df, test_df

def train_ultra_lightweight_model():
    """
    Train an ultra-lightweight model for extreme memory constraints
    """
    print("=== Training Ultra-Lightweight Japanese Sentiment Model ===")
    print("Target: Fly.io deployment with 256MB memory (extreme optimization)")
    
    try:
        train_df, val_df, test_df = load_processed_data()
        
        print("\n--- Ultra-lite Configuration ---")
        print("Vectorizer: HashingVectorizer(n_features=2**18, alternate_sign=False)")
        print("Classifier: SGDClassifier(loss='log_loss', random_state=42)")
        print("Memory: Minimal footprint, no vocabulary storage")
        
        vectorizer = HashingVectorizer(
            n_features=2**18,  # 262,144 features
            alternate_sign=False,
            ngram_range=(1, 2),
            lowercase=True
        )
        
        classifier = SGDClassifier(
            loss='log_loss',  # Logistic regression equivalent
            random_state=42,
            max_iter=1000,
            alpha=0.0001
        )
        
        label_encoder = LabelEncoder()
        
        all_labels = pd.concat([train_df['sentiment'], val_df['sentiment'], test_df['sentiment']])
        label_encoder.fit(all_labels)
        
        y_train = label_encoder.transform(train_df['sentiment'])
        y_val = label_encoder.transform(val_df['sentiment'])
        y_test = label_encoder.transform(test_df['sentiment'])
        
        print(f"\nTraining on {len(train_df)} samples...")
        print(f"Validation on {len(val_df)} samples...")
        print(f"Test on {len(test_df)} samples...")
        
        X_train = vectorizer.transform(train_df['text'])
        X_val = vectorizer.transform(val_df['text'])
        X_test = vectorizer.transform(test_df['text'])
        
        print("\nTraining ultra-lightweight classifier...")
        classifier.fit(X_train, y_train)
        
        val_predictions = classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions, average='weighted')
        
        print(f"\nValidation Results:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"F1 Score: {val_f1:.4f}")
        
        test_predictions = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions, average='weighted')
        
        print(f"\nTest Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"F1 Score: {test_f1:.4f}")
        
        cm = confusion_matrix(y_test, test_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Ultra-Lightweight Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        cm_plot_path = models_dir / "ultra_lightweight_confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        model_name = "japanese_sentiment_model_ultra"
        
        vectorizer_path = models_dir / f"{model_name}_vectorizer.pkl"
        classifier_path = models_dir / f"{model_name}_classifier.pkl"
        metadata_path = models_dir / f"{model_name}_metadata.json"
        
        joblib.dump(vectorizer, vectorizer_path, compress=('gzip', 9))
        joblib.dump(classifier, classifier_path, compress=('gzip', 9))
        
        metadata = {
            'model_name': model_name,
            'model_type': 'Ultra-lightweight',
            'vectorizer_type': 'HashingVectorizer',
            'classifier_type': 'SGDClassifier',
            'sentiment_labels': label_encoder.classes_.tolist(),
            'label_to_index': {label: int(idx) for idx, label in enumerate(label_encoder.classes_)},
            'index_to_label': {int(idx): label for idx, label in enumerate(label_encoder.classes_)},
            'vectorizer_params': {
                'n_features': 2**18,
                'alternate_sign': False,
                'ngram_range': '(1, 2)'
            },
            'classifier_params': {
                'loss': 'log_loss',
                'max_iter': 1000,
                'alpha': 0.0001
            },
            'training_accuracy': float(val_accuracy),
            'training_f1': float(val_f1),
            'test_accuracy': float(test_accuracy),
            'test_f1': float(test_f1),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        vectorizer_size = vectorizer_path.stat().st_size / (1024 * 1024)
        classifier_size = classifier_path.stat().st_size / (1024 * 1024)
        total_size = vectorizer_size + classifier_size
        
        print(f"\nModel File Sizes:")
        print(f"Vectorizer: {vectorizer_size:.2f} MB")
        print(f"Classifier: {classifier_size:.2f} MB")
        print(f"Total: {total_size:.2f} MB")
        
        ultra_report = {
            'model_type': 'Ultra-lightweight',
            'optimization_target': 'Fly.io 256MB extreme memory constraints',
            'vectorizer_config': metadata['vectorizer_params'],
            'classifier_config': metadata['classifier_params'],
            'memory_optimization': 'HashingVectorizer (no vocabulary storage)',
            'compression': 'gzip level 9',
            'validation_results': {
                'accuracy': float(val_accuracy),
                'f1_score': float(val_f1)
            },
            'test_results': {
                'accuracy': float(test_accuracy),
                'f1_score': float(test_f1),
                'confusion_matrix': cm.tolist()
            },
            'model_paths': {
                'vectorizer': str(vectorizer_path),
                'classifier': str(classifier_path),
                'metadata': str(metadata_path)
            },
            'file_sizes_mb': {
                'vectorizer': float(vectorizer_size),
                'classifier': float(classifier_size),
                'total': float(total_size)
            },
            'confusion_matrix_plot': str(cm_plot_path),
            'evaluation_time': datetime.now().isoformat()
        }
        
        report_path = models_dir / "ultra_lightweight_model_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(ultra_report, f, ensure_ascii=False, indent=2)
        
        print(f"\nUltra-lightweight model report saved to: {report_path}")
        
        print("\n=== Testing Ultra-Lightweight Model Predictions ===")
        
        test_texts = [
            "この商品は本当に素晴らしいです！",
            "最悪の商品でした。二度と買いません。",
            "普通の商品だと思います。"
        ]
        
        for text in test_texts:
            text_vector = vectorizer.transform([text])
            prediction = classifier.predict(text_vector)[0]
            probabilities = classifier.predict_proba(text_vector)[0]
            
            sentiment_label = label_encoder.classes_[prediction]
            confidence = float(probabilities[prediction])
            
            print(f"Text: {text}")
            print(f"Prediction: {sentiment_label} (confidence: {confidence:.3f})")
            print()
        
        print("=== Ultra-Lightweight Model Training Complete ===")
        print(f"Memory footprint: ~{total_size:.1f}MB model files")
        print("Ready for 256MB Fly.io deployment!")
        
        return ultra_report
        
    except Exception as e:
        print(f"Error during ultra-lightweight model training: {e}")
        raise

if __name__ == "__main__":
    train_ultra_lightweight_model()
