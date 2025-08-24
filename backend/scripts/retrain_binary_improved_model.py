#!/usr/bin/env python3
"""
Script to retrain an improved binary sentiment classification model
Using the original Hugging Face dataset with enhanced preprocessing and optimization
"""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
from datetime import datetime
import re

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available, using synthetic data")

def preprocess_japanese_text(text):
    """Enhanced preprocessing for Japanese text"""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    text = re.sub(r'[！]{2,}', '！', text)
    text = re.sub(r'[？]{2,}', '？', text)
    text = re.sub(r'[。]{2,}', '。', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text

def load_improved_dataset():
    """Load and preprocess the original Hugging Face dataset with improvements"""
    print("=== Loading Improved Dataset ===")
    
    datasets_available = DATASETS_AVAILABLE
    if datasets_available:
        try:
            print("Loading sepidmnorozy/Japanese_sentiment dataset...")
            dataset = load_dataset("sepidmnorozy/Japanese_sentiment")
            
            texts = []
            labels = []
            
            for item in dataset['train']:
                text = preprocess_japanese_text(item['text'])
                if len(text) >= 5 and len(text) <= 500:
                    texts.append(text)
                    labels.append(item['label'])
            
            print(f"Loaded {len(texts)} samples from Hugging Face dataset")
            
        except Exception as e:
            print(f"Error loading Hugging Face dataset: {e}")
            print("Falling back to synthetic data...")
            datasets_available = False
    
    if not datasets_available:
        print("Using enhanced synthetic dataset...")
        data = [
            ("この商品は本当に素晴らしいです！最高の品質で大満足です。", 1),
            ("とても良い商品だと思います。おすすめします。", 1),
            ("素晴らしい体験でした。期待以上の結果です。", 1),
            ("完璧な商品です。文句なしの品質です。", 1),
            ("感動しました。本当に良い買い物でした。", 1),
            ("優れた品質です。価格以上の価値があります。", 1),
            ("とても満足しています。また購入したいです。", 1),
            ("期待通りの商品でした。良い選択でした。", 1),
            ("品質が良く、使いやすいです。", 1),
            ("コストパフォーマンスが良いです。", 1),
            
            ("最悪の商品でした。二度と買いません。", 0),
            ("お金の無駄でした。全く使えません。", 0),
            ("ひどい品質です。期待外れでした。", 0),
            ("失敗でした。がっかりしました。", 0),
            ("問題が多すぎます。不満です。", 0),
            ("品質が悪く、すぐに壊れました。", 0),
            ("価格に見合わない商品です。", 0),
            ("使いにくく、機能も不十分です。", 0),
            ("サポートも悪く、対応が遅いです。", 0),
            ("返品したいくらいひどい商品です。", 0),
        ]
        
        expanded_data = []
        for text, label in data:
            for _ in range(50):
                expanded_data.append((text, label))
        
        texts, labels = zip(*expanded_data)
        texts = list(texts)
        labels = list(labels)
    
    return texts, labels

def retrain_improved_binary_model():
    """Retrain an improved binary sentiment classification model"""
    print("=== Retraining Improved Binary Sentiment Model ===")
    
    texts, labels = load_improved_dataset()
    
    print(f"Dataset size: {len(texts)}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    vectorizer = HashingVectorizer(
        n_features=262144,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        norm='l2'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"Class weights: {class_weight_dict}")
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000, 2000],
        'solver': ['liblinear', 'lbfgs']
    }
    
    base_model = LogisticRegression(
        class_weight=class_weight_dict,
        random_state=42
    )
    
    print("Performing grid search for optimal hyperparameters...")
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_vec, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- Model Evaluation ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ネガティブ', 'ポジティブ']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    weights_path = models_dir / "weights.npz"
    np.savez(
        weights_path,
        coef=best_model.coef_.astype(np.float32),
        intercept=best_model.intercept_.astype(np.float32)
    )
    
    metadata = {
        "model_name": "lightweight_numpy_binary",
        "model_type": "Ultra-lightweight-numpy-binary",
        "vectorizer_type": "HashingVectorizer",
        "classifier_type": "Custom-numpy",
        "sentiment_labels": ["ネガティブ", "ポジティブ"],
        "label_to_index": {"ネガティブ": 0, "ポジティブ": 1},
        "index_to_label": {"0": "ネガティブ", "1": "ポジティブ"},
        "vectorizer_params": {
            "n_features": 262144,
            "alternate_sign": False,
            "ngram_range": [1, 2]
        },
        "weights_info": {
            "coef_shape": list(best_model.coef_.shape),
            "intercept_shape": list(best_model.intercept_.shape),
            "dtype": "float32"
        },
        "class_weights": class_weight_dict,
        "best_params": grid_search.best_params_,
        "test_accuracy": float(test_accuracy),
        "test_f1_score": float(test_f1),
        "saved_at": datetime.now().isoformat()
    }
    
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nModel saved to {weights_path}")
    print(f"Metadata saved to {metadata_path}")
    
    print("\n--- Testing Sample Predictions ---")
    test_texts = [
        "この商品は本当に素晴らしいです！",
        "最悪の商品でした。",
        "とても良い商品だと思います。",
        "ひどい品質です。"
    ]
    
    for text in test_texts:
        X_vec = vectorizer.transform([text])
        logits = X_vec @ best_model.coef_.T + best_model.intercept_
        probabilities = 1 / (1 + np.exp(-logits[0]))
        prediction = (probabilities > 0.5).astype(int)[0]
        confidence = probabilities[0] if prediction == 1 else 1 - probabilities[0]
        
        predicted_label = "ポジティブ" if prediction == 1 else "ネガティブ"
        print(f"Text: {text}")
        print(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
        print()
    
    print("=== Improved Binary Model Training Complete ===")

if __name__ == "__main__":
    retrain_improved_binary_model()
