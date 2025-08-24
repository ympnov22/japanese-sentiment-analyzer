#!/usr/bin/env python3
"""
Script to retrain the model for 3-class sentiment classification
"""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

def create_sample_data():
    """Create sample training data for 3-class sentiment classification"""
    
    data = [
        ("この商品は本当に素晴らしいです！", "ポジティブ"),
        ("最高の品質で大満足です。", "ポジティブ"),
        ("とても良い商品だと思います。", "ポジティブ"),
        ("素晴らしい体験でした。", "ポジティブ"),
        ("おすすめします！", "ポジティブ"),
        ("完璧な商品です。", "ポジティブ"),
        ("期待以上でした。", "ポジティブ"),
        ("とても満足しています。", "ポジティブ"),
        ("優れた品質です。", "ポジティブ"),
        ("感動しました。", "ポジティブ"),
        
        ("最悪の商品でした。", "ネガティブ"),
        ("二度と買いません。", "ネガティブ"),
        ("お金の無駄でした。", "ネガティブ"),
        ("ひどい品質です。", "ネガティブ"),
        ("期待外れでした。", "ネガティブ"),
        ("全く使えません。", "ネガティブ"),
        ("失敗でした。", "ネガティブ"),
        ("がっかりしました。", "ネガティブ"),
        ("問題が多すぎます。", "ネガティブ"),
        ("不満です。", "ネガティブ"),
        
        ("普通の商品だと思います。", "ニュートラル"),
        ("特に良くも悪くもありません。", "ニュートラル"),
        ("まあまあです。", "ニュートラル"),
        ("可もなく不可もなく。", "ニュートラル"),
        ("標準的な品質です。", "ニュートラル"),
        ("普通です。", "ニュートラル"),
        ("特に印象に残りません。", "ニュートラル"),
        ("平均的だと思います。", "ニュートラル"),
        ("そこそこです。", "ニュートラル"),
        ("一般的な商品です。", "ニュートラル"),
    ]
    
    expanded_data = []
    for text, label in data:
        expanded_data.append((text, label))
        for i in range(3):
            expanded_data.append((text, label))
    
    return expanded_data

def retrain_model():
    """Retrain the model for 3-class sentiment classification"""
    print("=== Retraining Model for 3-Class Sentiment Classification ===")
    
    data = create_sample_data()
    texts, labels = zip(*data)
    
    print(f"Training data size: {len(texts)}")
    print(f"Label distribution: {dict(Counter(labels))}")
    
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {str(idx): label for idx, label in enumerate(unique_labels)}
    
    print(f"Label mapping: {label_to_index}")
    
    y = np.array([label_to_index[label] for label in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    
    vectorizer = HashingVectorizer(
        n_features=262144,
        alternate_sign=False,
        ngram_range=(1, 2)
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
    
    model = LogisticRegression(
        class_weight=class_weight_dict,
        random_state=42,
        max_iter=1000
    )
    
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    weights_path = models_dir / "weights.npz"
    np.savez(
        weights_path,
        coef=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float32)
    )
    
    metadata = {
        "model_name": "lightweight_numpy_3class",
        "model_type": "Ultra-lightweight-numpy-3class",
        "vectorizer_type": "HashingVectorizer",
        "classifier_type": "Custom-numpy",
        "sentiment_labels": unique_labels,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "vectorizer_params": {
            "n_features": 262144,
            "alternate_sign": False,
            "ngram_range": [1, 2]
        },
        "weights_info": {
            "coef_shape": list(model.coef_.shape),
            "intercept_shape": list(model.intercept_.shape),
            "dtype": "float32"
        },
        "class_weights": class_weight_dict,
        "saved_at": datetime.now().isoformat()
    }
    
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nModel saved to {weights_path}")
    print(f"Metadata saved to {metadata_path}")
    
    print("\n--- Testing Predictions ---")
    test_texts = [
        "この商品は本当に素晴らしいです！",
        "普通の商品だと思います。",
        "最悪の商品でした。"
    ]
    
    for text in test_texts:
        X_vec = vectorizer.transform([text])
        logits = X_vec @ model.coef_.T + model.intercept_
        
        if len(unique_labels) == 3:
            exp_logits = np.exp(logits[0] - np.max(logits[0]))
            probabilities = exp_logits / np.sum(exp_logits)
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
        else:
            probabilities = 1 / (1 + np.exp(-logits[0]))
            prediction = (probabilities > 0.5).astype(int)[0]
            confidence = probabilities[0] if prediction == 1 else 1 - probabilities[0]
        
        predicted_label = unique_labels[prediction]
        print(f"Text: {text}")
        print(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
        print()
    
    print("=== Retraining Complete ===")

if __name__ == "__main__":
    retrain_model()
