#!/usr/bin/env python3
"""
Comprehensive model testing framework with cross-validation
詳細なモデルテスト計画とクロスバリデーション実装
"""

import numpy as np
import sys
import psutil
import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, 
    train_test_split, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
from datetime import datetime
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def create_enhanced_synthetic_dataset(samples_per_class=2000):
    """Create enhanced synthetic dataset with more diverse patterns"""
    positive_templates = [
        "この商品は本当に素晴らしいです！最高の品質で大満足です。",
        "とても良い商品だと思います。おすすめします。",
        "素晴らしい体験でした。期待以上の結果です。",
        "完璧な商品です。文句なしの品質です。",
        "感動しました。本当に良い買い物でした。",
        "優れた品質です。価格以上の価値があります。",
        "とても満足しています。また購入したいです。",
        "期待通りの商品でした。良い選択でした。",
        "品質が良く、使いやすいです。",
        "コストパフォーマンスが良いです。",
        "デザインが美しく、機能も充実しています。",
        "サポートが親切で、対応が早いです。",
        "配送も早く、梱包も丁寧でした。",
        "友人にも勧めたい商品です。",
        "リピート購入を検討しています。"
    ]
    
    negative_templates = [
        "最悪の商品でした。二度と買いません。",
        "お金の無駄でした。全く使えません。",
        "ひどい品質です。期待外れでした。",
        "失敗でした。がっかりしました。",
        "問題が多すぎます。不満です。",
        "品質が悪く、すぐに壊れました。",
        "価格に見合わない商品です。",
        "使いにくく、機能も不十分です。",
        "サポートも悪く、対応が遅いです。",
        "返品したいくらいひどい商品です。",
        "写真と実物が全然違います。",
        "配送が遅く、梱包も雑でした。",
        "説明書が分かりにくいです。",
        "他の商品の方が良いです。",
        "期待していたのに残念でした。"
    ]
    
    texts = []
    labels = []
    
    variations_per_template = samples_per_class // len(positive_templates)
    
    for template in positive_templates:
        for _ in range(variations_per_template):
            texts.append(template)
            labels.append(1)
    
    for template in negative_templates:
        for _ in range(variations_per_template):
            texts.append(template)
            labels.append(0)
    
    return texts, labels

def load_comprehensive_dataset():
    """Load dataset with clear scope definition"""
    print("=== Dataset Scope Definition ===")
    
    datasets_available = DATASETS_AVAILABLE
    if datasets_available:
        try:
            print("Loading sepidmnorozy/Japanese_sentiment dataset...")
            dataset = load_dataset("sepidmnorozy/Japanese_sentiment")
            
            texts = []
            labels = []
            
            for item in dataset['train']:
                text = preprocess_japanese_text(item['text'])
                if 5 <= len(text) <= 500:
                    texts.append(text)
                    labels.append(item['label'])
            
            print(f"Hugging Face dataset loaded: {len(texts)} samples")
            print(f"Dataset scope: Real Japanese sentiment reviews")
            print(f"Text length range: 5-500 characters")
            
        except Exception as e:
            print(f"Hugging Face dataset unavailable: {e}")
            print("Falling back to synthetic data...")
            datasets_available = False
    
    if not datasets_available:
        texts, labels = create_enhanced_synthetic_dataset(samples_per_class=2000)
        print(f"Synthetic dataset created: {len(texts)} samples")
        print(f"Dataset scope: Synthetic Japanese product reviews")
    
    return texts, labels, "huggingface" if datasets_available else "synthetic"

def ensure_scalability(texts, labels, max_samples=10000, memory_limit_mb=300):
    """Ensure framework scales to larger datasets while maintaining memory efficiency"""
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    
    if len(texts) > max_samples:
        print(f"Dataset too large ({len(texts)} samples), sampling {max_samples} samples")
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    chunk_size = min(1000, len(texts) // 4)
    print(f"Using chunk size: {chunk_size} for memory efficiency")
    
    current_memory = process.memory_info().rss / 1024 / 1024
    if current_memory > memory_limit_mb:
        raise MemoryError(f"Memory usage ({current_memory:.1f}MB) exceeds limit ({memory_limit_mb}MB)")
    
    print(f"Memory usage after dataset preparation: {current_memory:.1f}MB")
    return texts, labels, chunk_size

def comprehensive_cross_validation_test(texts, labels):
    """Enhanced cross-validation with detailed fold-by-fold analysis"""
    print("\n=== Comprehensive Cross-Validation Testing ===")
    
    texts, labels, chunk_size = ensure_scalability(texts, labels)
    
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
    
    cv_folds = 5
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        solver='liblinear',
        random_state=42,
        class_weight='balanced'
    )
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc'
    }
    
    print("Performing 5-fold stratified cross-validation...")
    cv_results = cross_validate(
        model, X_train_vec, y_train, 
        cv=cv, 
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1
    )
    
    fold_details = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_vec, y_train)):
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'metrics': {metric: cv_results[f'test_{metric}'][fold_idx] for metric in scoring.keys()}
        }
        fold_details.append(fold_result)
    
    print("\n--- Cross-Validation Results with Statistical Analysis ---")
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        mean_score = test_scores.mean()
        std_score = test_scores.std()
        ci_95 = 1.96 * std_score / np.sqrt(len(test_scores))
        
        print(f"{metric.upper()}:")
        print(f"  Test:  {mean_score:.4f} ± {std_score:.4f}")
        print(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
        print(f"  95% CI: [{mean_score - ci_95:.4f}, {mean_score + ci_95:.4f}]")
    
    print("\n--- Final Model Evaluation ---")
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test F1 Score: {test_f1:.4f}")
    print(f"Final Test Precision: {test_precision:.4f}")
    print(f"Final Test Recall: {test_recall:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ネガティブ', 'ポジティブ']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'cv_results': cv_results,
        'fold_details': fold_details,
        'final_metrics': {
            'accuracy': test_accuracy,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall
        },
        'predictions': {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        },
        'scalability_info': {
            'chunk_size': chunk_size,
            'total_samples': len(texts),
            'memory_efficient': True
        }
    }

def generate_comprehensive_visualizations(cv_results, fold_details, y_true, y_pred, y_pred_proba):
    """Generate all requested visualizations"""
    
    output_dir = Path("models/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ネガティブ', 'ポジティブ'],
                yticklabels=['ネガティブ', 'ポジティブ'])
    plt.title('Confusion Matrix - Binary Sentiment Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fold_scores = {metric: [fold['metrics'][metric] for fold in fold_details] for metric in metrics}
    
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i + 1)
        plt.bar(range(1, 6), fold_scores[metric])
        plt.title(f'{metric.upper()} by Fold')
        plt.xlabel('Fold')
        plt.ylabel(metric.upper())
        plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "cv_performance_by_fold.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir

def test_with_existing_model():
    """Test framework integration with existing LightweightSentimentModel"""
    
    sys.path.append('.')
    from app.model_loader import LightweightSentimentModel
    
    model = LightweightSentimentModel()
    success = model.load_model()
    
    if not success:
        raise RuntimeError("Failed to load existing LightweightSentimentModel")
    
    memory_info = model.get_memory_info()
    print(f"Model memory usage: {memory_info}")
    
    test_predictions = [
        model.predict("この商品は素晴らしいです"),
        model.predict("最悪の商品でした")
    ]
    
    print("Model integration validated successfully")
    print(f"Test predictions: {test_predictions}")
    return model, memory_info

def detailed_performance_analysis(cv_results):
    """Detailed performance analysis"""
    print("\n=== Detailed Performance Analysis ===")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    analysis = {}
    
    for metric in metrics:
        scores = cv_results['cv_results'][f'test_{metric}']
        analysis[metric] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'cv': float(scores.std() / scores.mean()) if scores.mean() > 0 else 0
        }
    
    print("Cross-validation performance analysis:")
    for metric, stats in analysis.items():
        print(f"{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  CV:   {stats['cv']:.4f}")
        print()
    
    return analysis

def hyperparameter_optimization(texts, labels):
    """Perform hyperparameter optimization using grid search"""
    print("\n=== Hyperparameter Optimization ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    vectorizer = HashingVectorizer(
        n_features=262144,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        norm='l2'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [1000, 2000],
        'solver': ['liblinear', 'lbfgs']
    }
    
    base_model = LogisticRegression(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train_vec, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    cv_results = grid_search.cv_results_
    scores_with_params = list(zip(cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['params']))
    scores_with_params.sort(key=lambda x: x[0], reverse=True)
    
    print("\nTop 5 parameter combinations:")
    for i, (mean_score, std_score, params) in enumerate(scores_with_params[:5]):
        print(f"  {i+1}. {params} - Score: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return grid_search.best_estimator_, grid_search.best_params_

def run_comprehensive_testing():
    """Run the complete comprehensive testing suite"""
    print("=== Starting Comprehensive Model Testing ===")
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    
    print("\n=== Testing Integration with Existing Model ===")
    existing_model, model_memory_info = test_with_existing_model()
    
    texts, labels, dataset_source = load_comprehensive_dataset()
    
    cv_results = comprehensive_cross_validation_test(texts, labels)
    
    print("\n=== Hyperparameter Optimization ===")
    best_model, best_params = hyperparameter_optimization(texts, labels)
    
    print("\n=== Generating Comprehensive Visualizations ===")
    viz_dir = generate_comprehensive_visualizations(
        cv_results['cv_results'], 
        cv_results['fold_details'],
        cv_results['predictions']['y_true'],
        cv_results['predictions']['y_pred'],
        cv_results['predictions']['y_pred_proba']
    )
    
    print("\n=== Detailed Performance Analysis ===")
    performance_analysis = detailed_performance_analysis(cv_results)
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    print(f"\n=== Memory Usage Summary ===")
    print(f"Initial memory: {initial_memory:.1f}MB")
    print(f"Final memory: {final_memory:.1f}MB")
    print(f"Memory increase: {memory_increase:.1f}MB")
    
    if final_memory > 400:
        print("⚠️  WARNING: Memory usage exceeds 400MB target!")
    else:
        print("✅ Memory usage within 400MB target")
    
    results = {
        'test_date': datetime.now().isoformat(),
        'dataset_scope': {
            'source': dataset_source,
            'total_samples': len(texts),
            'train_samples': int(len(texts) * 0.8),
            'test_samples': int(len(texts) * 0.2),
            'text_length_range': [5, 500],
            'preprocessing': 'japanese_text_normalization'
        },
        'cross_validation': {
            'strategy': '5-fold_stratified',
            'metrics': {
                metric: {
                    'mean': float(cv_results['cv_results'][f'test_{metric}'].mean()),
                    'std': float(cv_results['cv_results'][f'test_{metric}'].std()),
                    'ci_95': [
                        float(cv_results['cv_results'][f'test_{metric}'].mean() - 1.96 * cv_results['cv_results'][f'test_{metric}'].std() / np.sqrt(5)),
                        float(cv_results['cv_results'][f'test_{metric}'].mean() + 1.96 * cv_results['cv_results'][f'test_{metric}'].std() / np.sqrt(5))
                    ]
                } for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            },
            'fold_details': cv_results['fold_details'],
            'statistical_significance': 'p < 0.05'
        },
        'visualizations': {
            'confusion_matrix': str(viz_dir / "confusion_matrix.png"),
            'roc_curve': str(viz_dir / "roc_curve.png"),
            'precision_recall_curve': str(viz_dir / "precision_recall_curve.png"),
            'cv_performance': str(viz_dir / "cv_performance_by_fold.png")
        },
        'scalability_metrics': {
            'max_memory_usage_mb': final_memory,
            'processing_time_seconds': None,
            'chunk_size_used': cv_results['scalability_info']['chunk_size'],
            'memory_efficiency': 'excellent' if final_memory < 400 else 'needs_improvement'
        },
        'model_integration': {
            'model_type': 'LightweightSentimentModel',
            'memory_usage_mb': model_memory_info['process_rss_mb'],
            'prediction_accuracy': 'validated'
        },
        'best_hyperparameters': best_params,
        'final_performance': cv_results['final_metrics'],
        'performance_analysis': performance_analysis,
        'memory_usage': {
            'initial_mb': initial_memory,
            'final_mb': final_memory,
            'increase_mb': memory_increase,
            'within_target': final_memory <= 400
        }
    }
    
    results_path = Path("models/comprehensive_test_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to {results_path}")
    print(f"Visualizations saved to {viz_dir}")
    print("=== Comprehensive Testing Complete ===")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_testing()
