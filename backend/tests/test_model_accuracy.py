#!/usr/bin/env python3
"""
Model accuracy tests for Japanese sentiment analysis
Tests for bias detection, baseline comparison, and detailed error analysis
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.model_loader import LightweightSentimentModel

class TestModelAccuracy:
    """Test model accuracy and bias detection"""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures"""
        cls.model = LightweightSentimentModel()
        cls.model_loaded = cls.model.load_model()
        
        cls.test_data = None
        data_path = Path(__file__).parent.parent / "data" / "test.csv"
        if data_path.exists():
            cls.test_data = pd.read_csv(data_path)
    
    def test_sanity_check(self):
        """固定テストでバイアス検出 - 両方が同じラベルならバイアス検出"""
        if not self.model_loaded:
            pytest.skip("Model not loaded, skipping sanity test")
        
        positive_text = "最高に嬉しい！"
        negative_text = "最悪で腹が立つ。"
        
        positive_result = self.model.predict(positive_text)
        negative_result = self.model.predict(negative_text)
        
        print(f"\nSanity Test Results:")
        print(f"Positive text: '{positive_text}' -> {positive_result['result']} (score: {positive_result['score']:.3f})")
        print(f"Negative text: '{negative_text}' -> {negative_result['result']} (score: {negative_result['score']:.3f})")
        
        if positive_result['result'] == negative_result['result']:
            pytest.fail(f"BIAS DETECTED: Both texts classified as '{positive_result['result']}' - Model shows clear bias!")
        
        assert positive_result['result'] in ['ポジティブ', 'ネガティブ'], "Invalid sentiment label"
        assert negative_result['result'] in ['ポジティブ', 'ネガティブ'], "Invalid sentiment label"
        
        if positive_result['result'] != 'ポジティブ' or negative_result['result'] != 'ネガティブ':
            print(f"WARNING: Expected positive->ポジティブ, negative->ネガティブ, but got different results")
    
    def test_baseline_comparison(self):
        """DummyClassifier(strategy="most_frequent")との比較"""
        if not self.model_loaded or self.test_data is None:
            pytest.skip("Model not loaded or test data not available")
        
        X_test = self.test_data['text'].values
        y_test = self.test_data['sentiment'].values
        
        dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
        X_dummy = np.arange(len(X_test)).reshape(-1, 1)  # ダミーの2D array
        dummy_clf.fit(X_dummy, y_test)
        dummy_predictions = dummy_clf.predict(X_dummy)
        
        model_predictions = []
        model_scores = []
        
        for text in X_test:
            try:
                result = self.model.predict(text)
                model_predictions.append(result['result'])
                model_scores.append(result['score'])
            except Exception as e:
                print(f"Error predicting '{text[:30]}...': {e}")
                model_predictions.append('ポジティブ')  # デフォルト値
                model_scores.append(0.5)
        
        dummy_accuracy = accuracy_score(y_test, dummy_predictions)
        dummy_f1_macro = f1_score(y_test, dummy_predictions, average='macro')
        
        model_accuracy = accuracy_score(y_test, model_predictions)
        model_f1_macro = f1_score(y_test, model_predictions, average='macro')
        
        print(f"\nBaseline Comparison:")
        print(f"DummyClassifier - Accuracy: {dummy_accuracy:.3f}, Macro F1: {dummy_f1_macro:.3f}")
        print(f"Current Model  - Accuracy: {model_accuracy:.3f}, Macro F1: {model_f1_macro:.3f}")
        print(f"Improvement    - Accuracy: {model_accuracy - dummy_accuracy:.3f}, Macro F1: {model_f1_macro - dummy_f1_macro:.3f}")
        
        assert model_f1_macro > dummy_f1_macro, f"Model F1 ({model_f1_macro:.3f}) should exceed baseline ({dummy_f1_macro:.3f})"
        
        return {
            'dummy_accuracy': dummy_accuracy,
            'dummy_f1_macro': dummy_f1_macro,
            'model_accuracy': model_accuracy,
            'model_f1_macro': model_f1_macro
        }
    
    def test_accuracy_metrics(self):
        """accuracy, precision, recall, macro F1の評価"""
        if not self.model_loaded or self.test_data is None:
            pytest.skip("Model not loaded or test data not available")
        
        X_test = self.test_data['text'].values
        y_test = self.test_data['sentiment'].values
        
        predictions = []
        scores = []
        
        for text in X_test:
            try:
                result = self.model.predict(text)
                predictions.append(result['result'])
                scores.append(result['score'])
            except Exception as e:
                print(f"Error predicting '{text[:30]}...': {e}")
                predictions.append('ポジティブ')
                scores.append(0.5)
        
        accuracy = accuracy_score(y_test, predictions)
        precision_macro = precision_score(y_test, predictions, average='macro')
        recall_macro = recall_score(y_test, predictions, average='macro')
        f1_macro = f1_score(y_test, predictions, average='macro')
        
        print(f"\nDetailed Accuracy Metrics:")
        print(f"Accuracy:           {accuracy:.3f}")
        print(f"Precision (macro):  {precision_macro:.3f}")
        print(f"Recall (macro):     {recall_macro:.3f}")
        print(f"F1 Score (macro):   {f1_macro:.3f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        scores_array = np.array(scores)
        print(f"\nScore Distribution:")
        print(f"Min: {scores_array.min():.3f}, Max: {scores_array.max():.3f}")
        print(f"Mean: {scores_array.mean():.3f}, Std: {scores_array.std():.3f}")
        
        assert accuracy > 0.3, f"Accuracy too low: {accuracy:.3f}"
        assert f1_macro > 0.2, f"F1 score too low: {f1_macro:.3f}"
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'score_distribution': {
                'min': float(scores_array.min()),
                'max': float(scores_array.max()),
                'mean': float(scores_array.mean()),
                'std': float(scores_array.std())
            }
        }
    
    def test_confusion_matrix_output(self):
        """混同行列と確率分布ヒストグラム出力"""
        if not self.model_loaded or self.test_data is None:
            pytest.skip("Model not loaded or test data not available")
        
        X_test = self.test_data['text'].values
        y_test = self.test_data['sentiment'].values
        
        predictions = []
        scores = []
        
        for text in X_test:
            try:
                result = self.model.predict(text)
                predictions.append(result['result'])
                scores.append(result['score'])
            except Exception as e:
                predictions.append('ポジティブ')
                scores.append(0.5)
        
        cm = confusion_matrix(y_test, predictions)
        labels = sorted(list(set(y_test) | set(predictions)))
        
        print(f"\nConfusion Matrix:")
        print(f"Labels: {labels}")
        print(cm)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Score Distribution (All)')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        scores_by_class = {}
        for true_label in labels:
            mask = np.array(y_test) == true_label
            class_scores = np.array(scores)[mask]
            scores_by_class[true_label] = class_scores
            plt.hist(class_scores, bins=15, alpha=0.6, label=true_label, edgecolor='black')
        
        plt.title('Score Distribution by True Class')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "score_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
        
        scores_array = np.array(scores)
        score_range = scores_array.max() - scores_array.min()
        
        print(f"\nScore Range Analysis:")
        print(f"Range: {score_range:.3f} (min: {scores_array.min():.3f}, max: {scores_array.max():.3f})")
        
        if score_range < 0.2:
            print("WARNING: Score range is very narrow, indicating potential bias")
        
        return {
            'confusion_matrix': cm.tolist(),
            'labels': labels,
            'score_range': float(score_range)
        }
    
    def test_error_analysis(self):
        """誤分類Top20件を出力し、coef_を用いて特徴語リストを確認"""
        if not self.model_loaded or self.test_data is None:
            pytest.skip("Model not loaded or test data not available")
        
        X_test = self.test_data['text'].values
        y_test = self.test_data['sentiment'].values
        
        predictions = []
        scores = []
        errors = []
        
        for i, text in enumerate(X_test):
            try:
                result = self.model.predict(text)
                pred_label = result['result']
                pred_score = result['score']
                
                predictions.append(pred_label)
                scores.append(pred_score)
                
                if pred_label != y_test[i]:
                    errors.append({
                        'index': i,
                        'text': text,
                        'true_label': y_test[i],
                        'pred_label': pred_label,
                        'confidence': pred_score
                    })
            except Exception as e:
                predictions.append('ポジティブ')
                scores.append(0.5)
        
        errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nError Analysis:")
        print(f"Total errors: {len(errors)} / {len(X_test)} ({len(errors)/len(X_test)*100:.1f}%)")
        
        print(f"\nTop 20 Misclassifications (by confidence):")
        for i, error in enumerate(errors_sorted[:20]):
            print(f"{i+1:2d}. True: {error['true_label']:10s} | Pred: {error['pred_label']:10s} | "
                  f"Conf: {error['confidence']:.3f} | Text: {error['text'][:60]}...")
        
        if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
            print(f"\nFeature Analysis (Top/Bottom coefficients):")
            
            if hasattr(self.model, 'vectorizer') and self.model.vectorizer is not None:
                try:
                    sample_text = "サンプルテキスト"
                    sample_vector = self.model.vectorizer.transform([sample_text])
                    
                    coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
                    
                    top_indices = np.argsort(coef)[-20:][::-1]
                    bottom_indices = np.argsort(coef)[:20]
                    
                    print(f"Top 20 positive coefficients (indices): {top_indices}")
                    print(f"Values: {coef[top_indices]}")
                    print(f"Bottom 20 negative coefficients (indices): {bottom_indices}")
                    print(f"Values: {coef[bottom_indices]}")
                    
                except Exception as e:
                    print(f"Could not analyze features: {e}")
            else:
                print("Vectorizer not available for feature analysis")
        else:
            print("Model coefficients not available for feature analysis")
        
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        error_report = {
            'total_errors': len(errors),
            'total_samples': len(X_test),
            'error_rate': len(errors) / len(X_test),
            'top_20_errors': errors_sorted[:20]
        }
        
        with open(output_dir / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(error_report, f, ensure_ascii=False, indent=2)
        
        print(f"\nError analysis saved to {output_dir}/error_analysis.json")
        
        return error_report

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
