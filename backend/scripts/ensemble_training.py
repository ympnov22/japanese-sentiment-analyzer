#!/usr/bin/env python3
"""
Ensemble Training Script for Japanese Sentiment Analysis
Implements Voting and Stacking ensemble methods for improved performance
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

class EnsembleTrainer:
    """
    Ensemble trainer for Japanese sentiment analysis
    Implements Voting and Stacking ensemble methods
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.sentiment_labels = ['ネガティブ', 'ポジティブ']
        self.label_to_index = {label: idx for idx, label in enumerate(self.sentiment_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.sentiment_labels)}
        
        self.base_models = {}
        self.ensemble_models = {}
        self.vectorizer = None
        self.is_trained = False
        
    def prepare_labels(self, labels):
        """Convert sentiment labels to numeric indices"""
        if isinstance(labels.iloc[0], str):
            return labels.map(self.label_to_index).values
        return labels.values
    
    def create_base_models(self):
        """Create base models for ensemble"""
        print("Creating base models for ensemble...")
        
        lr_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.95,
                max_features=30000,
                sublinear_tf=True,
                norm='l2',
                lowercase=True
            )),
            ('classifier', LogisticRegression(
                C=10.0,  # Best parameter from Phase 2
                class_weight='balanced',
                max_iter=200,
                n_jobs=-1,
                random_state=42
            ))
        ])
        
        svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                min_df=3,
                max_df=0.9,
                max_features=20000,
                sublinear_tf=True,
                norm='l2',
                lowercase=True
            )),
            ('classifier', SVC(
                C=1.0,
                kernel='linear',
                class_weight='balanced',
                probability=True,  # Required for ensemble
                random_state=42
            ))
        ])
        
        rf_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                max_features=15000,
                sublinear_tf=True,
                norm='l2',
                lowercase=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ))
        ])
        
        self.base_models = {
            'logistic_regression': lr_pipeline,
            'svm': svm_pipeline,
            'random_forest': rf_pipeline
        }
        
        print(f"Created {len(self.base_models)} base models")
        return self.base_models
    
    def train_base_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train individual base models and evaluate performance"""
        print("\n=== Training Base Models ===")
        
        base_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.base_models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, y_pred, average='macro')
                val_accuracy = accuracy_score(y_val, y_pred)
            else:
                val_f1 = cv_scores.mean()
                val_accuracy = None
            
            base_results[name] = {
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'training_time': training_time
            }
            
            print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Val F1: {val_f1:.4f}")
            print(f"  Training time: {training_time:.2f}s")
        
        return base_results
    
    def create_ensemble_models(self):
        """Create ensemble models using trained base models"""
        print("\n=== Creating Ensemble Models ===")
        
        voting_ensemble = VotingClassifier(
            estimators=[
                ('lr', self.base_models['logistic_regression']),
                ('svm', self.base_models['svm']),
                ('rf', self.base_models['random_forest'])
            ],
            voting='soft',  # Use probability predictions
            n_jobs=-1
        )
        
        stacking_ensemble = StackingClassifier(
            estimators=[
                ('lr', self.base_models['logistic_regression']),
                ('svm', self.base_models['svm']),
                ('rf', self.base_models['random_forest'])
            ],
            final_estimator=LogisticRegression(
                class_weight='balanced',
                random_state=42
            ),
            cv=3,  # Cross-validation for meta-features
            n_jobs=-1
        )
        
        self.ensemble_models = {
            'voting': voting_ensemble,
            'stacking': stacking_ensemble
        }
        
        print(f"Created {len(self.ensemble_models)} ensemble models")
        return self.ensemble_models
    
    def train_ensemble_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble models and evaluate performance"""
        print("\n=== Training Ensemble Models ===")
        
        ensemble_results = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Fewer folds for speed
        
        for name, model in self.ensemble_models.items():
            print(f"\nTraining {name} ensemble...")
            start_time = time.time()
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                
                val_f1 = f1_score(y_val, y_pred, average='macro')
                val_accuracy = accuracy_score(y_val, y_pred)
                val_precision = precision_score(y_val, y_pred, average='macro')
                val_recall = recall_score(y_val, y_pred, average='macro')
            else:
                val_f1 = cv_scores.mean()
                val_accuracy = None
                val_precision = None
                val_recall = None
            
            ensemble_results[name] = {
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'training_time': training_time
            }
            
            print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Val F1: {val_f1:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Training time: {training_time:.2f}s")
        
        return ensemble_results
    
    def compare_models(self, base_results, ensemble_results):
        """Compare base models vs ensemble models"""
        print("\n=== Model Comparison ===")
        
        all_results = {}
        all_results.update({f"base_{k}": v for k, v in base_results.items()})
        all_results.update({f"ensemble_{k}": v for k, v in ensemble_results.items()})
        
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['val_f1'], reverse=True)
        
        print("\nModel Performance Ranking (by Validation F1):")
        print("-" * 60)
        for i, (name, results) in enumerate(sorted_results, 1):
            print(f"{i:2d}. {name:20s} | F1: {results['val_f1']:.4f} | Time: {results['training_time']:6.2f}s")
        
        best_base_f1 = max(base_results.values(), key=lambda x: x['val_f1'])['val_f1']
        best_ensemble_f1 = max(ensemble_results.values(), key=lambda x: x['val_f1'])['val_f1']
        improvement = best_ensemble_f1 - best_base_f1
        
        print(f"\nEnsemble Improvement:")
        print(f"Best Base Model F1:     {best_base_f1:.4f}")
        print(f"Best Ensemble F1:       {best_ensemble_f1:.4f}")
        print(f"Improvement:            +{improvement:.4f} ({improvement/best_base_f1*100:+.2f}%)")
        
        return sorted_results, improvement
    
    def save_ensemble_models(self, model_name="ensemble_sentiment_model"):
        """Save trained ensemble models"""
        print(f"\n=== Saving Ensemble Models ===")
        
        saved_paths = {}
        
        for ensemble_name, model in self.ensemble_models.items():
            model_path = self.model_dir / f"{model_name}_{ensemble_name}.pkl"
            joblib.dump(model, model_path, compress=('gzip', 3))
            saved_paths[ensemble_name] = str(model_path)
            print(f"Saved {ensemble_name} ensemble to: {model_path}")
        
        metadata = {
            'model_name': model_name,
            'model_type': 'Ensemble',
            'ensemble_types': list(self.ensemble_models.keys()),
            'sentiment_labels': self.sentiment_labels,
            'label_to_index': self.label_to_index,
            'index_to_label': self.index_to_label,
            'base_models': list(self.base_models.keys()),
            'save_time': datetime.now().isoformat()
        }
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        saved_paths['metadata'] = str(metadata_path)
        print(f"Metadata saved to: {metadata_path}")
        
        return saved_paths
    
    def train(self, train_df, val_df=None):
        """Main training pipeline for ensemble models"""
        print("\n=== Ensemble Training Pipeline Started ===")
        
        X_train = train_df['text'].values
        y_train = self.prepare_labels(train_df['sentiment'])
        
        if val_df is not None:
            X_val = val_df['text'].values
            y_val = self.prepare_labels(val_df['sentiment'])
        else:
            X_val, y_val = None, None
        
        print(f"Training data: {len(X_train)} samples")
        if val_df is not None:
            print(f"Validation data: {len(X_val)} samples")
        
        self.create_base_models()
        base_results = self.train_base_models(X_train, y_train, X_val, y_val)
        
        self.create_ensemble_models()
        ensemble_results = self.train_ensemble_models(X_train, y_train, X_val, y_val)
        
        model_ranking, improvement = self.compare_models(base_results, ensemble_results)
        
        self.is_trained = True
        
        training_results = {
            'base_results': base_results,
            'ensemble_results': ensemble_results,
            'model_ranking': model_ranking,
            'ensemble_improvement': improvement,
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'training_time': datetime.now().isoformat()
        }
        
        print("=== Ensemble Training Pipeline Completed ===\n")
        
        return training_results

def load_data():
    """Load training and validation data"""
    data_dir = Path("data")
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    
    print(f"Loaded training data: {len(train_df)} samples")
    print(f"Loaded validation data: {len(val_df)} samples")
    
    return train_df, val_df

def main():
    """Main ensemble training script"""
    print("=== Japanese Sentiment Analysis Ensemble Training ===")
    
    try:
        train_df, val_df = load_data()
        
        trainer = EnsembleTrainer()
        
        results = trainer.train(train_df, val_df)
        
        saved_paths = trainer.save_ensemble_models()
        
        print("\n=== Training Summary ===")
        print(f"Ensemble improvement: +{results['ensemble_improvement']:.4f}")
        print(f"Models saved to: {list(saved_paths.values())}")
        
        print("\n=== Ensemble Training Complete ===")
        print("Ready for ensemble evaluation!")
        
    except Exception as e:
        print(f"Error during ensemble training: {e}")
        raise

if __name__ == "__main__":
    main()
