"""
Model Training Script for Japanese Sentiment Analysis
Implements TF-IDF vectorization and Logistic Regression model training
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class JapaneseSentimentModel:
    """
    Japanese Sentiment Analysis Model using TF-IDF and Logistic Regression
    """
    
    def __init__(self, model_dir="models"):
        """
        Initialize the sentiment analysis model
        
        Args:
            model_dir (str): Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_trained = False
        
        self.sentiment_labels = ["ネガティブ", "ニュートラル", "ポジティブ"]
        self.label_to_index = {label: idx for idx, label in enumerate(self.sentiment_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.sentiment_labels)}
    
    def create_tfidf_vectorizer(self, max_features=10000, ngram_range=(1, 2)):
        """
        Create and configure TF-IDF vectorizer for Japanese text
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): N-gram range for feature extraction
            
        Returns:
            TfidfVectorizer: Configured vectorizer
        """
        print(f"Creating TF-IDF vectorizer with max_features={max_features}, ngram_range={ngram_range}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words=None,  # No built-in Japanese stop words in sklearn
            token_pattern=r'(?u)\b\w+\b',  # Basic tokenization
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        return self.vectorizer
    
    def create_classifier(self, random_state=42):
        """
        Create and configure Logistic Regression classifier
        
        Args:
            random_state (int): Random state for reproducibility
            
        Returns:
            LogisticRegression: Configured classifier
        """
        print(f"Creating Logistic Regression classifier with random_state={random_state}")
        
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced'
        )
        
        return self.classifier
    
    def prepare_labels(self, sentiment_labels):
        """
        Convert sentiment labels to numeric format
        
        Args:
            sentiment_labels (pd.Series): Sentiment labels
            
        Returns:
            np.array: Numeric labels
        """
        print("Converting sentiment labels to numeric format")
        
        numeric_labels = sentiment_labels.map(self.label_to_index)
        
        print(f"Label distribution:")
        for label, count in sentiment_labels.value_counts().items():
            print(f"  {label}: {count} ({count/len(sentiment_labels)*100:.1f}%)")
        
        return numeric_labels.values
    
    def train(self, train_df, val_df=None):
        """
        Train the sentiment analysis model
        
        Args:
            train_df (pd.DataFrame): Training dataset
            val_df (pd.DataFrame): Validation dataset (optional)
            
        Returns:
            dict: Training results and metrics
        """
        print("\n=== Model Training Started ===")
        
        X_train_text = train_df['text'].values
        y_train = self.prepare_labels(train_df['sentiment'])
        
        print(f"Training data: {len(X_train_text)} samples")
        
        if self.vectorizer is None:
            self.create_tfidf_vectorizer()
        
        print("Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
        
        if self.classifier is None:
            self.create_classifier()
        
        print("Training Logistic Regression classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        y_train_pred = self.classifier.predict(X_train_tfidf)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        self.is_trained = True
        
        val_results = None
        if val_df is not None:
            print("Evaluating on validation set...")
            val_results = self.evaluate(val_df, dataset_name="Validation")
        
        training_results = {
            'train_accuracy': train_accuracy,
            'train_samples': len(X_train_text),
            'feature_count': X_train_tfidf.shape[1],
            'validation_results': val_results,
            'training_time': datetime.now().isoformat()
        }
        
        print("=== Model Training Completed ===\n")
        
        return training_results
    
    def evaluate(self, test_df, dataset_name="Test"):
        """
        Evaluate the model on test data
        
        Args:
            test_df (pd.DataFrame): Test dataset
            dataset_name (str): Name of the dataset for reporting
            
        Returns:
            dict: Evaluation metrics and results
        """
        print(f"\n=== {dataset_name} Set Evaluation ===")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_test_text = test_df['text'].values
        y_test = self.prepare_labels(test_df['sentiment'])
        
        print(f"{dataset_name} data: {len(X_test_text)} samples")
        
        X_test_tfidf = self.vectorizer.transform(X_test_text)
        
        y_pred = self.classifier.predict(X_test_tfidf)
        y_pred_proba = self.classifier.predict_proba(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        cm = confusion_matrix(y_test, y_pred)
        
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.sentiment_labels,
            output_dict=True
        )
        
        print(f"{dataset_name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"Predicted →")
        print(f"Actual ↓   {'':>12} {'':>12} {'':>12}")
        for i, true_label in enumerate(self.sentiment_labels):
            row_str = f"{true_label:>12}"
            for j in range(len(self.sentiment_labels)):
                row_str += f"{cm[i][j]:>12}"
            print(row_str)
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'sample_count': len(X_test_text),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        print(f"=== {dataset_name} Set Evaluation Completed ===\n")
        
        return evaluation_results
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            dict: Prediction result with sentiment and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        text_tfidf = self.vectorizer.transform([text])
        
        prediction = self.classifier.predict(text_tfidf)[0]
        probabilities = self.classifier.predict_proba(text_tfidf)[0]
        
        sentiment = self.index_to_label[prediction]
        confidence = float(probabilities[prediction])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(self.sentiment_labels, probabilities)
            }
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        texts_tfidf = self.vectorizer.transform(texts)
        
        predictions = self.classifier.predict(texts_tfidf)
        probabilities = self.classifier.predict_proba(texts_tfidf)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            sentiment = self.index_to_label[pred]
            confidence = float(probs[pred])
            
            results.append({
                'text': texts[i],
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.sentiment_labels, probs)
                }
            })
        
        return results
    
    def save_model(self, model_name="japanese_sentiment_model"):
        """
        Save the trained model and vectorizer
        
        Args:
            model_name (str): Base name for saved files
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        print(f"\n=== Saving Model ===")
        
        vectorizer_path = self.model_dir / f"{model_name}_vectorizer.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        classifier_path = self.model_dir / f"{model_name}_classifier.pkl"
        joblib.dump(self.classifier, classifier_path)
        print(f"Classifier saved to: {classifier_path}")
        
        vectorizer_params = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                           for k, v in self.vectorizer.get_params().items()}
        classifier_params = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                           for k, v in self.classifier.get_params().items()}
        
        metadata = {
            'model_name': model_name,
            'sentiment_labels': self.sentiment_labels,
            'label_to_index': self.label_to_index,
            'index_to_label': self.index_to_label,
            'vectorizer_params': vectorizer_params,
            'classifier_params': classifier_params,
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'save_time': datetime.now().isoformat()
        }
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        print("=== Model Saving Completed ===\n")
        
        return {
            'vectorizer_path': str(vectorizer_path),
            'classifier_path': str(classifier_path),
            'metadata_path': str(metadata_path)
        }
    
    def load_model(self, model_name="japanese_sentiment_model"):
        """
        Load a saved model and vectorizer
        
        Args:
            model_name (str): Base name of saved files
        """
        print(f"\n=== Loading Model ===")
        
        vectorizer_path = self.model_dir / f"{model_name}_vectorizer.pkl"
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded from: {vectorizer_path}")
        
        classifier_path = self.model_dir / f"{model_name}_classifier.pkl"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
        
        self.classifier = joblib.load(classifier_path)
        print(f"Classifier loaded from: {classifier_path}")
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"Metadata loaded from: {metadata_path}")
            
            self.sentiment_labels = metadata['sentiment_labels']
            self.label_to_index = metadata['label_to_index']
            self.index_to_label = {int(k): v for k, v in metadata['index_to_label'].items()}
        
        self.is_trained = True
        print("=== Model Loading Completed ===\n")
        
        return True
    
    def create_confusion_matrix_plot(self, confusion_matrix, save_path=None):
        """
        Create and save confusion matrix visualization
        
        Args:
            confusion_matrix (np.array): Confusion matrix
            save_path (str): Path to save the plot
            
        Returns:
            str: Path to saved plot
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.sentiment_labels,
            yticklabels=self.sentiment_labels
        )
        
        plt.title('Confusion Matrix - Japanese Sentiment Analysis')
        plt.xlabel('Predicted Sentiment')
        plt.ylabel('Actual Sentiment')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.model_dir / 'confusion_matrix.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to: {save_path}")
        
        return str(save_path)

def load_processed_data(data_dir="data"):
    """
    Load processed datasets
    
    Args:
        data_dir (str): Directory containing processed data
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    data_path = Path(data_dir)
    
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    print(f"Loaded datasets:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def main():
    """
    Main model training pipeline
    """
    print("=== Japanese Sentiment Analysis Model Training ===")
    
    try:
        train_df, val_df, test_df = load_processed_data()
        
        model = JapaneseSentimentModel()
        
        training_results = model.train(train_df, val_df)
        
        test_results = model.evaluate(test_df, dataset_name="Test")
        
        cm_plot_path = model.create_confusion_matrix_plot(
            np.array(test_results['confusion_matrix'])
        )
        
        model_paths = model.save_model()
        
        evaluation_report = {
            'training_results': training_results,
            'test_results': test_results,
            'model_paths': model_paths,
            'confusion_matrix_plot': cm_plot_path,
            'evaluation_time': datetime.now().isoformat()
        }
        
        report_path = Path("models") / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation report saved to: {report_path}")
        
        print("\n=== Testing Prediction Functionality ===")
        
        test_texts = [
            "この商品は本当に素晴らしいです！",
            "普通の商品だと思います。",
            "最悪の商品でした。二度と買いません。"
        ]
        
        for text in test_texts:
            result = model.predict(text)
            print(f"Text: {text}")
            print(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            print()
        
        print("=== Model Training Pipeline Complete ===")
        
        return evaluation_report
        
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()
