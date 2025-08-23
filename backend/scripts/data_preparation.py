"""
Data Preparation Script for Japanese Sentiment Analysis
Fetches and preprocesses the Hugging Face amazon-japanese-reviews dataset
"""

import pandas as pd
from datasets import load_dataset
import json
import os
from pathlib import Path

def fetch_dataset(sample_size=10000):
    """
    Fetch the Hugging Face dataset and sample it
    
    Args:
        sample_size (int): Number of samples to use for training
        
    Returns:
        pd.DataFrame: Sampled dataset
    """
    print(f"Fetching Hugging Face dataset: sepidmnorozy/Japanese_sentiment")
    
    dataset = load_dataset("sepidmnorozy/Japanese_sentiment")
    
    df = pd.DataFrame(dataset['train'])
    
    print(f"Original dataset size: {len(df)} records")
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled dataset size: {len(df)} records")
    
    return df

def explore_data(df):
    """
    Explore the dataset structure and statistics
    
    Args:
        df (pd.DataFrame): Dataset to explore
        
    Returns:
        dict: Data exploration results
    """
    print("\n=== Data Exploration ===")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nSample records:")
    print(df.head())
    
    if 'label' in df.columns:
        print(f"\nLabel distribution:")
        label_counts = df['label'].value_counts().sort_index()
        print(label_counts)
        
        label_percentages = (label_counts / len(df) * 100).round(2)
        print(f"\nLabel percentages:")
        for label, percentage in label_percentages.items():
            print(f"Label {label}: {percentage}%")
    
    if 'text' in df.columns:
        df['text_length'] = df['text'].str.len()
        print(f"\nText length statistics:")
        print(df['text_length'].describe())
    
    exploration_results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'label_distribution': df['label'].value_counts().sort_index().to_dict() if 'label' in df.columns else None,
        'text_length_stats': df['text_length'].describe().to_dict() if 'text' in df.columns else None
    }
    
    return exploration_results

def convert_labels(df):
    """
    Convert binary labels to 3-class sentiment labels
    Note: This dataset has binary labels (0=negative, 1=positive)
    We'll create a balanced 3-class dataset by splitting positive samples
    
    Args:
        df (pd.DataFrame): Dataset with label column (0/1)
        
    Returns:
        pd.DataFrame: Dataset with sentiment labels
    """
    print("\n=== Label Conversion ===")
    
    print("Original label distribution:")
    original_counts = df['label'].value_counts().sort_index()
    print(original_counts)
    
    positive_samples = df[df['label'] == 1].copy()
    negative_samples = df[df['label'] == 0].copy()
    
    neutral_count = len(positive_samples) // 3
    
    positive_samples = positive_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    
    neutral_samples = positive_samples[:neutral_count].copy()
    neutral_samples['sentiment'] = "ニュートラル"
    
    positive_samples = positive_samples[neutral_count:].copy()
    positive_samples['sentiment'] = "ポジティブ"
    
    negative_samples['sentiment'] = "ネガティブ"
    
    df = pd.concat([negative_samples, neutral_samples, positive_samples], ignore_index=True)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Label conversion results:")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    
    sentiment_percentages = (sentiment_counts / len(df) * 100).round(2)
    print(f"\nSentiment distribution:")
    for sentiment, percentage in sentiment_percentages.items():
        print(f"{sentiment}: {percentage}%")
    
    print(f"\nFinal dataset size after conversion: {len(df)} records")
    
    return df

def clean_data(df):
    """
    Clean and preprocess the text data
    
    Args:
        df (pd.DataFrame): Dataset to clean
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("\n=== Data Cleaning ===")
    
    initial_size = len(df)
    print(f"Initial dataset size: {initial_size}")
    
    df = df.dropna(subset=['text'])
    print(f"After removing missing text: {len(df)} records")
    
    df = df[df['text'].str.strip() != '']
    print(f"After removing empty text: {len(df)} records")
    
    df = df[df['text'].str.len() >= 5]
    print(f"After removing very short text: {len(df)} records")
    
    df = df[df['text'].str.len() <= 1000]
    print(f"After removing very long text: {len(df)} records")
    
    df['text'] = df['text'].str.strip()
    
    final_size = len(df)
    removed_count = initial_size - final_size
    print(f"Final dataset size: {final_size}")
    print(f"Removed {removed_count} records ({removed_count/initial_size*100:.2f}%)")
    
    return df

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets
    
    Args:
        df (pd.DataFrame): Dataset to split
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("\n=== Data Splitting ===")
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    print(f"Training set: {len(train_df)} records ({len(train_df)/n*100:.1f}%)")
    print(f"Validation set: {len(val_df)} records ({len(val_df)/n*100:.1f}%)")
    print(f"Test set: {len(test_df)} records ({len(test_df)/n*100:.1f}%)")
    
    for name, dataset in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{name} set sentiment distribution:")
        sentiment_counts = dataset['sentiment'].value_counts()
        sentiment_percentages = (sentiment_counts / len(dataset) * 100).round(1)
        for sentiment, percentage in sentiment_percentages.items():
            print(f"  {sentiment}: {percentage}%")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, exploration_results, output_dir="data"):
    """
    Save processed datasets and exploration results
    
    Args:
        train_df (pd.DataFrame): Training dataset
        val_df (pd.DataFrame): Validation dataset
        test_df (pd.DataFrame): Test dataset
        exploration_results (dict): Data exploration results
        output_dir (str): Output directory
    """
    print(f"\n=== Saving Processed Data ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"Saved datasets to {output_dir}/:")
    print(f"  - train.csv: {len(train_df)} records")
    print(f"  - val.csv: {len(val_df)} records")
    print(f"  - test.csv: {len(test_df)} records")
    
    with open(output_path / "data_exploration.json", "w", encoding="utf-8") as f:
        json.dump(exploration_results, f, ensure_ascii=False, indent=2)
    
    print(f"  - data_exploration.json: exploration results")
    
    summary = {
        "total_records": len(train_df) + len(val_df) + len(test_df),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "sentiment_distribution": {
            "train": train_df['sentiment'].value_counts().to_dict(),
            "val": val_df['sentiment'].value_counts().to_dict(),
            "test": test_df['sentiment'].value_counts().to_dict()
        }
    }
    
    with open(output_path / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"  - data_summary.json: summary statistics")

def main():
    """
    Main data preparation pipeline
    """
    print("=== Japanese Sentiment Analysis Data Preparation ===")
    
    try:
        df = fetch_dataset(sample_size=10000)
        
        exploration_results = explore_data(df)
        
        df = convert_labels(df)
        
        df = clean_data(df)
        
        train_df, val_df, test_df = split_data(df)
        
        save_processed_data(train_df, val_df, test_df, exploration_results)
        
        print("\n=== Data Preparation Complete ===")
        print("Ready for model training!")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise

if __name__ == "__main__":
    main()
