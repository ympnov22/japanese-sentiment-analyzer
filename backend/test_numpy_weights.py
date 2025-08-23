#!/usr/bin/env python3
"""Test script to verify numpy weight files are working correctly"""

import numpy as np
import json
from pathlib import Path

def test_numpy_weights():
    """Test loading and inspecting numpy weight files"""
    models_dir = Path("models")
    weights_path = models_dir / "weights.npz"
    metadata_path = models_dir / "model_metadata.json"
    
    print("=== Testing Numpy Weight Files ===")
    
    print(f"Weights file exists: {weights_path.exists()}")
    print(f"Metadata file exists: {metadata_path.exists()}")
    
    if weights_path.exists():
        weights = np.load(weights_path)
        print(f"\nWeight file contents: {list(weights.keys())}")
        print(f"Coef shape: {weights['coef'].shape}")
        print(f"Intercept shape: {weights['intercept'].shape}")
        
        total_bytes = sum([weights[k].nbytes for k in weights.keys()])
        total_mb = total_bytes / (1024 * 1024)
        print(f"Total memory usage: {total_mb:.2f} MB")
        
        print(f"Coef dtype: {weights['coef'].dtype}")
        print(f"Intercept dtype: {weights['intercept'].dtype}")
        
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"\nMetadata model type: {metadata.get('model_type')}")
        print(f"Vectorizer params: {metadata.get('vectorizer_params')}")
        print(f"Sentiment labels: {metadata.get('sentiment_labels')}")
        print(f"Classes from metadata: {metadata.get('sentiment_labels')}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_numpy_weights()
