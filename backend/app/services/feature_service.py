#!/usr/bin/env python3
"""
Feature engineering service for Japanese text analysis
"""

import re
from typing import Dict, Any, List
import numpy as np

class JapaneseFeatureExtractor:
    """Feature extractor for Japanese text analysis"""
    
    def __init__(self):
        self.tokenizer = None
        self._init_tokenizer()
        
    def _init_tokenizer(self):
        """Initialize Janome tokenizer with error handling"""
        try:
            from janome.tokenizer import Tokenizer
            self.tokenizer = Tokenizer()
        except ImportError:
            print("Warning: Janome not available. POS features will be disabled.")
            self.tokenizer = None
    
    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """Extract statistical features from Japanese text"""
        if not text:
            return {}
            
        features = {
            'character_count': len(text),
            'sentence_count': len(re.findall(r'[。！？]', text)),
            'exclamation_count': text.count('！') + text.count('!'),
            'question_count': text.count('？') + text.count('?'),
            'punctuation_ratio': len(re.findall(r'[、。！？]', text)) / len(text) if text else 0,
            'hiragana_ratio': len(re.findall(r'[ひ-ゟ]', text)) / len(text) if text else 0,
            'katakana_ratio': len(re.findall(r'[ア-ヿ]', text)) / len(text) if text else 0,
            'kanji_ratio': len(re.findall(r'[一-龯]', text)) / len(text) if text else 0,
        }
        
        features['japanese_ratio'] = features['hiragana_ratio'] + features['katakana_ratio'] + features['kanji_ratio']
        features['avg_sentence_length'] = features['character_count'] / max(1, features['sentence_count'])
        
        return features
    
    def extract_pos_features(self, text: str) -> Dict[str, float]:
        """Extract part-of-speech features using Janome"""
        if not self.tokenizer or not text:
            return {}
            
        try:
            tokens = list(self.tokenizer.tokenize(text))
            if not tokens:
                return {}
                
            pos_counts = {}
            for token in tokens:
                pos = token.part_of_speech.split(',')[0]
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            total_tokens = len(tokens)
            pos_ratios = {f'{pos}_ratio': count/total_tokens for pos, count in pos_counts.items()}
            
            pos_ratios['token_count'] = total_tokens
            
            return pos_ratios
        except Exception as e:
            print(f"Warning: POS feature extraction failed: {e}")
            return {}
    
    def extract_sentiment_indicators(self, text: str) -> Dict[str, float]:
        """Extract sentiment-specific indicators"""
        if not text:
            return {}
            
        positive_patterns = [
            r'[良好素晴優秀最高]',  # Good, excellent, best
            r'[嬉楽喜]',  # Happy, joy
            r'ありがと',  # Thank you
            r'[満足成功]',  # Satisfaction, success
        ]
        
        negative_patterns = [
            r'[悪最悪駄目]',  # Bad, worst, no good
            r'[怒腹立]',  # Anger
            r'[失敗問題]',  # Failure, problem
            r'[不満不安心配]',  # Dissatisfaction, anxiety, worry
        ]
        
        positive_count = sum(len(re.findall(pattern, text)) for pattern in positive_patterns)
        negative_count = sum(len(re.findall(pattern, text)) for pattern in negative_patterns)
        
        features = {
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_polarity': positive_count - negative_count,
            'sentiment_intensity': positive_count + negative_count,
        }
        
        text_length = len(text)
        if text_length > 0:
            features['positive_density'] = positive_count / text_length
            features['negative_density'] = negative_count / text_length
            features['sentiment_density'] = features['sentiment_intensity'] / text_length
        else:
            features['positive_density'] = 0
            features['negative_density'] = 0
            features['sentiment_density'] = 0
            
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, Any]:
        """Extract all available features for a text"""
        features = {}
        
        features.update(self.extract_statistical_features(text))
        
        features.update(self.extract_pos_features(text))
        
        features.update(self.extract_sentiment_indicators(text))
        
        return features
    
    def estimate_word_count(self, text: str) -> int:
        """Estimate word count for Japanese text"""
        if not text:
            return 0
            
        word_boundaries = len(re.findall(r'[\s、。！？]+', text))
        
        char_count = len(text.strip())
        estimated_words = max(1, char_count // 3 + word_boundaries)
        
        return estimated_words
