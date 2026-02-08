"""
Data loader for MixRAGRec training and evaluation.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class DataLoader:
    """Load and manage training/evaluation data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get('data', {})
        
        self.train_data = []
        self.eval_data = []
        self.test_data = []
        
        self.current_epoch = 0
        self.current_idx = 0
        
    def load_data(self):
        """Load train/eval/test data from files"""
        # Load training data
        train_path = self.data_config.get('train_path')
        if train_path:
            self.train_data = self._load_json(train_path)
            print(f"Loaded {len(self.train_data)} training samples")
            
        # Load evaluation data
        eval_path = self.data_config.get('eval_path')
        if eval_path:
            self.eval_data = self._load_json(eval_path)
            print(f"Loaded {len(self.eval_data)} evaluation samples")
            
        # Load test data
        test_path = self.data_config.get('test_path')
        if test_path:
            self.test_data = self._load_json(test_path)
            print(f"Loaded {len(self.test_data)} test samples")
            
    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        file_path = Path(path)
        if not file_path.exists():
            print(f"Warning: Data file not found: {path}")
            return []
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []
            
    def get_train_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of training data"""
        if not self.train_data:
            return []
            
        # Shuffle at start of each epoch
        if self.current_idx == 0:
            random.shuffle(self.train_data)
            
        # Get batch
        end_idx = min(self.current_idx + batch_size, len(self.train_data))
        batch = self.train_data[self.current_idx:end_idx]
        
        # Update index
        self.current_idx = end_idx
        if self.current_idx >= len(self.train_data):
            self.current_idx = 0
            self.current_epoch += 1
            
        return batch
        
    def sample_train_data(self, n_samples: int) -> List[Dict[str, Any]]:
        """Sample n random training samples"""
        if not self.train_data:
            return []
        return random.sample(self.train_data, min(n_samples, len(self.train_data)))
        
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get all evaluation data"""
        return self.eval_data
        
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get all test data"""
        return self.test_data
        
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format a data sample for model input"""
        formatted = {
            'query': '',
            'options': {},
            'ground_truth': None,
            'user_context': {}
        }
        
        # Extract query
        if 'query' in sample:
            formatted['query'] = sample['query']
        elif 'input' in sample:
            formatted['query'] = sample['input']
        elif 'question' in sample:
            formatted['query'] = sample['question']
            
        # Extract options
        if 'options' in sample:
            formatted['options'] = sample['options']
        elif 'choices' in sample:
            formatted['options'] = {
                chr(65 + i): choice for i, choice in enumerate(sample['choices'])
            }
            
        # Extract ground truth
        if 'answer' in sample:
            formatted['ground_truth'] = sample['answer']
        elif 'label' in sample:
            formatted['ground_truth'] = sample['label']
        elif 'ground_truth' in sample:
            formatted['ground_truth'] = sample['ground_truth']
            
        # Extract user context
        if 'user_context' in sample:
            formatted['user_context'] = sample['user_context']
        elif 'context' in sample:
            formatted['user_context'] = sample['context'] if isinstance(sample['context'], dict) else {'context': sample['context']}
            
        return formatted
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get data statistics"""
        return {
            'train_size': len(self.train_data),
            'eval_size': len(self.eval_data),
            'test_size': len(self.test_data),
            'current_epoch': self.current_epoch
        }
