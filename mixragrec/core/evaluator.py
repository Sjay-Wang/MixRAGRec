"""
Basic evaluator for MixRAGRec recommendation quality.
"""

from typing import Dict, Any, List
import numpy as np


class Evaluator:
    """Basic evaluator for MixRAGRec recommendations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        reward_config = config.get('reward', {})
        self.relevance_weight = reward_config.get('relevance_weight', 0.4)
        self.diversity_weight = reward_config.get('diversity_weight', 0.3)
        self.quality_weight = reward_config.get('quality_weight', 0.3)
    
    def evaluate_result(self, result: Dict[str, Any]) -> float:
        """
        Evaluate a single recommendation result
        
        Args:
            result: Pipeline result
            
        Returns:
            Overall score
        """
        if not result.get('pipeline_success'):
            return 0.0
        
        # Extract metrics
        confidence = result.get('generation_confidence', 0.0)
        diversity = result.get('diversity_score', 0.0)
        quality = result.get('quality_score', 0.0)
        
        # Weighted score
        score = (
            self.relevance_weight * confidence +
            self.diversity_weight * diversity +
            self.quality_weight * quality
        )
        
        return score
    
    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of results
        
        Args:
            results: List of pipeline results
            
        Returns:
            Aggregated metrics
        """
        scores = [self.evaluate_result(r) for r in results]
        success_count = sum(1 for r in results if r.get('pipeline_success'))
        
        metrics = {
            'avg_score': np.mean(scores) if scores else 0.0,
            'success_rate': success_count / len(results) if results else 0.0,
            'total_evaluated': len(results)
        }
        
        return metrics
