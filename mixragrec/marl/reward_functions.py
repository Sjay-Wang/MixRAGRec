"""
Reward Functions for MixRAGRec Multi-Agent Training

Implements:
- R_rec: Recommendation reward (accuracy-based)
- R_MIG: Marginal Information Gain reward (information gain - cost)
- Total reward: R = R_rec + λ·R_MIG
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer


class RewardCalculator:
    """
    Calculates rewards for multi-agent reinforcement learning.
    Implements R = R_rec + λ·R_MIG
    """
    
    # Expert costs (1-4, using smaller differences to encourage exploration)
    EXPERT_COSTS = {
        1: 0.0,   # DirectGenerator - no retrieval, fastest
        2: 0.1,   # TripleRetriever - simple retrieval
        3: 0.2,   # SubgraphRetriever - 2-hop subgraph
        4: 0.3,   # ConnectedGraphRetriever - PageRank + MST
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        reward_config = config.get('reward', {})
        self.lambda_weight = reward_config.get('lambda_mig', 0.5)  # λ for R_MIG
        self.eta = reward_config.get('eta_cost', 0.1)  # η for cost penalty
        
        # For computing output distributions
        self.encoder = None
        self.baseline_expert_id = 1  # Expert 1 (DirectGenerator) as baseline
        
    def load_encoder(self):
        """Load sentence encoder for computing output distributions"""
        if self.encoder is None:
            model_name = self.config.get('models', {}).get('encoder', {}).get('model_name', 
                                                                              'sentence-transformers/all-MiniLM-L6-v2')
            self.encoder = SentenceTransformer(model_name)
    
    def compute_R_rec(self, 
                     predicted_answer: str, 
                     ground_truth: str,
                     confidence: float = 1.0) -> float:
        """
        Compute recommendation reward (R_rec)
        
        Based on prediction accuracy with optional confidence weighting.
        
        Args:
            predicted_answer: Model's predicted answer (A-T)
            ground_truth: Correct answer (A-T)
            confidence: Model's confidence score [0,1]
            
        Returns:
            Recommendation reward
        """
        if predicted_answer == ground_truth:
            # Correct prediction
            R_rec = 1.0 * confidence  # Scale by confidence
        else:
            # Incorrect prediction  
            R_rec = -0.1  # Small negative reward
        
        return R_rec
    
    def compute_output_distribution(self,
                                   recommendation_text: str,
                                   options: Dict[str, str],
                                   temperature: float = 1.0) -> np.ndarray:
        """
        Compute probability distribution over 20 candidate movies
        
        Uses semantic similarity between recommendation and each option.
        
        Args:
            recommendation_text: Generated recommendation text
            options: Dictionary of {letter: movie_name}
            temperature: Temperature for softmax (lower = more peaked)
            
        Returns:
            Probability distribution over options (length 20)
        """
        # Check input validity
        if options is None or len(options) == 0:
            # Return uniform distribution
            return np.ones(20) / 20
        
        if not recommendation_text:
            # Return uniform distribution
            return np.ones(len(options)) / len(options)
        
        self.load_encoder()
        
        # Encode recommendation
        rec_embedding = self.encoder.encode(recommendation_text, normalize_embeddings=True)
        
        # Encode all options
        option_letters = sorted(options.keys())  # A-T in order
        option_texts = [options[letter] for letter in option_letters]
        option_embeddings = self.encoder.encode(option_texts, normalize_embeddings=True)
        
        # Compute similarities (cosine similarity via dot product)
        similarities = np.dot(option_embeddings, rec_embedding)
        
        # Convert to probabilities via softmax with temperature
        logits = similarities / temperature
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities
    
    def compute_kl_divergence(self,
                             p: np.ndarray,
                             q: np.ndarray,
                             epsilon: float = 1e-10) -> float:
        """
        Compute KL divergence: D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
        
        Args:
            p: Probability distribution P (expert)
            q: Probability distribution Q (baseline)
            epsilon: Small value to avoid log(0)
            
        Returns:
            KL divergence value
        """
        # Add epsilon to avoid division by zero
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)
        
        # Compute KL divergence
        kl_div = entropy(p, q)  # scipy.stats.entropy computes KL divergence
        
        return float(kl_div)
    
    def compute_delta_I(self,
                       expert_output: str,
                       expert_options: Dict[str, str],
                       baseline_output: str,
                       baseline_options: Dict[str, str],
                       expert_distribution: Optional[Dict[str, float]] = None,
                       baseline_distribution: Optional[Dict[str, float]] = None) -> float:
        """
        Compute information gain ΔI(e) = D_KL(P_expert || P_baseline)
        
        Prioritizes LLM output logits distribution (more accurate),
        falls back to semantic similarity if not available.
        
        Args:
            expert_output: Expert's recommendation text
            expert_options: Options dict
            baseline_output: Baseline expert's recommendation text
            baseline_options: Options dict
            expert_distribution: LLM output option probability distribution (optional, preferred)
            baseline_distribution: Baseline option probability distribution (optional, preferred)
            
        Returns:
            Information gain
        """
        # Prefer LLM output logits distribution
        if expert_distribution is not None and baseline_distribution is not None:
            # Convert dict to ordered numpy array
            option_letters = sorted(expert_distribution.keys())
            P_expert = np.array([expert_distribution.get(l, 1.0/len(option_letters)) for l in option_letters])
            P_baseline = np.array([baseline_distribution.get(l, 1.0/len(option_letters)) for l in option_letters])
            
            # Normalize to ensure valid probability distribution
            P_expert = P_expert / P_expert.sum()
            P_baseline = P_baseline / P_baseline.sum()
        else:
            # Fallback to semantic similarity method
            P_expert = self.compute_output_distribution(expert_output, expert_options)
            P_baseline = self.compute_output_distribution(baseline_output, baseline_options)
        
        # Compute KL divergence
        delta_I = self.compute_kl_divergence(P_expert, P_baseline)
        
        return delta_I
    
    def compute_cost(self, expert_id: int) -> float:
        """
        Compute computational cost for expert
        
        Args:
            expert_id: Expert ID (1-4)
            
        Returns:
            Normalized cost [0, 1]
        """
        return self.EXPERT_COSTS.get(expert_id, 0.5)
    
    def get_expert_cost(self, expert_id: int) -> float:
        """Alias for compute_cost"""
        return self.compute_cost(expert_id)
    
    def compute_reward(self,
                      prediction: str,
                      ground_truth: str,
                      expert_id: int = 1,
                      confidence: float = 1.0) -> float:
        """
        Simple reward computation (R_rec only, no R_MIG)
        
        For quick reward calculation when baseline output is not available.
        
        Args:
            prediction: Predicted answer
            ground_truth: Correct answer
            expert_id: Expert ID used
            confidence: Model confidence
            
        Returns:
            Reward value
        """
        # Base reward from correctness
        R_rec = self.compute_R_rec(prediction, ground_truth, confidence)
        
        # Subtract cost penalty
        cost_penalty = self.eta * self.compute_cost(expert_id)
        
        return R_rec - cost_penalty
    
    def compute_R_MIG(self,
                     expert_id: int,
                     expert_output: str,
                     expert_options: Dict[str, str],
                     baseline_output: str,
                     baseline_options: Dict[str, str],
                     expert_distribution: Optional[Dict[str, float]] = None,
                     baseline_distribution: Optional[Dict[str, float]] = None) -> float:
        """
        Compute Marginal Information Gain reward
        
        R_MIG = E[ΔI(e)] - η·C(e)
        
        Prioritizes LLM logits distribution for information gain computation.
        
        Args:
            expert_id: Expert ID being evaluated
            expert_output: Expert's recommendation
            expert_options: Options dict
            baseline_output: Baseline expert's recommendation
            baseline_options: Options dict
            expert_distribution: LLM output option probability distribution (optional, preferred)
            baseline_distribution: Baseline option probability distribution (optional, preferred)
            
        Returns:
            Marginal information gain reward
        """
        # Compute information gain
        delta_I = self.compute_delta_I(
            expert_output, expert_options,
            baseline_output, baseline_options,
            expert_distribution, baseline_distribution
        )
        
        # Compute cost
        cost = self.compute_cost(expert_id)
        
        # Marginal information gain
        R_MIG = delta_I - self.eta * cost
        
        return R_MIG
    
    def compute_total_reward(self,
                            predicted_answer: str,
                            ground_truth: str,
                            expert_id: int,
                            expert_output: str,
                            expert_options: Dict[str, str],
                            baseline_output: Optional[str] = None,
                            baseline_options: Optional[Dict[str, str]] = None,
                            confidence: float = 1.0,
                            expert_distribution: Optional[Dict[str, float]] = None,
                            baseline_distribution: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute total reward: R = R_rec + λ·R_MIG
        
        Prioritizes LLM logits distribution for R_MIG computation (more accurate).
        
        Args:
            predicted_answer: Predicted answer (A-T)
            ground_truth: Correct answer (A-T)
            expert_id: Expert ID used
            expert_output: Expert's recommendation text
            expert_options: Options dictionary
            baseline_output: Baseline expert's output (for R_MIG)
            baseline_options: Baseline options
            confidence: Model confidence
            expert_distribution: LLM output option probability distribution (optional, preferred)
            baseline_distribution: Baseline option probability distribution (optional, preferred)
            
        Returns:
            Dictionary with reward components
        """
        # Compute R_rec
        R_rec = self.compute_R_rec(predicted_answer, ground_truth, confidence)
        
        # Compute R_MIG (if baseline provided and all required data available)
        R_MIG = 0.0
        delta_I = 0.0
        if (baseline_output is not None and 
            baseline_options is not None and 
            expert_options is not None and
            expert_id != self.baseline_expert_id):
            try:
                R_MIG = self.compute_R_MIG(
                    expert_id, expert_output, expert_options,
                    baseline_output, baseline_options,
                    expert_distribution, baseline_distribution
                )
                delta_I = R_MIG + self.eta * self.compute_cost(expert_id)
            except Exception as e:
                print(f"Warning: R_MIG computation failed: {e}")
                R_MIG = 0.0
                delta_I = 0.0
        
        # Total reward
        R_total = R_rec + self.lambda_weight * R_MIG
        
        return {
            'R_rec': R_rec,
            'R_MIG': R_MIG,
            'R_total': R_total,
            'lambda': self.lambda_weight,
            'eta': self.eta,
            'delta_I': delta_I,
            'cost': self.compute_cost(expert_id)
        }
    
    def compute_batch_rewards(self,
                             predictions: List[str],
                             ground_truths: List[str],
                             expert_ids: List[int],
                             expert_outputs: List[str],
                             options_list: List[Dict[str, str]],
                             baseline_outputs: Optional[List[str]] = None,
                             confidences: Optional[List[float]] = None) -> List[Dict[str, float]]:
        """
        Compute rewards for a batch of predictions
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of correct answers
            expert_ids: List of expert IDs used
            expert_outputs: List of expert outputs
            options_list: List of options dicts
            baseline_outputs: List of baseline outputs (optional)
            confidences: List of confidence scores (optional)
            
        Returns:
            List of reward dictionaries
        """
        if confidences is None:
            confidences = [1.0] * len(predictions)
        
        if baseline_outputs is None:
            baseline_outputs = [None] * len(predictions)
        
        rewards = []
        
        for i in range(len(predictions)):
            reward_dict = self.compute_total_reward(
                predicted_answer=predictions[i],
                ground_truth=ground_truths[i],
                expert_id=expert_ids[i],
                expert_output=expert_outputs[i],
                expert_options=options_list[i],
                baseline_output=baseline_outputs[i],
                baseline_options=options_list[i],
                confidence=confidences[i]
            )
            rewards.append(reward_dict)
        
        return rewards


def test_reward_calculator():
    """Test reward calculator"""
    print("="*80)
    print("Testing Reward Calculator")
    print("="*80)
    
    config = {
        'reward': {
            'lambda_mig': 0.5,
            'eta_cost': 0.1
        },
        'models': {
            'encoder': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
            }
        }
    }
    
    calculator = RewardCalculator(config)
    
    # Test data
    options = {
        'A': "The Godfather",
        'B': "Star Wars",
        'C': "The Matrix",
        'D': "Inception"
    }
    
    expert_rec = "I recommend The Matrix because it matches your preference for sci-fi films."
    baseline_rec = "Based on your history, I suggest checking out these options."
    
    # Test 1: R_rec
    print("\n1. Testing R_rec...")
    R_rec_correct = calculator.compute_R_rec('C', 'C', confidence=0.9)
    R_rec_wrong = calculator.compute_R_rec('A', 'C', confidence=0.8)
    print(f"  Correct prediction: {R_rec_correct:.3f}")
    print(f"  Wrong prediction: {R_rec_wrong:.3f}")
    
    # Test 2: Output distribution
    print("\n2. Testing output distribution...")
    dist = calculator.compute_output_distribution(expert_rec, options)
    print(f"  Distribution: {dist}")
    print(f"  Sum: {dist.sum():.3f} (should be 1.0)")
    print(f"  Max prob on 'C' (The Matrix): {dist[2]:.3f}")
    
    # Test 3: KL divergence
    print("\n3. Testing KL divergence...")
    expert_dist = calculator.compute_output_distribution(expert_rec, options)
    baseline_dist = calculator.compute_output_distribution(baseline_rec, options)
    kl_div = calculator.compute_kl_divergence(expert_dist, baseline_dist)
    print(f"  KL divergence: {kl_div:.3f}")
    print(f"  Interpretation: {'High' if kl_div > 0.5 else 'Moderate' if kl_div > 0.1 else 'Low'} information gain")
    
    # Test 4: Cost
    print("\n4. Testing costs...")
    for expert_id in range(1, 5):
        cost = calculator.compute_cost(expert_id)
        print(f"  Expert {expert_id}: {cost:.2f}")
    
    # Test 5: R_MIG
    print("\n5. Testing R_MIG...")
    R_MIG = calculator.compute_R_MIG(3, expert_rec, options, baseline_rec, options)
    print(f"  R_MIG for Expert 3: {R_MIG:.3f}")
    print(f"  Components: ΔI={kl_div:.3f}, η·C={calculator.eta * calculator.compute_cost(3):.3f}")
    
    # Test 6: Total reward
    print("\n6. Testing total reward...")
    reward_dict = calculator.compute_total_reward(
        predicted_answer='C',
        ground_truth='C',
        expert_id=3,
        expert_output=expert_rec,
        expert_options=options,
        baseline_output=baseline_rec,
        baseline_options=options,
        confidence=0.9
    )
    
    print(f"  R_rec: {reward_dict['R_rec']:.3f}")
    print(f"  R_MIG: {reward_dict['R_MIG']:.3f}")
    print(f"  R_total: {reward_dict['R_total']:.3f}")
    print(f"  ΔI: {reward_dict['delta_I']:.3f}")
    print(f"  Cost: {reward_dict['cost']:.3f}")
    
    print("\n" + "="*80)
    print("✓ Reward calculator test complete!")
    print("="*80)


if __name__ == "__main__":
    test_reward_calculator()
