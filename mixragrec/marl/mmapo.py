"""
MMAPO: Mixture-of-Experts Multi-Agent Policy Optimization

Mixed training framework for multi-agent recommendation system:
- Expert Selector Agent: Policy gradient optimization with value network
- Knowledge Alignment Agent: Policy optimization with value head
- Recommendation Agent: Preference-based optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import random
import threading
from concurrent.futures import ThreadPoolExecutor


class MMAPO:
    """
    Mixture-of-Experts Multi-Agent Policy Optimization
    
    Mixed training framework:
    - Agent 1 (Expert Selector): Policy gradient with GAE
    - Agent 2 (Knowledge Alignment): Policy optimization with Value Head
    - Agent 3 (Recommendation): Preference-based optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Policy optimization configuration (for Expert Selector and Knowledge Aligner)
        policy_config = config.get('marl', {}).get('policy', {})
        self.gamma = policy_config.get('gamma', 0.99)
        self.gae_lambda = policy_config.get('gae_lambda', 0.95)
        self.clip_epsilon = policy_config.get('clip_epsilon', 0.2)
        self.entropy_coef = policy_config.get('entropy_coef', 0.01)
        self.value_loss_coef = policy_config.get('value_loss_coef', 0.5)
        self.max_grad_norm = policy_config.get('max_grad_norm', 0.5)
        self.optimization_epochs = policy_config.get('optimization_epochs', 4)
        
        # Preference optimization configuration (for Recommendation Agent)
        preference_config = config.get('marl', {}).get('preference', {})
        self.preference_beta = preference_config.get('beta', 0.1)
        self.use_hard_negatives = preference_config.get('use_hard_negatives', False)
        self.num_hard_negatives = preference_config.get('num_hard_negatives', 3)
        
        self.device = config.get('device', 'cpu')
        
        # Parallel update configuration (for multi-GPU)
        self.enable_parallel_update = config.get('marl', {}).get('enable_parallel_update', False)
        
    # ═══════════════════════════════════════════════════════════════════
    # Expert Selector Agent Update
    # ═══════════════════════════════════════════════════════════════════
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                    next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        deltas = rewards + self.gamma * next_values * (~dones) - values
        
        gae = []
        running_gae = 0.0
        
        for t in reversed(range(len(deltas))):
            if dones[t]:
                running_gae = deltas[t].item() if torch.is_tensor(deltas[t]) else deltas[t]
            else:
                delta_val = deltas[t].item() if torch.is_tensor(deltas[t]) else deltas[t]
                running_gae = delta_val + self.gamma * self.gae_lambda * running_gae
            gae.insert(0, running_gae)
        
        gae = torch.tensor(gae, dtype=torch.float32, device=self.device)
        
        # Normalize
        if len(gae) > 1:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        
        return gae
    
    def update_expert_selector(self, 
                               agent,
                               episode_data: List[Dict]) -> Dict[str, float]:
        """
        Update Expert Selector Agent using policy gradient optimization
        
        Args:
            agent: ExpertSelectorAgent instance
            episode_data: Experience data for the selector
            
        Returns:
            Training statistics
        """
        if len(episode_data) == 0:
            return {}
        
        # Prepare data
        states = torch.stack([exp['state'] for exp in episode_data]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in episode_data], dtype=torch.long, device=self.device)
        old_log_probs = torch.stack([exp['log_prob'].squeeze() if exp['log_prob'].dim() > 0 else exp['log_prob'] 
                                     for exp in episode_data]).to(self.device)
        values = torch.stack([exp['value'].squeeze() if exp['value'].dim() > 0 else exp['value'] 
                             for exp in episode_data]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in episode_data], dtype=torch.float32, device=self.device)
        dones = torch.tensor([exp.get('done', True) for exp in episode_data], dtype=torch.bool, device=self.device)
        
        # next_values
        if 'next_value' in episode_data[0]:
            next_values = torch.stack([exp['next_value'].squeeze() if exp['next_value'].dim() > 0 else exp['next_value'] 
                                      for exp in episode_data]).to(self.device)
        else:
            next_values = torch.zeros_like(values)
        
        # Compute GAE
        advantages = self.compute_gae(rewards, values, next_values, dones)
        returns = advantages + values
        
        stats = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        
        # Multiple optimization epochs
        for epoch in range(self.optimization_epochs):
            # Forward pass
            action_probs = agent.policy_net(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_pred = agent.value_net(states).squeeze()
            value_loss = F.mse_loss(value_pred, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            agent.policy_optimizer.zero_grad()
            agent.value_optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(agent.policy_net.parameters()) + list(agent.value_net.parameters()),
                self.max_grad_norm
            )
            
            agent.policy_optimizer.step()
            agent.value_optimizer.step()
            
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.item())
        
        return {k: np.mean(v) for k, v in stats.items()}
    
    # ═══════════════════════════════════════════════════════════════════
    # Knowledge Alignment Agent Update
    # ═══════════════════════════════════════════════════════════════════
    
    def update_knowledge_alignment_agent(self,
                                         agent,
                                         episode_data: List[Dict]) -> Dict[str, float]:
        """
        Update Knowledge Alignment Agent using policy optimization (batch processing)
        
        Args:
            agent: KnowledgeAlignmentAgent instance
            episode_data: Experience data for the alignment agent
            
        Returns:
            Training statistics
        """
        if len(episode_data) == 0:
            return {}
        
        if agent.model is None or agent.optimizer is None:
            return {}
        
        if agent.value_head is None or agent.value_optimizer is None:
            print("  Warning: Value Head not initialized, falling back to REINFORCE")
            return self._update_alignment_reinforce(agent, episode_data)
        
        stats = {'policy_loss': [], 'value_loss': [], 'total_loss': []}
        
        # Filter valid experiences (must have prompt and action)
        valid_experiences = []
        for exp in episode_data:
            prompt = exp.get('prompt', '')
            action = exp.get('action', '')
            if prompt and action and len(action.strip()) > 0:
                valid_experiences.append(exp)
        
        if len(valid_experiences) == 0:
            print("  Warning: No valid experiences for Knowledge Alignment Agent (all actions empty)")
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Extract data
        prompts = [exp.get('prompt', '') for exp in valid_experiences]
        actions = [exp.get('action', '') for exp in valid_experiences]
        rewards = torch.tensor([exp.get('reward', 0.0) for exp in valid_experiences], 
                              dtype=torch.float32, device=agent.device)
        
        batch_size = len(prompts)
        
        # Compute old_log_probs and old_values in batch
        try:
            with torch.no_grad():
                old_log_probs = agent.compute_log_prob_batch(prompts, actions)
                old_values = agent.get_value_batch(prompts, requires_grad=False)
        except Exception as e:
            print(f"  Warning: Batch computation failed, using zeros: {e}")
            old_log_probs = torch.zeros(batch_size, device=agent.device)
            old_values = torch.zeros(batch_size, device=agent.device)
        
        # Compute advantages
        if old_values is not None and len(old_values) > 0:
            advantages = rewards - old_values
        else:
            baseline = rewards.mean()
            advantages = rewards - baseline
        
        # Normalize
        if len(advantages) > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)
            else:
                advantages = advantages - advantages.mean()
        
        old_lp_tensor = old_log_probs
        
        # Update in mini-batches
        mini_batch_size = 8
        num_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
        
        for epoch in range(self.optimization_epochs):
            agent.optimizer.zero_grad()
            agent.value_optimizer.zero_grad()
            
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            valid_count = 0
            
            for batch_start in range(0, batch_size, mini_batch_size):
                batch_end = min(batch_start + mini_batch_size, batch_size)
                batch_prompts = prompts[batch_start:batch_end]
                batch_actions = actions[batch_start:batch_end]
                batch_advantages = advantages[batch_start:batch_end]
                batch_rewards = rewards[batch_start:batch_end]
                batch_old_lp = old_lp_tensor[batch_start:batch_end]
                
                try:
                    # Batch compute log prob
                    new_log_probs = agent.compute_log_prob_batch(batch_prompts, batch_actions)
                    
                    if not new_log_probs.requires_grad:
                        # Fall back to sequential computation
                        batch_log_probs = []
                        for prompt, action in zip(batch_prompts, batch_actions):
                            batch_log_probs.append(agent.compute_log_prob(prompt, action))
                        new_log_probs = torch.stack(batch_log_probs)
                    
                    # Ensure dimension consistency
                    new_log_probs = new_log_probs.view(-1)
                    batch_old_lp_flat = batch_old_lp.view(-1)
                    batch_advantages_flat = batch_advantages.view(-1)
                    batch_rewards_flat = batch_rewards.view(-1)
                    
                    # Policy ratio
                    ratio = torch.exp(new_log_probs - batch_old_lp_flat)
                    ratio = torch.clamp(ratio, 0.01, 100.0)
                    
                    # Clipped objective
                    surr1 = ratio * batch_advantages_flat
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages_flat
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss - batch compute
                    new_values = agent.get_value_batch(batch_prompts)
                    new_values_flat = new_values.view(-1)
                    value_loss = F.mse_loss(new_values_flat, batch_rewards_flat)
                    
                    if torch.isnan(policy_loss) or torch.isnan(value_loss):
                        continue
                    
                    # Gradient scaling
                    actual_batch_size = len(batch_prompts)
                    scaled_loss = (policy_loss + self.value_loss_coef * value_loss) * actual_batch_size
                    scaled_loss.backward()
                    
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    valid_count += 1
                    
                except Exception as e:
                    continue
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update parameters
            if valid_count > 0:
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(agent.value_head.parameters(), self.max_grad_norm)
                
                agent.optimizer.step()
                agent.value_optimizer.step()
                
                stats['policy_loss'].append(epoch_policy_loss / valid_count)
                stats['value_loss'].append(epoch_value_loss / valid_count)
                stats['total_loss'].append((epoch_policy_loss + epoch_value_loss) / valid_count)
        
        return {k: np.mean(v) if v else 0.0 for k, v in stats.items()}
    
    def _update_alignment_reinforce(self, agent, episode_data: List[Dict]) -> Dict[str, float]:
        """
        Fallback: REINFORCE update for alignment agent
        """
        if len(episode_data) == 0 or agent.model is None:
            return {}
        
        rewards = torch.tensor([exp.get('reward', 0.0) for exp in episode_data], 
                              dtype=torch.float32, device=agent.device)
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        valid_count = 0
        
        for i, exp in enumerate(episode_data):
            prompt = exp.get('prompt', '')
            action = exp.get('action', '')
            
            if not prompt or not action:
                continue
            
            try:
                log_prob = agent.compute_log_prob(prompt, action)
                loss = -log_prob * advantages[i]
                total_loss += loss
                valid_count += 1
            except Exception as e:
                continue
        
        if valid_count > 0:
            total_loss = total_loss / valid_count
            agent.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), self.max_grad_norm)
            agent.optimizer.step()
            
            return {'policy_loss': total_loss.item()}
        
        return {}
    
    # ═══════════════════════════════════════════════════════════════════
    # Recommendation Agent Update
    # ═══════════════════════════════════════════════════════════════════
    
    def update_recommendation_agent(self,
                                    agent,
                                    episode_data: List[Dict]) -> Dict[str, float]:
        """
        Update Recommendation Agent using preference-based optimization (batch processing)
        
        Args:
            agent: RecommendationAgent instance
            episode_data: Experience data containing prompt, ground_truth, options
            
        Returns:
            Training statistics
        """
        if len(episode_data) == 0:
            return {}
        
        if agent.model is None or agent.optimizer is None:
            return {}
        
        if not hasattr(agent, 'use_peft') or not agent.use_peft:
            print("  Warning: Preference optimization requires PEFT model, skipping update")
            return {}
                
        # Prepare batch data (with Hard Negative Sampling support)
        prompts = []
        y_wins = []
        y_loses_list = []
        hard_neg_count = 0
        
        for exp in episode_data:
            prompt = exp.get('prompt', '')
            ground_truth = exp.get('ground_truth', '')
            options = exp.get('options', {})
            option_distribution = exp.get('option_distribution', {})
            
            if not prompt or not ground_truth or not options:
                continue
            
            wrong_options = [opt for opt in options.keys() if opt != ground_truth]
            if not wrong_options:
                continue
            
            # Hard Negative Sampling: select top-K wrong options with highest predicted probability
            if self.use_hard_negatives and option_distribution and len(option_distribution) > 0:
                # Sort by probability descending
                sorted_options = sorted(option_distribution.items(), key=lambda x: x[1], reverse=True)
                
                # Find top-K hard negatives
                hard_negatives = []
                for opt, prob in sorted_options:
                    if opt != ground_truth and opt in wrong_options:
                        hard_negatives.append(opt)
                        if len(hard_negatives) >= self.num_hard_negatives:
                            break
                
                if hard_negatives:
                    for neg in hard_negatives:
                        prompts.append(prompt)
                        y_wins.append(ground_truth)
                        y_loses_list.append(neg)
                    hard_neg_count += len(hard_negatives)
                else:
                    # Fallback: random selection
                    prompts.append(prompt)
                    y_wins.append(ground_truth)
                    y_loses_list.append(random.choice(wrong_options))
            else:
                # Random negative sampling (default)
                prompts.append(prompt)
                y_wins.append(ground_truth)
                y_loses_list.append(random.choice(wrong_options))
        
        if len(prompts) == 0:
            return {}
        
        # Log Hard Negative usage
        if self.use_hard_negatives:
            print(f"  Hard Negatives: {hard_neg_count}/{len(prompts)} ({hard_neg_count/len(prompts)*100:.1f}%)")
        
        # Process in mini-batches
        mini_batch_size = 8
        total_loss = 0.0
        valid_count = 0
        
        agent.optimizer.zero_grad()
        
        for batch_start in range(0, len(prompts), mini_batch_size):
            batch_end = min(batch_start + mini_batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_y_wins = y_wins[batch_start:batch_end]
            batch_y_loses = y_loses_list[batch_start:batch_end]
            batch_size_actual = len(batch_prompts)
            
            try:
                # Batch compute preference loss
                loss = agent.compute_preference_loss_batch(batch_prompts, batch_y_wins, batch_y_loses)
                
                if torch.isnan(loss) or torch.isinf(loss) or not loss.requires_grad:
                    continue
                
                # Gradient scaling
                scaled_loss = loss * batch_size_actual
                scaled_loss.backward()
                
                total_loss += loss.item()
                valid_count += 1
                
                del loss, scaled_loss
                
            except Exception as e:
                # Fallback to sequential processing
                for prompt, y_win, y_lose in zip(batch_prompts, batch_y_wins, batch_y_loses):
                    try:
                        loss = agent.compute_preference_loss(prompt, y_win, y_lose)
                        if not torch.isnan(loss) and loss.requires_grad:
                            loss.backward()
                            total_loss += loss.item()
                            valid_count += 1
                    except:
                        continue
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update parameters
        if valid_count > 0:
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), self.max_grad_norm)
            agent.optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'recommendation_loss': total_loss / valid_count,
                'num_samples': len(prompts),
                'preference_loss': total_loss / valid_count,
                'hard_neg_ratio': hard_neg_count / len(prompts) if len(prompts) > 0 else 0.0,
            }
        
        return {}
    
    # ═══════════════════════════════════════════════════════════════════
    # Unified Training Entry
    # ═══════════════════════════════════════════════════════════════════
    
    def train_step(self, agents: Dict[str, Any], episode_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        MMAPO unified training step
        
        Training strategy:
        - Expert Selector: Policy gradient with GAE
        - Knowledge Alignment: Policy optimization with Value Head
        - Recommendation: Preference-based optimization
        
        Args:
            agents: Dictionary of all agents
            episode_data: Experience data for all agents
            
        Returns:
            Training statistics
        """
        stats = {}
        
        # ═══════════════════════════════════════════════════════
        # 1. Update Expert Selector
        # ═══════════════════════════════════════════════════════
        if 'expert_selector' in agents:
            selector_data = episode_data.get('expert_selector', [])
            if len(selector_data) > 0:
                stats['expert_selector'] = self.update_expert_selector(
                    agents['expert_selector'],
                    selector_data
                )
        
        # ═══════════════════════════════════════════════════════
        # 2 & 3. Update Knowledge Alignment and Recommendation Agents
        # If parallel update enabled and models are on different GPUs, run in parallel
        # ═══════════════════════════════════════════════════════
        
        alignment_data = episode_data.get('knowledge_alignment', [])
        recommender_data = episode_data.get('recommender', [])
        
        has_alignment = 'knowledge_alignment' in agents and len(alignment_data) > 0
        has_recommender = 'recommender' in agents and len(recommender_data) > 0
        
        if self.enable_parallel_update and has_alignment and has_recommender:
            # Parallel update for two LLM agents (on different GPUs)
            alignment_result = {}
            recommender_result = {}
            
            def update_alignment():
                nonlocal alignment_result
                try:
                    alignment_result = self.update_knowledge_alignment_agent(
                        agents['knowledge_alignment'],
                        alignment_data
                    )
                except Exception as e:
                    print(f"⚠ Parallel alignment update failed: {e}")
                    alignment_result = {'error': str(e)}
            
            def update_recommender():
                nonlocal recommender_result
                try:
                    recommender_result = self.update_recommendation_agent(
                        agents['recommender'],
                        recommender_data
                    )
                except Exception as e:
                    print(f"⚠ Parallel recommender update failed: {e}")
                    recommender_result = {'error': str(e)}
            
            # Create and start threads
            t1 = threading.Thread(target=update_alignment)
            t2 = threading.Thread(target=update_recommender)
            
            t1.start()
            t2.start()
            
            # Wait for both threads to complete
            t1.join()
            t2.join()
            
            stats['knowledge_alignment'] = alignment_result
            stats['recommender'] = recommender_result
            
        else:
            # Sequential update (single GPU or parallel disabled)
            if has_alignment:
                stats['knowledge_alignment'] = self.update_knowledge_alignment_agent(
                    agents['knowledge_alignment'],
                    alignment_data
                )
        
            if has_recommender:
                stats['recommender'] = self.update_recommendation_agent(
                    agents['recommender'],
                    recommender_data
                )
        
        return stats


def test_mmapo():
    """Test MMAPO framework"""
    print("="*80)
    print("Testing MMAPO Framework (Mixture-of-Experts Multi-Agent Policy Optimization)")
    print("="*80)
    
    config = {
        'marl': {
            'policy': {
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_loss_coef': 0.5,
                'max_grad_norm': 0.5,
                'optimization_epochs': 4
            },
            'preference': {
                'beta': 0.1
            }
        },
        'device': 'cpu'
    }
    
    mmapo = MMAPO(config)
    
    # Simulated experience data
    episode_data = {
        'expert_selector': [
            {
                'state': torch.randn(64),
                'action': 2,
                'log_prob': torch.tensor(-1.2),
                'value': torch.tensor(0.5),
                'next_value': torch.tensor(0.6),
                'reward': 0.8,
                'done': True
            }
            for _ in range(10)
        ],
        'knowledge_alignment': [
            {
                'prompt': 'test prompt',
                'action': 'test action',
                'reward': 0.7,
                'log_prob': torch.tensor(-0.5),
                'value': torch.tensor(0.3)
            }
            for _ in range(10)
        ],
        'recommender': [
            {
                'prompt': 'test prompt',
                'ground_truth': 'A',
                'predicted_option': 'B',
                'options': {'A': 'Movie A', 'B': 'Movie B', 'C': 'Movie C'}
            }
            for _ in range(10)
        ]
    }
    
    print("\nMMAPO framework initialized with:")
    print(f"  - Expert Selector: Policy gradient (gamma={mmapo.gamma}, clip_eps={mmapo.clip_epsilon})")
    print(f"  - Knowledge Alignment: Policy optimization (epochs={mmapo.optimization_epochs})")
    print(f"  - Recommendation: Preference-based (beta={mmapo.preference_beta})")
    
    print("\n✓ MMAPO framework test complete!")


if __name__ == "__main__":
    test_mmapo()
