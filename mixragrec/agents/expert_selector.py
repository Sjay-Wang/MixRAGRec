"""
Expert Selector Agent that uses reinforcement learning to select 
the optimal retriever from available expert options.

Expert IDs: 1-4
1: DirectGenerator - No retrieval, use LLM internal knowledge
2: TripleRetriever - Simple triple-based retrieval
3: SubgraphRetriever - 2-hop subgraph retrieval
4: ConnectedGraphRetriever - PageRank + MST based retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from collections import deque
import random

from .base_agent import BaseAgent, AgentState


class PolicyNetwork(nn.Module):
    """Policy network for expert selection"""
    
    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        # Output layer
        self.output_layer = nn.Linear(input_dim, action_dim)
        self.hidden_network = nn.Sequential(*layers)
        
        # Uniform initialization for equal initial probabilities
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, state):
        hidden = self.hidden_network(state)
        logits = self.output_layer(hidden)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for state value estimation"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)


class ExpertSelectorAgent(BaseAgent):
    """Expert Selector Agent using reinforcement learning"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # Environment configuration
        self.num_experts = config.get('agents', {}).get('num_experts', 4)
        
        # Network configuration (state_dim lives under models.rl_selector)
        selector_config = config.get('models', {}).get('rl_selector', {})
        self.state_dim = selector_config.get('state_dim', 64)
        self.hidden_dims = selector_config.get('hidden_dims', [256, 128])
        self.learning_rate = selector_config.get('learning_rate', 3e-4)
        
        # Initialize networks
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.num_experts,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Exploration parameters (increased for better exploration)
        self.epsilon = 0.5  # Initial exploration rate 50%
        self.epsilon_decay = 0.9995  # Slower decay
        self.epsilon_min = 0.1  # Minimum exploration rate 10%
        
        # Exploration bonus parameters (encourage less-used experts)
        self.exploration_bonus_weight = 0.2
        
        # Statistics
        self.action_counts = np.zeros(self.num_experts)
        self.total_steps = 0
        
    def reset(self):
        """Reset agent state"""
        self.current_state = None
        self.episode_history = deque(maxlen=1000)
        
    def _encode_state(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Encode observation into state feature vector"""
        # Extract query features
        query = observation.get('query', '')
        query_features = self._extract_query_features(query)
        
        # Extract context features
        context_features = self._extract_context_features(observation)
        
        # Extract history features
        history_features = self._extract_history_features()
        
        # Concatenate all features
        state_vector = np.concatenate([
            query_features,
            context_features, 
            history_features
        ])
        
        # Ensure correct feature dimension
        if len(state_vector) != self.state_dim:
            if len(state_vector) < self.state_dim:
                state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)))
            else:
                state_vector = state_vector[:self.state_dim]
                
        return torch.FloatTensor(state_vector).to(self.device)
    
    def _extract_query_features(self, query: str) -> np.ndarray:
        """Extract query features"""
        features = [
            len(query),
            len(query.split()),
            1 if any(word in query.lower() for word in ['recommend', 'suggest']) else 0,
            1 if any(word in query.lower() for word in ['movie', 'film']) else 0,
            1 if any(word in query.lower() for word in ['actor', 'director']) else 0,
        ]
        
        target_len = self.state_dim // 4
        if len(features) < target_len:
            features.extend([0] * (target_len - len(features)))
        else:
            features = features[:target_len]
            
        return np.array(features, dtype=np.float32)
    
    def _extract_context_features(self, observation: Dict[str, Any]) -> np.ndarray:
        """Extract context features"""
        features = [
            observation.get('session_length', 0),
            observation.get('user_preference_score', 0.5),
            observation.get('time_of_day', 12) / 24.0,
            observation.get('previous_satisfaction', 0.5),
        ]
        
        target_len = self.state_dim // 4
        if len(features) < target_len:
            features.extend([0] * (target_len - len(features)))
        else:
            features = features[:target_len]
            
        return np.array(features, dtype=np.float32)
    
    def _extract_history_features(self) -> np.ndarray:
        """Extract history features"""
        if self.total_steps == 0:
            action_dist = np.ones(self.num_experts) / self.num_experts
        else:
            action_dist = self.action_counts / self.total_steps
            
        # deque doesn't support slicing, convert to list
        recent_rewards = [exp.get('reward', 0) for exp in list(self.episode_history)[-10:]]
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)
        else:
            avg_reward = 0
            max_reward = 0
            
        features = list(action_dist) + [avg_reward, max_reward]
        
        target_len = self.state_dim // 2
        if len(features) < target_len:
            features.extend([0] * (target_len - len(features)))
        else:
            features = features[:target_len]
            
        return np.array(features, dtype=np.float32)
    
    def step(self, observation: Dict[str, Any], training: bool = True) -> int:
        """
        Select expert action
        
        Args:
            observation: Observation data
            training: Whether in training mode. Set to False during evaluation to avoid affecting epsilon and statistics
            
        Returns:
            Selected expert ID (1-4)
        """
        state = self._encode_state(observation)
        self.current_state = state
        
        # epsilon-greedy exploration (greedy during evaluation)
        if training and random.random() < self.epsilon:
            # Random action: 0 to num_experts-1, then add 1 to get 1-4
            action = random.randint(0, self.num_experts - 1)
        else:
            with torch.no_grad():
                action_probs = self.policy_net(state.unsqueeze(0))
                action = torch.argmax(action_probs, dim=-1).item()
        
        # Only update statistics and decay epsilon during training
        if training:
            self.action_counts[action] += 1
            self.total_steps += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Return expert ID (1-4) instead of action index (0-3)
        return action + 1
    
    def compute_exploration_bonus(self, action: int) -> float:
        """
        Compute exploration bonus, encouraging less-used experts
        
        Uses count-based exploration bonus: bonus = β / sqrt(N(a) + 1)
        where N(a) is the count of action a
        
        Args:
            action: Action index (0-3)
        """
        action_count = self.action_counts[action]
        bonus = self.exploration_bonus_weight / np.sqrt(action_count + 1)
        return float(bonus)
    
    def update(self, experience: Dict[str, Any]):
        """Update agent parameters (with exploration bonus)"""
        # Action in experience should be expert_id (1-4), convert to index (0-3)
        action = experience.get('action', 1) - 1  # Convert from 1-4 to 0-3
        exploration_bonus = self.compute_exploration_bonus(action)
        
        # Create augmented experience with exploration bonus
        augmented_experience = experience.copy()
        original_reward = experience.get('reward', 0.0)
        augmented_experience['reward'] = original_reward + exploration_bonus
        augmented_experience['action'] = action  # Store as index
        
        self.memory.append(augmented_experience)
        self.episode_history.append(experience)  # Original experience for history
        
        if len(self.memory) >= self.batch_size:
            self._train_networks()
    
    def _train_networks(self):
        """Train policy and value networks"""
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool).to(self.device)
        
        # Compute target values
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            targets = rewards + 0.99 * next_values * (~dones)
        
        # Train value network
        current_values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(current_values, targets)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Train policy network
        action_probs = self.policy_net(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
        advantages = targets - current_values.detach()
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def get_action_space_size(self) -> int:
        return self.num_experts
    
    def get_observation_space_size(self) -> int:
        return self.state_dim
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        base_metrics = super().get_metrics()
        
        selector_metrics = {
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'action_distribution': (self.action_counts / max(1, self.total_steps)).tolist(),
            'memory_size': len(self.memory),
        }
        
        if self.episode_history:
            # deque doesn't support slicing, convert to list
            recent_rewards = [exp.get('reward', 0) for exp in list(self.episode_history)[-100:]]
            selector_metrics.update({
                'avg_recent_reward': np.mean(recent_rewards),
                'std_recent_reward': np.std(recent_rewards),
            })
        
        base_metrics.update(selector_metrics)
        return base_metrics
    
    def save_checkpoint(self, path: str):
        """Save Expert Selector complete state"""
        checkpoint = {
            # Network weights
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            
            # Optimizer states
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            
            # Training state
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'action_counts': self.action_counts.tolist(),
            
            # Experience replay (save last 1000)
            'memory': list(self.memory)[-1000:] if len(self.memory) > 0 else [],
            
            # Configuration
            'config': {
                'state_dim': self.state_dim,
                'num_experts': self.num_experts,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str, map_location=None):
        """Load Expert Selector complete state
        
        Args:
            path: checkpoint file path
            map_location: device mapping for cross-device loading
        """
        import os
        if not os.path.exists(path):
            print(f"  ⚠ No Expert Selector checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, weights_only=False, map_location=map_location)
        
        # Load network weights
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        
        # Load optimizer states
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        # Load training state
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.action_counts = np.array(checkpoint['action_counts'])
        
        # Restore experience
        if 'memory' in checkpoint:
            for exp in checkpoint['memory']:
                self.memory.append(exp)
        
        print(f"  ✓ Expert Selector loaded from {path}")
