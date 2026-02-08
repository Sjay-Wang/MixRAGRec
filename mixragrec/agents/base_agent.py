"""
Base agent class defining the interface for all agents in the MixRAGRec framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import numpy as np


class AgentState:
    """Agent state encapsulation"""
    
    def __init__(self, 
                 observation: Any,
                 reward: float = 0.0,
                 done: bool = False,
                 info: Optional[Dict[str, Any]] = None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'observation': self.observation,
            'reward': self.reward,
            'done': self.done,
            'info': self.info
        }


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.device = config.get('device', 'cpu')
        self.current_state = None
        self.episode_history = []
        
    @abstractmethod
    def reset(self):
        """Reset agent state"""
        pass
        
    @abstractmethod
    def step(self, observation: Any) -> Any:
        """Execute one action step"""
        pass
        
    @abstractmethod
    def update(self, experience: Dict[str, Any]):
        """Update agent parameters"""
        pass
        
    def get_action_space_size(self) -> int:
        """Get action space size"""
        return 1
        
    def get_observation_space_size(self) -> int:
        """Get observation space size"""
        return 1
        
    def save_checkpoint(self, path: str):
        """Save agent checkpoint"""
        pass
        
    def load_checkpoint(self, path: str, map_location=None):
        """Load agent checkpoint
        
        Args:
            path: checkpoint file path
            map_location: device mapping for cross-device loading
        """
        pass
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            'agent_id': self.agent_id,
            'episode_length': len(self.episode_history)
        }
