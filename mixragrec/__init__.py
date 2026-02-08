"""
MixRAGRec: Mixture-of-Experts Multi-Agent RAG for Recommendation

A multi-agent recommendation system that combines:
- Expert Selector Agent: Dynamically selects optimal retrieval strategy
- Knowledge Alignment Agent: Transforms structured KG into natural language
- Recommendation Agent: Generates personalized recommendations

Training uses MMAPO (Mixture-of-Experts Multi-Agent Policy Optimization).
"""

__version__ = "1.0.0"

from .agents import (
    AgentManager,
    ExpertSelectorAgent,
    KnowledgeAlignmentAgent,
    RecommendationAgent
)
from .marl import MMAPO, RewardCalculator
from .kg import KGRetriever
from .utils import ConfigLoader, DataLoader
from .core import RecommendationPipeline, Evaluator

__all__ = [
    'AgentManager',
    'ExpertSelectorAgent',
    'KnowledgeAlignmentAgent',
    'RecommendationAgent',
    'MMAPO',
    'RewardCalculator',
    'KGRetriever',
    'ConfigLoader',
    'DataLoader',
    'RecommendationPipeline',
    'Evaluator'
]
