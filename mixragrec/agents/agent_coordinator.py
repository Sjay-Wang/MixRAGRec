"""
Agent Manager that coordinates the three collaborative agents in the MixRAGRec framework.

This is the central coordination component that orchestrates:
1. Expert Selector Agent (RL-based retriever selection)
2. Knowledge Alignment Agent (LLM-based knowledge refinement)
3. Recommendation Agent (LLM-based recommendation generation)

Expert IDs: 1-4 (Direct, Triple, Subgraph, Connected)
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from .base_agent import BaseAgent, AgentState
from .expert_selector import ExpertSelectorAgent as RLSelectorAgent
from .knowledge_aligner import KnowledgeAlignmentAgent
from .recommender import RecommendationAgent
from ..kg.retrieval import KGRetriever


class AgentManager:
    """
    MixRAGRec Framework:
    - Expert Selector (rl_selector): RL-based expert selection (Experts 1-4)
    - Knowledge Aligner (llm_refiner): Knowledge alignment and refinement
    - Recommender (llm_generator): Recommendation generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cpu')
        
        self.retrieval_manager = KGRetriever(config)
        
        self.rl_selector = RLSelectorAgent('expert_selector', config)
        self.llm_refiner = KnowledgeAlignmentAgent('knowledge_alignment', config)
        self.llm_generator = RecommendationAgent('recommendation', config)
        
        self.agents = {
            'expert_selector': self.rl_selector,
            'knowledge_alignment': self.llm_refiner,
            'recommender': self.llm_generator
        }
        
        self.interaction_history = []
        self.episode_data = {}
        
        self.episode_rewards = []
        self.pipeline_success_rate = []
        
    def initialize(self):
        """"""
        print("Initializing MixRAGRec Agent Manager...")
        
        print("- Initializing retrieval manager...")
        self.retrieval_manager.initialize()
        
        print("- Initializing LLM models...")
        self.llm_refiner.initialize_model()
        self.llm_generator.initialize_model()
        
        self.reset_all_agents()
        
        print("Agent Manager initialization complete!")
        
    def reset_all_agents(self):
        """"""
        for agent in self.agents.values():
            agent.reset()
        self.episode_data = {}
        
    def execute_pipeline(self, 
                        user_query: str,
                        user_context: Optional[Dict[str, Any]] = None,
                        conversation_history: Optional[List[Dict[str, Any]]] = None,
                        training: bool = True,
                        expert_id: Optional[int] = None) -> Dict[str, Any]:
        """
        expert_id: If set (1-4), use this expert for retrieval and skip RL selection.
                   Used for R_MIG baseline (expert_id=1) or when caller specifies an expert.
        """
        
        if user_context is None:
            user_context = {}
        if conversation_history is None:
            conversation_history = []
            
        pipeline_result = {
            'user_query': user_query,
            'user_context': user_context,
            'conversation_history': conversation_history,
            'pipeline_success': False,
            'error_message': None
        }
        
        try:
            if expert_id is not None:
                selected_retriever_id = expert_id
            else:
                selector_observation = {
                    'query': user_query,
                    'user_context': user_context,
                    'session_length': len(conversation_history),
                    'user_preference_score': user_context.get('preference_score', 0.5),
                    'previous_satisfaction': self._get_previous_satisfaction(conversation_history)
                }
                selected_retriever_id = self.rl_selector.step(selector_observation, training=training)
            
            retrieval_query = self._extract_watching_history(user_query)
            if not retrieval_query:
                retrieval_query = user_query
            
            kg_config = self.config.get('knowledge_graph', {})
            retrieval_top_m = kg_config.get('retrieval_top_m', 10)
            
            retrieval_results = self.retrieval_manager.retrieve(
                query=retrieval_query,
                expert_id=selected_retriever_id,
                top_k=retrieval_top_m
            )
            
            formatted_results = []
            if hasattr(retrieval_results, 'retrieved_knowledge'):
                formatted_results.append({
                    'document': retrieval_results.retrieved_knowledge,
                    'score': retrieval_results.confidence if hasattr(retrieval_results, 'confidence') else 1.0,
                    'metadata': retrieval_results.metadata if hasattr(retrieval_results, 'metadata') else {}
                })
            elif hasattr(retrieval_results, 'documents') and hasattr(retrieval_results, 'scores'):
                for doc, score in zip(retrieval_results.documents, retrieval_results.scores):
                    formatted_results.append({
                        'document': doc,
                        'score': score if score is not None else 0.0
                    })
            else:
                formatted_results = [{
                    'document': str(retrieval_results) if retrieval_results else '',
                    'score': 0.0
                }]
            
            retrieved_doc = formatted_results[0]['document'] if formatted_results else ''
            
            if selected_retriever_id == 1:
                # Expert 1 (DirectGenerator) doesn't need knowledge alignment
                refined_knowledge = ""
                refinement_confidence = 1.0
            else:
                # Always use Knowledge Alignment Agent for other experts
                alignment_observation = {
                    'retrieval_results': formatted_results,
                    'watching_history': self._extract_watching_history(user_query),
                    'user_context': user_context
                }
                
                refinement_result = self.llm_refiner.step(alignment_observation)
                refined_knowledge = refinement_result['refined_knowledge']
                refinement_confidence = refinement_result['confidence']
            
            generator_observation = {
                'refined_knowledge': refined_knowledge,
                'original_query': user_query,
                'user_context': user_context,
                'conversation_history': conversation_history
            }
            
            generation_result = self.llm_generator.step(generator_observation)
            final_recommendations = generation_result['recommendations']
            generation_confidence = generation_result['confidence']
            diversity_score = generation_result['diversity_score']
            quality_score = generation_result['quality_score']
            explanation = generation_result.get('explanation', '')
            explanation_quality = generation_result.get('explanation_quality', 0.0)
            predicted_option = generation_result.get('predicted_option', None)
            option_distribution = generation_result.get('option_distribution', None)
            
            recommender_prompt = self.llm_generator._construct_option_prompt(
                refined_knowledge=refined_knowledge,
                original_query=user_query,
                options=user_context.get('options', {}),
                user_context=user_context
            ) if hasattr(self.llm_generator, '_construct_option_prompt') else user_query
            
            pipeline_result.update({
                'selected_retriever_id': selected_retriever_id,
                'retrieval_results': formatted_results,
                'refined_knowledge': refined_knowledge,
                'refinement_confidence': refinement_confidence,
                'explanation': explanation,
                'explanation_quality': explanation_quality,
                'final_recommendations': final_recommendations,
                'generation_confidence': generation_confidence,
                'diversity_score': diversity_score,
                'quality_score': quality_score,
                'overall_confidence': (refinement_confidence + generation_confidence + explanation_quality) / 3,
                'pipeline_success': True,
                'predicted_option': predicted_option,
                'option_distribution': option_distribution,
                'recommender_prompt': recommender_prompt
            })
            
            interaction_record = {
                'timestamp': len(self.interaction_history),
                'user_query': user_query,
                'selected_retriever': selected_retriever_id,
                'refinement_confidence': refinement_confidence,
                'generation_confidence': generation_confidence,
                'diversity_score': diversity_score,
                'quality_score': quality_score
            }
            self.interaction_history.append(interaction_record)
            
        except Exception as e:
            import traceback
            print(f"Error in pipeline execution: {e}")
            print("=== Full Traceback ===")
            traceback.print_exc()
            pipeline_result.update({
                'pipeline_success': False,
                'error_message': str(e),
                'final_recommendations': f"I apologize, but I encountered an error while processing your request: {str(e)}"
            })
        
        return pipeline_result
    
    def _get_previous_satisfaction(self, conversation_history: List[Dict[str, Any]]) -> float:
        """"""
        if not conversation_history:
            return 0.5
        
        positive_keywords = ['good', 'great', 'helpful', 'perfect', 'excellent', 'thanks']
        negative_keywords = ['bad', 'wrong', 'unhelpful', 'terrible', 'awful', 'no']
        
        satisfaction_scores = []
        for interaction in conversation_history[-3:]:
            user_msg = interaction.get('user', '').lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in user_msg)
            negative_count = sum(1 for keyword in negative_keywords if keyword in user_msg)
            
            if positive_count > negative_count:
                satisfaction_scores.append(0.8)
            elif negative_count > positive_count:
                satisfaction_scores.append(0.2)
            else:
                satisfaction_scores.append(0.5)
        
        return np.mean(satisfaction_scores) if satisfaction_scores else 0.5
    
    def _extract_watching_history(self, query: str) -> str:
        """"""
        import re
        
        pattern = r'Watching history:\s*\{([^}]+)\}'
        match = re.search(pattern, query)
        
        if match:
            movies_str = match.group(1)
            movies = re.findall(r'"([^"]+)"', movies_str)
            if movies:
                return "movies: " + ", ".join(movies)
            return f"movies: {movies_str}"
        
        movie_pattern = r'"([^"]+)"'
        movies = re.findall(movie_pattern, query)
        
        options_start = query.find("Options:")
        if options_start > 0 and movies:
            query_before_options = query[:options_start]
            movies = re.findall(movie_pattern, query_before_options)
        
        if movies:
            return "movies: " + ", ".join(movies[:10])
        
        return ""
    
    def compute_reward(self, 
                      pipeline_result: Dict[str, Any],
                      user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """"""
        if not pipeline_result.get('pipeline_success', False):
            return {
                'expert_selector': -1.0,
                'knowledge_aligner': -1.0,
                'recommender': -1.0,
                'overall': -1.0
            }
        
        refinement_confidence = pipeline_result.get('refinement_confidence', 0.0)
        generation_confidence = pipeline_result.get('generation_confidence', 0.0)
        diversity_score = pipeline_result.get('diversity_score', 0.0)
        quality_score = pipeline_result.get('quality_score', 0.0)
        
        refinement_confidence = refinement_confidence if refinement_confidence is not None else 0.0
        generation_confidence = generation_confidence if generation_confidence is not None else 0.0
        diversity_score = diversity_score if diversity_score is not None else 0.0
        quality_score = quality_score if quality_score is not None else 0.0
        
        base_rewards = {
            'expert_selector': 0.5,
            'knowledge_aligner': refinement_confidence,
            'recommender': (generation_confidence + diversity_score + quality_score) / 3
        }
        
        feedback_multiplier = 1.0
        if user_feedback:
            satisfaction = user_feedback.get('satisfaction', 0.5) or 0.5
            usefulness = user_feedback.get('usefulness', 0.5) or 0.5
            feedback_multiplier = (satisfaction + usefulness) / 2
        
        final_rewards = {}
        for agent_id, base_reward in base_rewards.items():
            final_rewards[agent_id] = base_reward * feedback_multiplier
            
        downstream_performance = (final_rewards['knowledge_aligner'] + final_rewards['recommender']) / 2
        final_rewards['expert_selector'] = downstream_performance
        
        final_rewards['overall'] = np.mean(list(final_rewards.values()))
        
        return final_rewards
    
    def update_agents(self, 
                     pipeline_result: Dict[str, Any],
                     rewards: Dict[str, float]):
        """"""
        
        experiences = {}
        
        selector_experience = {
            'state': self.rl_selector.current_state,
            'action': pipeline_result.get('selected_retriever_id', 1),  # Default to Expert 1
            'reward': rewards['expert_selector'],
            'next_state': self.rl_selector.current_state,
            'done': True
        }
        experiences['expert_selector'] = selector_experience
        
        refiner_experience = {
            'refined_knowledge': pipeline_result.get('refined_knowledge', ''),
            'reward': rewards['knowledge_aligner'],
            'confidence': pipeline_result.get('refinement_confidence', 0.0)
        }
        experiences['knowledge_aligner'] = refiner_experience
        
        generator_experience = {
            'recommendations': pipeline_result.get('final_recommendations', ''),
            'reward': rewards['recommender'],
            'confidence': pipeline_result.get('generation_confidence', 0.0),
            'diversity_score': pipeline_result.get('diversity_score', 0.0),
            'quality_score': pipeline_result.get('quality_score', 0.0)
        }
        experiences['recommender'] = generator_experience
        
        for agent_id, experience in experiences.items():
            if agent_id in self.agents:
                self.agents[agent_id].update(experience)
        
        self.episode_rewards.append(rewards['overall'])
        
        success = 1.0 if pipeline_result.get('pipeline_success', False) else 0.0
        self.pipeline_success_rate.append(success)
    
    def update_retrieval_index(self, documents: List[str]):
        """"""
        self.retrieval_manager.update_all_indices(documents)
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """"""
        metrics = {
            'total_interactions': len(self.interaction_history),
            'agent_metrics': {}
        }
        
        for agent_id, agent in self.agents.items():
            metrics['agent_metrics'][agent_id] = agent.get_metrics()
        
        metrics['retrieval_stats'] = self.retrieval_manager.get_retriever_stats()
        
        if self.episode_rewards:
            metrics.update({
                'avg_episode_reward': np.mean(self.episode_rewards[-100:]),
                'std_episode_reward': np.std(self.episode_rewards[-100:]),
                'total_episodes': len(self.episode_rewards)
            })
            
        if self.pipeline_success_rate:
            metrics['pipeline_success_rate'] = np.mean(self.pipeline_success_rate[-100:])
            
        if self.interaction_history:
            recent_interactions = self.interaction_history[-50:]
            metrics.update({
                'avg_refinement_confidence': np.mean([i['refinement_confidence'] for i in recent_interactions]),
                'avg_generation_confidence': np.mean([i['generation_confidence'] for i in recent_interactions]),
                'avg_diversity_score': np.mean([i['diversity_score'] for i in recent_interactions]),
                'avg_quality_score': np.mean([i['quality_score'] for i in recent_interactions])
            })
        
        return metrics
    
    def set_training_mode(self, training: bool):
        """"""
        self.llm_refiner.set_training_mode(training)
        self.llm_generator.set_training_mode(training)
        
    def save_checkpoint(self, checkpoint_path: str):
        """"""
        checkpoint = {
            'config': self.config,
            'interaction_history': self.interaction_history,
            'episode_rewards': self.episode_rewards,
            'pipeline_success_rate': self.pipeline_success_rate
        }
        
        for agent_id, agent in self.agents.items():
            agent.save_checkpoint(f"{checkpoint_path}_{agent_id}.pt")
            
        torch.save(checkpoint, f"{checkpoint_path}_system.pt")
        
    def load_checkpoint(self, checkpoint_path: str, device: str = None):
        """"""
        if device is None:
            device = self.config.get('device', 'cuda:0')
        map_location = torch.device(device) if device else None
        
        try:
            # weights_only=False for PyTorch 2.6+ compatibility
            checkpoint = torch.load(f"{checkpoint_path}_system.pt", weights_only=False, map_location=map_location)
            self.interaction_history = checkpoint.get('interaction_history', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.pipeline_success_rate = checkpoint.get('pipeline_success_rate', [])
            
            for agent_id, agent in self.agents.items():
                agent.load_checkpoint(f"{checkpoint_path}_{agent_id}.pt", map_location=map_location)
                
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
