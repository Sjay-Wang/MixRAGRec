import json
import sys
import os
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime


def set_seed(seed: int):
    """"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mixragrec.agents import AgentManager
from mixragrec.marl import MMAPO, RewardCalculator
from mixragrec.utils import ConfigLoader


class FullTrainer:
    """Complete trainer with model saving and loading"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize trainer"""
        print("="*80)
        print("MixRAGRec Full Training System")
        print("="*80)
        
        # Load config
        self.config = ConfigLoader.load_config(config_path)
        
        seed = self.config.get('seed', 42)
        set_seed(seed)
        
        exp_config = self.config.get('experiment', {})
        self.exp_model_name = exp_config.get('model_name', 'MixRAGRec')
        self.exp_dataset = exp_config.get('dataset', 'ml1m')
        self.exp_llm = exp_config.get('llm', 'llama-8b')
        self.exp_suffix = exp_config.get('suffix', '')
        
        self.experiment_name = f"{self.exp_model_name}_{self.exp_dataset}_{self.exp_llm}"
        if self.exp_suffix:
            self.experiment_name = f"{self.experiment_name}_{self.exp_suffix}"
        
        dataset_config = self.config.get('dataset', {}).get(self.exp_dataset, {})
        if not dataset_config:
            raise ValueError(f"Dataset '{self.exp_dataset}' not found in config. Available: {list(self.config.get('dataset', {}).keys())}")
        
        self.data_path = dataset_config.get('data_file', "data/movielens/10000_data_id_20.json")
        self.train_ratio = dataset_config.get('train_ratio', 0.9)
        self.domain = dataset_config.get('domain', 'movie')
        
        kg_db_path = dataset_config.get('kg_db_path')
        if kg_db_path:
            self.config['knowledge_graph']['kg_db_path'] = kg_db_path
        
        kg_indices_path = dataset_config.get('kg_indices_path')
        if kg_indices_path:
            self.config['knowledge_graph']['kg_indices_path'] = kg_indices_path
        
        self.config['domain'] = self.domain
        
        self.model_save_dir = Path("saved_models") / self.experiment_name
        self.checkpoint_dir = Path("trained_checkpoints")
        self.log_dir = Path("logs")
        
        # Create directories
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Save directory: {self.model_save_dir}")
        
        # Load data
        print("\n[1/5] Loading dataset...")
        self.load_dataset()
        
        # Initialize system
        print("\n[2/5] Initializing MixRAGRec framework...")
        self.agent_manager = AgentManager(self.config)
        self.agent_manager.initialize()
        
        # Initialize reward calculator
        print("\n[3/5] Initializing reward calculator...")
        self.reward_calculator = RewardCalculator(self.config)
        print(f"  λ (lambda_mig): {self.reward_calculator.lambda_weight}")
        print(f"  η (eta_cost): {self.reward_calculator.eta}")
        
        # Initialize MMAPO trainer
        print("\n[4/5] Initializing MMAPO trainer...")
        self.mmapo = MMAPO(self.config)
        print(f"  ✓ MMAPO framework initialized")
        print(f"    - Policy Optimization (Expert Selector & Knowledge Aligner): epochs={self.mmapo.optimization_epochs}, clip_ε={self.mmapo.clip_epsilon}")
        print(f"    - Preference Optimization (Recommender): β={self.mmapo.preference_beta}")
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_top3_accuracy = 0.0
        self.best_top5_accuracy = 0.0
        self.training_history = []
        
        print("\n[5/5] System ready!")
        print("="*80)
    
    def load_dataset(self):
        """Load and split dataset"""
        print(f"  Dataset: {self.exp_dataset}")
        print(f"  Data file: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        total_size = len(data)
        train_size = int(total_size * self.train_ratio)
        self.train_data = data[:train_size]
        self.test_data = data[train_size:]
        
        print(f"  Total samples: {total_size:,}")
        print(f"  Train set: {len(self.train_data):,} samples ({self.train_ratio:.0%})")
        print(f"  Test set: {len(self.test_data):,} samples ({1-self.train_ratio:.0%})")
        print(f"  KG database: {self.config['knowledge_graph']['kg_db_path']}")
    
    def preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess a sample"""
        return {
            'task_prompt': self.create_prompt(sample),
            'retrieval_query': sample['input'],
            'watching_history': sample['input'],
            'options': self.parse_options(sample['questions']),
            'correct_answer': sample['output'],
            'sequence_ids': sample.get('sequence_ids', '')
        }
    
    def create_prompt(self, sample: Dict) -> str:
        """"""
        if self.domain == 'music':
            instruction = "Given the user's listening history, select an artist that the user is most likely to enjoy from the options."
            history_label = "Listening history"
            item_type = "artist"
        else:  # movie
            instruction = "Given the user's watching history, select a film that is most likely to interest the user from the options."
            history_label = "Watching history"
            item_type = "movie"
        
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction: {instruction}

{history_label}: {{{sample['input']}}}.

Options: {{{sample['questions']}}}.

Select a {item_type} from options A to T that the user is most likely to be interested in."""
    
    def train_epoch(self, epoch: int, batch_size: int = 10, total_epochs: int = None):
        """
        Train one epoch using MMAPO framework
        Args:
            batch_size: Number of samples before each update
            total_epochs: Total number of epochs (for display)
        """
        total_samples = len(self.train_data)
        start_idx = (epoch * batch_size) % total_samples
        end_idx = start_idx + batch_size
        
        if end_idx <= total_samples:
            batch = self.train_data[start_idx:end_idx]
        else:
            batch = self.train_data[start_idx:] + self.train_data[:end_idx - total_samples]
        
        epoch_rewards = []
        epoch_accuracy = []
        epoch_top3_accuracy = []
        epoch_top5_accuracy = []
        
        # ═══════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════
        episode_data = {
            'expert_selector': [],
            'knowledge_alignment': [],
            'recommender': []
        }
        
        epoch_info = f"Epoch {epoch+1}/{total_epochs}" if total_epochs else f"Epoch {epoch}"
        
        for sample in tqdm(batch, desc=epoch_info):
            try:
                # Create prompt
                prompt = self.create_prompt(sample)
                
                parsed_options = self.parse_options(sample['questions'])
                
                result = self.agent_manager.execute_pipeline(
                    user_query=prompt,
                    user_context={'options': parsed_options},
                    conversation_history=[]
                )
                
                if result.get('pipeline_success'):
                    predicted = result.get('predicted_option')
                    recommendations = result.get('final_recommendations', '')
                    if predicted is None:
                        predicted = list(parsed_options.keys())[0] if parsed_options else 'A'
                        print(f"  ⚠ Constrained decoding returned None, using default: {predicted}")
                    
                    expert_distribution = result.get('option_distribution')
                    
                    correct = (predicted == sample['output'])
                    
                    if correct:
                        top3_correct = True
                        top5_correct = True
                    elif expert_distribution and len(expert_distribution) > 0:
                        sorted_options = sorted(expert_distribution.items(), key=lambda x: x[1], reverse=True)
                        top3_options = [opt for opt, _ in sorted_options[:3]]
                        top5_options = [opt for opt, _ in sorted_options[:5]]
                        top3_correct = sample['output'] in top3_options
                        top5_correct = sample['output'] in top5_options
                    else:
                        top3_correct = False
                        top5_correct = False
                    
                    # Compute reward
                    expert_id = result.get('selected_retriever_id', 1)  # Default to Expert 1
                    
                    # Get baseline if needed for R_MIG
                    baseline_output = None
                    baseline_distribution = None
                    if expert_id != 1:  # Expert 1 is baseline (DirectGenerator)
                        baseline_result = self.agent_manager.execute_pipeline(
                            user_query=prompt,
                            user_context={'options': parsed_options},
                            conversation_history=[],
                            expert_id=1
                        )
                        baseline_output = baseline_result.get('final_recommendations', '')
                        baseline_distribution = baseline_result.get('option_distribution')
                    
                    reward_dict = self.reward_calculator.compute_total_reward(
                        predicted_answer=predicted,
                        ground_truth=sample['output'],
                        expert_id=expert_id,
                        expert_output=recommendations,
                        expert_options=parsed_options,
                        baseline_output=baseline_output,
                        baseline_options=parsed_options if baseline_output is not None else None,
                        confidence=result.get('overall_confidence', 1.0),
                        expert_distribution=expert_distribution,
                        baseline_distribution=baseline_distribution
                    )
                    
                    # ═══════════════════════════════════════════════════════
                    # ═══════════════════════════════════════════════════════
                    
                    rl_selector = self.agent_manager.rl_selector
                    if rl_selector.current_state is not None:
                        # Note: expert_id is 1-4, but action_probs indices are 0-3
                        action_idx = expert_id - 1  # Convert from 1-4 to 0-3
                        with torch.no_grad():
                            action_probs = rl_selector.policy_net(rl_selector.current_state)
                            dist = torch.distributions.Categorical(action_probs)
                            action_log_prob = dist.log_prob(torch.tensor(action_idx, device=rl_selector.current_state.device))
                            
                            value = rl_selector.value_net(rl_selector.current_state).squeeze() if hasattr(rl_selector, 'value_net') else torch.tensor(0.0)
                            next_value = value
                        
                        episode_data['expert_selector'].append({
                            'state': rl_selector.current_state.cpu(),
                            'action': action_idx,  # Store as index (0-3), not expert_id (1-4)
                            'log_prob': action_log_prob.cpu(),
                            'value': value.cpu(),
                            'next_value': next_value.cpu(),
                            'reward': reward_dict.get('R_total', 0.0),
                            'done': True
                        })
                    
                    # Knowledge Aligner experience
                    aligner_prompt = f"Align knowledge for query: {prompt[:200]}..."
                    refined_knowledge = result.get('refined_knowledge', '')
                    aligner_reward = reward_dict.get('R_total', 0.0) * result.get('refinement_confidence', 0.5)
                    
                    episode_data['knowledge_alignment'].append({
                        'prompt': aligner_prompt,
                        'action': refined_knowledge,
                        'reward': aligner_reward,
                    })
                    
                    recommender_prompt = result.get('recommender_prompt', prompt)
                    episode_data['recommender'].append({
                        'prompt': recommender_prompt,
                        'ground_truth': sample['output'],
                        'predicted_option': predicted,
                        'options': parsed_options,
                        'option_distribution': expert_distribution,
                        'reward': reward_dict.get('R_rec', 0.0)
                    })
                    
                    epoch_rewards.append(reward_dict['R_total'])
                    epoch_accuracy.append(1.0 if correct else 0.0)
                    epoch_top3_accuracy.append(1.0 if top3_correct else 0.0)
                    epoch_top5_accuracy.append(1.0 if top5_correct else 0.0)
            

            except Exception as e:
                print(f"\nError on sample: {e}")
                continue

        
        # ═══════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════
        import time
        phase2_start = time.time()
        print(f"\nPhase 2: Updating agents with MMAPO...")
        
        if len(episode_data['expert_selector']) > 0:
            training_agents = {
                'expert_selector': self.agent_manager.rl_selector,
                'knowledge_alignment': self.agent_manager.llm_refiner,
                'recommender': self.agent_manager.llm_generator
            }
            
            mmapo_stats = self.mmapo.train_step(
                agents=training_agents,
                episode_data=episode_data
            )
            
            phase2_time = time.time() - phase2_start
            
            if 'expert_selector' in mmapo_stats and mmapo_stats['expert_selector']:
                print(f"  Expert Selector (Policy) - Policy Loss: {mmapo_stats['expert_selector'].get('policy_loss', 0):.4f}, "
                      f"Value Loss: {mmapo_stats['expert_selector'].get('value_loss', 0):.4f}")
            if 'knowledge_alignment' in mmapo_stats and mmapo_stats['knowledge_alignment']:
                print(f"  Knowledge Aligner (Policy) - Policy Loss: {mmapo_stats['knowledge_alignment'].get('policy_loss', 0):.4f}, "
                      f"Value Loss: {mmapo_stats['knowledge_alignment'].get('value_loss', 0):.4f}")
            if 'recommender' in mmapo_stats and mmapo_stats['recommender']:
                pref_stats = mmapo_stats['recommender']
                loss_scale = pref_stats.get('loss_scale', 1.0)
                scale_info = f", Scale: {loss_scale:.2f}" if loss_scale < 1.0 else ""
                print(f"  Recommender (Preference) - Loss: {pref_stats.get('preference_loss', 0):.4f}, "
                      f"Hard%: {pref_stats.get('hard_neg_ratio', 0):.0%}{scale_info}")
            print(f"  Phase 2 Time: {phase2_time:.1f}s")
        else:
            mmapo_stats = {}
            print("  ⚠ No experiences collected, skipping MMAPO update")
        
        # Epoch statistics
        stats = {
            'epoch': epoch,
            'avg_reward': np.mean(epoch_rewards) if epoch_rewards else 0.0,
            'accuracy': np.mean(epoch_accuracy) if epoch_accuracy else 0.0,
            'top3_accuracy': np.mean(epoch_top3_accuracy) if epoch_top3_accuracy else 0.0,
            'top5_accuracy': np.mean(epoch_top5_accuracy) if epoch_top5_accuracy else 0.0,
            'num_samples': len(epoch_rewards),
            'mmapo_stats': mmapo_stats
        }
        
        self.training_history.append(stats)
        
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
        
        print(f"  Train Metrics - Acc: {stats['accuracy']:.3f} (Top-3: {stats['top3_accuracy']:.3f}, Top-5: {stats['top5_accuracy']:.3f}), Reward: {stats['avg_reward']:.3f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return stats
    
    def evaluate(self, num_samples: int = None):
        """
        Evaluate on test set
        
        Args:
            num_samples: Number of test samples (None for all)
        """
        test_samples = self.test_data[:num_samples] if num_samples else self.test_data
        
        print(f"\nEvaluating on {len(test_samples)} test samples...")
        
        correct = 0
        top3_correct = 0
        top5_correct = 0
        total_rewards = []
        results = []
        
        for sample in tqdm(test_samples, desc="Evaluating"):
            try:
                prompt = self.create_prompt(sample)
                parsed_opts = self.parse_options(sample['questions'])
                
                result = self.agent_manager.execute_pipeline(
                    user_query=prompt,
                    user_context={'options': parsed_opts},
                    conversation_history=[],
                    training=False
                )
                
                if result.get('pipeline_success'):
                    predicted = result.get('predicted_option')
                    if predicted is None:
                        predicted = list(parsed_opts.keys())[0] if parsed_opts else 'A'
                    
                    is_correct = (predicted == sample['output'])
                    if is_correct:
                        correct += 1
                        top3_correct += 1
                        top5_correct += 1
                    else:
                        option_distribution = result.get('option_distribution')
                        if option_distribution:
                            sorted_options = sorted(option_distribution.items(), key=lambda x: x[1], reverse=True)
                            top3_options = [opt for opt, _ in sorted_options[:3]]
                            top5_options = [opt for opt, _ in sorted_options[:5]]
                            if sample['output'] in top3_options:
                                top3_correct += 1
                            if sample['output'] in top5_options:
                                top5_correct += 1
                    
                    expert_id = result.get('selected_retriever_id', 1)
                    
                    # Simple reward (no baseline for eval speed)
                    reward = 1.0 if is_correct else 0.0
                    total_rewards.append(reward)
                    
                    results.append({
                        'predicted': predicted,
                        'ground_truth': sample['output'],
                        'correct': is_correct,
                        'expert_id': expert_id
                    })
            
            except Exception as e:
                print(f"\nError: {e}")
                continue
        
        accuracy = correct / len(results) if results else 0.0
        top3_accuracy = top3_correct / len(results) if results else 0.0
        top5_accuracy = top5_correct / len(results) if results else 0.0
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        
        eval_results = {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'avg_reward': avg_reward,
            'correct': correct,
            'top3_correct': top3_correct,
            'top5_correct': top5_correct,
            'total': len(results),
            'results': results
        }
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {accuracy:.3f} ({correct}/{len(results)})")
        print(f"  Top-3 Accuracy: {top3_accuracy:.3f} ({top3_correct}/{len(results)})")
        print(f"  Top-5 Accuracy: {top5_accuracy:.3f} ({top5_correct}/{len(results)})")
        print(f"  Avg Reward: {avg_reward:.3f}")
        
        # Expert usage distribution
        expert_usage = {}
        for r in results:
            eid = r['expert_id']
            expert_usage[eid] = expert_usage.get(eid, 0) + 1
        
        print(f"  Expert Usage: {expert_usage}")
        
        return eval_results
    
    def save_models(self, epoch: int, samples_per_epoch: int = None, num_epochs: int = None, 
                    is_best: bool = False, accuracy: float = None, top3_accuracy: float = None, 
                    top5_accuracy: float = None):
        """Save trained models with comprehensive metadata for reproducibility
        
        Args:
            epoch: Current epoch number
            samples_per_epoch: Samples per epoch (for metadata)
            num_epochs: Total epochs (for metadata)
            is_best: If True, also save to 'best' directory
            accuracy: Top-1 accuracy (optional)
            top3_accuracy: Top-3 accuracy (optional)
            top5_accuracy: Top-5 accuracy (optional)
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = Path(self.data_path).stem  # e.g., "10000_data_id_20"
        
        save_path = self.model_save_dir / f"epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving models to {save_path}...")
        
        # Save agent coordinator state
        self.agent_manager.save_checkpoint(str(save_path / "agents"))
        
        marl_config = self.config.get('marl', {})
        policy_config = marl_config.get('policy', {})
        preference_config = marl_config.get('preference', {})
        reward_config = self.config.get('reward', {})
        model_config = self.config.get('models', {})
        
        # Save comprehensive training state
        training_state = {
            'epoch': epoch,
            'best_accuracy': self.best_accuracy,
            'best_top3_accuracy': self.best_top3_accuracy,
            'best_top5_accuracy': self.best_top5_accuracy,
            'accuracy': accuracy or self.best_accuracy,
            'top3_accuracy': top3_accuracy or 0.0,
            'top5_accuracy': top5_accuracy or 0.0,
            'experiment_name': self.experiment_name,
            'training_history': self.training_history,
            'timestamp': timestamp,
            
            'dataset': {
                'path': self.data_path,
                'name': dataset_name,
                'train_size': len(self.train_data),
                'test_size': len(self.test_data),
            },
            
            'training_params': {
                'samples_per_epoch': samples_per_epoch,
                'num_epochs': num_epochs,
                'seed': self.config.get('seed', 42),
            },
            
            'marl_params': {
                # Policy optimization parameters
                'policy': {
                    'learning_rate': policy_config.get('learning_rate', 0.0003),
                    'gamma': policy_config.get('gamma', 0.99),
                    'gae_lambda': policy_config.get('gae_lambda', 0.95),
                    'clip_epsilon': policy_config.get('clip_epsilon', 0.2),
                    'entropy_coef': policy_config.get('entropy_coef', 0.01),
                    'value_loss_coef': policy_config.get('value_loss_coef', 0.5),
                    'optimization_epochs': policy_config.get('optimization_epochs', 4),
                },
                # Preference optimization parameters
                'preference': {
                    'beta': preference_config.get('beta', 0.1),
                    'num_hard_negatives': preference_config.get('num_hard_negatives', 5),
                    'use_hard_negatives': preference_config.get('use_hard_negatives', True),
                },
            },
            
            'reward_params': {
                'lambda_mig': reward_config.get('lambda_mig', 0.3),
                'eta_cost': reward_config.get('eta_cost', 0.0),
            },
            
            'model_params': {
                'knowledge_aligner': {
                    'model_name': model_config.get('knowledge_aligner', {}).get('model_name', 'unknown'),
                    'use_quantization': model_config.get('knowledge_aligner', {}).get('use_quantization', False),
                },
                'recommender': {
                    'model_name': model_config.get('recommender', {}).get('model_name', 'unknown'),
                    'use_quantization': model_config.get('recommender', {}).get('use_quantization', False),
                    'answer_max_tokens': model_config.get('recommender', {}).get('answer_max_tokens', 1),
                },
                'rl_selector': {
                    'hidden_dims': model_config.get('rl_selector', {}).get('hidden_dims', [256, 128]),
                    'learning_rate': model_config.get('rl_selector', {}).get('learning_rate', 0.0003),
                },
            },
            
            'full_config': self.config
        }
        
        torch.save(training_state, save_path / "training_state.pt")
        
        with open(save_path / "metadata.txt", 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MixRAGRec Model Checkpoint\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Epoch: {epoch}\n\n")
            
            f.write("【Accuracy (This Checkpoint)】\n")
            f.write(f"  Top-1: {accuracy or self.best_accuracy:.4f}\n")
            f.write(f"  Top-3: {top3_accuracy or 0.0:.4f}\n")
            f.write(f"  Top-5: {top5_accuracy or 0.0:.4f}\n\n")
            
            f.write("【Dataset】\n")
            f.write(f"  Path: {self.data_path}\n")
            f.write(f"  Name: {dataset_name}\n")
            f.write(f"  Train Size: {len(self.train_data)}\n")
            f.write(f"  Test Size: {len(self.test_data)}\n\n")
            
            f.write("【Training Params】\n")
            f.write(f"  Samples per Epoch: {samples_per_epoch}\n")
            f.write(f"  Num Epochs: {num_epochs}\n")
            f.write(f"  Seed: {self.config.get('seed', 42)}\n\n")
            
            f.write("【Policy Optimization Params (Expert Selector & Knowledge Aligner)】\n")
            f.write(f"  Learning Rate: {policy_config.get('learning_rate', 0.0003)}\n")
            f.write(f"  Gamma: {policy_config.get('gamma', 0.99)}\n")
            f.write(f"  Clip Epsilon: {policy_config.get('clip_epsilon', 0.2)}\n")
            f.write(f"  Optimization Epochs: {policy_config.get('optimization_epochs', 4)}\n\n")
            
            f.write("【Preference Optimization Params (Recommender)】\n")
            f.write(f"  Beta: {preference_config.get('beta', 0.1)}\n")
            f.write(f"  Num Hard Negatives: {preference_config.get('num_hard_negatives', 5)}\n")
            f.write(f"  Use Hard Negatives: {preference_config.get('use_hard_negatives', True)}\n\n")
            
            f.write("【Reward Params】\n")
            f.write(f"  Lambda (MIG weight): {reward_config.get('lambda_mig', 0.3)}\n")
            f.write(f"  Eta (Cost penalty): {reward_config.get('eta_cost', 0.0)}\n\n")
            
            f.write("【Models】\n")
            f.write(f"  Knowledge Aligner: {model_config.get('knowledge_aligner', {}).get('model_name', 'unknown')}\n")
            f.write(f"  Recommender: {model_config.get('recommender', {}).get('model_name', 'unknown')}\n")
            f.write("=" * 60 + "\n")
        
        print(f"✓ Models saved with metadata")
        
        # ═══════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════
        if is_best:
            import shutil
            best_path = self.model_save_dir / "best"
            
            if best_path.exists():
                shutil.rmtree(best_path)
            
            shutil.copytree(save_path, best_path)
            
            with open(best_path / "metadata.txt", 'a') as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(">>> THIS IS THE BEST MODEL <<<\n")
                f.write(f"Best Accuracy achieved at Epoch {epoch}\n")
                f.write("=" * 60 + "\n")
            
            print(f"✓ Best model also saved to {best_path}")
    
    def load_models(self, epoch, verbose: bool = True):
        """Load trained models with metadata display
        
        Args:
            epoch: Epoch number (int) or 'best' (str) to load best model
            verbose: Whether to print metadata
        """
        if epoch == 'best' or epoch == -1:
            load_path = self.model_save_dir / "best"
            epoch_str = "best"
        else:
            load_path = self.model_save_dir / f"epoch_{epoch}"
            epoch_str = f"epoch_{epoch}"
        
        if not load_path.exists():
            print(f"✗ No saved models found at {load_path}")
            if epoch == 'best' or epoch == -1:
                print("  Hint: Best model is saved when accuracy improves during training")
            return False
        
        print(f"\nLoading models from {load_path}...")
        
        # Load agent states
        self.agent_manager.load_checkpoint(str(load_path / "agents"))
        
        # Load training state (weights_only=False for PyTorch 2.6+ compatibility)
        training_state = torch.load(load_path / "training_state.pt", weights_only=False)
        self.current_epoch = training_state['epoch']
        self.best_accuracy = training_state['best_accuracy']
        self.best_top3_accuracy = training_state.get('best_top3_accuracy', 0.0)
        self.best_top5_accuracy = training_state.get('best_top5_accuracy', 0.0)
        self.training_history = training_state.get('training_history', [])
        
        if verbose:
            print(f"\n{'='*60}")
            print("【Loaded Model Metadata】")
            print(f"{'='*60}")
            
            exp_name = training_state.get('experiment_name', 'N/A')
            print(f"  Experiment: {exp_name}")
            print(f"  Epoch: {training_state.get('epoch', 'N/A')}")
            print(f"  Saved at: {training_state.get('timestamp', 'N/A')}")
            
            print(f"\n【Accuracy】")
            print(f"  Top-1: {training_state.get('accuracy', training_state.get('best_accuracy', 0)):.4f}")
            print(f"  Top-3: {training_state.get('top3_accuracy', 0):.4f}")
            print(f"  Top-5: {training_state.get('top5_accuracy', 0):.4f}")
            
            dataset_info = training_state.get('dataset', {})
            if dataset_info:
                print(f"\n【Dataset】")
                print(f"  Name: {dataset_info.get('name', 'N/A')}")
                print(f"  Path: {dataset_info.get('path', 'N/A')}")
                print(f"  Train/Test: {dataset_info.get('train_size', 'N/A')}/{dataset_info.get('test_size', 'N/A')}")
            
            training_params = training_state.get('training_params', {})
            if training_params:
                print(f"\n【Training Params】")
                print(f"  Samples/Epoch: {training_params.get('samples_per_epoch', 'N/A')}")
                print(f"  Num Epochs: {training_params.get('num_epochs', 'N/A')}")
                print(f"  Seed: {training_params.get('seed', 'N/A')}")
            
            marl_params = training_state.get('marl_params', {})
            if marl_params:
                policy = marl_params.get('policy', {})
                preference = marl_params.get('preference', {})
                print(f"\n【MARL Params】")
                print(f"  Policy - LR: {policy.get('learning_rate', 'N/A')}, γ: {policy.get('gamma', 'N/A')}, ε: {policy.get('clip_epsilon', 'N/A')}")
                print(f"  Preference - β: {preference.get('beta', 'N/A')}")
            
            reward_params = training_state.get('reward_params', {})
            if reward_params:
                print(f"\n【Reward Params】")
                print(f"  λ (MIG): {reward_params.get('lambda_mig', 'N/A')}, η (Cost): {reward_params.get('eta_cost', 'N/A')}")
            
            print(f"{'='*60}")
        
        print(f"✓ Models loaded from {epoch_str}")
        return True
    
    def train(self, num_epochs: int = 10, samples_per_epoch: int = 100, eval_interval: int = 5):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs
            samples_per_epoch: Samples per epoch
            eval_interval: Evaluate every N epochs
        """
        print("\n" + "="*80)
        print(f"Starting Training: {num_epochs} epochs x {samples_per_epoch} samples")
        print("="*80)
        print(f"Total training samples: {num_epochs * samples_per_epoch}")
        print(f"Reward function: R = R_rec + {self.reward_calculator.lambda_weight}·R_MIG")
        print(f"Baseline expert: Expert 1 (DirectGenerator)")
        print("="*80)
        
        for epoch in range(num_epochs):
            # Train one epoch
            epoch_stats = self.train_epoch(epoch, batch_size=samples_per_epoch, total_epochs=num_epochs)
            
            # Periodic evaluation
            if (epoch + 1) % eval_interval == 0:
                eval_results = self.evaluate(num_samples=1000)
                
                # Update best accuracy
                is_new_best = False
                if eval_results['accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_results['accuracy']
                    is_new_best = True
                
                if eval_results.get('top3_accuracy', 0.0) > self.best_top3_accuracy:
                    self.best_top3_accuracy = eval_results['top3_accuracy']
                
                if eval_results.get('top5_accuracy', 0.0) > self.best_top5_accuracy:
                    self.best_top5_accuracy = eval_results['top5_accuracy']
                
                # Save best model when Top-1 accuracy improves
                if is_new_best:
                    self.save_models(
                        epoch, 
                        samples_per_epoch=samples_per_epoch, 
                        num_epochs=num_epochs, 
                        is_best=True,
                        accuracy=eval_results['accuracy'],
                        top3_accuracy=eval_results.get('top3_accuracy', 0.0),
                        top5_accuracy=eval_results.get('top5_accuracy', 0.0)
                    )
        
        # Final evaluation
        print("\n" + "="*80)
        print("Training Complete - Final Evaluation")
        print("="*80)
        
        final_results = self.evaluate(num_samples=500)
        
        # Save final model
        self.save_models(
            num_epochs - 1, 
            samples_per_epoch=samples_per_epoch, 
            num_epochs=num_epochs,
            accuracy=final_results['accuracy'],
            top3_accuracy=final_results.get('top3_accuracy', 0.0),
            top5_accuracy=final_results.get('top5_accuracy', 0.0)
        )
        
        # Save training log
        self.save_training_log(final_results)
        
        return final_results
    
    def save_training_log(self, final_results: Dict):
        """Save complete training log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_data = {
            'timestamp': timestamp,
            'config': self.config,
            'training_history': self.training_history,
            'final_results': final_results,
            'best_accuracy': self.best_accuracy,
            'best_top3_accuracy': self.best_top3_accuracy,
            'best_top5_accuracy': self.best_top5_accuracy,
            'reward_function': {
                'formula': 'R = R_rec + lambda*R_MIG',
                'R_MIG': 'delta_I - eta*Cost',
                'lambda': self.reward_calculator.lambda_weight,
                'eta': self.reward_calculator.eta
            }
        }
        
        log_path = self.log_dir / f"training_{timestamp}.json"
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n✓ Training log saved: {log_path}")
    
    def parse_options(self, options_str: str) -> Dict[str, str]:
        """Parse options string 'A: Movie1, B: Movie2, ...' (movie names may contain commas)."""
        import re
        options = {}
        if not options_str:
            return options
        parts = re.split(r', (?=[A-T]: )', options_str, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if ': ' in part:
                letter, movie = part.split(': ', 1)
                letter = letter.strip().upper()
                movie = movie.strip()
                if len(letter) == 1 and letter in 'ABCDEFGHIJKLMNOPQRST':
                    options[letter] = movie
        return options
    
    def extract_answer(self, text: str, options: Dict[str, str] = None) -> str:
        """
        Extract answer from generated text
        
        Args:
            text: Generated recommendations
            options: Dictionary of movie options (optional, for better extraction)
        
        Returns:
            Predicted letter (A-T)
        """
        if not text:
            return 'M'  # Default to middle option
        
        text_upper = text.upper()
        
        # Method 1: Look for explicit "ANSWER: X" or "SELECT X" patterns
        import re
        answer_patterns = [
            r'ANSWER:\s*([A-T])',
            r'SELECT\s+([A-T])',
            r'OPTION\s+([A-T])',
            r'CHOOSE\s+([A-T])',
            r'\(([A-T])\)',
            r'^([A-T])[\s:]',  # Start with letter
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)
        
        # Method 2: Check movie names if options provided
        if options:
            movie_mentions = {}
            for letter, movie in options.items():
                # Count how many times each movie is mentioned
                count = text_upper.count(movie.upper())
                if count > 0:
                    movie_mentions[letter] = count
            
            if movie_mentions:
                # Return most mentioned movie
                return max(movie_mentions, key=movie_mentions.get)
        
        # Method 3: Look for any standalone letter
        for letter in 'ABCDEFGHIJKLMNOPQRST':
            if f" {letter} " in text_upper or f" {letter}." in text_upper:
                return letter
        
        # Default to middle option
        return 'M'


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MixRAGRec model")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--samples-per-epoch', type=int, default=100, help='Samples per epoch')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--test-only', action='store_true', help='Only evaluate, no training')
    parser.add_argument('--load-epoch', type=int, help='Load model from specific epoch')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FullTrainer(args.config)
    
    # Load checkpoint if specified
    if args.load_epoch is not None:
        trainer.load_models(args.load_epoch)
    
    if args.test_only:
        # Evaluate only
        print("\n" + "="*80)
        print("Test-Only Mode")
        print("="*80)
        
        results = trainer.evaluate()
        
        print("\n" + "="*80)
        print("Evaluation Complete")
        print("="*80)
        print(f"Final Accuracy: {results['accuracy']:.3f}")
        print(f"Correct: {results['correct']}/{results['total']}")
        
    else:
        # Full training
        final_results = trainer.train(
            num_epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            eval_interval=args.eval_interval
        )
        
        print("\n" + "="*80)
        print("Training Summary")
        print("="*80)
        print(f"Final Accuracies:")
        print(f"  Top-1: {final_results['accuracy']:.3f}")
        print(f"  Top-3: {final_results.get('top3_accuracy', 0.0):.3f}")
        print(f"  Top-5: {final_results.get('top5_accuracy', 0.0):.3f}")
        print(f"\nModels saved in: {trainer.model_save_dir}")
        print(f"Logs saved in: {trainer.log_dir}")
        print("="*80)


if __name__ == "__main__":
    main()
