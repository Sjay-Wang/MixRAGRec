"""
Metrics collection and analysis for MixRAGRec framework.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import os


class MetricsCollector:
    """"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        self.episode_metrics = deque(maxlen=max_history)
        self.mmapo_metrics = deque(maxlen=max_history)  # MMAPO training metrics
        self.evaluation_metrics = deque(maxlen=max_history)
        
        self.reward_history = deque(maxlen=max_history)
        self.success_rate_history = deque(maxlen=max_history)
        self.confidence_history = deque(maxlen=max_history)
        
    def add_episode_metrics(self, episode: int, metrics: Dict[str, Any]):
        """"""
        metrics_with_episode = {'episode': episode, **metrics}
        self.episode_metrics.append(metrics_with_episode)
        
        if 'total_reward' in metrics:
            self.reward_history.append(metrics['total_reward'])
        
        if 'pipeline_success' in metrics:
            self.success_rate_history.append(1.0 if metrics['pipeline_success'] else 0.0)
        
        if 'generation_confidence' in metrics:
            self.confidence_history.append(metrics['generation_confidence'])
    
    def add_mmapo_metrics(self, episode: int, metrics: Dict[str, Any]):
        """"""
        metrics_with_episode = {'episode': episode, **metrics}
        self.mmapo_metrics.append(metrics_with_episode)
    
    # Alias for compatibility
    def add_grpo_metrics(self, episode: int, metrics: Dict[str, Any]):
        """"""
        self.add_mmapo_metrics(episode, metrics)
    
    def add_evaluation_metrics(self, episode: int, metrics: Dict[str, Any]):
        """"""
        metrics_with_episode = {'episode': episode, **metrics}
        self.evaluation_metrics.append(metrics_with_episode)
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """"""
        if not self.episode_metrics:
            return {}
        
        recent_episodes = list(self.episode_metrics)[-window:]
        
        metrics = {}
        
        if self.success_rate_history:
            recent_success = list(self.success_rate_history)[-window:]
            metrics['success_rate'] = np.mean(recent_success)
        
        if self.reward_history:
            recent_rewards = list(self.reward_history)[-window:]
            metrics['avg_reward'] = np.mean(recent_rewards)
            metrics['std_reward'] = np.std(recent_rewards)
        
        if self.confidence_history:
            recent_confidence = list(self.confidence_history)[-window:]
            metrics['avg_confidence'] = np.mean(recent_confidence)
        
        episode_lengths = [ep.get('episode_length', 1) for ep in recent_episodes if 'episode_length' in ep]
        if episode_lengths:
            metrics['avg_episode_length'] = np.mean(episode_lengths)
        
        return metrics
    
    def get_training_trends(self) -> Dict[str, List[float]]:
        """"""
        trends = {}
        
        if self.reward_history:
            trends['rewards'] = list(self.reward_history)
        
        if self.success_rate_history:
            trends['success_rates'] = list(self.success_rate_history)
        
        if self.confidence_history:
            trends['confidences'] = list(self.confidence_history)
        
        return trends
    
    def compute_moving_average(self, metric_name: str, window: int = 50) -> List[float]:
        """"""
        if metric_name == 'reward' and self.reward_history:
            values = list(self.reward_history)
        elif metric_name == 'success_rate' and self.success_rate_history:
            values = list(self.success_rate_history)
        elif metric_name == 'confidence' and self.confidence_history:
            values = list(self.confidence_history)
        else:
            return []
        
        if len(values) < window:
            return values
        
        moving_avg = []
        for i in range(len(values) - window + 1):
            avg = np.mean(values[i:i + window])
            moving_avg.append(avg)
        
        return moving_avg
    
    def get_summary(self) -> Dict[str, Any]:
        """"""
        summary = {
            'total_episodes': len(self.episode_metrics),
            'total_mmapo_updates': len(self.mmapo_metrics),
            'total_evaluations': len(self.evaluation_metrics)
        }
        
        recent_performance = self.get_recent_performance()
        summary['recent_performance'] = recent_performance
        
        if self.reward_history:
            summary['reward_stats'] = {
                'mean': np.mean(self.reward_history),
                'std': np.std(self.reward_history),
                'min': np.min(self.reward_history),
                'max': np.max(self.reward_history)
            }
        
        if self.success_rate_history:
            summary['overall_success_rate'] = np.mean(self.success_rate_history)
        
        if self.episode_metrics:
            best_episode = max(self.episode_metrics, 
                             key=lambda x: x.get('total_reward', 0))
            summary['best_episode'] = {
                'episode': best_episode.get('episode', 0),
                'reward': best_episode.get('total_reward', 0),
                'success': best_episode.get('pipeline_success', False)
            }
        
        return summary
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """"""
        if not self.episode_metrics:
            print("No metrics to plot")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MixRAGRec Training Metrics', fontsize=16)
        
        if self.reward_history:
            axes[0, 0].plot(list(self.reward_history), alpha=0.3, color='blue', label='Raw')
            moving_avg = self.compute_moving_average('reward', 50)
            if moving_avg:
                axes[0, 0].plot(range(49, len(self.reward_history)), moving_avg, 
                               color='red', linewidth=2, label='Moving Avg (50)')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if self.success_rate_history:
            moving_avg = self.compute_moving_average('success_rate', 50)
            if moving_avg:
                axes[0, 1].plot(range(49, len(self.success_rate_history)), moving_avg,
                               color='green', linewidth=2, label='Success Rate (50-ep avg)')
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        if self.confidence_history:
            moving_avg = self.compute_moving_average('confidence', 50)
            if moving_avg:
                axes[1, 0].plot(range(49, len(self.confidence_history)), moving_avg,
                               color='purple', linewidth=2, label='Confidence (50-ep avg)')
            axes[1, 0].set_title('Generation Confidence')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Confidence')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        if self.reward_history:
            axes[1, 1].hist(list(self.reward_history), bins=30, alpha=0.7, color='orange')
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_agent_performance(self, save_path: Optional[str] = None):
        """"""
        if not self.episode_metrics:
            print("No metrics to plot")
            return
        
        agent_metrics = defaultdict(list)
        episodes = []
        
        for ep_data in self.episode_metrics:
            if 'episode' in ep_data:
                episodes.append(ep_data['episode'])
                
                for agent in ['expert_selector', 'knowledge_aligner', 'recommender']:
                    policy_loss_key = f'{agent}_policy_loss'
                    if policy_loss_key in ep_data:
                        agent_metrics[agent].append(ep_data[policy_loss_key])
                    else:
                        agent_metrics[agent].append(np.nan)
        
        if not episodes:
            print("No episode data found")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Agent Performance Metrics', fontsize=16)
        
        colors = ['blue', 'red', 'green']
        agent_names = ['Expert Selector', 'Knowledge Aligner', 'Recommender']
        
        for i, (agent, name, color) in enumerate(zip(['expert_selector', 'knowledge_aligner', 'recommender'], 
                                                    agent_names, colors)):
            if agent in agent_metrics and agent_metrics[agent]:
                valid_data = [(ep, loss) for ep, loss in zip(episodes, agent_metrics[agent]) 
                             if not np.isnan(loss)]
                
                if valid_data:
                    valid_episodes, valid_losses = zip(*valid_data)
                    axes[i].plot(valid_episodes, valid_losses, color=color, alpha=0.7)
                    axes[i].set_title(f'{name} Policy Loss')
                    axes[i].set_xlabel('Episode')
                    axes[i].set_ylabel('Policy Loss')
                    axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Agent performance plots saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_metrics(self, output_dir: str):
        """"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_training_curves(os.path.join(output_dir, "training_curves.png"))
        self.plot_agent_performance(os.path.join(output_dir, "agent_performance.png"))
        
        summary = self.get_summary()
        
        report_path = os.path.join(output_dir, "training_report.txt")
        with open(report_path, 'w') as f:
            f.write("MixRAGRec Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Episodes: {summary.get('total_episodes', 0)}\n")
            f.write(f"Total MMAPO Updates: {summary.get('total_mmapo_updates', 0)}\n")
            f.write(f"Total Evaluations: {summary.get('total_evaluations', 0)}\n\n")
            
            if 'recent_performance' in summary:
                f.write("Recent Performance (last 100 episodes):\n")
                for key, value in summary['recent_performance'].items():
                    f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            if 'reward_stats' in summary:
                f.write("Overall Reward Statistics:\n")
                for key, value in summary['reward_stats'].items():
                    f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            f.write(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.4f}\n")
            
            if 'best_episode' in summary:
                best = summary['best_episode']
                f.write(f"\nBest Episode: {best['episode']}\n")
                f.write(f"  Reward: {best['reward']:.4f}\n")
                f.write(f"  Success: {best['success']}\n")
        
        print(f"Metrics exported to {output_dir}")
