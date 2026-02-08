"""
Recommendation Agent for MixRAGRec
Combines aligned knowledge with query to generate recommendations.

Modified: Added Constrained Decoding to force the final token to be one of A-T options.

Training: Preference-based optimization (MMAPO framework)
- Policy: LLM + LoRA (trainable)
- Reference: Original LLM weights (frozen, for KL constraint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, LogitsProcessor, LogitsProcessorList

from .base_agent import BaseAgent, AgentState


class FinalTokenConstraintProcessor(LogitsProcessor):
    """"""
    
    def __init__(self, tokenizer, valid_options: List[str] = None, max_new_tokens: int = 50):
        """
        Args:
        """
        self.tokenizer = tokenizer
        self.valid_options = valid_options or list('ABCDEFGHIJKLMNOPQRST')
        self.max_new_tokens = max_new_tokens
        
        self.valid_token_ids = set()
        for option in self.valid_options:
            tokens = tokenizer.encode(option, add_special_tokens=False)
            if tokens:
                self.valid_token_ids.add(tokens[0])
            tokens_with_space = tokenizer.encode(f" {option}", add_special_tokens=False)
            if tokens_with_space:
                self.valid_token_ids.add(tokens_with_space[-1])
        
        self.valid_token_ids = list(self.valid_token_ids)
        self.current_length = 0
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """"""
        batch_size = input_ids.shape[0]
        
        generated_length = input_ids.shape[1] - self.current_length if hasattr(self, 'initial_length') else 0
        
        if generated_length >= self.max_new_tokens - 1:
            mask = torch.full_like(scores, float('-inf'))
            for token_id in self.valid_token_ids:
                if token_id < scores.shape[-1]:
                    mask[:, token_id] = 0
            scores = scores + mask
        
        return scores
    
    def reset(self, initial_length: int):
        """"""
        self.initial_length = initial_length


class FirstTokenConstraintProcessor(LogitsProcessor):
    """"""
    
    def __init__(self, tokenizer, valid_options: List[str] = None):
        """
        Args:
        """
        self.tokenizer = tokenizer
        self.valid_options = valid_options or list('ABCDEFGHIJKLMNOPQRST')
        
        self.valid_token_ids = []
        for option in self.valid_options:
            tokens = tokenizer.encode(option, add_special_tokens=False)
            if tokens:
                self.valid_token_ids.append(tokens[0])
            tokens_with_space = tokenizer.encode(f" {option}", add_special_tokens=False)
            if tokens_with_space:
                self.valid_token_ids.append(tokens_with_space[-1])
        
        self.valid_token_ids = list(set(self.valid_token_ids))
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.valid_token_ids:
            if token_id < scores.shape[-1]:
                mask[:, token_id] = 0
        return scores + mask
    
    def reset(self):
        """"""
        pass


class LastTokenConstraintProcessor(LogitsProcessor):
    """"""
    
    def __init__(self, tokenizer, valid_options: List[str] = None, 
                 max_new_tokens: int = 50, boost_factor: float = 2.0):
        """
        Args:
        """
        self.tokenizer = tokenizer
        self.valid_options = valid_options or list('ABCDEFGHIJKLMNOPQRST')
        self.max_new_tokens = max_new_tokens
        self.boost_factor = boost_factor
        self.generated_count = 0
        
        self.valid_token_ids = []
        for option in self.valid_options:
            tokens = tokenizer.encode(option, add_special_tokens=False)
            if tokens:
                self.valid_token_ids.append(tokens[0])
            tokens_with_space = tokenizer.encode(f" {option}", add_special_tokens=False)
            if tokens_with_space:
                self.valid_token_ids.append(tokens_with_space[-1])
        
        self.valid_token_ids = list(set(self.valid_token_ids))
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.generated_count += 1
        
        if self.generated_count >= self.max_new_tokens:
            mask = torch.full_like(scores, float('-inf'))
            for token_id in self.valid_token_ids:
                if token_id < scores.shape[-1]:
                    mask[:, token_id] = 0
            return scores + mask
        
        elif self.generated_count >= self.max_new_tokens - 3:
            for token_id in self.valid_token_ids:
                if token_id < scores.shape[-1]:
                    scores[:, token_id] += self.boost_factor
        
        return scores
    
    def reset(self):
        """"""
        self.generated_count = 0


class RecommendationAgent(BaseAgent):
    """
    - Policy: LLM + LoRA (trainable)
    - Reference: Original LLM weights (frozen, for KL constraint)
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        recommender_config = config.get('models', {}).get('recommender', {})
        self.model_name = recommender_config.get('model_name', 't5-small')
        self.model_type = recommender_config.get('model_type', 'seq2seq')
        self.use_quantization = recommender_config.get('use_quantization', False)
        self.quantization_bits = recommender_config.get('quantization_bits', 8)
        self.max_input_length = recommender_config.get('max_input_length', 1024)
        self.max_output_length = recommender_config.get('max_output_length', 512)
        self.max_new_tokens = recommender_config.get('max_new_tokens', 512)
        self.temperature = recommender_config.get('temperature', 0.8)
        self.device_id = recommender_config.get('device_id', None)  # GPU ID for multi-GPU
        
        self.explanation_max_length = recommender_config.get('explanation_max_tokens', 256)
        self.enable_explanation = recommender_config.get('enable_explanation', True)
        
        self.answer_max_tokens = recommender_config.get('answer_max_tokens', 100)
        self.enable_constrained_decoding = recommender_config.get('enable_constrained_decoding', True)
        self.valid_options = list('ABCDEFGHIJKLMNOPQRST')
        
        # Preference optimization temperature (beta)
        self.preference_beta = recommender_config.get('preference_beta', 0.1)
        
        self.tokenizer = None
        self.model = None
        self.ref_model = None  # Reference model (frozen)
        self.use_peft = False  # Whether using PEFT (LoRA)
        self.generation_config = None
        self.explanation_config = None
        self.option_generation_config = None
        
        self.policy_gradient_enabled = True
        self.baseline_value = 0.0
        self.baseline_alpha = 0.1
        
        self.episode_experiences = []
        self.training_mode = True
        
        self.generation_scores = []
        self.recommendation_qualities = []
        self.diversity_scores = []
        self.explanation_qualities = []
        
    def reset(self):
        """"""
        from collections import deque
        self.current_state = None
        self.episode_history = deque(maxlen=1000)
        self.episode_experiences = deque(maxlen=1000)
        
    def initialize_model(self):
        """"""
        try:
            from ..utils.llm_loader import LLMLoader
            
            hf_token = self.config.get('models', {}).get('recommender', {}).get('hf_token')
            self.model, self.tokenizer, actual_device = LLMLoader.load_model_and_tokenizer(
                model_name=self.model_name,
                model_type=self.model_type,
                use_quantization=self.use_quantization,
                quantization_bits=self.quantization_bits,
                device=self.device,
                hf_token=hf_token,
                device_id=self.device_id
            )
            
            self.device = actual_device
            
            self.generation_config = LLMLoader.create_generation_config(
                model_type=self.model_type,
                max_length=self.max_output_length if self.model_type == 'seq2seq' else None,
                max_new_tokens=self.max_new_tokens if self.model_type == 'causal' else None,
                temperature=self.temperature,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            if self.training_mode and self.model is not None:
                try:
                    from peft import get_peft_model, LoraConfig, TaskType
                    
                    if self.model_type == 'causal':
                        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                    else:
                        target_modules = ["q", "v", "k", "o"]
                    
                    recommender_cfg = self.config.get('models', {}).get('recommender', {})
                    lora_r = recommender_cfg.get('lora_r', 8)
                    lora_alpha = recommender_cfg.get('lora_alpha', 16)
                    lora_dropout = recommender_cfg.get('lora_dropout', 0.05)
                    
                    lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM if self.model_type == 'causal' else TaskType.SEQ_2_SEQ_LM
                    )
                    
                    self.model = get_peft_model(self.model, lora_config)
                    self.model.print_trainable_parameters()
                    
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=1e-4,
                        weight_decay=0.01
                    )
                    
                    self.ref_model = self.model  # Same model, but with disabled adapters for reference
                    self.use_peft = True  # Mark as using PEFT
                    
                    print(f"  ✓ LoRA enabled for Recommender")
                    print(f"  ✓ Optimizer created (AdamW, lr=1e-4)")
                    print(f"  ✓ Preference reference: using disabled LoRA adapters")
                    
                except ImportError:
                    print(f"  ⚠ peft not available, using full model")
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=5e-5,
                        weight_decay=0.01
                    )
                except Exception as e:
                    print(f"  ⚠ Failed to setup LoRA: {e}")
                    self.optimizer = None
            else:
                self.optimizer = None
            
            self.explanation_config = LLMLoader.create_generation_config(
                model_type=self.model_type,
                max_length=self.explanation_max_length if self.model_type == 'seq2seq' else None,
                max_new_tokens=self.explanation_max_length if self.model_type == 'causal' else None,
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            self.answer_max_tokens = 1
            self.option_generation_config = LLMLoader.create_generation_config(
                model_type=self.model_type,
                max_length=1 if self.model_type == 'seq2seq' else None,
                max_new_tokens=1 if self.model_type == 'causal' else None,
                temperature=0.1,
                top_p=1.0,
                top_k=20,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            if self.enable_constrained_decoding:
                self.constraint_processor = FirstTokenConstraintProcessor(
                    tokenizer=self.tokenizer,
                    valid_options=self.valid_options
                )
                print(f"  ✓ Fast constrained decoding enabled (1 token only, direct A-T prediction)")
            else:
                self.constraint_processor = None
            
            print(f"Initialized Recommendation Agent with {self.model_name} ({self.model_type})")
            
        except Exception as e:
            print(f"Warning: Failed to load {self.model_name}: {e}")
            print("Using template-based generation")
            self.tokenizer = None
            self.model = None
            
    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        refined_knowledge = observation.get('refined_knowledge', '')
        original_query = observation.get('original_query', '')
        user_context = observation.get('user_context', {})
        conversation_history = observation.get('conversation_history', [])
        
        
        options = user_context.get('options', {})
        
        if isinstance(options, str):
            options = self._parse_options_str(options)
        
        if options and self.enable_constrained_decoding:
            prediction_result = self.predict_option(
                refined_knowledge=refined_knowledge,
                original_query=original_query,
                options=options,
                user_context=user_context
            )
            
            output = {
                'recommendations': prediction_result['full_output'],
                'confidence': prediction_result['confidence'],
                'diversity_score': 0.0,
                'quality_score': prediction_result['confidence'],
                'explanation': prediction_result['explanation'],
                'explanation_quality': prediction_result['confidence'],
                'predicted_option': prediction_result['predicted_option'],
                'option_distribution': prediction_result.get('option_distribution', {}),
                'metadata': {
                    'refined_knowledge_length': len(refined_knowledge),
                    'explanation_length': len(prediction_result['explanation']),
                    'recommendation_length': len(prediction_result['full_output']),
                    'query': original_query,
                    'mode': 'option_prediction',
                    'constrained_decoding': True,
                    'max_tokens': self.answer_max_tokens
                }
            }
            
            if self.training_mode:
                experience = {
                    'observation': observation,
                    'action': prediction_result['predicted_option'],
                    'explanation': prediction_result['explanation'],
                    'confidence': prediction_result['confidence'],
                    'timestamp': len(self.episode_experiences)
                }
                self.episode_experiences.append(experience)
            
            return output
        
        explanation = self._generate_explanation(
            refined_knowledge,
            original_query,
            user_context
        )
        
        recommendations = self._generate_recommendations(
            refined_knowledge,
            explanation,
            original_query,
            user_context,
            conversation_history
        )
        
        confidence = self._calculate_generation_confidence(recommendations, refined_knowledge)
        diversity_score = self._calculate_diversity_score(recommendations)
        quality_score = self._calculate_quality_score(recommendations, original_query)
        explanation_quality = self._calculate_explanation_quality(explanation, refined_knowledge)
        
        output = {
            'recommendations': recommendations,
            'confidence': confidence,
            'diversity_score': diversity_score,
            'quality_score': quality_score,
            'explanation': explanation,
            'explanation_quality': explanation_quality,
            'predicted_option': None,
            'metadata': {
                'refined_knowledge_length': len(refined_knowledge),
                'explanation_length': len(explanation),
                'recommendation_length': len(recommendations),
                'query': original_query,
                'mode': 'free_generation',
                'explanation_enabled': self.enable_explanation
            }
        }
        
        if self.training_mode:
            experience = {
                'observation': observation,
                'action': recommendations,
                'explanation': explanation,
                'confidence': confidence,
                'diversity_score': diversity_score,
                'quality_score': quality_score,
                'explanation_quality': explanation_quality,
                'timestamp': len(self.episode_experiences)
            }
            self.episode_experiences.append(experience)
        
        return output
    
    def _generate_explanation(self,
                           refined_knowledge: str,
                           original_query: str,
                           user_context: Dict[str, Any]) -> str:
        """
        Args:
        Returns:
        """
        if not self.enable_explanation:
            return ""
        
        if self.model is None:
            return self._template_based_explanation(refined_knowledge, original_query, user_context)
        
        explanation_prompt = self._construct_explanation_prompt(
            refined_knowledge,
            original_query,
            user_context
        )
        
        try:
            inputs = self.tokenizer(
                explanation_prompt,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                gen_kwargs = {
                    'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'temperature': 0.7,
                    'do_sample': True,
                    'top_p': 0.9,
                    'top_k': 50,
                }
                
                if self.model_type == 'seq2seq':
                    gen_kwargs['max_length'] = self.explanation_max_length
                else:  # causal
                    gen_kwargs['max_new_tokens'] = min(self.explanation_max_length, 256)
                
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            explanation = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return explanation
            
        except Exception as e:
            print(f"Error in explanation generation: {e}")
            return self._template_based_explanation(refined_knowledge, original_query, user_context)
    
    def _construct_explanation_prompt(self,
                                   refined_knowledge: str,
                                   original_query: str,
                                   user_context: Dict[str, Any]) -> str:
        """"""
        user_type = user_context.get('user_type', 'general')
        preferences = user_context.get('preferences', 'general preferences')
        
        prompt = f"""Task: Analyze the knowledge and reason about what recommendations would be most suitable.

User Query: {original_query}
User Type: {user_type}
User Preferences: {preferences}

Retrieved Knowledge:
{refined_knowledge[:800]}

Instructions:
1. Analyze the user's query to understand their intent
2. Examine the knowledge to identify relevant movies and patterns
3. Consider which movies best match the user's preferences
4. Consider diversity and variety in your analysis
5. Explain your analysis process step-by-step

Analysis:"""
        
        return prompt
    
    def _template_based_explanation(self,
                                  refined_knowledge: str,
                                  original_query: str,
                                  user_context: Dict[str, Any]) -> str:
        """"""
        explanation_parts = []
        
        explanation_parts.append("Let me analyze your request:")
        explanation_parts.append(f"1. Query Analysis: You're asking about '{original_query[:100]}'")
        
        knowledge_preview = refined_knowledge[:200] + "..." if len(refined_knowledge) > 200 else refined_knowledge
        explanation_parts.append(f"2. Available Knowledge: Found relevant information including {knowledge_preview}")
        
        user_prefs = user_context.get('preferences', 'your preferences')
        explanation_parts.append(f"3. Matching Analysis: Based on {user_prefs}, I'll focus on movies that best align with your interests")
        
        explanation_parts.append("4. Recommendation Strategy: I'll provide diverse options that match the themes and characteristics you're looking for")
        
        return "\n".join(explanation_parts)
    
    
    def compute_log_prob_for_option(self, prompt: str, option: str, use_ref: bool = False) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        if self.model is None:
            return torch.tensor(0.0, device=self.device)
        
        try:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_input_length,
                padding=True
            ).to(self.device)
            
            option_tokens = self.tokenizer.encode(option, add_special_tokens=False)
            if not option_tokens:
                return torch.tensor(0.0, device=self.device)
            option_token_id = option_tokens[0]
            
            with torch.set_grad_enabled(not use_ref):
                if use_ref and hasattr(self, 'use_peft') and self.use_peft:
                    with self.model.disable_adapter():
                        outputs = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask']
                        )
                else:
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                
                last_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                
                log_probs = F.log_softmax(last_logits, dim=-1)
                
                option_log_prob = log_probs[:, option_token_id]
                
            return option_log_prob.squeeze()
            
        except Exception as e:
            print(f"  Warning: compute_log_prob_for_option failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def compute_log_prob_for_options_batch(self, prompts: List[str], options: List[str], use_ref: bool = False) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        if self.model is None or len(prompts) == 0:
            return torch.zeros(len(prompts), device=self.device)
        
        batch_size = len(prompts)
        
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_input_length,
                padding=True
            ).to(self.device)
            
            option_token_ids = []
            for opt in options:
                opt_tokens = self.tokenizer.encode(opt, add_special_tokens=False)
                option_token_ids.append(opt_tokens[0] if opt_tokens else 0)
            option_token_ids = torch.tensor(option_token_ids, device=self.device)
            
            with torch.set_grad_enabled(not use_ref):
                if use_ref and hasattr(self, 'use_peft') and self.use_peft:
                    with self.model.disable_adapter():
                        outputs = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask']
                        )
                else:
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                
                seq_lengths = inputs['attention_mask'].sum(dim=1) - 1  # [batch]
                
                last_logits = outputs.logits[
                    torch.arange(batch_size, device=self.device),
                    seq_lengths
                ]  # [batch, vocab_size]
                
                log_probs_all = F.log_softmax(last_logits, dim=-1)  # [batch, vocab_size]
                
                log_probs = log_probs_all.gather(dim=-1, index=option_token_ids.unsqueeze(-1)).view(-1)
                
            return log_probs
            
        except Exception as e:
            print(f"  Warning: compute_log_prob_for_options_batch failed: {e}")
            return torch.zeros(batch_size, device=self.device, requires_grad=True)
    
    def compute_preference_loss_batch(self, prompts: List[str], y_wins: List[str], y_loses: List[str]) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        if self.model is None or len(prompts) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if not hasattr(self, 'use_peft') or not self.use_peft:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            log_pi_win = self.compute_log_prob_for_options_batch(prompts, y_wins, use_ref=False)
            log_pi_lose = self.compute_log_prob_for_options_batch(prompts, y_loses, use_ref=False)
            
            with torch.no_grad():
                log_ref_win = self.compute_log_prob_for_options_batch(prompts, y_wins, use_ref=True)
                log_ref_lose = self.compute_log_prob_for_options_batch(prompts, y_loses, use_ref=True)
            
            # Preference margin
            margin = (log_pi_win - log_ref_win) - (log_pi_lose - log_ref_lose)
            
            # Preference loss
            losses = -F.logsigmoid(self.preference_beta * margin)
            
            valid_mask = ~(torch.isnan(losses) | torch.isinf(losses))
            if valid_mask.sum() > 0:
                return losses[valid_mask].mean()
            else:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        except Exception as e:
            print(f"  Warning: compute_preference_loss_batch failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def compute_preference_loss(self, prompt: str, y_win: str, y_lose: str) -> torch.Tensor:
        """
        L = -log σ(β * [log π(y_win)/π_ref(y_win) - log π(y_lose)/π_ref(y_lose)])
        Args:
        Returns:
            loss: Preference loss
        """
        if self.model is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if not hasattr(self, 'use_peft') or not self.use_peft:
            print("  Warning: Preference optimization requires PEFT model with disable_adapter support")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            log_pi_win = self.compute_log_prob_for_option(prompt, y_win, use_ref=False)
            log_pi_lose = self.compute_log_prob_for_option(prompt, y_lose, use_ref=False)
            
            with torch.no_grad():
                log_ref_win = self.compute_log_prob_for_option(prompt, y_win, use_ref=True)
                log_ref_lose = self.compute_log_prob_for_option(prompt, y_lose, use_ref=True)
            
            # Preference margin
            margin = (log_pi_win - log_ref_win) - (log_pi_lose - log_ref_lose)
            
            # Preference loss
            loss = -F.logsigmoid(self.preference_beta * margin)
            
            return loss
            
        except Exception as e:
            print(f"  Warning: compute_preference_loss failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def get_all_option_log_probs(self, prompt: str, options: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Args:
        Returns:
        """
        log_probs = {}
        for option_letter in options.keys():
            log_probs[option_letter] = self.compute_log_prob_for_option(prompt, option_letter, use_ref=False)
        return log_probs
    
    
    def predict_option(self, 
                      refined_knowledge: str,
                      original_query: str,
                      options: Dict[str, str],
                      user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Args:
        Returns:
        """
        if isinstance(options, str):
            options = self._parse_options_str(options)
        
        if self.model is None:
            return self._template_based_option_prediction(refined_knowledge, original_query, options)
        
        prompt = self._construct_option_prompt(refined_knowledge, original_query, options, user_context)
        
        try:
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # ═══════════════════════════════════════════════════════
            # ═══════════════════════════════════════════════════════
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if self.model_type == 'causal':
                    seq_len = inputs['attention_mask'].sum(dim=1) - 1  # [batch_size]
                    last_logits = outputs.logits[0, seq_len.item(), :]  # [vocab_size]
                else:
                    gen_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    last_logits = gen_outputs.scores[0][0]  # [vocab_size]
            
            option_distribution = {}
            option_logits = {}
            
            for letter in options.keys():
                tokens = self.tokenizer.encode(letter, add_special_tokens=False)
                if tokens:
                    token_id = tokens[0]
                    if token_id < last_logits.shape[-1]:
                        option_logits[letter] = last_logits[token_id].item()
            
            if not option_logits:
                return self._template_based_option_prediction(refined_knowledge, original_query, options)
            
            logit_values = torch.tensor([option_logits[l] for l in sorted(option_logits.keys())])
            logit_values = logit_values - logit_values.max()
            probs = F.softmax(logit_values, dim=-1)
            
            for i, letter in enumerate(sorted(option_logits.keys())):
                option_distribution[letter] = probs[i].item()
            
            predicted_option = max(option_distribution.items(), key=lambda x: x[1])[0]
            confidence = option_distribution[predicted_option]
            
            return {
                'predicted_option': predicted_option,
                'explanation': '',
                'confidence': confidence,
                'full_output': predicted_option,
                'generated_only': predicted_option,
                'option_distribution': option_distribution
            }
            
        except Exception as e:
            import traceback
            print(f"  Warning: Fast option prediction failed: {e}")
            traceback.print_exc()
            return {
                'predicted_option': 'A',
                'explanation': '',
                'confidence': 0.05,
                'full_output': '',
                'generated_only': '',
                'option_distribution': {opt: 0.05 for opt in options}
            }
    
    def _construct_option_prompt(self,
                                refined_knowledge: str,
                                original_query: str,
                                options: Dict[str, str],
                                user_context: Optional[Dict[str, Any]] = None) -> str:
        """"""
        options_text = " ".join([f"{letter}:{movie[:30]}" for letter, movie in sorted(options.items())])
        
        if "Watching history:" in original_query:
            history_start = original_query.find("Watching history:")
            history_end = original_query.find("Options:", history_start) if "Options:" in original_query else len(original_query)
            watching_history = original_query[history_start:history_end].strip()
        else:
            watching_history = original_query[:500]
        
        if refined_knowledge and refined_knowledge.strip():
            knowledge_part = f"Knowledge: {refined_knowledge[:500]}\n"
        else:
            knowledge_part = ""
        
        prompt = f"""Based on the user's watching history, select the best movie from options A-T.

{watching_history}

{knowledge_part}Options: {options_text}

Answer with ONE letter (A-T): Analysis and Answer:"""
        
        return prompt
    
    def _extract_option_from_generation(self, 
                                        generated_text: str, 
                                        options: Dict[str, str]) -> str:
        """"""
        import re
        
        text = generated_text.strip()
        if text and text[-1].upper() in options:
            return text[-1].upper()
        
        patterns = [
            r'(?:Answer|Option|Choice|Select|Pick|Recommend)[:\s]*([A-T])\b',
            r'\b([A-T])\s*(?:is the best|would be|is my)',
            r'(?:I choose|I select|I recommend)[:\s]*([A-T])\b',
            r'\b([A-T])\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in options:
                    return letter
        
        option_mentions = {}
        text_upper = text.upper()
        for letter, movie in options.items():
            letter_count = text_upper.count(f" {letter} ") + text_upper.count(f"({letter})") + text_upper.count(f"{letter}:")
            movie_count = text_upper.count(movie.upper()[:20])
            option_mentions[letter] = letter_count * 2 + movie_count
        
        if option_mentions:
            best = max(option_mentions, key=option_mentions.get)
            if option_mentions[best] > 0:
                return best
        
        return list(options.keys())[0] if options else 'A'
    
    def _get_best_option_from_scores(self, 
                                     scores: torch.Tensor, 
                                     options: Dict[str, str]) -> str:
        """
        Args:
        Returns:
        """
        try:
            option_logits = {}
            for letter in options.keys():
                tokens = self.tokenizer.encode(letter, add_special_tokens=False)
                if tokens:
                    token_id = tokens[0]
                    if token_id < scores.shape[-1]:
                        option_logits[letter] = scores[0, token_id].item()
            
            if option_logits:
                return max(option_logits, key=option_logits.get)
            
        except Exception as e:
            pass
        
        return list(options.keys())[0] if options else 'A'
    
    def _compute_option_distribution(self, 
                                     scores: tuple,
                                     options: Dict[str, str]) -> Dict[str, float]:
        """
        Args:
        Returns:
        """
        uniform_dist = {letter: 1.0 / len(options) for letter in options}
        
        try:
            if not scores or len(scores) == 0:
                return uniform_dist
            
            last_scores = scores[-1]  # shape: (batch_size, vocab_size)
            last_logits = last_scores[0]
            
            option_token_ids = {}
            for letter in options.keys():
                tokens = self.tokenizer.encode(letter, add_special_tokens=False)
                if tokens:
                    option_token_ids[letter] = tokens[0]
            
            if not option_token_ids:
                return uniform_dist
            
            logit_values = []
            for letter in sorted(option_token_ids.keys()):
                token_id = option_token_ids[letter]
                logit = last_logits[token_id].item()
                if not torch.isfinite(torch.tensor(logit)):
                    logit = 0.0
                logit_values.append(logit)
            
            option_logits = torch.tensor(logit_values, dtype=torch.float32)
            
            option_logits = option_logits - option_logits.max()
            
            option_probs = F.softmax(option_logits, dim=-1)
            
            if torch.isnan(option_probs).any():
                return uniform_dist
            
            distribution = {}
            for i, letter in enumerate(sorted(option_token_ids.keys())):
                prob = float(option_probs[i].item())
                if not (0.0 <= prob <= 1.0) or not torch.isfinite(torch.tensor(prob)):
                    prob = uniform_dist[letter]
                distribution[letter] = prob
            
            return distribution
            
        except Exception as e:
            print(f"Error computing option distribution: {e}")
            return uniform_dist
    
    def _calculate_option_confidence(self, 
                                    outputs: torch.Tensor, 
                                    inputs: Dict[str, torch.Tensor],
                                    predicted_option: str) -> float:
        """"""
        try:
            if self.model is None:
                return 0.5
            
            with torch.no_grad():
                model_outputs = self.model(
                    input_ids=outputs,
                    attention_mask=torch.ones_like(outputs)
                )
                last_logits = model_outputs.logits[0, -2, :]
                
                probs = F.softmax(last_logits, dim=-1)
                
                option_tokens = self.tokenizer.encode(predicted_option, add_special_tokens=False)
                if option_tokens:
                    confidence = probs[option_tokens[0]].item()
                    return float(confidence)
            
            return 0.5
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
    
    def _template_based_option_prediction(self,
                                         refined_knowledge: str,
                                         original_query: str,
                                         options: Dict[str, str]) -> Dict[str, Any]:
        """"""
        query_lower = original_query.lower()
        knowledge_lower = refined_knowledge.lower()
        
        best_option = None
        best_score = -1
        
        for letter, movie in options.items():
            movie_words = movie.lower().split()
            score = sum(1 for word in movie_words if word in query_lower or word in knowledge_lower)
            if score > best_score:
                best_score = score
                best_option = letter
        
        if best_option is None:
            best_option = 'A'
        
        scores = {}
        for letter in options:
            movie_words = options[letter].lower().split()
            scores[letter] = max(1, sum(1 for word in movie_words 
                                       if word in query_lower or word in knowledge_lower))
        total_score = sum(scores.values())
        option_distribution = {letter: score / total_score for letter, score in scores.items()}
        
        return {
            'predicted_option': best_option,
            'explanation': f"Based on keyword matching, option {best_option} ({options.get(best_option, '')}) seems most relevant.",
            'confidence': 0.3,
            'full_output': f"Template-based prediction: {best_option}",
            'option_distribution': option_distribution
        }
    
    def _parse_options_str(self, options_str: str) -> Dict[str, str]:
        """
        Parse "A: Movie1, B: Movie2, ..." (movie names may contain commas, e.g. "Great Race, The").
        Splits only at option boundaries: ", " followed by a letter A-T and ": ".
        """
        import re
        options = {}
        if not options_str:
            return options
        normalized = options_str.replace('\n', ', ')
        parts = re.split(r', (?=[A-T]: )', normalized, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if ': ' in part:
                letter, movie = part.split(': ', 1)
                letter = letter.strip().upper()
                movie = movie.strip()
                if len(letter) == 1 and letter in 'ABCDEFGHIJKLMNOPQRST':
                    options[letter] = movie
        return options
    
    def _generate_recommendations(self,
                                 refined_knowledge: str,
                                 explanation: str,
                                 original_query: str,
                                 user_context: Dict[str, Any],
                                 conversation_history: List[Dict[str, Any]]) -> str:
        """
        Args:
        Returns:
        """
        if self.model is None:
            return self._template_based_generation(
                refined_knowledge, explanation, original_query, user_context
            )
        
        recommendation_prompt = self._construct_recommendation_prompt(
            refined_knowledge,
            explanation,
            original_query,
            user_context,
            conversation_history
        )
        
        try:
            inputs = self.tokenizer(
                recommendation_prompt,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                gen_kwargs = {
                    'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'temperature': self.temperature,
                    'do_sample': True,
                    'top_p': 0.9,
                    'top_k': 50,
                }
                
                if self.model_type == 'seq2seq':
                    gen_kwargs['max_length'] = self.max_output_length
                else:  # causal
                    gen_kwargs['max_new_tokens'] = min(self.max_new_tokens, 512)
                
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            recommendations = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return recommendations
            
        except Exception as e:
            print(f"Error in recommendation generation: {e}")
            return self._template_based_generation(
                refined_knowledge, explanation, original_query, user_context
            )
    
    def _construct_recommendation_prompt(self,
                                        refined_knowledge: str,
                                        explanation: str,
                                        original_query: str,
                                        user_context: Dict[str, Any],
                                        conversation_history: List[Dict[str, Any]]) -> str:
        """"""
        user_type = user_context.get('user_type', 'general')
        preferences = user_context.get('preferences', 'general preferences')
        
        domain = self.config.get('domain', 'movie')
        if domain == 'music':
            rec_type = 'music'
            item_type = 'artist'
        else:
            rec_type = 'movie'
            item_type = 'movie'
        
        history_summary = ""
        if conversation_history:
            recent_history = conversation_history[-2:]
            history_items = []
            for item in recent_history:
                user_msg = item.get('user', '')
                bot_msg = item.get('bot', '')
                if user_msg and bot_msg:
                    history_items.append(f"User: {user_msg}\nBot: {bot_msg}")
            if history_items:
                history_summary = "\n".join(history_items)
        
        prompt = f"""Task: Generate personalized {rec_type} recommendations based on analysis and knowledge.

User Query: {original_query}
User Type: {user_type}
User Preferences: {preferences}

{f'Conversation History:{chr(10)}{history_summary}{chr(10)}' if history_summary else ''}
Knowledge from Knowledge Graph:
{refined_knowledge[:600]}

Analysis:
{explanation}

Instructions:
1. Use the analysis to guide your recommendations
2. Generate 3-5 specific {item_type} recommendations
3. Explain WHY each {item_type} is recommended based on the analysis
4. Tailor to user preferences
5. Ensure diversity in recommendations
6. Be specific and actionable

Personalized {rec_type.capitalize()} Recommendations:"""
        
        return prompt
    
    def _template_based_generation(self,
                                   refined_knowledge: str,
                                   explanation: str,
                                   original_query: str,
                                   user_context: Dict[str, Any]) -> str:
        """"""
        rec_parts = []
        
        rec_parts.append(f"Based on your request for '{original_query}', here are my personalized recommendations:")
        rec_parts.append("")
        
        movies_found = self._extract_movies_from_knowledge(refined_knowledge)
        
        if movies_found:
            rec_parts.append("Recommended Movies:")
            rec_parts.append("")
            
            for i, movie_info in enumerate(movies_found[:5], 1):
                movie_name = movie_info['name']
                facts = movie_info.get('facts', [])
                
                rec_parts.append(f"{i}. **{movie_name}**")
                
                if facts:
                    rec_parts.append(f"   Why: {facts[0]}")
                else:
                    rec_parts.append(f"   This movie matches your interest in {user_context.get('preferences', 'quality films')}")
                
                rec_parts.append("")
        else:
            rec_parts.append("1. Based on the knowledge graph, I recommend exploring movies with similar themes")
            rec_parts.append("2. Consider films from the same directors or production companies")
            rec_parts.append("3. Look for movies in related genres")
        
        rec_parts.append("")
        rec_parts.append(f"These recommendations are tailored for {user_context.get('user_type', 'movie')} lovers.")
        rec_parts.append("Let me know if you'd like more details about any of these options!")
        
        return "\n".join(rec_parts)
    
    def _extract_movies_from_knowledge(self, knowledge: str) -> List[Dict[str, Any]]:
        """"""
        movies = []
        lines = knowledge.split('\n')
        
        current_movie = None
        current_facts = []
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')) or line.strip().startswith('**'):
                if current_movie:
                    movies.append({
                        'name': current_movie,
                        'facts': current_facts
                    })
                
                movie_name = line.strip()
                for prefix in ['1.', '2.', '3.', '4.', '5.', '**']:
                    movie_name = movie_name.replace(prefix, '')
                movie_name = movie_name.replace('**', '').replace(':', '').strip()
                
                if '(Movie)' in movie_name:
                    movie_name = movie_name.split('(Movie)')[0].strip()
                if '(Resource)' in movie_name:
                    movie_name = movie_name.split('(Resource)')[0].strip()
                if '[score:' in movie_name:
                    movie_name = movie_name.split('[score:')[0].strip()
                
                current_movie = movie_name
                current_facts = []
            
            elif line.strip().startswith(('-', '•', '  -')) and current_movie:
                fact = line.strip().lstrip('-•').strip()
                current_facts.append(fact)
        
        if current_movie:
            movies.append({
                'name': current_movie,
                'facts': current_facts
            })
        
        return movies
    
    def _calculate_generation_confidence(self, recommendations: str, refined_knowledge: str) -> float:
        """"""
        if not recommendations or not refined_knowledge:
            return 0.0
        
        confidence_factors = []
        
        rec_length = len(recommendations)
        length_score = min(1.0, rec_length / 300.0) * 0.8 + 0.2
        confidence_factors.append(length_score)
        
        rec_words = set(recommendations.lower().split())
        knowledge_words = set(refined_knowledge.lower().split())
        overlap_ratio = len(rec_words & knowledge_words) / max(len(rec_words), 1)
        confidence_factors.append(min(1.0, overlap_ratio * 2))
        
        has_structure = any(char in recommendations for char in ['1.', '2.', '•', '-'])
        confidence_factors.append(0.8 if has_structure else 0.4)
        
        final_confidence = np.mean(confidence_factors)
        return float(np.clip(final_confidence, 0.0, 1.0))
    
    def _calculate_diversity_score(self, recommendations: str) -> float:
        """"""
        if not recommendations:
            return 0.0
        
        diversity_factors = []
        
        num_items = sum(1 for line in recommendations.split('\n') if any(f'{i}.' in line for i in range(1, 10)))
        diversity_factors.append(min(1.0, num_items / 3.0))
        
        words = recommendations.lower().split()
        unique_words = set(words)
        vocab_diversity = len(unique_words) / max(len(words), 1)
        diversity_factors.append(vocab_diversity)
        
        themes = ['director', 'actor', 'genre', 'producer', 'story', 'style']
        theme_mentions = sum(1 for theme in themes if theme in recommendations.lower())
        topic_diversity = min(1.0, theme_mentions / 3.0)
        diversity_factors.append(topic_diversity)
        
        return float(np.mean(diversity_factors))
    
    def _calculate_quality_score(self, recommendations: str, query: str) -> float:
        """"""
        if not recommendations or not query:
            return 0.0
        
        quality_factors = []
        
        query_words = set(query.lower().split())
        rec_words = set(recommendations.lower().split())
        relevance = len(query_words & rec_words) / max(len(query_words), 1)
        quality_factors.append(relevance)
        
        has_explanations = any(word in recommendations.lower() 
                             for word in ['why', 'because', 'reason', 'based on', 'features', 'known for'])
        quality_factors.append(0.8 if has_explanations else 0.4)
        
        has_specific_recs = any(char in recommendations for char in ['1.', '2.', '3.'])
        quality_factors.append(0.8 if has_specific_recs else 0.3)
        
        return float(np.mean(quality_factors))
    
    def _calculate_explanation_quality(self, explanation: str, knowledge: str) -> float:
        """"""
        if not explanation:
            return 0.0
        
        quality_factors = []
        
        length_score = min(1.0, len(explanation) / 150.0) * 0.8 + 0.2
        quality_factors.append(length_score)
        
        explanation_words = set(explanation.lower().split())
        knowledge_words = set(knowledge.lower().split())
        overlap = len(explanation_words & knowledge_words) / max(len(explanation_words), 1)
        quality_factors.append(min(1.0, overlap * 2))
        
        analysis_keywords = ['analyze', 'based on', 'consider', 'match', 'suitable', 'recommend']
        has_analysis = sum(1 for kw in analysis_keywords if kw in explanation.lower())
        quality_factors.append(min(1.0, has_analysis / 3.0))
        
        return float(np.mean(quality_factors))
    
    def update(self, experience: Dict[str, Any]):
        """"""
        if not self.training_mode or self.model is None:
            return
            
        self.episode_history.append(experience)
        
        reward = experience.get('reward', 0.0)
        
        self.baseline_value = self.baseline_alpha * reward + (1 - self.baseline_alpha) * self.baseline_value
        
        if 'recommendations' in experience:
            self.generation_scores.append(reward)
            
        if 'quality_score' in experience:
            self.recommendation_qualities.append(experience['quality_score'])
            
        if 'diversity_score' in experience:
            self.diversity_scores.append(experience['diversity_score'])
        
        if 'explanation_quality' in experience:
            self.explanation_qualities.append(experience['explanation_quality'])
    
    def compute_policy_gradient_loss(self, experiences: List[Dict[str, Any]], advantages: torch.Tensor) -> torch.Tensor:
        """"""
        if self.model is None:
            return torch.tensor(0.0, device=self.device)
        
        log_probs = []
        
        for exp in experiences:
            observation = exp['observation']
            action = exp['action']  # recommendations
            
            refined_knowledge = observation.get('refined_knowledge', '')
            explanation = exp.get('explanation', '')
            original_query = observation.get('original_query', '')
            user_context = observation.get('user_context', {})
            conversation_history = observation.get('conversation_history', [])
            
            input_text = self._construct_recommendation_prompt(
                refined_knowledge, explanation, original_query, user_context, conversation_history
            )
            
            try:
                inputs = self.tokenizer(
                    input_text,
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                targets = self.tokenizer(
                    action,
                    max_length=self.max_output_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(**inputs, labels=targets['input_ids'])
                log_prob = -outputs.loss
                log_probs.append(log_prob)
                
            except Exception as e:
                print(f"Error computing log prob: {e}")
                log_probs.append(torch.tensor(0.0, device=self.device))
        
        if not log_probs:
            return torch.tensor(0.0, device=self.device)
        
        log_probs = torch.stack(log_probs)
        
        policy_loss = -(log_probs * advantages).mean()
        
        return policy_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """"""
        base_metrics = super().get_metrics()
        
        generator_metrics = {
            'model_name': self.model_name,
            'training_mode': self.training_mode,
            'baseline_value': self.baseline_value,
            'episode_experiences': len(self.episode_experiences),
            'explanation_enabled': self.enable_explanation,
        }
        
        if self.generation_scores:
            generator_metrics.update({
                'avg_generation_score': np.mean(self.generation_scores[-100:]),
                'total_generations': len(self.generation_scores),
            })
            
        if self.recommendation_qualities:
            generator_metrics.update({
                'avg_quality_score': np.mean(self.recommendation_qualities[-100:]),
                'std_quality_score': np.std(self.recommendation_qualities[-100:]),
            })
            
        if self.diversity_scores:
            generator_metrics.update({
                'avg_diversity_score': np.mean(self.diversity_scores[-100:]),
                'std_diversity_score': np.std(self.diversity_scores[-100:]),
            })
        
        if self.explanation_qualities:
            generator_metrics.update({
                'avg_explanation_quality': np.mean(self.explanation_qualities[-100:]),
                'std_explanation_quality': np.std(self.explanation_qualities[-100:]),
            })
        
        base_metrics.update(generator_metrics)
        return base_metrics
    
    def save_checkpoint(self, path: str):
        """"""
        if self.model is None:
            return
        
        checkpoint = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            
            'generation_scores': self.generation_scores[-100:],
            'recommendation_qualities': self.recommendation_qualities[-100:],
            'baseline_value': self.baseline_value,
            
            'has_optimizer': self.optimizer is not None
        }
        
        if self.optimizer is not None:
            try:
                model_path = path.replace('.pt', '_lora')
                self.model.save_pretrained(model_path)
                checkpoint['lora_path'] = model_path
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                print(f"  ✓ Recommender LoRA saved to {model_path}")
            except:
                checkpoint['model_state_dict'] = self.model.state_dict()
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, map_location=None):
        """"""
        import os
        if not os.path.exists(path):
            print(f"  ⚠ No Recommender checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, weights_only=False, map_location=map_location)
        
        self.generation_scores = checkpoint.get('generation_scores', [])
        self.recommendation_qualities = checkpoint.get('recommendation_qualities', [])
        self.baseline_value = checkpoint.get('baseline_value', 0.0)
        
        if checkpoint.get('has_optimizer') and self.optimizer is not None:
            try:
                if 'lora_path' in checkpoint and os.path.exists(checkpoint['lora_path']):
                    lora_path = checkpoint['lora_path']
                    from peft import set_peft_model_state_dict
                    from safetensors.torch import load_file
                    
                    adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_path):
                        adapter_weights = load_file(adapter_path)
                        set_peft_model_state_dict(self.model, adapter_weights)
                        print(f"  ✓ Recommender LoRA loaded from {lora_path}")
                    else:
                        adapter_path = os.path.join(lora_path, "adapter_model.bin")
                        if os.path.exists(adapter_path):
                            adapter_weights = torch.load(adapter_path, weights_only=False, map_location=map_location)
                            set_peft_model_state_dict(self.model, adapter_weights)
                            print(f"  ✓ Recommender LoRA loaded from {lora_path}")
                        else:
                            print(f"  ⚠ LoRA adapter file not found in {lora_path}")
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  ✓ Recommender model state loaded")
                
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                print(f"  ✓ Recommender checkpoint loaded from {path}")
            except Exception as e:
                print(f"  ⚠ Failed to load Recommender state: {e}")
                import traceback
                traceback.print_exc()
    
    def set_training_mode(self, training: bool):
        """"""
        self.training_mode = training
        if self.model is not None:
            self.model.train() if training else self.model.eval()
