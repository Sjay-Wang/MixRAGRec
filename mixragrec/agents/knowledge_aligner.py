"""
Knowledge Preference Alignment Agent for MixRAGRec
Bridges the gap between structured KG knowledge and natural language for LLM consumption.

Training: Policy gradient optimization with Value Head for advantage estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from .base_agent import BaseAgent, AgentState


class ValueHead(nn.Module):
    """
    Value Head for policy optimization training
    Estimates state value from LLM hidden states
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: LLM hidden states [batch, seq_len, hidden_size]
        Returns:
            value: Estimated state value [batch, 1]
        """
        # Use last token's hidden state
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        
        x = self.dense(hidden_states)
        x = torch.tanh(x)
        x = self.dropout(x)
        value = self.out_proj(x)
        return value


class KnowledgeAlignmentAgent(BaseAgent):
    """
    - Policy: LLM + LoRA
    - Value: Value Head on LLM hidden states
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        aligner_config = config.get('models', {}).get('knowledge_aligner', {})
        self.model_name = aligner_config.get('model_name', 't5-small')
        self.model_type = aligner_config.get('model_type', 'seq2seq')
        self.use_quantization = aligner_config.get('use_quantization', False)
        self.quantization_bits = aligner_config.get('quantization_bits', 8)
        self.max_input_length = aligner_config.get('max_input_length', 512)
        self.max_output_length = aligner_config.get('max_output_length', 256)
        self.max_new_tokens = aligner_config.get('max_new_tokens', 256)
        self.temperature = aligner_config.get('temperature', 0.7)
        self.device_id = aligner_config.get('device_id', None)  # GPU ID for multi-GPU
        
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        self.value_head = None
        self.value_optimizer = None
        
        self.alignment_templates = {
            'triple': self._create_triple_alignment_template,
            'subgraph': self._create_subgraph_alignment_template,
            'connected_graph': self._create_connected_graph_alignment_template,
            'direct': self._create_direct_alignment_template
        }
        
        self.policy_gradient_enabled = True
        self.baseline_value = 0.0
        self.baseline_alpha = 0.1
        
        self.episode_experiences = []
        self.training_mode = True
        
        self.alignment_scores = []
        self.generation_lengths = []
        
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
            
            hf_token = self.config.get('models', {}).get('knowledge_aligner', {}).get('hf_token')
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
            
            if self.training_mode and self.model is not None:
                try:
                    from peft import get_peft_model, LoraConfig, TaskType
                    
                    if self.model_type == 'causal':
                        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                    else:
                        target_modules = ["q", "v", "k", "o"]
                    
                    aligner_cfg = self.config.get('models', {}).get('knowledge_aligner', {})
                    lora_r = aligner_cfg.get('lora_r', 8)
                    lora_alpha = aligner_cfg.get('lora_alpha', 16)
                    lora_dropout = aligner_cfg.get('lora_dropout', 0.05)
                    
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
                    
                    if hasattr(self.model, 'config'):
                        hidden_size = getattr(self.model.config, 'hidden_size', 768)
                    else:
                        hidden_size = 768
                    
                    self.value_head = ValueHead(hidden_size, dropout=0.1)
                    self.value_head = self.value_head.to(self.device)
                    
                    if self.use_quantization:
                        self.value_head = self.value_head.half()
                        print(f"  ✓ Value Head converted to Half precision (matching quantized model)")
                    
                    self.value_optimizer = torch.optim.AdamW(
                        self.value_head.parameters(),
                        lr=1e-4,
                        weight_decay=0.01
                    )
                    
                    print(f"  ✓ LoRA enabled for Knowledge Aligner")
                    print(f"  ✓ Policy Optimizer created (AdamW, lr=1e-4)")
                    print(f"  ✓ Value Head initialized (hidden_size={hidden_size})")
                    print(f"  ✓ Value Optimizer created (AdamW, lr=1e-4)")
                    
                except ImportError:
                    print(f"  ⚠ peft not available, using full model (will be slow!)")
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
            
            self.generation_config = LLMLoader.create_generation_config(
                model_type=self.model_type,
                max_length=self.max_output_length if self.model_type == 'seq2seq' else None,
                max_new_tokens=self.max_new_tokens if self.model_type == 'causal' else None,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            print(f"Initialized Knowledge Alignment Agent with {self.model_name} ({self.model_type})")
            
        except Exception as e:
            print(f"Warning: Failed to load {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
            print("Using template-based alignment")
            self.tokenizer = None
            self.model = None
            self.optimizer = None
    
    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        retrieval_results = observation.get('retrieval_results', [])
        watching_history = observation.get('watching_history', '')
        user_context = observation.get('user_context', {})
        
        if not retrieval_results:
            return {
                'refined_knowledge': '',
                'confidence': 0.0,
                'metadata': {'status': 'no_retrieval_results'}
            }
        
        retrieval_type = self._identify_retrieval_type(retrieval_results)
        
        aligned_knowledge = self._align_knowledge(
            retrieval_results, 
            watching_history,
            user_context,
            retrieval_type
        )
        
        confidence = self._calculate_confidence(aligned_knowledge, retrieval_results)
        
        output = {
            'refined_knowledge': aligned_knowledge,
            'confidence': confidence,
            'metadata': {
                'num_source_docs': len(retrieval_results),
                'alignment_length': len(aligned_knowledge),
                'retrieval_type': retrieval_type
            }
        }
        
        if self.training_mode:
            experience = {
                'observation': observation,
                'action': aligned_knowledge,
                'confidence': confidence,
                'timestamp': len(self.episode_experiences)
            }
            self.episode_experiences.append(experience)
        
        return output
    
    def get_value(self, state_text: str) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        if self.model is None or self.value_head is None:
            return torch.tensor([0.0], device=self.device)
        
        try:
            inputs = self.tokenizer(
                state_text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_input_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                if self.model_type == 'causal':
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        output_hidden_states=True
                    )
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        hidden_states = outputs.hidden_states[-1]
                    else:
                        return torch.tensor([0.0], device=self.device)
                else:
                    base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
                    if hasattr(base_model, 'encoder'):
                        outputs = base_model.encoder(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            output_hidden_states=True
                        )
                        hidden_states = outputs.hidden_states[-1]
                    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'encoder'):
                        outputs = base_model.model.encoder(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            output_hidden_states=True
                        )
                        hidden_states = outputs.hidden_states[-1]
                    else:
                        return torch.tensor([0.0], device=self.device)
            
            value_head_dtype = next(self.value_head.parameters()).dtype
            if hidden_states.dtype != value_head_dtype:
                hidden_states = hidden_states.to(value_head_dtype)
            
            value = self.value_head(hidden_states)
            return value.squeeze().float()
            
        except Exception as e:
            print(f"  Warning: get_value failed: {e}")
            return torch.tensor([0.0], device=self.device)
    
    def compute_log_prob(self, prompt: str, action: str) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        if self.model is None:
            return torch.tensor(0.0, device=self.device)
        
        try:
            prompt_tokens = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_input_length // 2
            )
            
            action_tokens = self.tokenizer(
                action,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_input_length // 2
            )
            
            if self.model_type == 'causal':
                full_input_ids = prompt_tokens['input_ids'] + action_tokens['input_ids']
                full_attention_mask = prompt_tokens['attention_mask'] + action_tokens['attention_mask']
                
                labels = [-100] * len(prompt_tokens['input_ids']) + action_tokens['input_ids']
                
                max_len = self.max_input_length
                full_input_ids = full_input_ids[:max_len]
                full_attention_mask = full_attention_mask[:max_len]
                labels = labels[:max_len]
                
                input_ids = torch.tensor([full_input_ids], device=self.device)
                attention_mask = torch.tensor([full_attention_mask], device=self.device)
                labels_tensor = torch.tensor([labels], device=self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_tensor
                )
                
                log_prob = -outputs.loss
                
            else:
                # Seq2Seq LM
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_input_length
                ).to(self.device)
                
                labels = self.tokenizer(
                    action,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_output_length
                ).to(self.device)
                
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels['input_ids']
                )
                
                log_prob = -outputs.loss
            
            return log_prob
            
        except Exception as e:
            print(f"  Warning: compute_log_prob failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def compute_log_prob_batch(self, prompts: List[str], actions: List[str], mini_batch_size: int = 8) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        if self.model is None or len(prompts) == 0:
            return torch.zeros(len(prompts), device=self.device, requires_grad=True)
        
        total_size = len(prompts)
        all_log_probs = []
        
        try:
            for batch_start in range(0, total_size, mini_batch_size):
                batch_end = min(batch_start + mini_batch_size, total_size)
                batch_prompts = prompts[batch_start:batch_end]
                batch_actions = actions[batch_start:batch_end]
                batch_size = len(batch_prompts)
                
                if self.model_type == 'causal':
                    all_input_ids = []
                    all_attention_masks = []
                    all_labels = []
                    max_len = 0
                    
                    for prompt, action in zip(batch_prompts, batch_actions):
                        prompt_tokens = self.tokenizer(
                            prompt,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=self.max_input_length // 2
                        )
                        action_tokens = self.tokenizer(
                            action,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.max_input_length // 2
                        )
                        
                        full_input_ids = prompt_tokens['input_ids'] + action_tokens['input_ids']
                        full_attention_mask = prompt_tokens['attention_mask'] + action_tokens['attention_mask']
                        labels = [-100] * len(prompt_tokens['input_ids']) + action_tokens['input_ids']
                        
                        full_input_ids = full_input_ids[:self.max_input_length]
                        full_attention_mask = full_attention_mask[:self.max_input_length]
                        labels = labels[:self.max_input_length]
                        
                        all_input_ids.append(full_input_ids)
                        all_attention_masks.append(full_attention_mask)
                        all_labels.append(labels)
                        max_len = max(max_len, len(full_input_ids))
                    
                    # Padding
                    pad_token_id = self.tokenizer.pad_token_id or 0
                    for i in range(batch_size):
                        pad_len = max_len - len(all_input_ids[i])
                        all_input_ids[i] = all_input_ids[i] + [pad_token_id] * pad_len
                        all_attention_masks[i] = all_attention_masks[i] + [0] * pad_len
                        all_labels[i] = all_labels[i] + [-100] * pad_len
                    
                    input_ids = torch.tensor(all_input_ids, device=self.device)
                    attention_mask = torch.tensor(all_attention_masks, device=self.device)
                    labels_tensor = torch.tensor(all_labels, device=self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels_tensor[:, 1:].contiguous()
                    
                    log_probs_per_token = F.log_softmax(shift_logits, dim=-1)
                    
                    for i in range(batch_size):
                        valid_mask = shift_labels[i] != -100
                        if valid_mask.sum() > 0:
                            token_log_probs = log_probs_per_token[i].gather(
                                dim=-1, 
                                index=shift_labels[i].clamp(min=0).unsqueeze(-1)
                            ).squeeze(-1)
                            sample_log_prob = (token_log_probs * valid_mask.float()).sum()
                            all_log_probs.append(sample_log_prob.view(1).squeeze(0))
                        else:
                            zero_lp = (log_probs_per_token[i, 0, 0] * 0.0)
                            all_log_probs.append(zero_lp)
                else:
                    for prompt, action in zip(batch_prompts, batch_actions):
                        all_log_probs.append(self.compute_log_prob(prompt, action))
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return torch.stack(all_log_probs)
                
        except Exception as e:
            print(f"  Warning: compute_log_prob_batch failed: {e}")
            return torch.zeros(total_size, device=self.device, requires_grad=True)
    
    def get_value_batch(self, state_texts: List[str], mini_batch_size: int = 8, requires_grad: bool = True) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        if self.model is None or self.value_head is None or len(state_texts) == 0:
            return torch.zeros(len(state_texts), device=self.device, requires_grad=requires_grad)
        
        total_size = len(state_texts)
        all_values = []
        
        try:
            for batch_start in range(0, total_size, mini_batch_size):
                batch_end = min(batch_start + mini_batch_size, total_size)
                batch_texts = state_texts[batch_start:batch_end]
                
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_input_length,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    if self.model_type == 'causal':
                        outputs = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            output_hidden_states=True
                        )
                        hidden_states = outputs.hidden_states[-1]
                        
                        seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
                        batch_size = hidden_states.size(0)
                        last_hidden = hidden_states[
                            torch.arange(batch_size, device=self.device), 
                            seq_lengths
                        ]
                    else:
                        outputs = self.model.encoder(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            output_hidden_states=True
                        )
                        hidden_states = outputs.hidden_states[-1]
                        last_hidden = hidden_states[:, -1, :]
                
                value_head_dtype = next(self.value_head.parameters()).dtype
                if last_hidden.dtype != value_head_dtype:
                    last_hidden = last_hidden.to(value_head_dtype)
                
                if requires_grad:
                    last_hidden = last_hidden.detach().requires_grad_(True)
                    values = self.value_head(last_hidden).view(-1)
                else:
                    with torch.no_grad():
                        values = self.value_head(last_hidden).view(-1)
                
                all_values.append(values)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return torch.cat(all_values, dim=0)
                
        except Exception as e:
            print(f"  Warning: get_value_batch failed: {e}")
            return torch.zeros(total_size, device=self.device)
    
    def _identify_retrieval_type(self, retrieval_results: List[Dict[str, Any]]) -> str:
        """
        Returns:
        """
        if not retrieval_results or len(retrieval_results) == 0:
            return 'direct'
        
        first_result = retrieval_results[0]
        
        if not first_result.get('document', ''):
            return 'direct'
        
        doc = first_result.get('document', '')
        
        if 'Retrieved factual triples:' in doc:
            return 'triple'
        elif 'relevant entities and their 2-hop subgraphs' in doc:
            return 'subgraph'
        elif 'Connected subgraph' in doc or 'Connected graph' in doc:
            return 'connected_graph'
        else:
            return 'subgraph'
    
    def _align_knowledge(self,
                        retrieval_results: List[Dict[str, Any]],
                        watching_history: str,
                        user_context: Dict[str, Any],
                        retrieval_type: str) -> str:
        """"""
        if retrieval_type == 'direct':
            return ""
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Knowledge Aligner LLM model not initialized. Call initialize_model() first.")
        
        if retrieval_type == 'triple':
            alignment_prompt = self._create_triple_alignment_template(retrieval_results, watching_history, user_context)
        elif retrieval_type == 'subgraph':
            alignment_prompt = self._create_subgraph_alignment_template(retrieval_results, watching_history, user_context)
        elif retrieval_type == 'connected_graph':
            alignment_prompt = self._create_connected_graph_alignment_template(retrieval_results, watching_history, user_context)
        else:
            alignment_prompt = self._create_direct_alignment_template(retrieval_results, watching_history, user_context)
        
        if not alignment_prompt:
            raise ValueError(f"Failed to create alignment prompt for retrieval_type: {retrieval_type}")
        
        aligned_knowledge = self._model_based_alignment(alignment_prompt)
        
        return aligned_knowledge
    
    def _create_triple_alignment_template(self,
                                         retrieval_results: List[Dict[str, Any]],
                                         query: str,
                                         user_context: Dict[str, Any]) -> str:
        """
        Triple-based Alignment Template
        """
        if not retrieval_results:
            return ""
        
        doc = retrieval_results[0].get('document', '')
        
        template = f"""Task: Convert the following factual triples into natural language knowledge suitable for movie recommendations.

User Query: {query}
User Preferences: {user_context.get('preferences', 'general user')}

Retrieved Triples:
{doc}

Instructions:
1. Convert each triple into a natural, fluent sentence
2. Group related triples together
3. Maintain factual accuracy
4. Make the text easy to understand
5. Focus on information relevant to the user's query

Aligned Knowledge (natural language):"""
        
        return template
    
    def _create_subgraph_alignment_template(self,
                                           retrieval_results: List[Dict[str, Any]],
                                           query: str,
                                           user_context: Dict[str, Any]) -> str:
        """
        Subgraph-based Alignment Template
        """
        if not retrieval_results:
            return ""
        
        doc = retrieval_results[0].get('document', '')
        
        template = f"""Task: Transform the following knowledge graph subgraph into natural language suitable for movie recommendations.

User Query: {query}
User Preferences: {user_context.get('preferences', 'general user')}

Retrieved Subgraph:
{doc}

Instructions:
1. Extract key entities and their relationships
2. Describe connectivity patterns and relational paths
3. Summarize the semantic meaning of the subgraph
4. Organize information by relevance to the query
5. Use natural, flowing language
6. Highlight important connections between movies, actors, directors, etc.

Aligned Knowledge (natural language summary):"""
        
        return template
    
    def _create_connected_graph_alignment_template(self,
                                                   retrieval_results: List[Dict[str, Any]],
                                                   query: str,
                                                   user_context: Dict[str, Any]) -> str:
        """
        Connected Graph Alignment Template
        """
        if not retrieval_results:
            return ""
        
        doc = retrieval_results[0].get('document', '')
        
        domain = self.config.get('domain', 'movie')
        rec_type = 'music' if domain == 'music' else 'movie'
        
        template = f"""Task: Transform the following connected knowledge graph into natural language for {rec_type} recommendations.

User Query: {query}
User Preferences: {user_context.get('preferences', 'general user')}

Connected Graph (with PageRank importance):
{doc}

Instructions:
1. Emphasize highly important nodes (high PageRank scores)
2. Describe the connected paths between entities
3. Explain why these connections are relevant
4. Summarize the graph structure's semantic meaning
5. Focus on recommendation-relevant information
6. Use clear, natural language

Aligned Knowledge (natural language narrative):"""
        
        return template
    
    def _create_direct_alignment_template(self,
                                         retrieval_results: List[Dict[str, Any]],
                                         query: str,
                                         user_context: Dict[str, Any]) -> str:
        """
        Direct mode template (no retrieval)
        """
        domain = self.config.get('domain', 'movie')
        rec_type = 'music' if domain == 'music' else 'movie'
        
        template = f"""Task: Generate {rec_type} recommendations based on the query without external knowledge.

User Query: {query}
User Preferences: {user_context.get('preferences', 'general user')}

Instructions:
Use your internal knowledge to provide recommendations.

Knowledge summary:"""
        
        return template
    
    def _model_based_alignment(self, alignment_prompt: str) -> str:
        """
        Args:
        Returns:
        """
        inputs = self.tokenizer(
            alignment_prompt,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        input_length = inputs['input_ids'].shape[1]
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            gen_kwargs = {
                'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'do_sample': False,
                'num_beams': 1,
            }
            
            if self.model_type == 'seq2seq':
                gen_kwargs['max_length'] = self.max_output_length
            else:  # causal
                gen_kwargs['max_new_tokens'] = self.max_new_tokens
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        if self.model_type == 'causal':
            generated_ids = outputs[0][input_length:]
            aligned_text = self.tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True
            )
        else:
            aligned_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
        
        return aligned_text.strip()
    
    def _template_based_alignment(self,
                                  retrieval_results: List[Dict[str, Any]],
                                  watching_history: str,
                                  user_context: Dict[str, Any],
                                  retrieval_type: str) -> str:
        """
        Args:
        Returns:
        """
        if not retrieval_results:
            return "No relevant knowledge found."
        
        doc = retrieval_results[0].get('document', '')
        
        if retrieval_type == 'direct':
            return ""
        
        elif retrieval_type == 'triple':
            return self._align_triples(doc, watching_history, user_context)
        
        elif retrieval_type == 'subgraph':
            return self._align_subgraph(doc, watching_history, user_context)
        
        elif retrieval_type == 'connected_graph':
            return self._align_connected_graph(doc, watching_history, user_context)
        
        else:
            return self._simple_alignment(doc, watching_history)
    
    def _align_triples(self, doc: str, watching_history: str, user_context: Dict[str, Any]) -> str:
        """
        Triple-based Alignment
        """
        aligned_parts = []
        
        if watching_history:
            aligned_parts.append(f"Based on {watching_history}")
            aligned_parts.append("")
        
        lines = doc.split('\n')
        triples = []
        
        for line in lines:
            if line.strip().startswith('•'):
                triple_text = line.strip()[1:].strip()
                if '(score:' in triple_text:
                    triple_text = triple_text.split('(score:')[0].strip()
                triples.append(triple_text)
        
        if not triples:
            return "No factual information found."
        
        movies_info = {}
        for triple in triples:
            if ' director ' in triple or ' producer ' in triple or ' starring ' in triple or ' distributor ' in triple:
                parts = triple.split(' director ')
                if len(parts) == 1:
                    parts = triple.split(' producer ')
                if len(parts) == 1:
                    parts = triple.split(' starring ')
                if len(parts) == 1:
                    parts = triple.split(' distributor ')
                
                if len(parts) >= 2:
                    movie = parts[0].strip()
                    if movie not in movies_info:
                        movies_info[movie] = []
                    movies_info[movie].append(triple)
                else:
                    if 'Other' not in movies_info:
                        movies_info['Other'] = []
                    movies_info['Other'].append(triple)
        
        aligned_parts.append("Relevant movie facts from knowledge graph:")
        aligned_parts.append("")
        
        for movie, facts in list(movies_info.items())[:5]:
            if movie != 'Other':
                aligned_parts.append(f"**{movie}**:")
                for fact in facts[:5]:
                    aligned_parts.append(f"  - {fact}")
                aligned_parts.append("")
        
        if 'Other' in movies_info:
            aligned_parts.append("Additional Facts:")
            for fact in movies_info['Other'][:3]:
                aligned_parts.append(f"  - {fact}")
        
        return "\n".join(aligned_parts)
    
    def _align_subgraph(self, doc: str, watching_history: str, user_context: Dict[str, Any]) -> str:
        """
        Subgraph-based Alignment
        """
        aligned_parts = []
        
        if watching_history:
            aligned_parts.append(f"Based on {watching_history}")
            aligned_parts.append("")
        
        lines = doc.split('\n')
        
        key_entities = []
        key_relations = []
        
        in_entities = False
        in_relationships = False
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')) and '[score:' in line:
                key_entities.append(line.strip())
                in_entities = True
                in_relationships = False
            elif 'Key relationships:' in line or 'Important relationships' in line:
                in_entities = False
                in_relationships = True
            elif in_relationships and line.strip().startswith('•'):
                relation = line.strip()[1:].strip()
                key_relations.append(relation)
        
        aligned_parts.append("Related entities and connections from knowledge graph:")
        aligned_parts.append("")
        
        if key_entities:
            aligned_parts.append("Main Movies Found:")
            for i, entity in enumerate(key_entities[:5], 1):
                if '(' in entity and ')' in entity:
                    parts = entity.split('(')
                    movie_name = parts[0].replace(f'{i}.', '').strip()
                    aligned_parts.append(f"  {i}. {movie_name}")
            aligned_parts.append("")
        
        if key_relations:
            aligned_parts.append("Key Facts About These Movies:")
            
            movies_dict = {}
            for relation in key_relations[:20]:
                if ' is directed by ' in relation or ' is produced by ' in relation or ' starring ' in relation:
                    movie = relation.split(' is directed by ')[0].split(' is produced by ')[0].split(' starring ')[0].strip()
                    if movie not in movies_dict:
                        movies_dict[movie] = []
                    movies_dict[movie].append(relation)
            
            for movie, facts in list(movies_dict.items())[:5]:
                aligned_parts.append(f"\n  {movie}:")
                for fact in facts[:4]:
                    aligned_parts.append(f"    - {fact}")
        
        aligned_parts.append("")
        aligned_parts.append(f"Knowledge network: {len(key_entities)} entities, {len(key_relations)} relationships.")
        
        return "\n".join(aligned_parts)
    
    def _align_connected_graph(self, doc: str, watching_history: str, user_context: Dict[str, Any]) -> str:
        """
        Connected Graph Alignment
        """
        aligned_parts = []
        
        if watching_history:
            aligned_parts.append(f"Based on {watching_history}")
            aligned_parts.append("")
        
        lines = doc.split('\n')
        
        seed_entities = []
        graph_relationships = []
        
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if 'Seed entities' in line:
                current_section = 'seed'
                continue
            elif 'important relationships' in line.lower():
                current_section = 'relationships'
                continue
            elif 'Connected graph statistics' in line or 'graph statistics' in line.lower():
                current_section = 'stats'
                continue
            
            if current_section == 'seed' and line_stripped:
                if line_stripped and (line_stripped[0].isdigit() or line_stripped.startswith('-')):
                    seed_entities.append(line_stripped)
            elif current_section == 'relationships' and line_stripped:
                if line_stripped.startswith('•'):
                    rel = line_stripped[1:].strip()
                    if '[relevance:' in rel:
                        rel = rel.split('[relevance:')[0].strip()
                    graph_relationships.append(rel)
                elif line_stripped.startswith('-'):
                    rel = line_stripped[1:].strip()
                    if '[relevance:' in rel:
                        rel = rel.split('[relevance:')[0].strip()
                    graph_relationships.append(rel)
        
        aligned_parts.append("Knowledge Graph Connections:")
        aligned_parts.append("")
        
        if seed_entities:
            aligned_parts.append("Related Entities:")
            for entity in seed_entities[:5]:
                clean_entity = entity
                if '[relevance:' in clean_entity:
                    clean_entity = clean_entity.split('[relevance:')[0].strip()
                if '[importance:' in clean_entity:
                    clean_entity = clean_entity.split('[importance:')[0].strip()
                aligned_parts.append(f"  {clean_entity}")
        
        if graph_relationships:
            aligned_parts.append("")
            aligned_parts.append("Key Relationships:")
            for i, rel in enumerate(graph_relationships[:10], 1):
                aligned_parts.append(f"  {i}. {rel}")
        
        if not graph_relationships:
            for line in lines:
                if line.strip().startswith('•'):
                    rel = line.strip()[1:].strip()
                    if '[relevance:' in rel:
                        rel = rel.split('[relevance:')[0].strip()
                    if rel and rel not in graph_relationships:
                        graph_relationships.append(rel)
            
            if graph_relationships:
                aligned_parts.append("")
                aligned_parts.append("Key Relationships:")
                for i, rel in enumerate(graph_relationships[:10], 1):
                    aligned_parts.append(f"  {i}. {rel}")
        
        return "\n".join(aligned_parts)
    
    def _simple_alignment(self, doc: str, watching_history: str = "") -> str:
        """"""
        if watching_history:
            return f"Based on {watching_history}\n\nRelevant knowledge from knowledge graph:\n{doc[:1000]}"
        return f"Relevant knowledge from knowledge graph:\n{doc[:1000]}"
    
    def _calculate_confidence(self, aligned_knowledge: str, retrieval_results: List[Dict[str, Any]]) -> float:
        """"""
        if not aligned_knowledge or not retrieval_results:
            return 0.0
        
        confidence_factors = []
        
        num_docs = len(retrieval_results)
        confidence_factors.append(min(1.0, num_docs / 5.0))
        
        text_length = len(aligned_knowledge)
        length_score = min(1.0, text_length / 200.0) * 0.8 + 0.2
        confidence_factors.append(length_score)
        
        if retrieval_results and isinstance(retrieval_results[0], dict):
            score = retrieval_results[0].get('score', 0.5)
            confidence_factors.append(score)
        else:
            confidence_factors.append(0.5)
        
        final_confidence = np.mean(confidence_factors)
        return float(np.clip(final_confidence, 0.0, 1.0))
    
    def update(self, experience: Dict[str, Any]):
        """"""
        if not self.training_mode or self.model is None:
            return
            
        self.episode_history.append(experience)
        
        reward = experience.get('reward', 0.0)
        
        self.baseline_value = self.baseline_alpha * reward + (1 - self.baseline_alpha) * self.baseline_value
        
        if 'refined_knowledge' in experience:
            refined_text = experience['refined_knowledge']
            self.alignment_scores.append(reward)
            self.generation_lengths.append(len(refined_text))
    
    def compute_policy_gradient_loss(self, experiences: List[Dict[str, Any]], advantages: torch.Tensor) -> torch.Tensor:
        """"""
        if self.model is None:
            return torch.tensor(0.0, device=self.device)
        
        log_probs = []
        
        for exp in experiences:
            observation = exp['observation']
            action = exp['action']  # aligned_knowledge
            
            retrieval_results = observation.get('retrieval_results', [])
            query = observation.get('query', '')
            user_context = observation.get('user_context', {})
            retrieval_type = self._identify_retrieval_type(retrieval_results)
            
            template_func = self.alignment_templates.get(retrieval_type, self._create_subgraph_alignment_template)
            input_text = template_func(retrieval_results, query, user_context)
            
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
        
        alignment_metrics = {
            'model_name': self.model_name,
            'training_mode': self.training_mode,
            'baseline_value': self.baseline_value,
            'episode_experiences': len(self.episode_experiences),
        }
        
        if self.alignment_scores:
            alignment_metrics.update({
                'avg_alignment_score': np.mean(self.alignment_scores[-100:]),
                'avg_generation_length': np.mean(self.generation_lengths[-100:]),
                'total_alignments': len(self.alignment_scores),
            })
        
        base_metrics.update(alignment_metrics)
        return base_metrics
    
    def save_checkpoint(self, path: str):
        """"""
        if self.model is None:
            return
        
        checkpoint = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            
            'alignment_scores': self.alignment_scores[-100:],
            'generation_lengths': self.generation_lengths[-100:],
            'baseline_value': self.baseline_value,
            
            'has_optimizer': self.optimizer is not None
        }
        
        if self.optimizer is not None:
            try:
                model_path = path.replace('.pt', '_lora')
                self.model.save_pretrained(model_path)
                checkpoint['lora_path'] = model_path
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                print(f"  ✓ Knowledge Aligner LoRA saved to {model_path}")
            except:
                checkpoint['model_state_dict'] = self.model.state_dict()
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if hasattr(self, 'value_head') and self.value_head is not None:
            checkpoint['value_head_state_dict'] = self.value_head.state_dict()
            if hasattr(self, 'value_optimizer') and self.value_optimizer is not None:
                checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, map_location=None):
        """"""
        import os
        if not os.path.exists(path):
            print(f"  ⚠ No Knowledge Aligner checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, weights_only=False, map_location=map_location)
        
        self.alignment_scores = checkpoint.get('alignment_scores', [])
        self.generation_lengths = checkpoint.get('generation_lengths', [])
        self.baseline_value = checkpoint.get('baseline_value', 0.0)
        
        if checkpoint.get('has_optimizer') and self.optimizer is not None:
            try:
                if 'lora_path' in checkpoint:
                    lora_path = checkpoint['lora_path']
                    if os.path.exists(lora_path):
                        from peft import set_peft_model_state_dict
                        from safetensors.torch import load_file
                        
                        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
                        if os.path.exists(adapter_path):
                            adapter_weights = load_file(adapter_path)
                            set_peft_model_state_dict(self.model, adapter_weights)
                            print(f"  ✓ Knowledge Aligner LoRA loaded from {lora_path}")
                        else:
                            adapter_path = os.path.join(lora_path, "adapter_model.bin")
                            if os.path.exists(adapter_path):
                                adapter_weights = torch.load(adapter_path, weights_only=False, map_location=map_location)
                                set_peft_model_state_dict(self.model, adapter_weights)
                                print(f"  ✓ Knowledge Aligner LoRA loaded from {lora_path}")
                            else:
                                print(f"  ⚠ LoRA adapter file not found in {lora_path}")
                    else:
                        print(f"  ⚠ LoRA path does not exist: {lora_path}")
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  ✓ Knowledge Aligner model state loaded")
                
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if hasattr(self, 'value_head') and 'value_head_state_dict' in checkpoint:
                    self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
                    if hasattr(self, 'value_optimizer') and 'value_optimizer_state_dict' in checkpoint:
                        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
                    print(f"  ✓ Knowledge Aligner Value Head loaded")
                
                print(f"  ✓ Knowledge Aligner checkpoint loaded from {path}")
            except Exception as e:
                print(f"  ⚠ Failed to load Knowledge Aligner state: {e}")
                import traceback
                traceback.print_exc()
    
    def set_training_mode(self, training: bool):
        """"""
        self.training_mode = training
        if self.model is not None:
            self.model.train() if training else self.model.eval()
