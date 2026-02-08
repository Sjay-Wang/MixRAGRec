"""
LLM model loader with support for both seq2seq and causal LMs.
Supports quantization for large models like Llama.

Part of MixRAGRec framework.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from typing import Dict, Any, Optional


class LLMLoader:
    """Universal LLM loader supporting multiple model types and quantization"""
    
    @staticmethod
    def load_model_and_tokenizer(
        model_name: str,
        model_type: str = "seq2seq",
        use_quantization: bool = False,
        quantization_bits: int = 8,
        device: str = "cuda",
        hf_token: str = None,
        device_id: int = None
    ):
        """
        Load model and tokenizer with optional quantization
        
        Args:
            model_name: HuggingFace model name
            model_type: "seq2seq" or "causal"
            use_quantization: Whether to use quantization
            quantization_bits: 4 or 8 bit quantization
            device: Device to load model on
            hf_token: HuggingFace token for gated models
            device_id: Specific GPU ID to load model on (for multi-GPU setup)
            
        Returns:
            (model, tokenizer, actual_device)
        """
        if device_id is not None:
            actual_device = f"cuda:{device_id}"
            print(f"Loading {model_name} ({model_type}) on GPU {device_id}...")
        else:
            actual_device = device
            print(f"Loading {model_name} ({model_type})...")
        
        # Load tokenizer (with token if provided)
        tokenizer_kwargs = {}
        if hf_token:
            tokenizer_kwargs['token'] = hf_token
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare quantization config if needed
        quantization_config = None
        if use_quantization and quantization_bits in [4, 8]:
            print(f"  Using {quantization_bits}-bit quantization")
            if quantization_bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:  # 4-bit
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        
        if use_quantization:
            if device_id is not None:
                device_map = {"": actual_device}
            else:
                device_map = "auto"
        else:
            device_map = None
        
        # Load model based on type
        model_kwargs = {
            "torch_dtype": torch.float16 if not use_quantization else None,
            "device_map": device_map,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Add HF token if provided
        if hf_token:
            model_kwargs["token"] = hf_token
        
        try:
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            elif model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            # Move to device if not using quantization
            if not use_quantization:
                model = model.to(actual_device)
            
            print(f"✓ Model loaded successfully on {actual_device}")
            
            return model, tokenizer, actual_device
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print(f"Falling back to T5-small...")
            
            # Fallback to T5-small
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            fallback_device = actual_device if not use_quantization else "cpu"
            model = model.to(fallback_device)
            
            return model, tokenizer, fallback_device
    
    @staticmethod
    def create_generation_config(
        model_type: str,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        pad_token_id: int = None,
        eos_token_id: int = None
    ) -> GenerationConfig:
        """
        Create generation configuration
        
        Args:
            model_type: "seq2seq" or "causal"
            max_length: Max total length (for seq2seq)
            max_new_tokens: Max new tokens (for causal)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            GenerationConfig
        """
        if model_type == "seq2seq":
            return GenerationConfig(
                max_length=max_length or 512,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        else:  # causal
            return GenerationConfig(
                max_new_tokens=max_new_tokens or 512,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=1.1,
            )
