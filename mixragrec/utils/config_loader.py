"""
Configuration loader for MixRAGRec framework.
"""

import yaml
import os
from typing import Dict, Any, Optional
import json


class ConfigLoader:
    """"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """"""
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        file_extension = os.path.splitext(config_path)[1].lower()
        
        if file_extension == '.yaml' or file_extension == '.yml':
            config = ConfigLoader._load_yaml(config_path)
        elif file_extension == '.json':
            config = ConfigLoader._load_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {file_extension}")
        
        config = ConfigLoader._apply_llm_preset(config)
        
        return config
    
    @staticmethod
    def _apply_llm_preset(config: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        llm_name = config.get('experiment', {}).get('llm', 'llama-8b')
        presets = config.get('llm_presets', {})
        
        if llm_name not in presets:
            print(f"⚠ LLM preset '{llm_name}' not found. Available: {list(presets.keys())}")
            return config
        
        preset = presets[llm_name]
        print(f"✓ Applying LLM preset: {llm_name} → {preset.get('model_name', 'unknown')}")
        
        if 'models' in config and 'knowledge_aligner' in config['models']:
            config['models']['knowledge_aligner']['model_name'] = preset.get('model_name')
            config['models']['knowledge_aligner']['model_type'] = preset.get('model_type', 'causal')
            config['models']['knowledge_aligner']['hf_token'] = preset.get('hf_token', '')
        
        if 'models' in config and 'recommender' in config['models']:
            config['models']['recommender']['model_name'] = preset.get('model_name')
            config['models']['recommender']['model_type'] = preset.get('model_type', 'causal')
            config['models']['recommender']['hf_token'] = preset.get('hf_token', '')
        
        config = ConfigLoader._sync_device_ids(config)
        
        return config
    
    @staticmethod
    def _sync_device_ids(config: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        global_device = config.get('device', 'cuda:0')
        
        if global_device.startswith('cuda:'):
            try:
                gpu_id = int(global_device.split(':')[1])
            except (IndexError, ValueError):
                gpu_id = 0
        elif global_device == 'cuda':
            gpu_id = 0
        else:
            gpu_id = None  # CPU
        
        if gpu_id is not None:
            print(f"✓ Syncing all components to GPU {gpu_id}")
            
            if 'models' in config and 'knowledge_aligner' in config['models']:
                config['models']['knowledge_aligner']['device_id'] = gpu_id
            
            if 'models' in config and 'recommender' in config['models']:
                config['models']['recommender']['device_id'] = gpu_id
        
        return config
    
    @staticmethod
    def _load_yaml(config_path: str) -> Dict[str, Any]:
        """"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def _load_json(config_path: str) -> Dict[str, Any]:
        """"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """"""
        required_sections = ['models', 'agents', 'marl', 'training', 'data']
        
        for section in required_sections:
            if section not in config:
                print(f"Warning: Missing required config section: {section}")
                return False
        
        if 'device' not in config:
            config['device'] = 'cpu'
            print("Warning: No device specified, using CPU")
        
        return True
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str):
        """"""
        file_extension = os.path.splitext(output_path)[1].lower()
        
        if file_extension == '.yaml' or file_extension == '.yml':
            ConfigLoader._save_yaml(config, output_path)
        elif file_extension == '.json':
            ConfigLoader._save_json(config, output_path)
        else:
            raise ValueError(f"Unsupported output format: {file_extension}")
    
    @staticmethod
    def _save_yaml(config: Dict[str, Any], output_path: str):
        """"""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _save_json(config: Dict[str, Any], output_path: str):
        """"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
