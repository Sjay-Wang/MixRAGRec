"""
Logger for MixRAGRec framework.
"""

import logging
import os
import json
from typing import Dict, Any
from datetime import datetime


class Logger:
    """"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_level = config.get('log_level', 'INFO')
        
        log_dir = config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"mixragrec_training_{timestamp}.log")
        metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.jsonl")
        
        self.logger = logging.getLogger('MixRAGRec')
        self.logger.setLevel(getattr(logging, self.log_level))
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.metrics_file = metrics_file
        
        self.logger.info("Logger initialized")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """"""
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(self.metrics_file, 'a') as f:
            json.dump(metrics, f, default=str)
            f.write('\n')
    
    def info(self, message: str):
        """"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """"""
        self.logger.debug(message)
