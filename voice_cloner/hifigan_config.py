import os
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class HiFiGANConfig:
    BASE_DIR = os.path.join('models', 'hifigan')
    
    MODEL_PATHS = {
        'universal_v1': {
            'dir': os.path.join(BASE_DIR, 'UNIVERSAL_V1'),
            'generator': 'g_02500000',
            'do_file': 'do_02500000',
            'config': 'config.json'
        },
        'lj_v1': {
            'dir': os.path.join(BASE_DIR, 'LJ_V1'),
            'generator': 'generator_v1',
            'config': 'config_v1.json'
        },
        'lj_v2': {
            'dir': os.path.join(BASE_DIR, 'LJ_V2'),
            'generator': 'generator_v2',
            'config': 'config_v2.json'
        },
        'lj_ft_t2_v1': {
            'dir': os.path.join(BASE_DIR, 'LJ_FT_T2_V1'),
            'generator': 'generator_v1',
            'config': 'config_v1.json'
        },
        'lj_ft_t2_v2': {
            'dir': os.path.join(BASE_DIR, 'LJ_FT_T2_V2'),
            'generator': 'generator_v2',
            'config': 'config_v2.json'
        },
        'lj_ft_t2_v3': {
            'dir': os.path.join(BASE_DIR, 'LJ_FT_T2_V3'),
            'generator': 'generator_v3',
            'config': 'config_v3.json'
        },
        'vctk_v1': {
            'dir': os.path.join(BASE_DIR, 'VCTK_V1'),
            'generator': 'generator_v1',
            'config': 'config_v1.json'
        },
        'vctk_v2': {
            'dir': os.path.join(BASE_DIR, 'VCTK_V2'),
            'generator': 'generator_v2',
            'config': 'config_v2.json'
        },
        'vctk_v3': {
            'dir': os.path.join(BASE_DIR, 'VCTK_V3'),
            'generator': 'generator_v3',
            'config': 'config_v3.json'
        }
    }
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Dict[str, str]:
        """Get the full paths for a specific model"""
        if model_name not in cls.MODEL_PATHS:
            raise ValueError(f"Model {model_name} not found. Available models: {list(cls.MODEL_PATHS.keys())}")
        
        model_info = cls.MODEL_PATHS[model_name]
        return {
            'generator_path': os.path.join(model_info['dir'], model_info['generator']),
            'config_path': os.path.join(model_info['dir'], model_info['config']),
            'do_path': os.path.join(model_info['dir'], model_info.get('do_file', ''))
        }