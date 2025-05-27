# config_loader.py
import copy
import os
from pathlib import Path

import yaml
from loguru import logger


def load_config(yaml_path=None):
    """
    load config from yaml file
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_experiment_configs(base_config, experiment_variations):
    """
    generate experiment configs from base config and experiment variations
    """
    configs = []
    for variation in experiment_variations:
        config = copy.deepcopy(base_config)
        
        for path, value in variation['params'].items():
            # like router.type
            parts = path.split('.')
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
        
        config['experiment_name'] = variation['name']
        configs.append(config)

    return configs

def save_temp_config(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{config['experiment_name']}.yaml")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path

# test
if __name__ == "__main__":
    cfg = load_config()
    print(cfg)  # print to see