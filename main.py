import argparse
import json
import os
import subprocess
import sys

import yaml
from loguru import logger

from config.config_loader import (generate_experiment_configs, load_config,
                                  save_temp_config)


def get_args():
    parser = argparse.ArgumentParser(description="Run multiple experiments sequentially")
    parser.add_argument(
        "--config", type=str, help="Path to experiment variations config file", default='config/experiment_variations.yaml'
    )
    args = parser.parse_args()
    
    return args

def run_experiment(config_path, save_dir: str):
    """运行单个实验"""
    logger.info(f"Running experiment with config: {config_path}, save on: {save_dir}")
    result = subprocess.run(
        [sys.executable, "app.py", "--config", config_path, "--save_dir", save_dir],
    )
    
    if result.returncode != 0:
        logger.error(f"Experiment failed: {result.stderr}")
        return False
    
    logger.info(f"Experiment completed successfully")
    return True

def main():
    args = get_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        experiment_config = yaml.safe_load(f)

    base_config_path = experiment_config.get('base_config', 'config/experts.yaml')
    save_dir = experiment_config.get('save_dir', 'results')
    base_config = load_config(base_config_path)
    experiment_variations = experiment_config.get('variations', [])
    
    # create temp dirs
    temp_dir = 'config/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    experiment_configs = generate_experiment_configs(base_config, experiment_variations)
    logger.info(f"Generated {len(experiment_configs)} experiment configs")
    
    for i, experiment_config in enumerate(experiment_configs):
        logger.info(f"Running experiment {i+1}/{len(experiment_configs)}: {experiment_config['experiment_name']}")
        config_path = save_temp_config(experiment_config, temp_dir)
        success = run_experiment(config_path, save_dir)
        
        if not success:
            logger.error(f"Experiment {experiment_config['experiment_name']} failed")
    
    logger.info("All experiments completed")
    
    
if __name__ == "__main__":
    main()