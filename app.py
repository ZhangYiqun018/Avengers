import argparse
import json
import os
import time

from loguru import logger

from config.config_loader import load_config
from core.experts.load_experts import load_experts
from core.inference import DirectGenerator
from core.routing import GPTRouter, RouterFactory, RouterType, StraightRouter
from evaluate.factory import EvaluatorFactory


def show_config(config: dict):
    logger.info("="*30)
    logger.info("Experiment Config:")
    if 'experiment_name' in config:
        logger.info(f"Experiment: {config['experiment_name']}")
    logger.info(f"Task: {config['experiments']['task']}")
    logger.info(f"Max workers: {config['experiments']['max_workers']}")
    logger.info(f"Mode: {config['experiments']['mode']}")
    logger.info(f"Router: {config['router']['type']}")
    logger.info(f"Generator: {config['generator']['type']}")
    logger.info(f"Use HTTP Cache: {config['experiments']['use_http_cache']}")
    logger.info("="*30)
    
    
def run_experiment(config: dict, save_dir: str = None):
    show_config(config)
    
    task = config['experiments']['task']
    max_workers = config['experiments']['max_workers']
    mode = config['experiments']['mode']
    use_http_cache = config['experiments']['use_http_cache']
    
    if use_http_cache:
        logger.warning("Use HTTP Cache, supported by hishel: https://github.com/karpetrosyan/hishel")
        logger.warning(f"Cache dir: {config['experiments']['cache_dir']}")
        
    # 2. load experts, TODO: thinking experts
    normal_experts, thinking_experts = load_experts(config)
    
    # 3. create router
    router = RouterFactory.create_router(
        normal_experts=normal_experts, thinking_experts=thinking_experts, router_config = config['router']
    )
    # 4. get evaluator
    if config['generator']['type'] == "fast_slow" and max_workers > 1:
        logger.warning(f"FastSlowGenerator does not recommend multi-threading, kv-cache may cause GPU boom.")
    evaluator = EvaluatorFactory(max_workers=max_workers, mode=mode).get_evaluator(task=task)
    
    # 5. evaluate
    results = evaluator.evaluate_loop(router=router, generator_config=config['generator'])
    
    results['config'] = config
    
    # 6. save results
    # 使用实验名称作为文件名的一部分（如果有）
    if save_dir is None:
        save_dir = "results"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    experiment_name = config.get('experiment_name', '')
    generator_type = config.get('generator', {}).get('type', '')
    
    os.makedirs(f"{save_dir}/{generator_type}", exist_ok=True)
    if experiment_name:
        filename = f"{save_dir}/{generator_type}/{task}-{experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    else:
        filename = f"{save_dir}/{generator_type}/{task}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    
    logger.info(f"Save result to {filename}")
    with open(filename, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return results
    
if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config file (relative to config directory)")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Path to save results")
    args = parser.parse_args()
    
    # 1. Load config
    config = load_config(args.config)
    run_experiment(config, args.save_dir)