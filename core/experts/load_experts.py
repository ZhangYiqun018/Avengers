from dataclasses import dataclass
from typing import List

from loguru import logger
from openai import OpenAI

from config.config_loader import load_config
import httpx
from core.cache import HttpCache

@dataclass
class Expert:
    model_name: str
    base_url: str
    api_key: str
    description: str
    client: OpenAI

def load_experts(config: dict) -> List[Expert]:
    use_http_cache = config['experiments']['use_http_cache']
    httpx_client = None
    if use_http_cache:
        # Prefer remote cache when configured; otherwise fall back to local file cache
        remote_cfg = config['experiments'].get('remote_cache')
        if remote_cfg:
            http_cache = HttpCache.remote(
                host=remote_cfg.get('host'),
                port=int(remote_cfg.get('port', 6379)),
                user=remote_cfg.get('user'),
                password=remote_cfg.get('password'),
                ttl=remote_cfg.get('ttl'),
                database=remote_cfg.get('database'),
            )
        else:
            cache_dir = config['experiments']['cache_dir']
            http_cache = HttpCache.local(cache_dir)
        httpx_client = http_cache.create_httpx_client()
        
    experts = []
    for model_config in config['experts']:
        client = OpenAI(
            base_url=model_config['base_url'],
            api_key=model_config['api_key'],
            http_client=httpx_client if use_http_cache else None
        )
        experts.append(Expert(
            model_name=model_config['name'],
            base_url=model_config['base_url'],
            api_key=model_config['api_key'],
            description=model_config['description'],
            client=client
        ))
    expert_names = [e.model_name for e in experts]
    logger.info(
        f"Load {len(experts)} experts: {expert_names}"
    )
    thinking_experts = []
    if 'thinking_experts' in config.keys():
        for model_config in config['thinking_experts']:
            client = OpenAI(
                base_url=model_config['base_url'],
                api_key=model_config['api_key']
            )
            thinking_experts.append(Expert(
                model_name=model_config['name'],
                base_url=model_config['base_url'],
                api_key=model_config['api_key'],
                description=model_config['description'],
                client=client
            ))
        logger.info(f"Load thinking expert: {[e.model_name for e in thinking_experts]}")
        
    return experts, thinking_experts
# test
if __name__ == "__main__":
    config = load_config()
    experts = load_experts(config)
    logger.info(experts)
