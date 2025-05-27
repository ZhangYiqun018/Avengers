from dataclasses import dataclass
from typing import List

from loguru import logger
from openai import OpenAI

from config.config_loader import load_config
import hishel, httpx
import json, hashlib
from typing import Optional
from hishel._utils import normalized_url

@dataclass
class Expert:
    model_name: str
    base_url: str
    api_key: str
    description: str
    client: OpenAI

def param_only_key(request: httpx.Request, body: Optional[bytes] = b"") -> str:
    INTERESTED_FIELDS = {"model", "temperature", "top_p", "n", "messages"}

    # 1) extract url path
    full_url = normalized_url(request.url)        
    url_obj  = httpx.URL(full_url)
    encoded_url = (url_obj.raw_path or b"/")      
    # 2) extract interested fields
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        payload = {}
    filtered = {k: payload.get(k) for k in sorted(INTERESTED_FIELDS)}
    encoded_body = json.dumps(
        filtered, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode()
    
    # 3) generate key
    key_parts = [request.method, encoded_url, encoded_body]

    try:                                          # use blake2b-128
        hasher = hashlib.blake2b(digest_size=16, usedforsecurity=False)
    except (TypeError, ValueError, AttributeError):
        hasher = hashlib.sha256(usedforsecurity=False)

    for part in key_parts:
        hasher.update(part)
    return hasher.hexdigest()

def init_http_cache(cache_dir: str):
    storage = hishel.FileStorage(base_path=cache_dir)
    base_transport = httpx.HTTPTransport()
    controller = hishel.Controller(
        cacheable_methods = ["GET", "POST"],
        cacheable_status_codes=[200],
        allow_stale=True,
        force_cache=True,
        key_generator=param_only_key
    )
    transport = hishel.CacheTransport(
        storage=storage,
        transport=base_transport,
        controller=controller
    )
    return transport

def load_experts(config: dict) -> List[Expert]:
    use_http_cache = config['experiments']['use_http_cache']
    if use_http_cache:
        cache_dir = config['experiments']['cache_dir']
        transport = init_http_cache(cache_dir)
        httpx_client = httpx.Client(transport=transport)
        
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