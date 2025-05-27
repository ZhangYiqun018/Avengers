from enum import Enum

from loguru import logger

from core.routing.base_router import BaseRouter
from core.routing.gpt_router import GPTRouter
from core.routing.straight_router import StraightRouter
from core.routing.routerdc_router import RouterDC
from core.routing.random_router import RandomRouter
from core.routing.elo_router import EloRouter
from core.routing.rank_router import RankRouter
from core.routing.symbolic_moe_router import SymbolicMoERouter
from core.routing.moa_router import MoARouter

class RouterType(Enum):
    GPT = "gpt"
    STRAIGHT = "straight"
    RANDOM = "random"
    ROUTERDC = "routerdc"
    ELO = "elo"
    RANK = "rank"
    SYMBOLIC_MOE = "symbolic_moe"
    MOA = "moa"
    
class RouterFactory:
    @staticmethod
    def create_router(normal_experts: list, thinking_experts: list, router_config: dict):
        router_type = router_config['type']
        
        if isinstance(router_type, str):
            router_type = RouterType(router_type)
        
        if router_type == RouterType.GPT:
            logger.info(f"Creating GPT router.")
            return GPTRouter(normal_experts, thinking_experts, router_config)
        elif router_type == RouterType.STRAIGHT:
            logger.info(f"Creating Straight router.")
            return StraightRouter(normal_experts, thinking_experts, router_config)
        elif router_type == RouterType.RANDOM:
            logger.info(f"Creating Random router.")
            return RandomRouter(normal_experts, thinking_experts, router_config)
        elif router_type == RouterType.ROUTERDC:
            logger.info(f"Creating RouterDC router.")
            return RouterDC(normal_experts, thinking_experts, router_config)
        elif router_type == RouterType.ELO:
            logger.info(f"Creating Elo router.")
            return EloRouter(normal_experts, thinking_experts, router_config)
        elif router_type == RouterType.RANK:
            logger.info(f"Creating Rank router.")
            return RankRouter(normal_experts, thinking_experts, router_config)
        elif router_type == RouterType.SYMBOLIC_MOE:
            logger.info(f"Creating Symbolic MoE router.")
            return SymbolicMoERouter(normal_experts, thinking_experts, router_config)
        elif router_type == RouterType.MOA:
            logger.info(f"Creating MoA router.")
            return MoARouter(normal_experts, thinking_experts, router_config)
        else:
            logger.error(f"Invalid router type: {router_type}")
            raise ValueError(f"Invalid router type: {router_type}")     