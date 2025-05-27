from typing import List

from loguru import logger

from core.experts.load_experts import Expert
from core.routing.base_router import BaseRouter, RouterOutput
import random

class RandomRouter(BaseRouter):
    def __init__(self, normal_experts: List[Expert], thinking_experts: List[Expert], router_config: dict):
        super().__init__(normal_experts, thinking_experts)
        self.config = router_config['random_router']
        self.max_router = self.config['max_router']
        available_models = self.config.get("available_models")
        self.candidate_models = []
        for expert in normal_experts:
            if expert.model_name in available_models:
                self.candidate_models.append(expert)
        assert len(self.candidate_models) == len(available_models)
        if not isinstance(self.max_router, int) or self.max_router <= 0:
            raise ValueError(f"max_router must be a positive integer, got {self.max_router}")
            
        if self.max_router > len(normal_experts):
            logger.warning(f"max_router ({self.max_router}) is greater than the number of normal experts ({len(normal_experts)})")
            raise ValueError(f"max_router ({self.max_router}) cannot be greater than the number of normal experts ({len(normal_experts)})")

    def route(self, question: str) -> RouterOutput:
        """随机选择指定数量的专家进行路由
        
        Args:
            question: 输入的问题
            
        Returns:
            RouterOutput: 包含随机选择的normal_experts和所有thinking_experts
        """
        selected_experts = random.sample(self.candidate_models, self.max_router)
        return RouterOutput(
            normal_experts=selected_experts,
            thinking_experts=self.thinking_experts
        )