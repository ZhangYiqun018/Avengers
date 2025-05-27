from typing import List

from loguru import logger

from core.experts.load_experts import Expert
from core.routing.base_router import BaseRouter, RouterOutput


class StraightRouter(BaseRouter):
    def __init__(self, normal_experts: List[Expert], thinking_experts: List[Expert], router_config: dict):
        super().__init__(normal_experts, thinking_experts)
        
        self.config = router_config['straight_router']
        model_name = self.config['model']
        self.expert = self.find_expert(model_name)
        
    def find_expert(self, model_name: str) -> Expert:
        self.expert = next((expert for expert in self.normal_experts if expert.model_name == model_name), None)
        if self.expert is None:
            logger.error(f"Expert with model name {model_name} not found")
            raise ValueError(f"Expert with model name {model_name} not found")
        return self.expert
        
    def route(self, question: str) -> RouterOutput:
        return RouterOutput(
            normal_experts=[self.expert],
            thinking_experts=self.thinking_experts
        )