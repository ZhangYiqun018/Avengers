from typing import List

from loguru import logger

from core.experts.load_experts import Expert
from core.routing.base_router import BaseRouter, RouterOutput


class MoARouter(BaseRouter):
    def __init__(self, normal_experts: List[Expert], thinking_experts: List[Expert], router_config: dict):
        super().__init__(normal_experts, thinking_experts)
        self.config = router_config['moa_router']
        self.proposers = self.config.get('proposers')
        self.aggregator = self.config.get('aggregator')
        
        self.experts = self.find_experts()
        
    def find_experts(self) -> List[Expert]:
        experts = []
        for proposer in self.proposers:
            for expert in self.normal_experts:
                if proposer == expert.model_name:
                    experts.append(expert)
                    break
        
        for expert in self.normal_experts:
            if expert.model_name == self.aggregator:
                experts.append(expert)
                break
        
        assert len(experts) == len(self.proposers) + 1, f"Number of experts must be equal to {len(self.proposers) + 1}"
        return experts
        
    def route(self, question: str) -> RouterOutput:
        return RouterOutput(
            normal_experts=self.experts,
            thinking_experts=[]
        )