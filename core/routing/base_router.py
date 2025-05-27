from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.experts.load_experts import Expert

@dataclass
class RouterOutput:
    normal_experts: List[Expert]
    thinking_experts: List[Expert]


class BaseRouter(ABC):
    def __init__(self, normal_experts: List[Expert], thinking_experts: List[Expert]):
        self.normal_experts = normal_experts
        self.thinking_experts = thinking_experts

    @abstractmethod
    def route(self, question: str) -> RouterOutput:
        return RouterOutput(
            normal_experts=[],
            thinking_experts=[]
        )