from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
@dataclass
class GeneratorOutput:
    first_output: str
    raw_output: List[str]
    prompt_tokens: int
    completion_tokens: int
    
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str):
        pass