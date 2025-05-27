import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List

from loguru import logger

BOXED_PATTERN = r"\\boxed\{([^}]*)\}"

class BaseEvaluator(ABC):
    def __init__(self, max_workers:int=8, mode: str="test"):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.max_workers = max_workers
        self.mode = mode
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data
    
    def update_tokens(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
    
    def fresh_tokens(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def extract_boxed_content(self, text: str) -> str:
        start_tag = r"\boxed{"
        start = text.find(start_tag)
        if start == -1:
            return ""

        start += len(start_tag)
        brace_count = 1  # 已经找到一个 {
        result = []

        for char in text[start:]:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            result.append(char)

        return ''.join(result).strip()

    def count_prediction_frequency(self, predictions: list):
        """
        Count the frequency of each prediction in the list of predictions for math or multi-choice tasks.
        """
        prediction_counts = Counter(predictions)
        total = len(predictions)
        frequency_stats = dict()
        for pred, count in prediction_counts.items():
            frequency_stats[pred] = {
                "count": count,
                "frequency": count / total
            }
        return frequency_stats
    
    def calculate_model_counts(self, results: list[dict]):
        # process model name
        position_model_counts = {}
        for result in results:
            model_name = result['model_name']
            if not isinstance(model_name, list):
                model_name = [model_name]
            for idx, model in enumerate(model_name):
                position = idx + 1
                if position not in position_model_counts:
                    position_model_counts[position] = Counter()
                position_model_counts[position][model] += 1
        
        # 输出每个位置的模型使用情况
        for position, model_counter in position_model_counts.items():
            logger.info(f"Position {position} model counts: {model_counter}")
            
        return position_model_counts
    
    def extract_normal_answer(self, text: str, answer_pattern: str) -> str:
        """
        Extract the answer from the text using the answer pattern.
        Like:
        - Answer: 123 -> 123
        - Answer:123 -> 123
        - Final Answer\n\nxxx -> xxx
        if failed, try to parse \\boxed{answer}
        """
        if len(text) <= 10 and 'Answer' not in text and 'box' not in text:
            return text.lstrip()
        
        if text is None:
            return ""
        
        # First, try to match using the provided answer_pattern
        matches = re.findall(answer_pattern, text)
        if matches:
            extracted_answer = matches[-1].strip()
            if extracted_answer.lower().startswith("answer: "):
                extracted_answer = extracted_answer[len("answer:"):].strip().lstrip()
            return extracted_answer
        
        # If no match is found, check for "Final Answer" format
        answer_pattern = answer_pattern.replace("Answer\s*:\s", "Final Answer\s\n+\s")
        final_answer_match = re.search(answer_pattern, text)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # If both patterns fail, try to extract boxed content
        return self.extract_boxed_content(text)
    
    @abstractmethod
    def load_data(self, split: str):
        pass

    @abstractmethod
    def evaluate(self, question: str, answer: str):
        pass

    