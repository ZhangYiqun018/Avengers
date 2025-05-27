import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from datasets import Dataset, disable_progress_bars
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.base_evaluator import BaseEvaluator
from evaluate.HumanEval.execution import check_correctness
from evaluate.HumanEval.utils import imports, sanitize

disable_progress_bars()

DATA_DIR = "data/HumanEval"

PROMPT = """You are an expert Python programmer, and here is your task:
{question}

Your code should pass these tests:
{test}
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class HumanEvalEvaluator(BaseEvaluator):
    def __init__(self, max_workers:int = 8, mode: str="test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "HumanEval"
        self.seed = 42
        self.imports = imports
        
    def load_data(self, split: str):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"HumanEval.jsonl"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        if self.mode == "test":
            # split data into train and test
            logger.warning(f"Split data into train and test for {self.task}")
            split_data = data.train_test_split(test_size=0.3)
            train_data = split_data["train"]
            data = split_data["test"]
        return data
    
    def format_prompt(self, item: Dict):
        # answer key: Answer
        prompt = PROMPT.format(
            question=item["prompt"],
            test=item["test"]
        )
        return {"task_prompt": prompt}
    
    def extract_raw_answer(self, raw_datas: list[str], test: str, entry_point: str) -> list[str]:
        extracted_answer = []
        for data in raw_datas:
            answer = self.extract_code_answer(text=data, test=test, entry_point=entry_point)
            if answer is None:
                answer = ""
            extracted_answer.append(answer)

        return extracted_answer
    
    def extract_code_answer(self, text: str, test: str, entry_point: str) -> str:
        extract_code = sanitize(text)
        code = "\n".join(self.imports) + "\n" + extract_code + "\n" + test + "\n" + f"check({entry_point})"
        
        return code
    
    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        # step 1. get router, get generator
        router_result = router.route(question=data['task_prompt'])
        generator = GeneratorFactory.create_generator(
            experts=router_result, generator_config=generator_config
        ) # type: ignore
        
        # step 2. generate & update token usage
        output: GeneratorOutput = generator.generate(question=data['task_prompt'])
        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        
        # step 3. extract answer
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output, test=data['test'], entry_point=data['entry_point'])
        
        # step 4. TODO: code majority voting
        prediction = full_prediction[0]
        is_correct = check_correctness(task_id = data['task_id'], completion_id=0, solution=prediction, time_out=10)['passed']
        
        return dict(
            index=index,
            query=data['task_prompt'],
            origin_query=data['prompt'],
            prediction=prediction,
            full_prediction=full_prediction,
            raw_output=output.raw_output,
            answer=None,
            is_correct=is_correct,
            model_name=generator.model
        )
    
    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        data = self.load_data(split="test")
        
        counter = 0
        results = []
        pbar = tqdm(total=len(data), desc=f"Evaluating {self.task} ...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.evaluate, index=idx, data=d, router=router, generator_config=generator_config) 
                for idx, d in enumerate(data)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['is_correct']:
                    counter += 1
                pbar.update(1)
        pbar.close()
        
        models = [result['model_name'] for result in results]
        
        model_counts = self.calculate_model_counts(results=results)
        logger.info(model_counts)
        
        acc = counter / len(data)
        end_time = time.time()
        logger.info(f"Task: {self.task}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")
        
        return {
            "performance": acc,
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": model_counts,
            "records": results,
        }
