import os
import re
import time
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from datasets import Dataset, disable_progress_bars, load_dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.base_evaluator import BaseEvaluator
from evaluate.LiveCodeBench.compute_code_generation_metrics import evaluate_generation

disable_progress_bars()

DATA_DIR = "/fs-computility/mabasic/shared/data/LiveCodeBench/code_generation_lite"

PROMPT = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Here is your task:
{question}
""".strip()

FORMATTING_WITHOUT_STARTER_CODE = "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

class LiveCodeBenchEvaluator(BaseEvaluator):
    def __init__(self, max_workers:int = 8, mode: str = "test", split: str = "v2"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = f"LiveCodeBench_{split}"
        self.seed = 42
        self.debug = False
        self.split = split
        logger.info(f"Loading data for {self.task} with split {self.split}")
        
    def load_data(self, split: str):
        assert split in ["v1", "v2", "v3", "v4", "v5", "v6"], f"Invalid split: {split}, {self.task} supports splits: v1, v2, v3, v4, v5, v6"
        
        data = load_dataset(
            DATA_DIR, split="test", version_tag=f"release_{split}", trust_remote_code=True
        )     
        data = data.remove_columns(
            ["platform", "contest_id", "contest_date", "difficulty", "private_test_cases", "question_id"]
        )
        
        data = data.map(lambda x: self.format_prompt(x))
        
        # split data into train and test
        if self.mode == "test":
            split_data = data.train_test_split(test_size=0.3)
            train_data = split_data["train"]
            data = split_data["test"]
            logger.info(f"Train data: {len(train_data)}")
            logger.info(f"Test data: {len(data)}")
        
        return data
    
    def format_prompt(self, item: Dict):
        public_test_cases = json.loads(item["public_test_cases"]) # type: ignore
        fn_name = json.loads(item["metadata"]).get("func_name", None)
        
        input = [t["input"] for t in public_test_cases]
        output = [t["output"] for t in public_test_cases]
        
        input_output = {
            "input_output": json.dumps(
                {
                    "inputs": input,
                    "outputs": output,
                    "fn_name": fn_name
                }
            )
        }
        # answer key: Answer
        prompt = PROMPT.format(
            question=item["question_content"],
        )
        prompt += FORMATTING_WITHOUT_STARTER_CODE
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        
        return {"task_prompt": prompt, "test": input_output}
    
    def extract_raw_answer(self, raw_datas: list[str]) -> list[str]:
        extracted_answer = []
        for data in raw_datas:
            answer = self.extract_code_answer(text=data)
            if answer is None:
                answer = ""
            extracted_answer.append(answer)

        return extracted_answer
    
    def extract_code_answer(self, text: str) -> str:
        outputlines = text.split("\n")
        if not outputlines:  # 处理分割后为空的情况
            return ""
        try:
            # 首先尝试查找 PYTHON] 标记
            indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
            
            # 如果没找到 PYTHON] 标记，则查找 ``` 标记
            if len(indexlines) < 2:
                indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
                
            # 如果找到了至少两个标记
            if len(indexlines) >= 2:
                code = "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
                return code.strip()  # 移除首尾空白字符
                
            return ""  # 如果没有找到足够的标记，返回空字符串
            
        except Exception as e:  # 明确指定异常类型，并记录错误
            print(f"Error extracting code: {str(e)}")  # 或使用proper logging
            return ""
        
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
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output)
        
        # step 4. TODO: code majority voting
        prediction = full_prediction[0]
        results, metadata = evaluate_generation(generations=full_prediction, sample=data['test'], debug=False, timeout=10)
        real_results = results[0]
        
        correct_number = 0
        for result in real_results:
            if result is True:
                correct_number += 1
        strict_score = True if correct_number == len(real_results) else False
        
        if len(real_results) == 0:
            soft_score = 0
        else:
            soft_score = correct_number / len(real_results)

        return dict(
            index=index,
            query=data['task_prompt'],
            origin_query=data['question_content'],
            prediction=prediction,
            full_prediction=full_prediction,
            raw_output=output.raw_output,
            answer=None,
            is_correct=strict_score,
            soft_score=soft_score,
            metadata=metadata,
            model_name=generator.model
        )
    
    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        data = self.load_data(split=self.split)

        counter = 0
        soft_score = 0
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
                soft_score += result['soft_score']
                pbar.update(1)
        pbar.close()
        
        models = [result['model_name'] for result in results]
        
        model_counts = Counter(models)
        logger.info(model_counts)
        
        acc = counter / len(data)
        soft_score = soft_score / len(data)
        end_time = time.time()
        logger.info(f"Task: {self.task}")
        logger.info(f"Pass@1: {acc}")
        logger.info(f"Soft score: {soft_score}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")
        
        return {
            "performance": acc,
            "soft_score": soft_score,
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": model_counts,
            "records": results,
        }
