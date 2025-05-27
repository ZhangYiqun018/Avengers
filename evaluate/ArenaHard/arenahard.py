import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from datasets import Dataset, disable_progress_bars
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/arena-hard"

PROMPT_FOUR_OPTIONS = """Answer the following question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{question}

A) {A}
B) {B}
C) {C}
D) {D}

Let's think step by step."""

class ArenaHardEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 8, mode: str="test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "ArenaHard"
        self.seed = 42
    
    def load_data(self, split="v2"):
        if split == "v2":
            data = self.load_jsonl(os.path.join(DATA_DIR, f"arena-hard-v2.jsonl"))
        else:
            raise ValueError(f"Invalid split: {split}")
        
        data = Dataset.from_list(data)
        logger.info(data)
        
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        pass
    
    def extract_raw_answer(self, raw_datas: list[str]) -> list[str]:
        pass
    
    def process_output(self, question: str, output: GeneratorOutput):
        prediction = output.raw_output[0]
        return [
            {"role": "user", "content": question},
            # Add an 'answer' key to meet ArenaHard's specific requirements
            {"role": "assistant", "content": {"answer": prediction}}
        ]
    
    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        # step 1. get router, get generator
        uid = data['uid']
        if uid != "6c69551e80664df5":
            router_result = router.route(question=data['prompt'])
            generator = GeneratorFactory.create_generator(
                experts=router_result, generator_config=generator_config
            ) # type: ignore

            output: GeneratorOutput = generator.generate(question=data['prompt'])
            model = generator.model
        else:
            logger.warning(f"Context length exceeded, unable to answer this question, uid: {uid}")
            output = GeneratorOutput(
                first_output="Context length exceeded, unable to answer this question",
                raw_output=["Context length exceeded, unable to answer this question"],
                prompt_tokens=0,
                completion_tokens=0
            )
            model = "None"
            
        messages = self.process_output(question=data['prompt'], output=output)
        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        
        return dict(
            uid=data['uid'],
            category=data['category'],
            subcategory=data['subcategory'],
            language=data['language'],
            ans_id=index,
            messages=messages,
            model="avengers",
            model_name=model,
            tstamp=time.time()
        )
    
    def save_records(self, records: list[dict]):
        # save records for evluate on ArenaHard official leaderboard
        # save to data/arena-hard/avengers-{timestamp}.jsonl
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(DATA_DIR, f"avengers-{timestamp}.jsonl"), "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    
    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        data = self.load_data(split="v2")
        
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
                pbar.update(1)
        pbar.close()
        
        model_counts = self.calculate_model_counts(results=results)
        
        end_time = time.time()
        logger.info(f"Task: {self.task}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")
        
        self.save_records(records=results)
        return {
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": model_counts,
            "records": results,
        }