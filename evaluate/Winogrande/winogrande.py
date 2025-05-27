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

DATA_DIR = "data/winogrande"

PROMPT = """You are given a *cloze* sentence containing a blank marked by underscores `___`. Only **one** of the two options correctly fills the blank while preserving commonsense.

After your reasoning, end with a line **exactly** in the form:  `Answer: $LETTER`, where `LETTER` is **A** or **B**.

Cloze Sentence:
{sentence}

Options:
A. {option1}
B. {option2}

Let's think step by step."""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-B])[.\s\n]?"

class WinograndeEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 8, mode: str="test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "Winogrande"
        self.seed = 42
    
    def load_data(self, split: str):
        if split == "train" or split == "test":
            logger.warning(f"Winogrande does not have {split} split, only valid split is available now.")
            split = "valid"
        
        data = self.load_jsonl(os.path.join(DATA_DIR, f"valid.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        if self.mode == "test":
            # split data into train and test
            logger.warning(f"Split data into train and test for {self.task}")
            split_data = data.train_test_split(test_size=0.3)
            train_data = split_data["train"]
            data = split_data["test"]
            logger.info(f"Calibration data: {len(train_data)}")
            logger.info(f"Test data: {len(data)}")
            
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        answer = item['answer']
        if answer == "2":
            project_answer = "B"
        elif answer == "1":
            project_answer = "A"
        else:
            raise ValueError(f"Invalid answer: {answer}")
        
        prompt = PROMPT.format(
            sentence = item["sentence"],
            option1 = item["option1"],
            option2 = item["option2"]
        )
        return {"prompt": prompt, "project_answer": project_answer}
    
    def extract_raw_answer(self, raw_datas: list[str]) -> list[str]:
        return [
            self.extract_normal_answer(text=data, answer_pattern=ANSWER_PATTERN) 
            for data in raw_datas
        ]
    
    def process_output(self, output: GeneratorOutput):
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output)
        prediction = Counter(full_prediction).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=full_prediction)
        
        return prediction, full_prediction, prediction_stats
    
    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        answer = data['project_answer']
        
        # step 1. get router, get generator
        router_result = router.route(question=data['prompt'])
        generator = GeneratorFactory.create_generator(
            experts=router_result, generator_config=generator_config
        ) # type: ignore
        
        # step 2. generate & update token usage
        if generator_config['type'] == 'model_switch':
            output: tuple[GeneratorOutput, GeneratorOutput] = generator.generate(question=data['prompt'])
            first_output, final_output = output
            prediction, full_prediction, prediction_stats = self.process_output(output=first_output)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                prediction, full_prediction, prediction_stats = self.process_output(output=final_output)
                output = final_output
            else:
                output = first_output
        elif generator_config['type'] == 'fast_slow':
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            prediction, full_prediction, prediction_stats = self.process_output(output=output)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                slow_output = generator.slow_generate(question=data['prompt'])
                prediction, full_prediction, prediction_stats = self.process_output(output=slow_output)
                if prediction == "":
                    logger.warning(f"slow_output is empty, use fast_output to replace.")
                    prediction, full_prediction, prediction_stats = self.process_output(output=output)
                else:
                    output = slow_output
        else:
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            prediction, full_prediction, prediction_stats = self.process_output(output=output)

        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        
        is_correct = answer == prediction
        
        return dict(
            index=index,
            query=data['prompt'],
            origin_query=data['sentence'] + " " + data['option1'] + " " + data['option2'],
            prediction=prediction,
            full_prediction=full_prediction,
            prediction_stats=prediction_stats,
            raw_output=output.raw_output,
            answer=answer,
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
        
        model_counts = self.calculate_model_counts(results=results)
        
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