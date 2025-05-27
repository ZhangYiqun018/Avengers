import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from datasets import Dataset, disable_progress_bars, load_dataset
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.base_evaluator import BaseEvaluator
from evaluate.K_and_K.scoring import parse_cot_eval, parse_answer, judge_answer, ensemble_answers

disable_progress_bars()

DATA_DIR = "data/K_and_K"

PROMPT = """Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...

### Question:
{question}
"""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-D])[.\s\n]?"

class KnightsAndKnavesEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int=8, mode: str="test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "Knights_and_Knaves"
        self.seed = 42
    
    def load_data(self, split: str):
        assert split in ["train", "test"]
        data = load_dataset(DATA_DIR, split=split)
        
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
        # format answer 
        answer = item["solution_text_format"].split("\n")
        answer = [a[3:].lstrip().strip() for a in answer if a.strip()]
        
        prompt = PROMPT.format(
            question = item["quiz"],
        )
        return {"prompt": prompt, "answer": answer}
    
    def extract_raw_answer(self, raw_datas: list[str]) -> list[str]:
        full_prediction = []
        for data in raw_datas:
            parsed_answer, is_success = parse_answer(pred_str=data)
            full_prediction.append(parsed_answer)
        return full_prediction
    
    def get_ensemble_answer(self, raw_datas: list[str]) -> tuple[str, dict]:
        prediction, prediction_states = ensemble_answers(raw_datas)
        return prediction, prediction_states
    
    def process_output(self, output: GeneratorOutput):
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output)
        prediction, prediction_stats = self.get_ensemble_answer(raw_datas=full_prediction)
        return prediction, full_prediction, prediction_stats
    
    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        answer = data["solution_text_format"].split("\n")
        answer = [a[3:].lstrip().strip() for a in answer if a.strip()]
        
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
                if prediction == "" or prediction is None:
                    logger.warning(f"[K&K] slow_output is empty, use fast_output to replace.")
                    prediction, full_prediction, prediction_stats = self.process_output(output=output)
                else:
                    output = slow_output
        else:
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            prediction, full_prediction, prediction_stats = self.process_output(output=output)
            
        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        
        # step 3. majority voting & records
        is_correct, wrong_reason, correct_ratio = judge_answer(
            pred_answer=prediction, reformat_gold_conditions=answer
        )
        return dict(
            index=index,
            query=data['prompt'],
            origin_query=data['quiz'],
            prediction=prediction,
            full_prediction=full_prediction,
            prediction_stats=prediction_stats,
            raw_output=output.raw_output,
            answer=answer,
            is_correct=is_correct,
            correct_ratio=correct_ratio,
            wrong_reason=wrong_reason,
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

        soft_score = sum([result['correct_ratio'] for result in results]) / len(results)
        acc = counter / len(data)
        end_time = time.time()
        
        logger.info(f"Task: {self.task}")
        logger.info(f"Strict Score: {acc}")
        logger.info(f"Soft Score: {soft_score}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")
        
        return {
            "performance": acc,
            "soft_performance": soft_score,
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": model_counts,
            "records": results,
        }