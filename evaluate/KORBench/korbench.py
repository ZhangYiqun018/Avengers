import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from datasets import Dataset, disable_progress_bars, concatenate_datasets
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.base_evaluator import BaseEvaluator
import yaml
import json
from evaluate.KORBench.eval_utils import evaluate_response_vs_answer, extract_single_answer

disable_progress_bars()

DATA_DIR = "data/KORBench"

def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError("No JSON or JSONL file found.")
    
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]
    
    if mapping_key:
        return {item[mapping_key]: item for item in data if mapping_key in item}
    else:
        return data
    
class KORBenchEvaluator(BaseEvaluator):
    def __init__(self, split: str = "full", max_workers: int=8, mode: str="test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = f"KORBench-{split}"
        self.seed = 42
        self.split = split
        self.max_workers = max_workers
    
    def load_yaml(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, split: str=None):
        task_list = ['cipher', 'operation', 'puzzle', 'counterfactual', 'logic']
        dataset = []
        for task in task_list:
            dataset.append(self.load_single_data(task=task))
        return concatenate_datasets(dataset)
    
    def load_single_data(self, task: str):
        if task in ['cipher', 'operation', 'puzzle', 'counterfactual', 'logic']:
            sample = self.load_jsonl(os.path.join(DATA_DIR, task, "sample.jsonl"))
            
            few_shot = read_json_or_jsonl(data_path=os.path.join(DATA_DIR, task), split='three-shot')
            rule = read_json_or_jsonl(data_path=os.path.join(DATA_DIR, task), split="rule", mapping_key="idx")
        else:
            raise ValueError(f"Invalid task: {task}")
        
        template = self.load_yaml(os.path.join(DATA_DIR, "three-shot.yaml"))
        
        data = Dataset.from_list(sample)
        
        data = data.map(
            lambda x: self.format_prompt(item=x, task=task, template=template, rule=rule, few_shot=few_shot)
        )
        logger.info(f"Loaded {task} data: {len(data)}")
        
        if self.mode == "test":
            # split data into train and test
            logger.warning(f"Split data into train and test for {self.task}")
            split_data = data.train_test_split(test_size=0.3)
            train_data = split_data["train"]
            data = split_data["test"]
            logger.info(f"Calibration data: {len(train_data)}")
            logger.info(f"Test data: {len(data)}")
            
        return data
    
    def format_prompt(self, item, task, template, rule, few_shot):
        rule_id = item['rule_id']
        rule_content = rule[rule_id]['rule_content']
        question = item['question']
        
        few_shot_qa = [
            i for fs in few_shot if fs['rule_id'] == rule_id for i in [fs['question'], fs['answer']]
        ]
        prompt_format = [rule_content, *few_shot_qa, question]
        prompt = template[f'{task}_prompt_format'][0].format(*prompt_format)
        
        return {"prompt": prompt, "question_type": task}
    
    def extract_raw_answer(self, raw_datas: list[str], question_type: str, rule_id: str, idx: str) -> list[str]:
        return [
            extract_single_answer(response=data, question_type=question_type, rule_id=rule_id, idx=idx) 
            for data in raw_datas
        ]
    
    def process_output(self, output: GeneratorOutput, question_type: str, rule_id: str, idx: str):
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output, question_type=question_type, rule_id=rule_id, idx=idx)
        prediction = Counter(full_prediction).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=full_prediction)
        return prediction, full_prediction, prediction_stats
    
    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        # prepare eval data
        answer = data['answer']
        question_type = data['question_type']
        rule_id = data['rule_id']
        idx = data['idx']
        
        # step 1. get router, get generator
        router_result = router.route(question=data['prompt'])
        generator = GeneratorFactory.create_generator(
            experts=router_result, generator_config=generator_config
        ) # type: ignore
        
        # step 2. generate & update token usage
        if generator_config['type'] == 'model_switch':
            output: tuple[GeneratorOutput, GeneratorOutput] = generator.generate(question=data['prompt'])
            first_output, final_output = output
            prediction, full_prediction, prediction_stats = self.process_output(output=first_output, question_type=question_type, rule_id=rule_id, idx=idx)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                prediction, full_prediction, prediction_stats = self.process_output(output=final_output, question_type=question_type, rule_id=rule_id, idx=idx)
                output = final_output
            else:
                output = first_output
        elif generator_config['type'] == 'fast_slow':
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            prediction, full_prediction, prediction_stats = self.process_output(output=output, question_type=question_type, rule_id=rule_id, idx=idx)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                slow_output = generator.slow_generate(question=data['prompt'])
                prediction, full_prediction, prediction_stats = self.process_output(output=slow_output, question_type=question_type, rule_id=rule_id, idx=idx)
                if prediction == "":
                    logger.warning(f"slow_output is empty, use fast_output to replace.")
                    prediction, full_prediction, prediction_stats = self.process_output(output=output, question_type=question_type, rule_id=rule_id, idx=idx)
                else:
                    output = slow_output
        else:
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            prediction, full_prediction, prediction_stats = self.process_output(output=output, question_type=question_type, rule_id=rule_id, idx=idx)

        # step 3. update token usage
        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        # step 4. calculate is_correct
        is_correct = evaluate_response_vs_answer(response=prediction, answer=answer, question_type=question_type, rule_id=rule_id, idx=idx)
        
        return dict(
            index=index,
            query=data['prompt'],
            origin_query=data['question'],
            prediction=prediction,
            full_prediction=full_prediction,
            prediction_stats=prediction_stats,
            raw_output=output.raw_output,
            answer=answer,
            is_correct=is_correct,
            question_type=question_type,
            model_name=generator.model
        )
    
    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        data = self.load_data(split=self.split)
        counter = 0
        counter_map = dict()
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
                question_type = result['question_type']
                if question_type not in counter_map:
                    counter_map[question_type] = {}
                if result['is_correct']:
                    counter_map[question_type] = {
                        "correct": counter_map[question_type].get("correct", 0) + 1,
                        "total": counter_map[question_type].get("total", 0) + 1
                    }
                    counter += 1
                else:
                    counter_map[question_type] = {
                        "correct": counter_map[question_type].get("correct", 0),
                        "total": counter_map[question_type].get("total", 0) + 1
                    }
                pbar.update(1)
        pbar.close()
        
        # calculate accuracy for each question type
        acc_map = {question_type: counter_map[question_type]['correct'] / counter_map[question_type]['total'] for question_type in counter_map}
        
        model_counts = self.calculate_model_counts(results=results)
        
        acc = counter / len(data)
        end_time = time.time()
        logger.info(f"Task: {self.task}")
        logger.info(f"Total Accuracy: {acc}")
        for question_type in counter_map:
            logger.info(f"{question_type} Accuracy: {acc_map[question_type]}")
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