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
from evaluate.deepscaler_rm import (extract_answer, grade_answer_mathd,
                                    grade_answer_sympy)

disable_progress_bars()

DATA_DIR = "data/MathBench"

CLOZE_PROMPT = """Solve the following math problem step by step. The last line of your response should only contain your final answer inside a \\boxed{} command.

{question}

Remember to put your final answer on the last line using the format \\boxed{$ANSWER} where $ANSWER is the answer to the problem.
""".strip()

SINGLE_CHOICE_PROMPT = """Answer the following question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{question}

A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Let's think step by step.
""".strip()

CLOZE_ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

SINGLE_CHOICE_ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-D])[.\s\n]?"

class MathBenchEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 8, mode: str="test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "MATHBENCH"
        self.seed = 42
    
    def load_data(self, split: str):
        # single choice
        single_choice_data = self.load_jsonl(os.path.join(DATA_DIR, f"college.jsonl"))
        single_choice_data = Dataset.from_list(single_choice_data)
        single_choice_data = single_choice_data.map(lambda x: self.format_prompt(x, type="single_choice"))
        
        if self.mode == "test":
            # split data into train and test
            logger.warning(f"Split data into train and test for {self.task}")
            split_single_choice_data = single_choice_data.train_test_split(test_size=0.3)
            
            single_choice_data = split_single_choice_data["test"]
            
        return single_choice_data
    
    def format_prompt(self, item: Dict, type: str):
        if type == "cloze":
            # [question: str, answer: str]
            prompt = CLOZE_PROMPT.replace("{question}", item["question"])
        elif type == "single_choice":
            # [question: str, options: list[str]]
            prompt = SINGLE_CHOICE_PROMPT.replace("{question}", item["question"]).replace("{option_a}", item["options"][0]).replace("{option_b}", item["options"][1]).replace("{option_c}", item["options"][2]).replace("{option_d}", item["options"][3])
        else:
            raise ValueError(f"Invalid type: {type}")
        
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_datas: list[str]) -> list[str]:
        extracted_answer = []
        for data in raw_datas:
            if "Final Answer" in data and "\\boxed" not in data:
                answer = self.extract_normal_answer(text=data, answer_pattern=CLOZE_ANSWER_PATTERN)
            else:
                answer = extract_answer(passage=data)
            if answer is None:
                answer = ""
            extracted_answer.append(answer)

        return extracted_answer
    
    def extract_single_choice_raw_answer(self, raw_datas: list[str]) -> list[str]:
        return [
            self.extract_normal_answer(text=data, answer_pattern=SINGLE_CHOICE_ANSWER_PATTERN) 
            for data in raw_datas
        ]
    
    def process_single_choice_output(self, output: GeneratorOutput):
        full_prediction = self.extract_single_choice_raw_answer(raw_datas=output.raw_output)
        prediction = Counter(full_prediction).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=full_prediction)
        
        return prediction, full_prediction, prediction_stats
    
    def process_output(self, output: GeneratorOutput):
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output)
        prediction = Counter(full_prediction).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=full_prediction)
        
        return prediction, full_prediction, prediction_stats
    
    def evaluate_cloze(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        answer = data['answer']
        
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
        
        if "\\boxed" in answer:
            ground_truth = extract_answer(passage=answer)
        else:
            ground_truth = answer
        
        if ground_truth is None or prediction is None or prediction == "":
            is_correct = False
        else:
            is_correct = grade_answer_mathd(prediction, ground_truth) \
                or grade_answer_sympy(prediction, ground_truth)
        
        return dict(
            index=index,
            query=data['prompt'],
            origin_query=data['question'],
            prediction=prediction,
            full_prediction=full_prediction,
            prediction_stats=prediction_stats,
            raw_output=output.raw_output,
            answer=ground_truth,
            is_correct=is_correct,
            model_name=generator.model
        )
    
    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        pass
    
    def evaluate_single_choice(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
        answer = data['answer']
        
        # step 1. get router, get generator
        router_result = router.route(question=data['prompt'])
        generator = GeneratorFactory.create_generator(
            experts=router_result, generator_config=generator_config
        ) # type: ignore
        
        # step 2. generate & update token usage
        if generator_config['type'] == 'model_switch':
            output: tuple[GeneratorOutput, GeneratorOutput] = generator.generate(question=data['prompt'])
            first_output, final_output = output
            prediction, full_prediction, prediction_stats = self.process_single_choice_output(output=first_output)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                prediction, full_prediction, prediction_stats = self.process_single_choice_output(output=final_output)
                output = final_output
            else:
                output = first_output
        elif generator_config['type'] == 'fast_slow':
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            prediction, full_prediction, prediction_stats = self.process_single_choice_output(output=output)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                slow_output = generator.slow_generate(question=data['prompt'])
                prediction, full_prediction, prediction_stats = self.process_single_choice_output(output=slow_output)
                if prediction == "":
                    logger.warning(f"slow_output is empty, use fast_output to replace.")
                    prediction, full_prediction, prediction_stats = self.process_single_choice_output(output=output)
                else:
                    output = slow_output
        else:
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            prediction, full_prediction, prediction_stats = self.process_single_choice_output(output=output)

        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        
        is_correct = answer == prediction
        
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
            model_name=generator.model
        )
    
    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        single_choice_data = self.load_data(split="test")
        results = []
        
        logger.info(single_choice_data)
        single_choice_counter = 0
        pbar = tqdm(total=len(single_choice_data), desc=f"Evaluating {self.task}: [single choice, college] ...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.evaluate_single_choice, index=idx, data=d, router=router, generator_config=generator_config) 
                for idx, d in enumerate(single_choice_data)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['is_correct']:
                    single_choice_counter += 1
                pbar.update(1)
        pbar.close()
        
        end_time = time.time()
        
        single_choice_acc = single_choice_counter / len(single_choice_data)
        position_model_counts = self.calculate_model_counts(results=results)
        
        acc = single_choice_acc
        logger.info(f"Task: {self.task}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"Position Model Counts: {position_model_counts}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")
        
        return {
            "performance": acc,
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": position_model_counts,
            "records": results,
        }