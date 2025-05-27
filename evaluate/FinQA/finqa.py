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


DATA_DIR = "data/FinQA"

# PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

# {Question}

# Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
# """.strip()

PROMPT = """Solve the following problem step by step. The last line of your response should only contain your final answer inside a \\boxed{} command.

{Question}

Remember to put your final answer on the last line using the format \\boxed{$ANSWER} where $ANSWER is the answer to the problem.
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class FinQAEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 8, mode: str="test"):
        super().__init__(max_workers=max_workers, mode=mode)
        self.task = "FinQA"
        self.seed = 42
    
    def load_data(self, split: str):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        
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
    
    def list_to_simple_table(self, table_data):
        if not table_data or not isinstance(table_data, list):
            return "Invalid table data"
        
        # Initialize the table string
        table_str = ""
        
        # Process each row
        for row in table_data:
            # Convert each cell in the row to string and strip whitespace
            row_str = [str(cell).strip() for cell in row]
            
            # Join cells with pipe separators and add to table
            table_str += "| " + " | ".join(row_str) + " |\n"
        
        return table_str
    
    def format_prompt(self, item: Dict):
        # make question
        pre_text = ''.join(item['pre_text'])
        table = item['table']
        table_str = self.list_to_simple_table(table)
        
        post_text = ''.join(item['post_text'])  
        question = item['question']
        Question = f"{pre_text}\n\nHere is the table:\n{table_str}\n\n{post_text}\n\nBased on the content and the table, please answer the following question: {question}\n"
        # answer key: answer
        prompt = PROMPT.replace("{Question}", Question)
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_datas: list[str]) -> list[str]:
        extracted_answer = []
        for data in raw_datas:
            answer = extract_answer(passage=data)
            if answer is None:
                answer = ""
            extracted_answer.append(answer)

        return extracted_answer
    
    def extract_number(self, text: str):
        pattern = r'-?\d+(?:\.\d+)?'

        match = re.search(pattern, text)
        if match:
            number = float(match.group(0))
            number = round(number, 0)
            return str(number)
        else:
            return text
        
    def process_output(self, output: GeneratorOutput):
        full_prediction = self.extract_raw_answer(raw_datas=output.raw_output)
        prediction = Counter(full_prediction).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=full_prediction)
        
        return prediction, full_prediction, prediction_stats
    
    def evaluate(self, index: int, data: dict, router: BaseRouter, generator_config: dict):
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
        
        # finqa has the number of decimal places, so we need to check if the prediction is close to the ground truth
        if not is_correct:
            formatted_prediction = self.extract_number(prediction)
            formatted_ground_truth = self.extract_number(ground_truth)
            if formatted_prediction == formatted_ground_truth:
                is_correct = True
            else:
                is_correct = False
        
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
