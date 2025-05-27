import os
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
from openai import OpenAI

from datasets import Dataset, disable_progress_bars
from loguru import logger
from tqdm import tqdm

from core.inference import GeneratorFactory, GeneratorOutput
from core.routing import BaseRouter
from evaluate.SimpleQA.prompts import GRADER_TEMPLATE
from evaluate.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/SimpleQA"

PROMPT = """Answer the following question. 
Question: {question}

If you are certain, answer with the fact only.
If you are uncertain, reply exactly: "NOT_ATTEMPTED
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-D])[.\s\n]?"

class SimpleQAEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 8):
        super().__init__(max_workers=max_workers)
        self.task = "SimpleQA"
        self.seed = 42
        self.grader = OpenAI(
            api_key=os.getenv("GRADER_API_KEY", "123"), 
            base_url=os.getenv("GRADER_BASE_URL", "http://172.30.4.29:8000/v1")
        )
        self.grader_model = os.getenv("GRADER_MODEL_NAME", "Qwen2.5-72B-Instruct")
        
    def load_data(self, split: str):
        with open(os.path.join(DATA_DIR, f"simple_qa_eval.json"), 'r') as f:
            data = json.load(f)
            f.close()
            
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            question = item["problem"],
        )
        return {"prompt": prompt}
    
    
    def _get_grader_result(self, question: str, answer: str, predicted_answer: str) -> str:
        prompt = GRADER_TEMPLATE.format(
            question = question,
            target = answer,
            predicted_answer = predicted_answer
        )
        response = self.grader.chat.completions.create(
            model=self.grader_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            top_p=1.0
        )
        response = response.choices[0].message.content
        match = re.search(r"(A|B|C)", response)
        result = match.group(0) if match else "C"
        
        return result

    def get_grader_result(self, question: str, answer: str, raw_datas: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=len(raw_datas)) as executor:
            futures = [
                executor.submit(self._get_grader_result, question=question, answer=answer, predicted_answer=data)
                for data in raw_datas
            ]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        return results
    
    def process_output(self, output: list[str]):
        full_prediction = output
        prediction = Counter(output).most_common(1)[0][0]
        prediction_stats = self.count_prediction_frequency(predictions=output)
        
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
            # get grader result
            first_grader_output = self.get_grader_result(question=data['prompt'], answer=answer, raw_datas=first_output.raw_output)
            prediction, full_prediction, prediction_stats = self.process_output(output=first_grader_output)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                final_grader_output = self.get_grader_result(question=data['prompt'], answer=answer, raw_datas=final_output.raw_output)
                prediction, full_prediction, prediction_stats = self.process_output(output=final_grader_output)
                output = final_output
            else:
                output = first_output
        elif generator_config['type'] == 'fast_slow':
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            grader_output = self.get_grader_result(question=data['prompt'], answer=answer, raw_datas=output.raw_output)
            prediction, full_prediction, prediction_stats = self.process_output(output=grader_output)
            consistency_rate = prediction_stats[prediction]['frequency']
            if consistency_rate < generator.consistency_rate_threshold:
                slow_output = generator.slow_generate(question=data['prompt'])
                slow_grader_output = self.get_grader_result(question=data['prompt'], answer=answer, raw_datas=slow_output.raw_output)
                prediction, full_prediction, prediction_stats = self.process_output(output=slow_grader_output)
                if prediction == "":
                    logger.warning(f"slow_output is empty, use fast_output to replace.")
                    prediction, full_prediction, prediction_stats = self.process_output(output=grader_output)
                else:
                    output = slow_output
        else:
            output: GeneratorOutput = generator.generate(question=data['prompt'])
            grader_output = self.get_grader_result(question=data['prompt'], answer=answer, raw_datas=output.raw_output)
            prediction, full_prediction, prediction_stats = self.process_output(output=grader_output)

        self.update_tokens(prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens)
        
        is_correct = prediction == "A"
        is_incorrect = prediction == "B"
        is_not_attempted = prediction == "C"
        
        return dict(
            index=index,
            query=data['prompt'],
            origin_query=data['problem'],
            prediction=prediction,
            full_prediction=full_prediction,
            prediction_stats=prediction_stats,
            raw_output=output.raw_output,
            answer=answer,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            is_not_attempted=is_not_attempted,
            model_name=generator.model
        )
    
    def evaluate_loop(self, router: BaseRouter, generator_config: dict):
        start_time = time.time()
        data = self.load_data(split="test").select(range(100))
        logger.info("Dataset stats:")
        logger.info(data)
        logger.info("Dataset example:")
        logger.info(data[0]['prompt'])
        logger.info("=" * 100)
        
        correct_counter = 0
        incorrect_counter = 0
        not_attempted_counter = 0
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
                    correct_counter += 1
                elif result['is_incorrect']:
                    incorrect_counter += 1
                elif result['is_not_attempted']:
                    not_attempted_counter += 1
                pbar.update(1)
        pbar.close()
        
        model_counts = self.calculate_model_counts(results=results)
        
        # calculate metrics
        is_correct = correct_counter / len(data)
        is_incorrect = incorrect_counter / len(data)
        is_not_attempted = not_attempted_counter / len(data)
        is_given_attempted = is_correct + is_incorrect
        accuracy_given_attempted = is_correct / is_given_attempted if is_given_attempted > 0 else 0
        denominator = accuracy_given_attempted + is_correct
        
        f1 = 2*accuracy_given_attempted*is_correct / denominator if denominator > 0 else 0
        acc = incorrect_counter / (incorrect_counter + correct_counter)
        
        end_time = time.time()
        logger.info(f"Task: {self.task}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"F1: {f1}")
        logger.info(f"Accuracy given attempted: {accuracy_given_attempted}")
        logger.info(f"Accuracy not attempted: {is_not_attempted}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        logger.info(f"Prompt tokens: {self.prompt_tokens}")
        logger.info(f"Completion tokens: {self.completion_tokens}")
        
        return {
            "performance": dict(
                accuracy=acc,
                f1=f1,
                accuracy_given_attempted=accuracy_given_attempted,
                accuracy_not_attempted=is_not_attempted,
                correct_counter=correct_counter,
                incorrect_counter=incorrect_counter,
                not_attempted_counter=not_attempted_counter,
            ),
            "time_taken": end_time - start_time,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_counts": model_counts,
            "records": results,
        }