from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from loguru import logger
from openai import NOT_GIVEN, OpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput


class ModelSwitchGenerator(BaseGenerator):
    def __init__(self, experts: List[Expert], generator_config: dict):
        self.name = self.__class__.__name__
        assert len(experts) > 1, "ModelSwitchGenerator requires at least two experts."
        # get clients and models from experts
        self.experts = experts
        self.first_client = experts[0].client
        self.first_model = experts[0].model_name
        self.second_client = experts[1].client
        self.second_model = experts[1].model_name
        self.model = [self.first_model, self.second_model]
        # get config
        self.config = generator_config
        self.samples = self.config.get("samples", 5)
        self.temperature = self.config.get("temperature", 0.7)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", NOT_GIVEN)
        self.consistency_rate_threshold = self.config.get("consistency_rate_threshold", 0.8)
        # define final results
        self.first_results = None
        self.second_results = None
        self.final_results = None
        
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying ModelSwitchGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None
    
    @retry(
        stop=stop_after_attempt(10),  # 最多重试10次
        wait=wait_exponential(multiplier=1, min=2, max=100),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def _generate(self, client: OpenAI, model: str, question: str) -> GeneratorOutput:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.samples,
                timeout=1000,
            )
            choices = response.choices
            usage = response.usage
            raw_output = [choice.message.content for choice in choices]
            assert len(raw_output) == self.samples, f"Expected {self.samples} samples, got {len(raw_output)}"
            
            return GeneratorOutput(
                first_output=choices[0].message.content,
                raw_output=raw_output,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens
            )
        except Exception as e:
            logger.error(f"Error in ModelSwitchGenerator._generate: {str(e)}, error model: {model}")
            raise  # 重新抛出异常，让重试装饰器捕获

    def generate(self, question: str) -> tuple[GeneratorOutput, GeneratorOutput]:
        results = dict()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_model = {
                executor.submit(self._generate, self.first_client, self.first_model, question): self.first_model,
                executor.submit(self._generate, self.second_client, self.second_model, question): self.second_model
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.error(f"Error in ModelSwitchGenerator.generate: {str(e)}, error model: {model}")
                    results[model] = None
        
        self.first_results = results.get(self.first_model, None)
        self.second_results = results.get(self.second_model, None)
        self.final_results = self.get_second_output()
        
        return self.first_results, self.final_results
    
    def get_second_output(self) -> GeneratorOutput:
        if self.first_results is None and self.second_results is None:
            raise ValueError("first_results and second_results is None")
        if self.first_results is None:
            return self.second_results
        if self.second_results is None:
            return self.first_results
        
        final_results = GeneratorOutput(
            first_output = self.second_results.first_output,
            raw_output = self.first_results.raw_output + self.second_results.raw_output,
            prompt_tokens = self.first_results.prompt_tokens + self.second_results.prompt_tokens,
            completion_tokens = self.first_results.completion_tokens + self.second_results.completion_tokens
        )
        
        return final_results