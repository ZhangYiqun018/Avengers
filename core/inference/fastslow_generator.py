from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from loguru import logger
from openai import NOT_GIVEN, OpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput


class FastSlowGenerator(BaseGenerator):
    def __init__(self, fast_expert: Expert, slow_expert: Expert, generator_config: dict):
        self.name = self.__class__.__name__
        # get clients and models from fast and slow experts
        self.fast_expert = fast_expert
        self.slow_expert = slow_expert
        self.fast_client = fast_expert.client
        self.fast_model = fast_expert.model_name
        self.slow_client = slow_expert.client
        self.slow_model = slow_expert.model_name
        self.model = [self.fast_model]
        
        # get fast and slow config
        self.config = generator_config
        self.fast_samples = self.config.get("fast_samples", 10)
        self.fast_temperature = self.config.get("fast_temperature", 0.7)
        self.fast_top_p = self.config.get("fast_top_p", 1.0)
        self.slow_samples = self.config.get("slow_samples", 1)
        self.slow_temperature = self.config.get("slow_temperature", 0.7)
        self.slow_top_p = self.config.get("slow_top_p", 1.0)
        self.consistency_rate_threshold = self.config.get("consistency_rate_threshold", 0.8)
        
        self.fast_max_retries = 20
        self.slow_max_retries = 3
        # define final results
        self.fast_results = None
        self.slow_results = None
        self.final_results = None
    
    def _generate_with_retry(self, client: OpenAI, model: str, question: str, mode: str = "fast") -> GeneratorOutput:
        max_retries = self.fast_max_retries if mode == "fast" else self.slow_max_retries
        
        def _log_retry(retry_state):
            exception = retry_state.outcome.exception()
            if exception:
                logger.warning(f"Retrying FastSlowGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
            return None
        
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type((Exception)),
            before_sleep=_log_retry
        )
        def _generate_impl():
            temperature = self.fast_temperature if mode == "fast" else self.slow_temperature
            top_p = self.fast_top_p if mode == "fast" else self.slow_top_p
            samples = self.fast_samples if mode == "fast" else self.slow_samples
            timeout = 2_000 if mode == "fast" else 200_000
            if mode == "slow":
                slow_prompt = "\nDon't make your reasoning and thinking too long.\n"
                question_with_prompt = question + slow_prompt
            else:
                question_with_prompt = question
                
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question_with_prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    n=samples,
                    timeout=timeout,
                )
                choices = response.choices
                usage = response.usage
                raw_output = [choice.message.content for choice in choices]
                assert len(raw_output) == samples, f"Mode={mode}, Expected {samples} samples, got {len(raw_output)}"
                
                return GeneratorOutput(
                    first_output=choices[0].message.content,
                    raw_output=raw_output,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
            except Exception as e:
                logger.warning(f"Error in FastSlowGenerator._generate: {str(e)}, error model: {model}, mode: {mode}")
                raise
                
        return _generate_impl()

    def generate(self, question: str) -> GeneratorOutput:
        self.fast_results = self._generate_with_retry(self.fast_client, self.fast_model, question, "fast")
        return self.fast_results
    
    def slow_generate(self, question: str) -> GeneratorOutput:
        try:
            self.slow_results = self._generate_with_retry(self.slow_client, self.slow_model, question, "slow")
            self.final_results = GeneratorOutput(
                first_output = self.slow_results.first_output,
                raw_output = self.slow_results.raw_output,
                prompt_tokens = self.slow_results.prompt_tokens + self.fast_results.prompt_tokens,
                completion_tokens = self.slow_results.completion_tokens + self.fast_results.completion_tokens
            )
            self.model = [self.slow_model]
            return self.final_results
        except Exception as e:
            logger.warning(f"Slow model generation failed after all retries: {str(e)}. Falling back to fast model results.")
            return self.fast_results