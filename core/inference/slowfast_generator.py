from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from loguru import logger
from openai import NOT_GIVEN, OpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput


SLOW_PROMPT="""Your goal is to provide a clear and detailed approach (within 2000 tokens) for solving the given question.  

Question: {question}

Provide your step-by-step reasoning strategy, clearly outlining:
1. How you interpret the question and its key components.
2. What background knowledge or concepts are relevant.
3. How you will structure your solution approach.

Note: Do NOT solve the question itself; only explain your problem-solving thought process clearly and concisely."""


FAST_PROMPT="""You are an agent tasked with solving the provided question step-by-step.

Question: {question}

Reasoning strategy (may be incomplete): {reasoning_steps}

Now, answer the question step by step."""

class SlowFastGenerator(BaseGenerator):
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
        ## fast config
        self.fast_samples = self.config.get("fast_samples", 10)
        self.fast_temperature = self.config.get("fast_temperature", 0.2)
        self.fast_top_p = self.config.get("fast_top_p", 1.0)
        ## slow config
        self.slow_samples = self.config.get("slow_samples", 1)
        self.slow_temperature = self.config.get("slow_temperature", 0.7)
        self.slow_top_p = self.config.get("slow_top_p", 1.0)
        
        # define final results
        self.fast_results = None
        self.slow_results = None
        self.final_results = None
        
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying FastSlowGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None
    
    @retry(
        stop=stop_after_attempt(20),  # 最多重试10次
        wait=wait_exponential(multiplier=1, min=10, max=120),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def _generate(self, client: OpenAI, model: str, question: str, mode: str = "fast", reasoning_steps: str = None) -> GeneratorOutput:
        if mode == "fast":
            # fast
            assert reasoning_steps is not None
            temperature = self.fast_temperature
            top_p = self.fast_top_p
            samples = self.fast_samples
            timeout = 1_000
            max_tokens = NOT_GIVEN
            prompt = FAST_PROMPT.format(question=question, reasoning_steps=reasoning_steps)
        else:
            # slow
            temperature = self.slow_temperature
            top_p = self.slow_top_p
            samples = 1
            timeout = 10_000
            max_tokens = 2_500
            prompt = SLOW_PROMPT.format(question=question)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                n=samples,
                max_tokens=max_tokens,
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
            logger.error(f"Error in FastSlowGenerator._generate: {str(e)}, error model: {model}, mode: {mode}")
            raise  # 重新抛出异常，让重试装饰器捕获

    def generate(self, question: str) -> GeneratorOutput:
        # first use slow model get reasoning steps
        slow_output = self._generate(self.slow_client, self.slow_model, question, "slow")
        reasoning_steps = slow_output.first_output
        slow_prompt_tokens = slow_output.prompt_tokens
        slow_completion_tokens = slow_output.completion_tokens
        # then use fast model to get final answer
        fast_output = self._generate(self.fast_client, self.fast_model, question, "fast", reasoning_steps)

        fast_output.completion_tokens += slow_completion_tokens
        fast_output.prompt_tokens += slow_prompt_tokens
        fast_output.raw_output += [reasoning_steps]
        return fast_output
    
    def slow_generate(self, question: str) -> GeneratorOutput:
        self.slow_results = self._generate(self.slow_client, self.slow_model, question, "slow")
        
        if self.slow_results is None:
            return self.fast_results
        self.final_results = GeneratorOutput(
            first_output = self.slow_results.first_output,
            raw_output = self.slow_results.raw_output,
            prompt_tokens = self.slow_results.prompt_tokens + self.fast_results.prompt_tokens,
            completion_tokens = self.slow_results.completion_tokens + self.fast_results.completion_tokens
        )
        self.model = [self.slow_model]
        return self.final_results