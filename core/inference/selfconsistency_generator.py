from openai import OpenAI, NOT_GIVEN
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput

class SelfConsistencyGenerator(BaseGenerator):
    def __init__(self, expert: Expert, generator_config: dict):
        self.client = expert.client
        self.model = expert.model_name
        self.config = generator_config
        self.samples = self.config.get("samples", 5)
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", NOT_GIVEN)
    # 定义重试日志记录函数
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying SelfConsistencyGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None
    
    @retry(
        stop=stop_after_attempt(10),  # 最多重试10次
        wait=wait_exponential(multiplier=1, min=2, max=100),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def generate(self, question: str) -> GeneratorOutput:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.samples,
                timeout=1_000,
            )
            choices = response.choices
            usage = response.usage
            raw_output = [choice.message.content for choice in choices]
            assert len(raw_output) == self.samples, f"Expected {self.samples} samples, got {len(raw_output)}"
            first_output = choices[0].message.content
            
            return GeneratorOutput(
                first_output=first_output, 
                raw_output=raw_output,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens
            )
        except Exception as e:
            logger.error(f"Error in SelfConsistencyGenerator.generate: {str(e)}")
            raise  # 重新抛出异常，让重试装饰器捕获