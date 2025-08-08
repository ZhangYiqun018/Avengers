from openai import OpenAI, NOT_GIVEN
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput

class DirectGenerator(BaseGenerator):
    def __init__(self, expert: Expert, generator_config: dict):
        self.client = expert.client
        self.model = expert.model_name
        self.config = generator_config
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", NOT_GIVEN)
        self.n = self.config.get("n", 1)
        
    # 定义重试日志记录函数
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying DirectGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None
    
    @retry(
        stop=stop_after_attempt(5),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=2, max=60),  # 指数退避策略：1*2^x 秒，最少2秒，最多60秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def generate_with_retry(self, question: str) -> GeneratorOutput:
        if "Distill" in self.model or "EXAOME" in self.model:
            question += "Don't make your reasoning and thinking too long.\n"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                timeout=500,
            )
            usage = response.usage
            choices = response.choices
            assert choices[0].message.content is not None, f"choices[0].message.content is None"
            
            return GeneratorOutput(
                first_output=choices[0].message.content,
                raw_output=[choice.message.content for choice in choices],
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens
            )
        except Exception as e:
            logger.error(f"Error in DirectGenerator.generate: {str(e)}, model_name: {self.model}")
            raise  # 重新抛出异常，让重试装饰器捕获
    
    def generate(self, question: str) -> GeneratorOutput:
        try:
            return self.generate_with_retry(question=question)
        except Exception as e:
            logger.error(
                f"Error in DirectGenerator.generate after all retries: "
                f"{str(e)}, model_name: {self.model}"
            )
            return GeneratorOutput(
                first_output="failed to generate",
                raw_output=["failed to generate"],
                prompt_tokens=0,
                completion_tokens=0
            )
