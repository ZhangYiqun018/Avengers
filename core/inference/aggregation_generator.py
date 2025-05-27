from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import tiktoken
from loguru import logger
from openai import NOT_GIVEN, OpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput

AGGREGATION_SYSTEM_PROMPT="""You are an aggregation agent designed to provide a coherent, accurate, and well-formatted final answer to a given question based on multiple responses from different models.

Your task is:
1. **Input Analysis:**
   - Clearly understand the original question provided.
   - Carefully review all provided responses from multiple models.

2. **Aggregation Process:**
   - Identify consensus, contradictions, and unique insights among the provided responses.
   - Give higher priority to responses supported by the majority of models or responses that show strong reasoning and credible evidence.
   - Clearly resolve contradictions by favoring the most logical, evidence-based, or well-supported answers.

3. **Output Formatting:**
   - Provide your final aggregated answer strictly following the format implied by the original question (e.g., if it's a yes/no question, your answer should clearly be 'Yes' or 'No'; if it asks for an explanation, your answer should clearly explain the reasoning).
   - Ensure clarity, conciseness, and readability in the final response.

Your final output must directly answer the original question with the most accurate, logically consistent, and well-supported conclusion derived from the provided inputs."""

AGGREGATION_USER_PROMPT_TEMPLATE="""
Question: {question}

{expert_results}"""

class AggregationGenerator(BaseGenerator):
    def __init__(self, experts: List[Expert], generator_config: dict):
        self.name = self.__class__.__name__
        # get clients and models from experts
        self.numbers = generator_config['numbers']
        self.aggregation_client = OpenAI(base_url=generator_config['aggregation_base_url'], api_key=generator_config['aggregation_api_key'])
        self.aggregation_model = generator_config['aggregation_model']
        assert len(experts) >= self.numbers, f"Number of experts must be greater than or equal to {self.numbers}"
        self.experts = experts[:self.numbers]

        self.model = [expert.model_name for expert in self.experts]
        # get config
        self.config = generator_config
        self.samples = self.config.get("samples", 5)
        self.temperature = self.config.get("temperature", 0.7)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", NOT_GIVEN)
        self.max_tokens = self.config.get("max_tokens", 3000)
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        # define final results
        self.results = None
        
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying AggregationGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None
    
    @retry(
        stop=stop_after_attempt(20),  # 最多重试10次
        wait=wait_exponential(multiplier=1, min=20, max=120),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def _aggregation_generate(self, question: str, expert_results: str, prompt_tokens: int, completion_tokens: int) -> GeneratorOutput:
        try:
            user_inputs = AGGREGATION_USER_PROMPT_TEMPLATE.format(question=question, expert_results=expert_results)
            
            response = self.aggregation_client.chat.completions.create(
                model=self.aggregation_model,
                messages=[
                    {"role": "system", "content": AGGREGATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_inputs}
                ],
                temperature=0.2,
                top_p=1.0,
                n=1,
                timeout=3_200,
            )
            choices = response.choices
            usage = response.usage
            first_output = choices[0].message.content
            result = GeneratorOutput(
                first_output=first_output,
                raw_output=[first_output],
                prompt_tokens=usage.prompt_tokens + prompt_tokens,
                completion_tokens=usage.completion_tokens + completion_tokens,
            )
            return result
        except Exception as e:
            logger.error(f"Error in AggregationGenerator._aggregation_generate: {str(e)}, question: {question}, expert_results: {expert_results}")
            raise
    
    @retry(
        stop=stop_after_attempt(10),  # 最多重试10次
        wait=wait_exponential(multiplier=1, min=10, max=120),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
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
                timeout=2000,
            )
            choices = response.choices
            usage = response.usage
            aggregation_raw_output = [choice.message.content for choice in choices]
            assert len(aggregation_raw_output) == self.samples, f"Expected {self.samples} samples, got {len(aggregation_raw_output)}"
            
            result = GeneratorOutput(
                first_output=choices[0].message.content,
                raw_output=aggregation_raw_output,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens
            )
            return result
        except Exception as e:
            logger.error(f"Error in AggregationGenerator._generate: {str(e)}, error model: {model}")
            raise  # 重新抛出异常，让重试装饰器捕获
    
    def _format_single_result(self, model_list: List[str], question: str, response_list: List[str], max_tokens: int, split_number: int = 1) -> str:
        question_tokens = len(self.encoding.encode(question))
        max_tokens = max_tokens - question_tokens
        # Calculate max tokens per response to ensure fairness
        max_tokens_per_response = max_tokens // split_number
        
        def process_single_response(args):
            model, output = args
            
            result_header = f"### Result of {model}\n"
            result_content = f"{output}\n\n"
            
            # Calculate tokens for this section
            header_tokens = len(self.encoding.encode(result_header))
            content_tokens = len(self.encoding.encode(result_content))
            
            # If this response exceeds its token limit, truncate it
            if header_tokens + content_tokens > max_tokens_per_response:
                logger.info(f"[MoAGenerator] Truncating result of {model} due to token limit ({max_tokens_per_response}).")
                # Calculate remaining tokens for this response
                remaining_tokens = max_tokens_per_response - header_tokens
                if remaining_tokens > 0:
                    # Get all tokens of the content
                    content_tokens_list = self.encoding.encode(result_content)
                    # Keep tokens from the end where the important conclusions are
                    truncated_tokens = content_tokens_list[-remaining_tokens:]
                    truncated_content = self.encoding.decode(truncated_tokens)
                    return result_header + truncated_content
            
            return result_header + result_content
        
        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            formatted_parts = list(executor.map(process_single_response, zip(model_list, response_list)))
        
        return "".join(formatted_parts)
        
    def generate(self, question: str) -> tuple[GeneratorOutput, GeneratorOutput]:
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            # get results from expert models
            future_to_model = {
                executor.submit(self._generate, expert.client, expert.model_name, question): expert.model_name
                for expert in self.experts
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    results.append({
                        "model": model,
                        "response": future.result()
                    })
                except Exception as e:
                    logger.error(f"Error in AggregationGenerator.generate: {str(e)}, error model: {model}")
                    results.append({
                        "model": model,
                        "response": "No response (error)"
                    })
        
        # format results
        model_list = [result["model"] for result in results]
        response_list = [result["response"].first_output for result in results]
        formatted_results = self._format_single_result(model_list=model_list, question=question, response_list=response_list, max_tokens=self.max_tokens, split_number=len(model_list))
        prompt_tokens = 0
        completion_tokens = 0
        
        for result in results:
            prompt_tokens += result["response"].prompt_tokens
            completion_tokens += result["response"].completion_tokens
        # get aggregation results
        aggregation_result = self._aggregation_generate(question=question, expert_results=formatted_results, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        
        return aggregation_result