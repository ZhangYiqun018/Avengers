from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import tiktoken
from loguru import logger
from openai import NOT_GIVEN, OpenAI
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from core.experts.load_experts import Expert
from core.inference.base_generator import BaseGenerator, GeneratorOutput

aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

class MoAGenerator(BaseGenerator):
    def __init__(self, experts: List[Expert], generator_config: dict):
        self.name = self.__class__.__name__
        self.config = generator_config
        # get clients and models from experts
        self.layers = generator_config.get("layers", 3)
        self.number_of_proposers = generator_config['number_of_proposers']
        self.number_of_aggregator = generator_config['number_of_aggregator']
        assert len(experts) == self.number_of_proposers + self.number_of_aggregator, f"Number of experts must be equal to {self.number_of_proposers + self.number_of_aggregator}"
        self.proposers = experts[:self.number_of_proposers]
        self.aggregator = experts[-1]
        # get parameters
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", NOT_GIVEN)
        self.proposer_max_tokens = self.config.get("proposer_max_tokens", 3000)
        self.aggregator_max_tokens = self.config.get("aggregator_max_tokens", 10000)
        
        self.model = [proposer.model_name for proposer in self.proposers] + [self.aggregator.model_name]
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        # define final results
        self.results = None
        
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying MoAGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None
    
    @retry(
        stop=stop_after_attempt(20),  # 最多重试10次
        wait=wait_exponential(multiplier=1, min=20, max=120),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def _aggregation_generate(self, client: OpenAI, model: str, system: str, question: str, prompt_tokens: int, completion_tokens: int) -> GeneratorOutput:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": question}
                ],
                temperature=0.2,
                top_p=1.0,
                n=1,
                timeout=300,
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
            logger.error(f"Error in MoAGenerator._aggregation_generate: {str(e)}, question: {question}")
            raise
    
    # @retry(
    #     stop=stop_after_attempt(10),  # 最多重试10次
    #     wait=wait_exponential(multiplier=1, min=10, max=120),  # 指数退避策略：1*2^x 秒，最少2秒，最多100秒
    #     retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
    #     before_sleep=_log_retry  # 重试前记录日志
    # )
    def _generate(self, client: OpenAI, model: str, question: str) -> GeneratorOutput:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
                timeout=100,
            )
            choices = response.choices
            usage = response.usage
            aggregation_raw_output = [choice.message.content for choice in choices]
                
            result = GeneratorOutput(
                first_output=choices[0].message.content,
                raw_output=aggregation_raw_output,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens
            )
            return result
        except Exception as e:
            logger.error(f"Error in MoAGenerator._generate: {str(e)}, error model: {model}")
            return GeneratorOutput(
                first_output="No response (error: {str(e)})",
                raw_output=["No response (error: {str(e)})"],
                prompt_tokens=0,
                completion_tokens=0
            )
    
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
        results = dict()
        prompt_tokens = 0
        completion_tokens = 0
        
        # layer 0
        with ThreadPoolExecutor(max_workers=self.number_of_proposers) as executor:
            # get results from proposer models
            future_to_model = {
                executor.submit(self._generate, proposer.client, proposer.model_name, question): proposer.model_name
                for proposer in self.proposers
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.error(f"[MoAGenerator] Layer 0 Error in MoAGenerator.generate: {str(e)}, error model: {model}")
                    results[model] = None
                    
        for layer in range(1, self.layers):
            # format results
            model_list = list(results.keys())
            response_list = [result.first_output for result in results.values()]
            formatted_results: str = self._format_single_result(model_list=model_list, response_list=response_list, question=question, max_tokens=self.proposer_max_tokens, split_number=len(model_list))

            # get results from proposer models
            for result in results.values():
                prompt_tokens += result.prompt_tokens
                completion_tokens += result.completion_tokens
            
            results = dict()
            with ThreadPoolExecutor(max_workers=self.number_of_proposers) as executor:
                future_to_model = {
                    executor.submit(
                        self._generate, client=proposer.client, model=proposer.model_name, 
                        question=aggreagator_system_prompt + formatted_results + f"Original question: {question}"): proposer.model_name
                    for proposer in self.proposers
                }
                for future in as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        results[model] = future.result()
                    except Exception as e:
                        logger.error(f"[MoAGenerator] Layer {layer} Error in MoAGenerator.generate: {str(e)}, error model: {model}")
                        results[model] = None
        
        model_list = list(results.keys())
        response_list = [result.first_output for result in results.values()]
        formatted_results = self._format_single_result(
            model_list=model_list, response_list=response_list, 
            question=question, max_tokens=self.aggregator_max_tokens, split_number=len(model_list)
        )
        # logger.info(f"[MoAGenerator] final aggregator start!")
        # get aggregation results
        aggregation_result = self._aggregation_generate(
            client=self.aggregator.client, model=self.aggregator.model_name, 
            system=aggreagator_system_prompt + "\n" + formatted_results,
            question=question, 
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        
        return aggregation_result