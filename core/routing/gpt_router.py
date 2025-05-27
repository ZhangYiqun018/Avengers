import json
import os
import random
import time
from typing import List, Dict

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from tenacity import (before_sleep_log, retry, retry_if_exception_type,
                      stop_after_attempt, wait_exponential)

from core.experts.load_experts import Expert
from core.routing.base_router import BaseRouter, RouterOutput

SYSTEM_PROMPT = """You are a helpful assistant that routes questions to the top {max_router} experts.

Here are the experts and their descriptions:
{experts}

Output in a JSON object.

The number of experts to route to is {max_router}.
NOTE: 
- DONT try to solve math problems with code!
- DONT try to solve emotional problems with medical experts!
"""

tools = [{
    "type": "function",
    "function": {
        "name": "route_to_experts",
        "description": "Route the query to appropriate expert(s) based on the content.",
        "parameters": {
            "type": "object",
            "properties": {
                "experts": {
                    "type": "array",
                    "description": "List of experts with reasoning and confidence scores",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query_type": {
                                "type": "string",
                                "enum": ["hard_math", "normal_math", "logic", "code", "medical", "general", "finance"],
                                "description": "The category of expertise required for handling the query. Must be one of the enum values"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Reasoning for selecting this expert"
                            },
                            "expert_name": {
                                "type": "string",
                                "description": "Name of the selected expert"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score (0 to 100) for this expert selection"
                            }
                        },
                        "required": ["expert_name"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["experts"],
            "additionalProperties": False
        },
    }
}]

class GPTRouterOutput(BaseModel):
    query_type: str
    reasoning: str
    expert_name: str
    confidence: float

class ListGPTRouterOutput(BaseModel):
    experts: List[GPTRouterOutput]

class GPTRouter(BaseRouter):
    def __init__(self, normal_experts: list, thinking_experts: list, router_config: dict):
        super().__init__(normal_experts, thinking_experts)
        self.config = router_config['gpt_router']
        
        self.max_router = self.config['max_router']
        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url']
        )
        self.model = self.config['model']
        
        # get available model
        self.candidate_models = []
        available_models = self.config.get("available_models")
        for expert in self.normal_experts:
            if expert.model_name in available_models:
                self.candidate_models.append(expert)
        assert len(self.candidate_models) == len(available_models)
        
        self.anonymize = self.config.get('anonymize', True)
        if self.anonymize:
            self._create_expert_id_mapping()
        
    def _create_expert_id_mapping(self):
        """创建固定的专家ID映射（不随机打乱顺序）"""
        expert_ids = [f"Expert_{i+1}" for i in range(len(self.candidate_models))]
        self.expert_id_map = {expert_id: expert for expert_id, expert in zip(expert_ids, self.candidate_models)}
        
    def _get_experts_details(self):
        """获取专家详情，根据匿名化设置决定是否使用ID"""
        if self.anonymize:
            # 使用匿名ID
            expert_details = "Expert name: Description\n"
            for expert_id, expert in self.expert_id_map.items():
                expert_details += f"- {expert_id}: {expert.description}\n"
        else:
            # 直接使用专家名称
            expert_details = "Expert name: Description\n"
            for expert in self.candidate_models:
                expert_details += f"- {expert.model_name}: {expert.description}\n"
        return expert_details
    
    # 定义重试日志记录函数
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None

    def _save_response(self, system: str, prompt: str, parsed_response: ListGPTRouterOutput | Dict, experts: List[Expert]):
        os.makedirs(".cache", exist_ok=True)
        # save the prompt and parsed_response to a file
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        if isinstance(parsed_response, ListGPTRouterOutput):
            parsed_response_str = parsed_response.model_dump_json()
        else:
            parsed_response_str = json.dumps(parsed_response)

        save_dict = {
            "timestamp": timestamp,
            "system": system,
            "prompt": prompt,
            "parsed_response": parsed_response_str,
            "experts": [expert.model_name for expert in experts]
        }
        prefix_model = self.model.split('/')[-1]
        with open(f".cache/routing_response_{prefix_model}.jsonl", "a") as f:
            f.write(json.dumps(save_dict) + "\n")
    
    @retry(
        stop=stop_after_attempt(20),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=5, max=10),  # 指数退避策略：1*2^x 秒，最少2秒，最多60秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def function_route(self, question: str) -> RouterOutput:
        try:
            user_prompt = f"Based on the following question, please route it to the appropriate expert(s) and provide your reasoning for each selection. The question is: " \
                + question + f"\nRemember your task is not answer the question, but to route it to the appropriate {self.max_router} expert(s)."
            
            system_prompt = SYSTEM_PROMPT.format(
                experts = self._get_experts_details(),
                max_router = self.max_router
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                timeout=300,
                tools=tools,
                tool_choice="auto"
            )
            assert response.choices[0].finish_reason == "tool_calls", "No tool calls found in the response"
            parsed_response = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            assert len(parsed_response['experts']) >= self.max_router, f"Expected {self.max_router} experts, got {len(parsed_response['experts'])}"
            experts = self._match_experts(parsed_response)
            self._save_response(system_prompt, user_prompt, parsed_response, experts)
            return RouterOutput(
                normal_experts=experts,
                thinking_experts=self.thinking_experts
            )
        
        except Exception as e:
            logger.warning(f"Error in function_route: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(10),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=5, max=60),  # 指数退避策略：1*2^x 秒，最少2秒，最多60秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def parsed_route(self, question: str) -> RouterOutput:
        try:
            user_prompt = question
            system_prompt = SYSTEM_PROMPT.format(
                experts = self._get_experts_details(),
                max_router = self.max_router
            )
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                timeout=300,
                response_format=ListGPTRouterOutput,
            )
            # parse response
            parsed_response = response.choices[0].message.parsed
            assert isinstance(parsed_response, ListGPTRouterOutput), f"Expected ListGPTRouterOutput, got {type(parsed_response)}"
            assert len(parsed_response.experts) == self.max_router, f"Expected {self.max_router} experts, got {len(parsed_response.experts)}"
            # match the experts to the experts list
            experts = self._match_experts(parsed_response)
            # save the response
            self._save_response(system_prompt, user_prompt, parsed_response, experts)
            
            return RouterOutput(
                normal_experts=experts,
                thinking_experts=self.thinking_experts
            )
        except Exception as e:
            logger.warning(f"Error in route function: {str(e)}")
            raise  # 重新抛出异常，让重试装饰器捕获
    
    def route(self, question: str):
        if 'gpt-4o' in self.model:
            return self.parsed_route(question)
        else:
            return self.function_route(question)
    
    def _match_experts(self, parsed_response):
        """根据匿名化设置匹配专家"""
        experts = []
        if isinstance(parsed_response, ListGPTRouterOutput):
            parsed_experts = parsed_response.experts
        elif isinstance(parsed_response, dict):
            parsed_experts = parsed_response['experts']
        else:
            raise TypeError(f"Unsupported type: {type(parsed_response)}")

        for expert_output in parsed_experts:
            if isinstance(expert_output, GPTRouterOutput):
                expert_name = expert_output.expert_name
            else:
                expert_name = expert_output['expert_name']
            
            if self.anonymize:
                # 当匿名化时，expert_name应该是一个ID (如 "Expert_1")
                if expert_name in self.expert_id_map:
                    experts.append(self.expert_id_map[expert_name])
                else:
                    logger.error(f"Expert ID {expert_name} not found in mapping")
                    raise ValueError(f"Expert ID {expert_name} not found in mapping")
            else:
                # 直接通过名称查找专家
                found = False
                for expert in self.candidate_models:
                    if expert.model_name == expert_name:
                        experts.append(expert)
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Expert name {expert_name} not found in experts list")
        
        return experts