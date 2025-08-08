from typing import Dict, List, Set
from dataclasses import dataclass

from config.config_loader import load_config
from core.experts.load_experts import load_experts
from core.app.models import ChatMessage

@dataclass
class ModelMapping:
    """模型映射结果"""
    model_type: str  # "expert" | "router"
    target: str      # 专家名称或路由方式
    generator: str   # 生成方式

class ModelResolver:
    """模型名称解析和映射"""
    
    def __init__(self, config_path: str = "config/experts.yaml"):
        self.config = load_config(config_path)
        self.experts, self.thinking_experts = load_experts(self.config)
        
        # 构建专家名称集合
        self.expert_names: Set[str] = {expert.model_name for expert in self.experts}
        self.thinking_expert_names: Set[str] = {expert.model_name for expert in self.thinking_experts}
        
        # 支持的路由方式
        self.router_types: Set[str] = {
            "straight", "gpt", "random", "rank", "elo", "routerdc", "symbolic_moe", "moa"
        }
        
        # 支持的生成方式
        self.generator_types: Set[str] = {
            "direct", "self_consistency", "model_switch", "fast_slow", "slow_fast", "aggregation", "moa"
        }
    
    def validate_model(self, model_name: str) -> bool:
        """验证模型名称是否存在"""
        return model_name in self.expert_names
    
    def validate_router(self, router_name: str) -> bool:
        """验证路由方法是否支持"""
        return router_name in self.router_types
    
    def validate_generator(self, generator_name: str) -> bool:
        """验证生成方法是否支持"""
        return generator_name in self.generator_types
    
    def get_expert_by_name(self, model_name: str):
        """根据名称获取专家模型"""
        return next((expert for expert in self.experts if expert.model_name == model_name), None)
    
    def get_available_models(self) -> List[str]:
        """获取所有可用的模型名称"""
        return sorted(list(self.expert_names))
    
    def get_available_routers(self) -> List[str]:
        """获取所有可用的路由方法"""
        return sorted(list(self.router_types))
    
    def get_available_generators(self) -> List[str]:
        """获取所有可用的生成方法"""
        return sorted(list(self.generator_types))

class MessageProcessor:
    """消息历史处理器"""
    
    @staticmethod
    def extract_question(messages: List[ChatMessage]) -> str:
        """从消息历史中提取问题字符串"""
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        # 简单策略：合并所有消息内容
        question_parts = []
        for msg in messages:
            if msg.role == "system":
                question_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                question_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                question_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(question_parts)
    
    @staticmethod
    def is_multi_turn(messages: List[ChatMessage]) -> bool:
        """判断是否为多轮对话"""
        # 简单判断：是否有assistant消息
        return any(msg.role == "assistant" for msg in messages)
    
    @staticmethod 
    def get_last_user_message(messages: List[ChatMessage]) -> str:
        """获取最后一条用户消息"""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        raise ValueError("No user message found")

# 会话状态管理 (用于多轮对话)
class SessionManager:
    """简单的会话管理器"""
    
    def __init__(self):
        self._sessions: Dict[str, str] = {}  # session_id -> selected_model
    
    def get_or_create_session(self, session_key: str, default_model: str) -> str:
        """获取或创建会话，返回应该使用的模型"""
        if session_key not in self._sessions:
            self._sessions[session_key] = default_model
        return self._sessions[session_key]
    
    def clear_session(self, session_key: str):
        """清除会话"""
        self._sessions.pop(session_key, None)
    
    def generate_session_key(self, messages: List[ChatMessage]) -> str:
        """根据消息历史生成会话标识"""
        # 简单策略：使用第一条用户消息的hash
        first_user_msg = next((msg.content for msg in messages if msg.role == "user"), "")
        return str(hash(first_user_msg))