from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    
    # Avengers扩展参数
    router: Optional[str] = "straight"  # 路由方法: straight, rank, random, etc.
    generator: Optional[str] = "direct"  # 生成方法: direct, self_consistency, etc.

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int 
    total_tokens: int
    cost: Optional[float] = 0.0

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "avengers"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ErrorResponse(BaseModel):
    error: Dict[str, Any]