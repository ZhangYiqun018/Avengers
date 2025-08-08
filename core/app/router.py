import time
import uuid
from loguru import logger

from fastapi import HTTPException

from core.app.models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ModelListResponse,
    ModelInfo,
    Choice, 
    Usage,
    ChatMessage,
    MessageRole
)
from core.app.service import ModelResolver, MessageProcessor, SessionManager

class AvengersService:
    """Avengers核心服务"""
    
    def __init__(self, config_path: str = "config/experts.yaml"):
        self.resolver = ModelResolver(config_path)
        self.processor = MessageProcessor()
        self.session_manager = SessionManager()
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """处理聊天完成请求"""
        
        # 1. 检查不支持的参数
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported"
            )
        
        # 2. 验证参数
        if not self.resolver.validate_model(request.model):
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model}' not found. Available models: {self.resolver.get_available_models()}"
            )
        
        if not self.resolver.validate_router(request.router):
            raise HTTPException(
                status_code=400,
                detail=f"Router '{request.router}' not supported. Available routers: {self.resolver.get_available_routers()}"
            )
        
        if not self.resolver.validate_generator(request.generator):
            raise HTTPException(
                status_code=400,
                detail=f"Generator '{request.generator}' not supported. Available generators: {self.resolver.get_available_generators()}"
            )
        
        # 3. 处理多轮对话
        if self.processor.is_multi_turn(request.messages):
            # 多轮对话：使用之前的模型，不走路由
            session_key = self.session_manager.generate_session_key(request.messages)
            question = self.processor.get_last_user_message(request.messages)
            
            # 获取会话中使用的模型 
            selected_expert_name = self.session_manager.get_or_create_session(
                session_key, request.model
            )
            
            selected_expert = self.resolver.get_expert_by_name(selected_expert_name)
            if not selected_expert:
                raise HTTPException(status_code=500, detail=f"Expert '{selected_expert_name}' not found")
            
            from core.inference.direct_generator import DirectGenerator
            generator_config = {
                "temperature": request.temperature, 
                "top_p": request.top_p,
                "n": request.n
            }
            generator = DirectGenerator(selected_expert, generator_config)
            
        else:
            # 单轮对话：根据router参数决定处理方式
            question = self.processor.extract_question(request.messages)
            
            if request.router == "straight":
                # straight路由：直接使用指定的模型
                selected_expert = self.resolver.get_expert_by_name(request.model)
                if not selected_expert:
                    raise HTTPException(status_code=400, detail=f"Expert '{request.model}' not found")
                
                from core.inference.direct_generator import DirectGenerator
                generator_config = {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "n": request.n,
                }
                generator = DirectGenerator(selected_expert, generator_config)
                
            else:
                # 使用其他路由系统
                from core.routing.factory import RouterFactory
                from core.inference.factory import GeneratorFactory
                
                # 构建路由配置
                router_config = {
                    "type": request.router,
                    f"{request.router}_router": {}
                }
                
                router = RouterFactory.create_router(
                    normal_experts=self.resolver.experts,
                    thinking_experts=self.resolver.thinking_experts,
                    router_config=router_config
                )
                
                # 路由选择专家
                router_output = router.route(question)
                
                # 创建生成器
                generator_config_dict = {
                    "type": request.generator,
                    request.generator: {
                        "temperature": request.temperature,
                        "top_p": request.top_p
                    }
                }
                
                generator = GeneratorFactory.create_generator(
                    experts=router_output,
                    generator_config=generator_config_dict
                )
        
        # 4. 生成响应
        try:
            result = generator.generate(question)
            
            # 5. 构造OpenAI格式的响应
            choices = []
            for i in range(request.n):
                choices.append(Choice(
                    index=i,
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=result.first_output
                    )
                ))
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=Usage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens
                )
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )
    
    async def list_models(self) -> ModelListResponse:
        """获取可用模型列表"""
        models = []
        for model_name in self.resolver.get_available_models():
            models.append(ModelInfo(id=model_name))
        
        return ModelListResponse(data=models)

# 全局服务实例
avengers_service = AvengersService()