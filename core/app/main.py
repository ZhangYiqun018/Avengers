from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from core.app.router import avengers_service
from core.app.models import ChatCompletionRequest, ErrorResponse

# 创建FastAPI应用
app = FastAPI(
    title="Avengers OpenAI Compatible API",
    description="OpenAI Compatible API for Avengers Ensemble Framework",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    from core.app.models import ErrorResponse
    error_response = ErrorResponse(
        error={
            "message": exc.detail,
            "type": "invalid_request_error", 
            "code": exc.status_code
        }
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )

# API路由
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天完成接口"""
    return await avengers_service.chat_completion(request)

@app.get("/v1/models") 
async def list_models():
    """获取可用模型列表"""
    return await avengers_service.list_models()

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Avengers OpenAI Compatible API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    logger.info("Starting Avengers OpenAI Compatible API Server...")
    uvicorn.run(
        "core.app.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )