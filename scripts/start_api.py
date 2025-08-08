#!/usr/bin/env python3
"""
启动Avengers OpenAI兼容API服务
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    from core.app.main import app
    import uvicorn
    
    print("Starting Avengers OpenAI Compatible API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Models List: http://localhost:8000/v1/models")
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    )