# Avengers OpenAI Compatible API

这个模块将Avengers框架包装成OpenAI兼容的API服务，方便集成到现有应用中。

## 功能特性

- **OpenAI兼容接口**: 支持 `/v1/chat/completions` 和 `/v1/models` 接口
- **灵活的模型映射**: 支持专家模型直接调用和路由策略
- **多轮对话支持**: 自动处理对话历史和状态管理
- **完整的错误处理**: 提供详细的错误信息

## 安装依赖

```bash
# 激活uv环境
source /Users/apple/Documents/Avengers/avengers/bin/activate

# 安装新增依赖
pip install fastapi>=0.100.0 uvicorn>=0.20.0
```

## 启动服务

```bash
# 方式1: 使用启动脚本
python scripts/start_api.py

# 方式2: 直接启动
python -m uvicorn core.app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后可访问:
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 模型列表: http://localhost:8000/v1/models

## 模型名称格式

### 1. 专家模型直接调用
使用配置文件中定义的专家模型名称:
```json
{
  "model": "openai/gpt-5-chat",
  "messages": [{"role": "user", "content": "你好"}]
}
```

### 2. 路由策略调用
使用路由方法名称:
```json
{
  "model": "straight",  // 直接路由到指定专家
  "model": "rank",      // 使用rank路由策略
  "model": "random"     // 随机路由策略
}
```

### 3. 路由+生成策略组合
```json
{
  "model": "rank-direct",         // rank路由 + direct生成
  "model": "straight-aggregation" // straight路由 + aggregation生成
}
```

## API测试

```bash
# 运行测试脚本
python scripts/test_api.py
```

## OpenAI SDK兼容性

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="fake-key"  # 不需要真实API key
)

response = client.chat.completions.create(
    model="openai/gpt-5-chat",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.2
)

print(response.choices[0].message.content)
```

## 多轮对话处理

- **单轮对话**: 正常执行路由和生成流程
- **多轮对话**: 自动保持专家模型一致性，不重新路由
- **会话管理**: 基于对话历史自动生成会话标识

## 注意事项

1. **不支持流式响应**: 设置 `stream=true` 会返回错误
2. **配置文件依赖**: 需要正确配置 `config/experts.yaml`
3. **专家模型可用性**: 确保配置的专家模型API可访问
4. **内存使用**: 服务会常驻内存，包含所有专家模型连接

## 错误处理

API会返回标准的OpenAI格式错误响应:
```json
{
  "error": {
    "message": "错误描述",
    "type": "invalid_request_error", 
    "code": 400
  }
}
```