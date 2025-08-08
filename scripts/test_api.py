#!/usr/bin/env python3
"""
测试Avengers OpenAI兼容API
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # 1. 测试健康检查
    print("=== 测试健康检查 ===")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
        return
    
    # 2. 测试模型列表
    print("\n=== 测试模型列表 ===")
    try:
        response = requests.get(f"{base_url}/v1/models")
        print(f"状态码: {response.status_code}")
        models = response.json()
        print(f"可用模型数量: {len(models.get('data', []))}")
        for model in models.get('data', [])[:5]:  # 只显示前5个
            print(f"  - {model['id']}")
    except Exception as e:
        print(f"获取模型列表失败: {e}")
    
    # 3. 测试聊天完成 - 专家模型
    print("\n=== 测试专家模型 ===")
    test_expert_chat(base_url)
    
    # 4. 测试聊天完成 - 路由模式
    print("\n=== 测试路由模式 ===")  
    test_router_chat(base_url)

def test_expert_chat(base_url):
    """测试专家模型"""
    payload = {
        "model": "openai/gpt-5-chat",  # 使用配置文件中的专家模型
        "messages": [
            {"role": "user", "content": "你好！请简单介绍一下你自己。"}
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"模型: {result['model']}")
            print(f"响应: {result['choices'][0]['message']['content'][:100]}...")
            print(f"Token使用: {result['usage']}")
        else:
            print(f"错误响应: {response.text}")
            
    except Exception as e:
        print(f"专家模型测试失败: {e}")

def test_router_chat(base_url):
    """测试路由模式"""
    payload = {
        "model": "straight",  # 使用straight路由
        "messages": [
            {"role": "user", "content": "解释一下什么是机器学习？"}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions", 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"模型: {result['model']}")
            print(f"响应: {result['choices'][0]['message']['content'][:100]}...")
            print(f"Token使用: {result['usage']}")
        else:
            print(f"错误响应: {response.text}")
            
    except Exception as e:
        print(f"路由模式测试失败: {e}")

def test_openai_sdk():
    """测试OpenAI SDK兼容性"""
    print("\n=== 测试OpenAI SDK兼容性 ===")
    try:
        import httpx
        from openai import OpenAI
        
        # Create custom transport to avoid proxy issues
        transport = httpx.HTTPTransport(
            verify=False,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
            ),
        )
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="fake-key",  # API key不重要，因为我们不验证
            http_client=httpx.Client(transport=transport, timeout=30.0)
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-5-chat",
            messages=[
                {"role": "user", "content": "用一句话解释什么是人工智能"}
            ],
            temperature=0.2,
            max_tokens=50
        )
        
        print("✓ OpenAI SDK兼容性测试成功")
        print(f"响应: {response.choices[0].message.content}")
        print(f"Token使用: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        
    except ImportError:
        print("OpenAI SDK未安装，跳过兼容性测试")
    except Exception as e:
        print(f"OpenAI SDK兼容性测试失败: {e}")

if __name__ == "__main__":
    print("启动API测试...")
    print("请确保API服务已经在 http://localhost:8000 启动")
    print("启动命令: python scripts/start_api.py")
    print()
    
    try:
        test_api()
        test_openai_sdk()
        print("\n测试完成！")
    except KeyboardInterrupt:
        print("\n测试已停止")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()