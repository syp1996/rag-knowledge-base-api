#!/usr/bin/env python3

import os
import sys
import asyncio
import httpx

# Add the app directory to path
sys.path.append('.')

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env文件加载成功")
except:
    print("⚠️  .env文件加载失败，将使用默认配置")

from app.embedding import embed_texts, PROVIDER

async def check_configuration():
    """检查配置状态"""
    
    print("\n🔧 RAG系统配置检查")
    print("=" * 50)
    
    # 检查embedding配置
    print(f"📡 Embedding提供商: {PROVIDER}")
    
    if PROVIDER == "dashscope":
        api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-279e04bee3d94a61884fd0c3969cf230')
        base_url = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model = os.getenv('DASHSCOPE_EMBED_MODEL', 'text-embedding-v4')
        
        print(f"🔑 API Key: {api_key[:20]}...")
        print(f"🌐 Base URL: {base_url}")
        print(f"🤖 Model: {model}")
        
        # 测试API连接
        print("\n🧪 测试embedding API...")
        try:
            vectors = await embed_texts(["测试文本"])
            print(f"✅ API连接正常，向量维度: {len(vectors[0])}")
        except Exception as e:
            print(f"❌ API连接失败: {e}")
            return False
    
    # 测试服务器健康状态
    print("\n🏥 测试服务器健康状态...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ 服务器状态: {health_data.get('status', 'unknown')}")
                components = health_data.get('components', {})
                print(f"   📊 MySQL: {components.get('mysql', 'unknown')}")
                print(f"   🗂️  Milvus: {components.get('milvus', 'unknown')}")
                print(f"   📁 Collection: {'存在' if components.get('collection_exists') else '不存在'}")
            else:
                print(f"⚠️  服务器响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
    
    # 测试搜索功能
    print("\n🔍 测试搜索功能...")
    try:
        async with httpx.AsyncClient() as client:
            search_response = await client.post(
                "http://localhost:8000/api/v1/search",
                json={"query": "测试查询", "top_k": 3},
                timeout=10
            )
            if search_response.status_code == 200:
                search_data = search_response.json()
                print(f"✅ 搜索功能正常，返回 {search_data.get('total_hits', 0)} 个结果")
            else:
                print(f"⚠️  搜索功能异常: {search_response.status_code}")
    except Exception as e:
        print(f"❌ 搜索功能测试失败: {e}")
    
    print("\n✅ 配置检查完成!")
    return True

if __name__ == "__main__":
    asyncio.run(check_configuration())
