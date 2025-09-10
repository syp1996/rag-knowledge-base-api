#!/usr/bin/env python3
"""
RAG Knowledge Base API 启动脚本
"""

import os
import sys
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    # 检查必要的环境变量
    required_vars = ["EMBED_PROVIDER"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
        print("Some features may not work properly")
        # 不强制退出，允许用户测试基础功能
    
    print("=== RAG Knowledge Base API with Document Management ===")
    print(f"API Service: http://localhost:{os.getenv('PORT', 8000)}")
    print(f"API Docs: http://localhost:{os.getenv('PORT', 8000)}/docs")
    print(f"Health Check: http://localhost:{os.getenv('PORT', 8000)}/health")
    print("=" * 60)
    print(f"Embed Provider: {os.getenv('EMBED_PROVIDER', 'Not Set')}")
    print(f"Database: {os.getenv('DB_HOST', 'localhost')}")
    print(f"Milvus: {os.getenv('MILVUS_URI', 'localhost:19530')}")
    print("=" * 60)
    
    # 启动服务器
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )

if __name__ == "__main__":
    main()