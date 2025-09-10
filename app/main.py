# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .ingest import router as ingest_router
from .search import router as search_router
from .api.auth import router as auth_router
from .api.users import router as users_router
from .api.categories import router as categories_router
from .api.documents import router as documents_router
import os

# 创建FastAPI应用
app = FastAPI(
    title="RAG Knowledge Base API",
    description="基于Milvus和MySQL的RAG知识库系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(ingest_router, prefix="/api/v1", tags=["文档入库"])
app.include_router(search_router, prefix="/api/v1", tags=["搜索检索"])

# 新增的管理系统路由
app.include_router(auth_router, prefix="/api/auth", tags=["认证"])
app.include_router(users_router, prefix="/api/users", tags=["用户管理"])
app.include_router(categories_router, prefix="/api/categories", tags=["分类管理"])
app.include_router(documents_router, prefix="/api/documents", tags=["文档管理"])

# 健康检查
@app.get("/", tags=["健康检查"])
async def root():
    return {
        "message": "RAG Knowledge Base API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["健康检查"])
async def health_check():
    """系统健康检查"""
    try:
        from .deps import SessionLocal, milvus
        
        # 检查MySQL连接
        db = SessionLocal()
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            mysql_status = "ok"
        except Exception as e:
            mysql_status = f"error: {str(e)}"
        finally:
            db.close()
        
        # 检查Milvus连接
        try:
            if milvus is not None:
                collections = milvus.list_collections()
                milvus_status = "ok"
                has_collection = "kb_chunks" in collections
            else:
                milvus_status = "unavailable"
                has_collection = False
        except Exception as e:
            milvus_status = f"error: {str(e)}"
            has_collection = False
        
        # 检查环境变量
        embed_provider = os.getenv("EMBED_PROVIDER", "not_set")
        
        return {
            "status": "healthy" if mysql_status == "ok" else "partial" if mysql_status == "ok" and milvus_status != "ok" else "unhealthy",
            "components": {
                "mysql": mysql_status,
                "milvus": milvus_status,
                "collection_exists": has_collection,
                "embed_provider": embed_provider
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred"
        }
    )

# 启动事件
@app.on_event("startup")
async def startup_event():
    print("RAG Knowledge Base API is starting...")
    
    # 检查必要的环境变量
    required_vars = ["EMBED_PROVIDER"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
    else:
        print("All required environment variables are set")
    
    print("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    print("RAG Knowledge Base API is shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    # 开发环境运行
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )