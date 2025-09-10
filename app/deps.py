# app/deps.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pymilvus import MilvusClient

# 数据库配置（使用您的阿里云MySQL配置）
DB_HOST = "rm-bp1ljbjb34n55su6uko.mysql.rds.aliyuncs.com"
DB_PORT = 3306
DB_NAME = "markdown_manager"
DB_USER = "markdown_user"
DB_PASSWORD = "Syp19960424"

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Milvus配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", None)

# SQLAlchemy引擎和会话
engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Milvus客户端（容错处理）
try:
    milvus = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    print("[OK] Milvus connection successful")
except Exception as e:
    print(f"[WARNING] Milvus connection failed: {e}")
    print("Document management features will work, but RAG features may be limited")
    milvus = None

# 依赖注入函数
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_milvus():
    if milvus is None:
        raise Exception("Milvus is not available. Please check your Milvus service.")
    return milvus