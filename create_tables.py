# create_tables.py
from sqlalchemy import create_engine, text
from app.models import Base
from app.deps import DB_URL

def create_tables():
    """创建数据库表"""
    engine = create_engine(DB_URL)
    
    # 删除已存在的表（开发阶段使用，生产环境需要谨慎）
    Base.metadata.drop_all(engine)
    
    # 创建所有表
    Base.metadata.create_all(engine)
    
    print("Database tables created successfully!")
    
    # 为documents表添加生成列（MySQL特定语法）
    with engine.connect() as conn:
        try:
            # 添加生成列
            conn.execute(text("""
                ALTER TABLE documents 
                ADD COLUMN content_text_generated LONGTEXT GENERATED ALWAYS AS (
                    CASE
                        WHEN JSON_VALID(content) AND JSON_EXTRACT(content,'$.markdown') IS NOT NULL
                        THEN JSON_UNQUOTE(JSON_EXTRACT(content,'$.markdown'))
                        WHEN JSON_VALID(content) AND JSON_EXTRACT(content,'$.html') IS NOT NULL
                        THEN JSON_UNQUOTE(JSON_EXTRACT(content,'$.html'))
                        WHEN JSON_VALID(content) AND JSON_EXTRACT(content,'$.text') IS NOT NULL
                        THEN JSON_UNQUOTE(JSON_EXTRACT(content,'$.text'))
                        ELSE NULL
                    END
                ) STORED
            """))
            conn.commit()
            print("Generated column added successfully!")
        except Exception as e:
            print(f"Note: Generated column may already exist or MySQL version doesn't support it: {e}")

if __name__ == "__main__":
    create_tables()