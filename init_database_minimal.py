# init_database_minimal.py
"""
数据库初始化脚本（不依赖Milvus）
专门用于文档管理系统的表结构创建
"""
from sqlalchemy import create_engine, text
from app.models import Base, User, Category
import sys

# 直接使用数据库配置，避免导入deps.py中的Milvus客户端
DB_HOST = "rm-bp1ljbjb34n55su6uko.mysql.rds.aliyuncs.com"
DB_PORT = 3306
DB_NAME = "markdown_manager"
DB_USER = "markdown_user"
DB_PASSWORD = "Syp19960424"

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def test_connection():
    """测试数据库连接"""
    print("Testing database connection...")
    
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.fetchone():
                print("[OK] Database connection successful!")
                return True, engine
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return False, None
    
    return False, None

def create_tables(engine):
    """创建数据库表"""
    print("Creating database tables...")
    
    # 删除已存在的表（开发阶段使用）
    print("Dropping existing tables...")
    try:
        Base.metadata.drop_all(engine)
        print("[OK] Existing tables dropped")
    except Exception as e:
        print(f"Note: Error dropping tables (may not exist): {e}")
    
    # 创建所有表
    print("Creating new tables...")
    Base.metadata.create_all(engine)
    
    print("[OK] Database tables created successfully!")

def create_fulltext_index(engine):
    """创建全文搜索索引"""
    print("Creating fulltext search index...")
    
    with engine.connect() as conn:
        try:
            # 为content_text字段添加全文索引
            conn.execute(text("""
                ALTER TABLE documents 
                ADD FULLTEXT(content_text, title, excerpt)
            """))
            conn.commit()
            print("[OK] Fulltext index created successfully!")
        except Exception as e:
            print(f"Note: Fulltext index creation failed (may already exist): {e}")

def create_initial_data(engine):
    """创建初始数据"""
    print("Creating initial data...")
    
    # 创建SessionLocal
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    
    db = SessionLocal()
    try:
        # 检查是否已有数据
        existing_users = db.query(User).count()
        if existing_users > 0:
            print("[OK] Initial data already exists, skipping...")
            return
        
        # 创建管理员用户
        admin_user = User(
            username="admin",
            email="admin@example.com",
            is_admin=True
        )
        admin_user.set_password("admin123")
        db.add(admin_user)
        
        # 创建默认用户
        default_user = User(
            username="default",
            email="default@example.com",
            is_admin=False
        )
        default_user.set_password("default123")
        db.add(default_user)
        
        # 创建Chrome插件用户
        plugin_user = User(
            username="chrome_plugin_user",
            email="chrome_plugin@example.com",
            is_admin=False
        )
        plugin_user.set_password("plugin123")
        db.add(plugin_user)
        
        # 创建默认分类
        categories = [
            Category(name="技术文档", description="技术相关的文档和资料"),
            Category(name="产品文档", description="产品功能和使用说明"),
            Category(name="会议记录", description="会议纪要和讨论记录"),
            Category(name="知识分享", description="团队知识分享和经验总结"),
            Category(name="其他", description="其他类型的文档")
        ]
        
        for category in categories:
            db.add(category)
        
        db.commit()
        print("[OK] Initial data created successfully!")
        
        print("\n[INFO] Default user accounts:")
        print("Admin: admin / admin123")
        print("User: default / default123") 
        print("Plugin: chrome_plugin_user / plugin123")
        
    except Exception as e:
        print(f"Error creating initial data: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def main():
    """主函数"""
    print("=== 文档管理系统数据库初始化脚本 ===\n")
    
    # 测试数据库连接
    success, engine = test_connection()
    if not success:
        print("请检查数据库配置和连接信息")
        sys.exit(1)
    
    try:
        # 创建表结构
        create_tables(engine)
        
        # 创建全文搜索索引
        create_fulltext_index(engine)
        
        # 创建初始数据
        create_initial_data(engine)
        
        print("\n[SUCCESS] Database initialization completed!")
        print("\nAPI Documentation: http://localhost:8000/docs")
        print("Next steps:")
        print("1. Run 'python run.py' to start API service")
        print("2. Visit http://localhost:8000/docs to view API documentation")
        print("3. Use default accounts to test functionality")
        print("\nNote: For RAG features, ensure Milvus service is running")
        
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()