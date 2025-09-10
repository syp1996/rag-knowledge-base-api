# init_database.py
"""
数据库初始化脚本
用于创建表结构、索引和初始数据
"""
from sqlalchemy import create_engine, text
from app.models import Base, User, Category
from app.deps import DB_URL, SessionLocal
import sys

def create_tables():
    """创建数据库表"""
    print("Creating database tables...")
    engine = create_engine(DB_URL)
    
    # 删除已存在的表（开发阶段使用）
    print("Dropping existing tables...")
    Base.metadata.drop_all(engine)
    
    # 创建所有表
    print("Creating new tables...")
    Base.metadata.create_all(engine)
    
    print("✓ Database tables created successfully!")
    
    return engine

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
            print("✓ Fulltext index created successfully!")
        except Exception as e:
            print(f"Note: Fulltext index creation failed (may already exist): {e}")

def create_initial_data():
    """创建初始数据"""
    print("Creating initial data...")
    
    db = SessionLocal()
    try:
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
        print("✓ Initial data created successfully!")
        
        print("\n默认用户账号信息：")
        print("管理员: admin / admin123")
        print("普通用户: default / default123")
        print("插件用户: chrome_plugin_user / plugin123")
        
    except Exception as e:
        print(f"Error creating initial data: {e}")
        db.rollback()
    finally:
        db.close()

def test_connection():
    """测试数据库连接"""
    print("Testing database connection...")
    
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.fetchone():
                print("✓ Database connection successful!")
                return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
    
    return False

def main():
    """主函数"""
    print("=== 数据库初始化脚本 ===\n")
    
    # 测试数据库连接
    if not test_connection():
        print("请检查数据库配置和连接信息")
        sys.exit(1)
    
    try:
        # 创建表结构
        engine = create_tables()
        
        # 创建全文搜索索引
        create_fulltext_index(engine)
        
        # 创建初始数据
        create_initial_data()
        
        print("\n🎉 数据库初始化完成！")
        print("\nAPI文档地址: http://localhost:8000/docs")
        print("下一步:")
        print("1. 运行 'python run.py' 启动API服务")
        print("2. 访问 http://localhost:8000/docs 查看API文档")
        print("3. 使用默认账号登录测试功能")
        
    except Exception as e:
        print(f"初始化过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()