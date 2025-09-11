#!/usr/bin/env python3
"""
Database status test script
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from app.deps import SessionLocal
from sqlalchemy import text

def test_database_status():
    """Test database status with the provided SQL queries"""
    db = SessionLocal()
    try:
        print("=== 数据库状态检查 ===\n")
        
        # 1) 嵌入表是否有数据
        print("1) 检查 document_embeddings 表数据量:")
        result = db.execute(text("SELECT COUNT(*) as count FROM document_embeddings"))
        count = result.fetchone()[0]
        print(f"   嵌入数据总数: {count}")
        
        # 2) 哪些 document 还没切块/嵌入
        print("\n2) 检查未嵌入的文档:")
        result = db.execute(text("""
            SELECT d.id, d.title
            FROM documents d
            LEFT JOIN document_embeddings e ON e.document_id = d.id
            GROUP BY d.id, d.title
            HAVING COUNT(e.id) = 0
        """))
        
        unembedded_docs = result.fetchall()
        if unembedded_docs:
            print(f"   发现 {len(unembedded_docs)} 个未嵌入的文档:")
            for doc in unembedded_docs:
                print(f"   - ID: {doc[0]}, Title: {doc[1]}")
        else:
            print("   所有文档都已嵌入")
            
        # 3) 检查表结构和可能的模型字段
        print("\n3) 检查 document_embeddings 表结构:")
        result = db.execute(text("DESCRIBE document_embeddings"))
        columns = result.fetchall()
        print("   表字段:")
        for col in columns:
            print(f"   - {col[0]}: {col[1]}")
            
        # 如果有模型相关字段，统计覆盖率
        column_names = [col[0] for col in columns]
        if 'embedding_model' in column_names:
            result = db.execute(text("""
                SELECT embedding_model, COUNT(*) AS chunks
                FROM document_embeddings
                GROUP BY embedding_model
            """))
            model_stats = result.fetchall()
            if model_stats:
                print("\n   嵌入模型覆盖率:")
                for model, chunks in model_stats:
                    print(f"   模型 {model}: {chunks} 个嵌入块")
        else:
            print("   表中没有 embedding_model 字段")
            
        # 额外检查：文档总数
        print("\n4) 额外信息:")
        result = db.execute(text("SELECT COUNT(*) as count FROM documents"))
        doc_count = result.fetchone()[0]
        print(f"   文档总数: {doc_count}")
        
        # 检查表是否存在
        result = db.execute(text("""
            SHOW TABLES LIKE 'document_embeddings'
        """))
        table_exists = result.fetchone() is not None
        print(f"   document_embeddings 表存在: {table_exists}")
        
    except Exception as e:
        print(f"数据库查询错误: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    test_database_status()