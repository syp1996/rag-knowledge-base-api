#!/usr/bin/env python3
"""
MySQL数据库设置脚本
连接到阿里云MySQL并创建文档管理系统的表结构
"""

import mysql.connector
from mysql.connector import Error

# 数据库配置
DB_CONFIG = {
    'host': 'rm-bp1ljbjb34n55su6uko.mysql.rds.aliyuncs.com',
    'port': 3306,
    'database': 'markdown_manager',
    'user': 'markdown_user',
    'password': 'Syp19960424'
}

# SQL语句 - 修正版本
SQL_STATEMENTS = [
    # 1. 创建新的文档片段表（避免与现有documents表冲突）
    """
    CREATE TABLE IF NOT EXISTS doc_chunks (
      id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
      document_id BIGINT UNSIGNED NOT NULL,
      chunk_index INT NOT NULL,
      content MEDIUMTEXT NOT NULL,
      token_count INT,
      milvus_pk BIGINT,
      metadata JSON,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    ) ENGINE=InnoDB
    """,
    
    # 2. 添加全文检索索引
    "ALTER TABLE doc_chunks ADD FULLTEXT KEY ft_content (content)",
    
    # 3. 创建性能优化索引 (移除IF NOT EXISTS，MySQL 5.x不支持)
    "CREATE INDEX idx_doc_source ON documents(title)",
    "CREATE INDEX idx_doc_created ON documents(created_at)", 
    "CREATE INDEX idx_chunk_document_id ON doc_chunks(document_id)",
    "CREATE INDEX idx_chunk_milvus_pk ON doc_chunks(milvus_pk)",
    "CREATE INDEX idx_chunk_created ON doc_chunks(created_at)"
]

def main():
    connection = None
    try:
        # 连接数据库
        print("Connecting to MySQL database...")
        connection = mysql.connector.connect(**DB_CONFIG)
        
        if connection.is_connected():
            print(f"Successfully connected to MySQL database: {DB_CONFIG['database']}")
            
            cursor = connection.cursor()
            
            # Execute SQL statements
            for i, sql in enumerate(SQL_STATEMENTS, 1):
                try:
                    print(f"Executing SQL statement {i}/{len(SQL_STATEMENTS)}...")
                    cursor.execute(sql)
                    print(f"  Success")
                except Error as e:
                    if "Duplicate key name" in str(e):
                        print(f"  Index already exists, skipping")
                    else:
                        print(f"  Error: {e}")
            
            # Commit transaction
            connection.commit()
            print("\nAll changes committed")
            
            # Show table structure
            print("\n=== Table Structure ===")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print("Created tables:")
            for table in tables:
                print(f"  - {table[0]}")
                
            # Show documents table structure
            print("\n--- documents table structure ---")
            cursor.execute("DESCRIBE documents")
            for row in cursor.fetchall():
                print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")
                
            # Show doc_chunks table structure
            print("\n--- doc_chunks table structure ---")
            cursor.execute("DESCRIBE doc_chunks")
            for row in cursor.fetchall():
                print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")
                
            cursor.close()
            
    except Error as e:
        print(f"Database connection error: {e}")
        
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("\nMySQL connection closed")

if __name__ == "__main__":
    main()