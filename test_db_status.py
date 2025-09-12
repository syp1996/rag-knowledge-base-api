#!/usr/bin/env python3
"""
Database status check script.

Checks whether documents have been chunked and vectorized.
Supports current schema using `doc_chunks` and will fallback to
legacy `document_embeddings` if found.
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
    """Check database status for chunking and vectorization"""
    db = SessionLocal()
    try:
        print("=== 数据库状态检查 ===\n")

        # 0) 统计文档总数
        result = db.execute(text("SELECT COUNT(*) as count FROM documents"))
        doc_count = result.fetchone()[0]
        print(f"文档总数: {doc_count}")

        # 判断使用哪张表：优先 doc_chunks，回退 document_embeddings
        table_name = None
        for candidate in ("doc_chunks", "document_embeddings"):
            r = db.execute(text("SHOW TABLES LIKE :t"), {"t": candidate})
            if r.fetchone():
                table_name = candidate
                break

        if not table_name:
            print("未找到 doc_chunks 或 document_embeddings 表，请先初始化数据库结构。")
            return

        print(f"\n使用的数据表: {table_name}")

        # 1) 总分块/嵌入条数
        result = db.execute(text(f"SELECT COUNT(*) as count FROM {table_name}"))
        total = result.fetchone()[0]
        print(f"总分块数/嵌入条数: {total}")

        # 2) 找出未被处理（无分块/嵌入）的文档
        if table_name == "doc_chunks":
            join_field = "document_id"
            id_field = "id"
        else:
            join_field = "document_id"
            id_field = "id"

        result = db.execute(text(f"""
            SELECT d.id, d.title
            FROM documents d
            LEFT JOIN {table_name} e ON e.{join_field} = d.id
            GROUP BY d.id, d.title
            HAVING COUNT(e.{id_field}) = 0
        """))
        unprocessed = result.fetchall()
        if unprocessed:
            print(f"\n未切分/未嵌入文档: {len(unprocessed)} 篇")
            for row in unprocessed[:20]:
                print(f" - ID: {row[0]}, 标题: {row[1]}")
            if len(unprocessed) > 20:
                print(f" ... 其余 {len(unprocessed) - 20} 篇省略")
        else:
            print("\n所有文档均已切分/嵌入")

        # 3) 若为 doc_chunks，检查是否缺少向量（milvus_pk 为空）
        if table_name == "doc_chunks":
            print("\n检查向量化状态（基于 milvus_pk）:")
            # 汇总每个文档的 chunk 总数与已向量化的数量
            result = db.execute(text("""
                SELECT 
                    d.id AS document_id,
                    d.title AS title,
                    COUNT(c.id) AS chunks_total,
                    SUM(CASE WHEN c.milvus_pk IS NOT NULL THEN 1 ELSE 0 END) AS vectorized
                FROM documents d
                LEFT JOIN doc_chunks c ON c.document_id = d.id
                GROUP BY d.id, d.title
                HAVING COUNT(c.id) > 0 AND SUM(CASE WHEN c.milvus_pk IS NOT NULL THEN 1 ELSE 0 END) < COUNT(c.id)
            """))
            partial = result.fetchall()
            if partial:
                print(f" - 存在 {len(partial)} 篇文档部分未向量化:")
                for row in partial[:20]:
                    doc_id, title, chunks_total, vectorized = row
                    print(f"   ID {doc_id} 《{title}》: {vectorized}/{chunks_total} 已写入向量库")
                if len(partial) > 20:
                    print(f"   ... 其余 {len(partial) - 20} 篇省略")
            else:
                print(" - 所有已有分块的文档均已完成向量写入")

            # 表结构概览
            print("\n表结构（doc_chunks）:")
            cols = db.execute(text("DESCRIBE doc_chunks")).fetchall()
            for col in cols:
                print(f" - {col[0]}: {col[1]}")
        else:
            # 旧表结构检查
            print("\n表结构（document_embeddings）:")
            cols = db.execute(text("DESCRIBE document_embeddings")).fetchall()
            for col in cols:
                print(f" - {col[0]}: {col[1]}")

            column_names = [c[0] for c in cols]
            if 'embedding_model' in column_names:
                result = db.execute(text("""
                    SELECT embedding_model, COUNT(*) AS chunks
                    FROM document_embeddings
                    GROUP BY embedding_model
                """))
                stats = result.fetchall()
                if stats:
                    print("\n嵌入模型覆盖率:")
                    for model, cnt in stats:
                        print(f" - 模型 {model}: {cnt} 个嵌入块")

    except Exception as e:
        print(f"数据库查询错误: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    test_database_status()
