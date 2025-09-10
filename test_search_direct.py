#!/usr/bin/env python3
"""
直接测试搜索功能
"""
import asyncio
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

async def test_search():
    from app.embedding import embed_query
    from app.deps import SessionLocal, milvus
    from sqlalchemy import text as sql_text
    
    query = "Milvus 是什么"
    print(f"Query: {query}")
    
    # 生成查询向量
    print("Generating query embedding...")
    query_vector = await embed_query(query)
    print(f"Query vector dimension: {len(query_vector)}")
    
    # 执行向量搜索
    search_params = {
        "metric_type": "COSINE", 
        "params": {"nprobe": 16}
    }
    
    print("Searching in Milvus...")
    hits = milvus.search(
        collection_name="kb_chunks",
        data=[query_vector],
        anns_field="vector",
        limit=5,
        search_params=search_params,
        output_fields=["doc_id", "chunk_index", "text"]
    )[0]
    
    print(f"Found {len(hits)} results:")
    for i, hit in enumerate(hits):
        score = hit["distance"]
        doc_id = hit["entity"]["doc_id"]
        chunk_index = hit["entity"]["chunk_index"]
        preview = hit["entity"]["text"]
        print(f"  {i+1}. Score: {score:.4f} | Doc: {doc_id} | Chunk: {chunk_index}")
        print(f"     Preview: {preview[:100]}...")
    
    # 获取MySQL完整内容
    if hits:
        doc_chunk_pairs = [(h["entity"]["doc_id"], h["entity"]["chunk_index"]) for h in hits]
        pairs_str = ",".join([f"({doc_id},{chunk_idx})" for doc_id, chunk_idx in doc_chunk_pairs])
        
        db = SessionLocal()
        try:
            mysql_result = db.execute(sql_text(f"""
                SELECT d.id, d.title, c.document_id, c.chunk_index, c.content
                FROM doc_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE (c.document_id, c.chunk_index) IN ({pairs_str})
            """))
            
            print("\nMySQL full content:")
            for row in mysql_result.fetchall():
                print(f"  Doc {row.document_id}: {row.title}")
                print(f"  Content: {row.content[:200]}...")
                
        except Exception as e:
            print(f"MySQL query error: {e}")
        finally:
            db.close()
    
    return len(hits)

if __name__ == "__main__":
    hits_count = asyncio.run(test_search())
    print(f"\nSearch completed, found {hits_count} results")