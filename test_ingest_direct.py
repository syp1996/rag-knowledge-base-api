#!/usr/bin/env python3
"""
直接测试入库功能，绕过API调用
"""
import asyncio
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

async def test_ingest():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from app.embedding import embed_texts
    from app.deps import SessionLocal, milvus
    from sqlalchemy import text as sql_text
    import tiktoken
    
    def token_len(s: str) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    
    # 读取测试文件
    with open("test.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"Raw text length: {len(raw_text)} characters")
    
    # 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, 
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
    )
    chunks = splitter.split_text(raw_text)
    print(f"Split into {len(chunks)} chunks")
    
    # 生成嵌入向量
    print("Generating embeddings...")
    vectors = await embed_texts(chunks)
    print(f"Generated {len(vectors)} vectors of dimension {len(vectors[0])}")
    
    # 数据库操作
    db = SessionLocal()
    try:
        # 插入文档（使用现有表结构，content为JSON字段）
        import json
        doc_result = db.execute(sql_text("""
            INSERT INTO documents(user_id, title, excerpt, content, tags_json, is_pinned)
            VALUES (:user_id, :title, :excerpt, :content, :tags, :is_pinned)
        """), {
            "user_id": 1,  # 假设测试用户ID为1
            "title": "Test Document",
            "excerpt": "Test document for RAG system",
            "content": json.dumps({"text": raw_text}),
            "tags": None,
            "is_pinned": 0
        })
        doc_id = doc_result.lastrowid
        print(f"Inserted document with ID: {doc_id}")
        
        # 准备Milvus数据
        milvus_rows = []
        for i, (content, vec) in enumerate(zip(chunks, vectors)):
            # 确保text字段不超过1000字符（考虑Unicode编码）
            preview_text = content[:500] if len(content) > 500 else content
            # 再次检查编码后的字节长度
            while len(preview_text.encode('utf-8')) > 1000 and len(preview_text) > 0:
                preview_text = preview_text[:-10]
            
            milvus_rows.append({
                "doc_id": int(doc_id),
                "chunk_index": i,
                "text": preview_text,
                "vector": vec
            })
            print(f"Chunk {i}: text length = {len(preview_text)}, bytes = {len(preview_text.encode('utf-8'))}")
        
        # 插入到Milvus
        print("Inserting to Milvus...")
        insert_result = milvus.insert(
            collection_name="kb_chunks", 
            data=milvus_rows
        )
        milvus.flush("kb_chunks")
        print(f"Milvus insert result: {insert_result}")
        
        # 插入chunk记录
        for i, content in enumerate(chunks):
            db.execute(sql_text("""
                INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk)
                VALUES (:doc_id, :chunk_index, :content, :token_count, :milvus_pk)
            """), {
                "doc_id": doc_id,
                "chunk_index": i,
                "content": content,
                "token_count": token_len(content),
                "milvus_pk": None
            })
        
        db.commit()
        print("Successfully ingested document!")
        return doc_id
        
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    doc_id = asyncio.run(test_ingest())
    print(f"Document ingested with ID: {doc_id}")
