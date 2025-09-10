# app/ingest.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .deps import get_db, get_milvus
from .embedding import embed_texts
import tiktoken
import json
from typing import Optional

router = APIRouter()

def token_len(s: str) -> int:
    """计算文本的token长度"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(None),
    tags: str = Form(None),
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    文档入库：切分 → 嵌入 → 写入 Milvus & MySQL
    """
    try:
        # 读取文件内容
        content = await file.read()
        raw_text = content.decode("utf-8", "ignore")
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="File is empty or cannot be decoded")
        
        # 文本切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, 
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from file")
        
        # 生成嵌入向量
        print(f"Generating embeddings for {len(chunks)} chunks...")
        vectors = await embed_texts(chunks)
        
        # 开始数据库事务
        # 1. 插入文档记录到documents表
        doc_result = db.execute(sql_text("""
            INSERT INTO documents(title, source, uri, tags_json)
            VALUES (:title, :source, :uri, :tags)
        """), {
            "title": title or file.filename,
            "source": "upload",
            "uri": file.filename,
            "tags": tags if tags else None
        })
        doc_id = doc_result.lastrowid
        
        # 2. 准备Milvus数据
        milvus_rows = []
        for i, (content, vec) in enumerate(zip(chunks, vectors)):
            milvus_rows.append({
                "doc_id": int(doc_id),
                "chunk_index": i,
                "text": content[:1000],  # VARCHAR长度限制
                "vector": vec
            })
        
        # 3. 插入到Milvus
        print(f"Inserting {len(milvus_rows)} vectors to Milvus...")
        insert_result = milvus_client.insert(
            collection_name="kb_chunks", 
            data=milvus_rows
        )
        milvus_client.flush("kb_chunks")
        
        # 4. 获取Milvus主键（可选，用于回填MySQL）
        milvus_pks = insert_result.primary_keys if hasattr(insert_result, 'primary_keys') else []
        
        # 5. 插入chunk记录到doc_chunks表
        for i, content in enumerate(chunks):
            milvus_pk = milvus_pks[i] if i < len(milvus_pks) else None
            db.execute(sql_text("""
                INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk)
                VALUES (:doc_id, :chunk_index, :content, :token_count, :milvus_pk)
            """), {
                "doc_id": doc_id,
                "chunk_index": i,
                "content": content,
                "token_count": token_len(content),
                "milvus_pk": milvus_pk
            })
        
        # 提交事务
        db.commit()
        
        return {
            "success": True,
            "message": "Document ingested successfully",
            "document_id": doc_id,
            "chunks_count": len(chunks),
            "total_tokens": sum(token_len(chunk) for chunk in chunks)
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@router.get("/documents")
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """列出已入库的文档"""
    try:
        # 查询文档列表
        result = db.execute(sql_text("""
            SELECT d.id, d.title, d.source, d.uri, d.tags_json, d.created_at,
                   COUNT(c.id) as chunks_count,
                   SUM(c.token_count) as total_tokens
            FROM documents d
            LEFT JOIN doc_chunks c ON d.id = c.document_id
            GROUP BY d.id, d.title, d.source, d.uri, d.tags_json, d.created_at
            ORDER BY d.created_at DESC
            LIMIT :limit OFFSET :offset
        """), {"limit": limit, "offset": offset})
        
        documents = []
        for row in result.fetchall():
            doc = {
                "id": row.id,
                "title": row.title,
                "source": row.source,
                "uri": row.uri,
                "tags": json.loads(row.tags_json) if row.tags_json else [],
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "chunks_count": row.chunks_count or 0,
                "total_tokens": row.total_tokens or 0
            }
            documents.append(doc)
        
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """删除文档及其相关数据"""
    try:
        # 检查文档是否存在
        doc_result = db.execute(sql_text("""
            SELECT id FROM documents WHERE id = :doc_id
        """), {"doc_id": doc_id})
        
        if not doc_result.fetchone():
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 从Milvus删除向量（按doc_id过滤）
        milvus_client.delete(
            collection_name="kb_chunks",
            filter=f"doc_id == {doc_id}"
        )
        
        # 从MySQL删除（CASCADE会自动删除chunks）
        db.execute(sql_text("""
            DELETE FROM documents WHERE id = :doc_id
        """), {"doc_id": doc_id})
        
        db.commit()
        
        return {"success": True, "message": f"Document {doc_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")