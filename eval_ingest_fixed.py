#!/usr/bin/env python3

import os
import sys
import glob
import json
from sqlalchemy import text
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Add the app directory to path
sys.path.append('.')

from app.deps import SessionLocal, milvus
from app.embedding import embed_texts

def token_len(s: str) -> int:
    """Calculate token length of text"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))

async def ingest_test_documents():
    """Ingest test documents properly with both documents and doc_chunks"""
    
    # Get all markdown files
    docs_dir = 'rag_recall_testpack_v1'
    md_files = glob.glob(f'{docs_dir}/*.md')
    md_files = [f for f in md_files if not f.endswith('README.md')]
    
    print(f'Found {len(md_files)} test documents to ingest')
    
    db = SessionLocal()
    milvus_client = milvus
    
    try:
        # First, clean up any existing test documents (IDs 57-65)
        print("Cleaning up existing test documents...")
        db.execute(text("DELETE FROM doc_chunks WHERE document_id BETWEEN 57 AND 65"))
        db.execute(text("DELETE FROM documents WHERE id BETWEEN 57 AND 65"))
        milvus_client.delete(collection_name="kb_chunks", filter="doc_id >= 57 and doc_id <= 65")
        db.commit()
        
        for file_path in md_files:
            filename = os.path.basename(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f'✗ Skipping empty file: {filename}')
                continue
            
            # Insert document into database
            doc_result = db.execute(text("""
                INSERT INTO documents(user_id, title, content, status, is_pinned, created_at, updated_at, tags_json)
                VALUES (1, :title, :content, 1, 0, NOW(), NOW(), :tags)
            """), {
                "title": filename.replace('.md', ''),
                "content": json.dumps({"text": content}),
                "tags": json.dumps(["test_data"])
            })
            
            doc_id = doc_result.lastrowid
            print(f'✓ Inserted document {filename} with ID {doc_id}')
            
            # Text splitting for embeddings
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900, 
                chunk_overlap=150,
                separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
            )
            chunks = splitter.split_text(content)
            
            if not chunks:
                print(f'  ✗ No chunks generated for {filename}')
                continue
            
            # Generate embeddings
            print(f'  Generating embeddings for {len(chunks)} chunks...')
            vectors = await embed_texts(chunks)
            
            # Insert chunks into doc_chunks table
            for i, chunk_content in enumerate(chunks):
                db.execute(text("""
                    INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, created_at)
                    VALUES (:doc_id, :chunk_index, :content, :token_count, NOW())
                """), {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "content": chunk_content,
                    "token_count": token_len(chunk_content)
                })
            
            # Prepare Milvus data
            milvus_rows = []
            for i, (chunk_content, vec) in enumerate(zip(chunks, vectors)):
                milvus_rows.append({
                    "doc_id": int(doc_id),
                    "chunk_index": i,
                    "text": chunk_content[:1000],  # VARCHAR length limit
                    "vector": vec
                })
            
            # Insert into Milvus
            print(f'  Inserting {len(milvus_rows)} vectors to Milvus...')
            insert_result = milvus_client.insert(
                collection_name="kb_chunks", 
                data=milvus_rows
            )
            milvus_client.flush("kb_chunks")
            
            print(f'  ✓ Completed {filename}: {len(chunks)} chunks')
        
        # Commit transaction
        db.commit()
        print(f'\n✓ Successfully ingested all {len(md_files)} test documents')
        
        # Verify the results
        result = db.execute(text("""
            SELECT d.id, d.title, COUNT(dc.id) as chunks_count
            FROM documents d
            LEFT JOIN doc_chunks dc ON d.id = dc.document_id
            WHERE d.id >= (SELECT MAX(id) - 10 FROM documents)
            GROUP BY d.id, d.title
            ORDER BY d.id DESC
        """))
        
        print("\nVerification - Recent documents with chunk counts:")
        for row in result:
            print(f"  Doc {row[0]} ({row[1]}): {row[2]} chunks")
        
    except Exception as e:
        db.rollback()
        print(f'✗ Error during ingestion: {e}')
        raise
    finally:
        db.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(ingest_test_documents())