#!/usr/bin/env python3
"""
Batch (re)vectorize documents into Milvus.

Scenarios covered:
1) Documents without any chunks yet -> split from documents.content, embed, insert Milvus + doc_chunks
2) Existing doc_chunks rows where milvus_pk is NULL -> embed the chunk content and insert into Milvus, then update milvus_pk

Usage:
  python scripts/batch_vectorize.py [--limit N]

Env:
  - Uses .env for DB, Milvus, and embedding provider (DashScope by default)
"""

import os
import sys
import json
import argparse
from pathlib import Path


def load_env():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print("✅ .env loaded")
    except Exception:
        print("⚠️  Could not load .env; continue with process env")


def truncate_utf8_bytes(s: str, max_bytes: int = 1000) -> str:
    if s is None:
        return ""
    b = s.encode('utf-8')
    if len(b) <= max_bytes:
        return s
    lo, hi = 0, len(s)
    res = s
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(s[:mid].encode('utf-8')) <= max_bytes:
            res = s[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return res


async def embed_in_batches(texts, batch_size: int = 10, delay: float = 0.2):
    from app.embedding import embed_texts
    import asyncio
    vectors_all = []
    for i in range(0, len(texts), batch_size):
        part = texts[i:i+batch_size]
        vecs = await embed_texts(part)
        vectors_all.extend(vecs)
        if delay:
            await asyncio.sleep(delay)
    return vectors_all


async def process_chunks_without_vectors(db, milvus, limit: int | None):
    from sqlalchemy import text as sql_text
    # Fetch chunks missing vectors
    q = """
        SELECT document_id, chunk_index, content
        FROM doc_chunks
        WHERE milvus_pk IS NULL
        ORDER BY document_id, chunk_index
    """
    if limit:
        q += " LIMIT :limit"
    rows = db.execute(sql_text(q), {"limit": limit} if limit else {}).fetchall()
    if not rows:
        print("No existing chunks missing vectors.")
        return 0
    print(f"Found {len(rows)} chunk(s) with missing vectors. Embedding and inserting...")
    # Group by document for stable order
    from collections import defaultdict
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[int(r.document_id)].append((int(r.chunk_index), str(r.content)))
    total_inserted = 0
    for doc_id, items in by_doc.items():
        items.sort(key=lambda x: x[0])
        texts = [c for _, c in items]
        vecs = await embed_in_batches(texts, batch_size=10, delay=0.2)
        milvus_rows = []
        for (chunk_idx, text), vec in zip(items, vecs):
            milvus_rows.append({
                "doc_id": int(doc_id),
                "chunk_index": int(chunk_idx),
                "text": truncate_utf8_bytes(text, 1000),
                "vector": vec,
            })
        res = milvus.insert(collection_name="kb_chunks", data=milvus_rows)
        milvus.flush("kb_chunks")
        # Try to reflect PKs back
        pks = []
        if hasattr(res, 'primary_keys'):
            pks = list(getattr(res, 'primary_keys'))
        elif isinstance(res, dict) and 'ids' in res:
            pks = list(res['ids'])
        if pks:
            for (chunk_idx, _), pk in zip(items, pks):
                db.execute(sql_text(
                    "UPDATE doc_chunks SET milvus_pk = :pk WHERE document_id = :doc_id AND chunk_index = :idx"
                ), {"pk": int(pk), "doc_id": int(doc_id), "idx": int(chunk_idx)})
            db.commit()
        total_inserted += len(milvus_rows)
        print(f"Doc {doc_id}: inserted {len(milvus_rows)} vectors.")
    print(f"Inserted total vectors for missing chunks: {total_inserted}")
    return total_inserted


def normalize_content_json(content_json: dict) -> str:
    if not content_json:
        return ""
    if isinstance(content_json, str):
        return content_json
    # Prefer markdown, then text, then strip tags from html
    if content_json.get('markdown'):
        return str(content_json['markdown'])
    if content_json.get('text'):
        return str(content_json['text'])
    if content_json.get('html'):
        import re
        return re.sub(r"<[^>]+>", "", str(content_json['html']))
    return ""


async def process_documents_without_chunks(db, milvus, limit: int | None):
    from sqlalchemy import text as sql_text
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # Find docs with zero chunks
    q = """
        SELECT d.id, d.title, d.content
        FROM documents d
        LEFT JOIN doc_chunks c ON c.document_id = d.id
        GROUP BY d.id
        HAVING COUNT(c.id) = 0
        ORDER BY d.id ASC
    """
    if limit:
        q += " LIMIT :limit"
    rows = db.execute(sql_text(q), {"limit": limit} if limit else {}).fetchall()
    if not rows:
        print("No documents without chunks.")
        return 0
    print(f"Found {len(rows)} document(s) without chunks. Embedding and inserting...")
    # Use env chunk params if provided
    try:
        size = int(os.getenv('CHUNK_SIZE', '500'))
    except Exception:
        size = 500
    try:
        overlap = int(os.getenv('CHUNK_OVERLAP', '120'))
    except Exception:
        overlap = 120
    if size <= 0:
        size = 500
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = max(0, size // 4)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
    )
    total_chunks = 0
    from sqlalchemy import text as sql_text
    for r in rows:
        doc_id = int(r.id)
        title = getattr(r, 'title', '')
        content = r.content
        try:
            content_json = json.loads(content) if isinstance(content, (bytes, bytearray)) else (content if isinstance(content, dict) else json.loads(content))
        except Exception:
            content_json = {"text": str(content) if content else ""}
        raw_text = normalize_content_json(content_json).strip()
        if not raw_text:
            print(f"Doc {doc_id} has empty content; skip.")
            continue
        chunks = splitter.split_text(raw_text)
        if not chunks:
            print(f"Doc {doc_id} split into 0 chunks; skip.")
            continue
        vecs = await embed_in_batches(chunks, batch_size=10, delay=0.2)
        milvus_rows = []
        for i, (content_text, vec) in enumerate(zip(chunks, vecs)):
            milvus_rows.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "text": truncate_utf8_bytes(content_text, 1000),
                "vector": vec,
            })
        res = milvus.insert(collection_name="kb_chunks", data=milvus_rows)
        milvus.flush("kb_chunks")
        # Insert doc_chunks rows
        for i, content_text in enumerate(chunks):
            db.execute(sql_text(
                "INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk) VALUES (:doc_id, :idx, :content, :tok, NULL)"
            ), {"doc_id": doc_id, "idx": i, "content": content_text, "tok": len(content_text)})
        db.commit()
        total_chunks += len(chunks)
        print(f"Doc {doc_id} '{title}' -> chunks {len(chunks)} inserted.")
    print(f"Inserted total chunks from documents: {total_chunks}")
    return total_chunks


async def run(limit: int | None):
    # Ensure project root import
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    load_env()
    from app.deps import SessionLocal, milvus
    db = SessionLocal()
    try:
        inserted_from_chunks = await process_chunks_without_vectors(db, milvus, limit)
        inserted_from_docs = await process_documents_without_chunks(db, milvus, limit)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Max rows per phase')
    args = parser.parse_args()
    import asyncio
    asyncio.run(run(args.limit))


if __name__ == '__main__':
    main()
