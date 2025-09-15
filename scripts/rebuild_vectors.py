#!/usr/bin/env python3
"""
Rebuild all vectors in Milvus from MySQL doc_chunks.

Steps:
 1) Optional: drop and recreate collection `kb_chunks`
 2) Read all rows from `doc_chunks` (document_id, chunk_index, content)
 3) Embed in batches using app.embedding
 4) Insert into Milvus and update doc_chunks.milvus_pk

Usage:
  python -X utf8 scripts/rebuild_vectors.py [--drop]
"""

import os
import sys
from pathlib import Path
import argparse

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_env():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


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


def ensure_collection(dim: int):
    from pymilvus import MilvusClient
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    token = os.getenv("MILVUS_TOKEN")
    client = MilvusClient(uri=uri, token=token)
    name = "kb_chunks"
    existing = set(client.list_collections())
    if name not in existing:
        schema = {
            "auto_id": True,
            "description": "RAG chunks",
            "fields": [
                {"name": "id", "data_type": "INT64", "is_primary": True, "auto_id": True},
                {"name": "doc_id", "data_type": "INT64"},
                {"name": "chunk_index", "data_type": "INT64"},
                {"name": "text", "data_type": "VARCHAR", "max_length": 1000},
                {"name": "vector", "data_type": "FLOAT_VECTOR", "dim": dim},
            ],
        }
        client.create_collection(collection_name=name, schema=schema)
    # Ensure index
    try:
        idxs = client.list_indexes(collection_name=name)
    except Exception:
        idxs = []
    if not idxs:
        client.create_index(
            collection_name=name,
            index_name="idx_vector_ivf",
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024},
            },
        )
    try:
        client.load_collection(name)
    except Exception:
        pass
    return client


def drop_collection_if_requested(drop: bool):
    if not drop:
        return
    from pymilvus import MilvusClient
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    token = os.getenv("MILVUS_TOKEN")
    client = MilvusClient(uri=uri, token=token)
    name = "kb_chunks"
    try:
        client.drop_collection(name)
        print("Dropped collection 'kb_chunks'.")
    except Exception as e:
        print(f"Drop skipped: {e}")


async def rebuild_all(drop: bool):
    load_env()
    try:
        dim = int(os.getenv("EMBED_DIM", "1024"))
    except Exception:
        dim = 1024
    if dim <= 0:
        dim = 1024

    drop_collection_if_requested(drop)
    client = ensure_collection(dim)

    # DB session
    from app.deps import SessionLocal
    db = SessionLocal()
    try:
        from sqlalchemy import text as sql_text
        rows = db.execute(sql_text(
            "SELECT document_id, chunk_index, content FROM doc_chunks ORDER BY document_id, chunk_index"
        )).fetchall()
    finally:
        db.close()

    if not rows:
        print("No rows in doc_chunks to rebuild.")
        return

    print(f"Rebuilding vectors for {len(rows)} chunks...")
    # Group into batches to keep PKs aligned per insert
    batch = 200
    import asyncio
    from sqlalchemy import text as sql_text
    db = SessionLocal()
    try:
        total = 0
        for i in range(0, len(rows), batch):
            part = rows[i:i+batch]
            texts = [str(r.content) for r in part]
            vecs = await embed_in_batches(texts, batch_size=10, delay=0.2)
            milvus_rows = []
            for r, vec in zip(part, vecs):
                milvus_rows.append({
                    "doc_id": int(r.document_id),
                    "chunk_index": int(r.chunk_index),
                    "text": truncate_utf8_bytes(str(r.content), 1000),
                    "vector": vec,
                })
            res = client.insert(collection_name="kb_chunks", data=milvus_rows)
            client.flush("kb_chunks")
            # Reflect PKs back when available
            pks = []
            if hasattr(res, 'primary_keys'):
                pks = list(getattr(res, 'primary_keys'))
            elif isinstance(res, dict) and 'ids' in res:
                pks = list(res['ids'])
            if pks:
                for row, pk in zip(part, pks):
                    db.execute(sql_text(
                        "UPDATE doc_chunks SET milvus_pk = :pk WHERE document_id = :doc AND chunk_index = :idx"
                    ), {"pk": int(pk), "doc": int(row.document_id), "idx": int(row.chunk_index)})
                db.commit()
            total += len(milvus_rows)
            print(f"Inserted {len(milvus_rows)} vectors... (total {total})")
    finally:
        db.close()
    print("Rebuild completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop', action='store_true', help='Drop and recreate collection before rebuild')
    args = parser.parse_args()
    import asyncio
    asyncio.run(rebuild_all(args.drop))


if __name__ == '__main__':
    main()

