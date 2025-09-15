#!/usr/bin/env python3
"""
Initialize Milvus for this project:
- Ensure collection `kb_chunks` exists with expected schema
- Create IVF_FLAT index with COSINE metric

Reads config from .env: MILVUS_URI, MILVUS_TOKEN, EMBED_DIM
"""

import os
import sys


def load_env():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


def main():
    load_env()
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    token = os.getenv("MILVUS_TOKEN")
    try:
        dim = int(os.getenv("EMBED_DIM", "1024"))
    except Exception:
        dim = 1024
    if dim <= 0:
        dim = 1024

    connections.connect(alias="default", uri=uri, token=token)
    name = "kb_chunks"

    if not utility.has_collection(name):
        print(f"Creating collection '{name}' with dim={dim}...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description="RAG chunks")
        coll = Collection(name=name, schema=schema)
        # Create index
        print("Creating IVF_FLAT index (COSINE)...")
        coll.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024},
            },
        )
        print("Index created.")
        coll.load()
        print("Collection created and loaded.")
    else:
        print(f"Collection '{name}' already exists.")
        coll = Collection(name)
        try:
            coll.load()
            print("Collection loaded.")
        except Exception as e:
            print(f"Load skipped: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Init failed: {e}")
        sys.exit(1)
