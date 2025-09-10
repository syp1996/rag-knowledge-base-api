# tools/init_milvus.py
import os
from pymilvus import MilvusClient, DataType

# 从环境变量读取配置
DIM = int(os.getenv("EMBED_DIM", "1024"))  # 例如 Cohere multilingual v3.0 -> 1024
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", None)

def init_milvus_collection():
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
    collection_name = "kb_chunks"
    
    # 检查集合是否已存在
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Dropping...")
        client.drop_collection(collection_name)
    
    # 创建集合（默认 metric=COSINE，可指定）
    schema = client.create_schema(
        auto_id=True,
        description="KB chunks for RAG system"
    )
    
    # 添加字段
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="doc_id", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
    
    # （可选）混合检索：稀疏向量字段
    # schema.add_field(field_name="sparse", datatype="SPARSE_FLOAT_VECTOR")
    
    # 创建集合
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        shards_num=2,
        # 指定默认度量（也可默认 COSINE）
        consistency_level="Bounded"
    )
    print(f"Collection '{collection_name}' created successfully")
    
    # 构建索引：IVF_FLAT（基础版）
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 2048}  # 视规模调优
    )
    
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.load_collection(collection_name)
    
    print("Index built & collection loaded successfully")
    
    return client

if __name__ == "__main__":
    try:
        client = init_milvus_collection()
        print("Milvus initialization completed!")
    except Exception as e:
        print(f"Error initializing Milvus: {e}")
        raise