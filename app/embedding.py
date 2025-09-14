# app/embedding.py
import os, httpx
from typing import List
from tenacity import retry, wait_random_exponential, stop_after_attempt

PROVIDER = os.getenv("EMBED_PROVIDER", "dashscope")

# 测试模式：返回模拟向量
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    调用第三方嵌入API获取文本向量
    支持多种供应商：cohere, openai, voyage
    """
    # 测试模式：返回模拟向量
    if TEST_MODE:
        import random
        vectors = []
        for text in texts:
            random.seed(hash(text) % 2**32)
            vector = [random.uniform(-1, 1) for _ in range(1024)]
            vectors.append(vector)
        return vectors
    if PROVIDER == "cohere":
        # Cohere embed v3（示例）: 维度按模型为 1024，多语/英文模型二选一
        url = "https://api.cohere.com/v1/embed"
        headers = {"Authorization": f"Bearer {os.environ['COHERE_API_KEY']}"}
        model = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json={
                "model": model,
                "texts": texts,
                "input_type": "search_document"  # 查询用 search_query
            })
            r.raise_for_status()
            return r.json()["embeddings"]

    if PROVIDER == "openai":
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json={"model": model, "input": texts})
            r.raise_for_status()
            data = r.json()["data"]
            return [d["embedding"] for d in data]

    if PROVIDER == "voyage":
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}"}
        model = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3")
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json={"model": model, "input": texts})
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]

    if PROVIDER == "dashscope":
        # 阿里云DashScope text-embedding-v4（OpenAI兼容模式）
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        url = f"{base_url}/embeddings"
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for dashscope embedding")
        headers = {"Authorization": f"Bearer {api_key}"}
        model = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")
        # 可选参数：维度、编码格式
        payload = {"model": model, "input": texts}
        try:
            dim = int(os.getenv("EMBED_DIM", "0"))
            if dim > 0:
                payload["dimensions"] = dim
        except Exception:
            pass
        payload["encoding_format"] = "float"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()["data"]
            return [d["embedding"] for d in data]

    raise ValueError(f"Unsupported embedding provider: {PROVIDER}")

async def embed_query(query: str) -> List[float]:
    """
    为查询文本生成嵌入向量
    """
    # 测试模式：返回模拟向量
    if TEST_MODE:
        import random
        random.seed(hash(query) % 2**32)
        return [random.uniform(-1, 1) for _ in range(1024)]
        
    if PROVIDER == "cohere":
        url = "https://api.cohere.com/v1/embed"
        headers = {"Authorization": f"Bearer {os.environ['COHERE_API_KEY']}"}
        model = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json={
                "model": model,
                "texts": [query],
                "input_type": "search_query"  # 查询模式
            })
            r.raise_for_status()
            return r.json()["embeddings"][0]
    
    # 其他提供商可以复用embed_texts
    return (await embed_texts([query]))[0]
