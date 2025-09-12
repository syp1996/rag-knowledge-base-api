import os
from typing import List, Tuple
import httpx

"""
Rerank utilities

Supports Aliyun DashScope/BaiLian rerank via HTTP API.
Configuration via env:
  - RERANK_PROVIDER (default: dashscope)
  - RERANK_MODEL    (default: text-rerank-v1)
  - DASHSCOPE_API_KEY (token for DashScope/BaiLian)
  - DASHSCOPE_BASE_URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)

Return value: list of (index, score) sorted by score desc.
"""


async def rerank_texts(query: str, texts: List[str], top_n: int | None = None) -> List[Tuple[int, float]]:
    provider = os.getenv("RERANK_PROVIDER", "dashscope").lower()
    if provider != "dashscope":
        raise ValueError(f"Unsupported RERANK_PROVIDER: {provider}")

    model = os.getenv("RERANK_MODEL", "text-rerank-v1")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY for rerank")

    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    # Prefer compatible endpoint if available; some deployments expose native API
    endpoints = [
        f"{base_url.rstrip('/')}/rerank",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank",
    ]

    payload = {
        "model": model,
        "query": query,
        "documents": texts,
        "top_n": top_n or len(texts),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err = None
    async with httpx.AsyncClient(timeout=60) as client:
        for url in endpoints:
            try:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                # Expected format: { data: { results: [ { index, relevance_score }, ... ] } } or flat list
                results = None
                if isinstance(data, dict):
                    if "results" in data:
                        results = data["results"]
                    elif "data" in data and isinstance(data["data"], dict) and "results" in data["data"]:
                        results = data["data"]["results"]
                if results is None:
                    # Try OpenAI-compatible style: choices with reranked indices is unlikely, but guard.
                    raise ValueError("Unexpected rerank response format")
                pairs: list[tuple[int, float]] = []
                for item in results:
                    idx = int(item.get("index"))
                    score = float(item.get("relevance_score", item.get("score", 0.0)))
                    pairs.append((idx, score))
                # Ensure desc order
                pairs.sort(key=lambda x: x[1], reverse=True)
                return pairs
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Rerank request failed: {last_err}")

