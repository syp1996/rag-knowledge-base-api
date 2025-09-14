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

    # 1) Prefer official DashScope SDK if available (as per console example)
    topn = top_n or len(texts)
    try:
        import dashscope  # type: ignore
        dashscope.api_key = api_key
        # SDK usage: dashscope.TextReRank.call(...)
        resp = dashscope.TextReRank.call(
            model=model,
            query=query,
            documents=texts,
            top_n=topn,
            return_documents=False,
        )
        # Extract results (prefer standard SDK .output.results)
        results = None
        out = getattr(resp, "output", None)
        if isinstance(out, dict) and "results" in out:
            results = out["results"]
        else:
            data = getattr(resp, "data", None)
            if isinstance(data, dict):
                if "results" in data:
                    results = data["results"]
                elif "output" in data and isinstance(data["output"], dict) and "results" in data["output"]:
                    results = data["output"]["results"]
        if results is None:
            raise RuntimeError("Unexpected DashScope rerank SDK response format")
        pairs: list[tuple[int, float]] = []
        for item in results:
            idx = int(item.get("index"))
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            pairs.append((idx, score))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
    except ImportError:
        # SDK not installed; fall back to HTTP
        pass
    except Exception as e:
        # If SDK path fails, surface the error for clarity
        raise RuntimeError(f"DashScope SDK rerank failed: {e}")
    
    # 2) HTTP fallback across multiple endpoint/payload shapes
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_path = os.getenv("RERANK_MODEL", model)
    endpoints = [
        f"{base_url.rstrip('/')}/rerank",
        f"{base_url.rstrip('/')}/rerank/{model_path}",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank",
        f"https://dashscope.aliyuncs.com/api/v1/services/rerank/{model_path}",
    ]

    # topn defined above
    # Prepare multiple payload shapes to adapt to different BaiLian deployments
    docs_list = texts
    docs_kv = [{"text": t} for t in texts]
    payloads = [
        {  # Cohere-like flat body
            "model": model,
            "query": query,
            "documents": docs_list,
            "top_n": topn,
        },
        {  # DashScope documented style (input/parameters)
            "model": model,
            "input": {
                "query": query,
                "documents": docs_list,
            },
            "parameters": {
                "top_n": topn,
            },
        },
        {  # Sometimes top_n is accepted under input
            "model": model,
            "input": {
                "query": query,
                "documents": docs_list,
                "top_n": topn,
            },
        },
        {  # Documents as objects with text field
            "model": model,
            "query": query,
            "documents": docs_kv,
            "top_n": topn,
        },
        {
            "model": model,
            "input": {"query": query, "documents": docs_kv},
            "parameters": {"top_n": topn},
        },
        {  # Some deployments expect a task indicator
            "model": model,
            "task": "rerank",
            "input": {"query": query, "documents": docs_kv},
            "parameters": {"top_n": topn},
        },
        {
            "model": model,
            "parameters": {"task": "rerank", "top_n": topn},
            "input": {"query": query, "documents": docs_kv},
        },
    ]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Token": api_key,
        "Content-Type": "application/json",
    }

    last_err = None
    async with httpx.AsyncClient(timeout=60) as client:
        for url in endpoints:
            for payload in payloads:
                try:
                    r = await client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    data = r.json()
                    # Expected formats to try
                    results = None
                    if isinstance(data, dict):
                        if "results" in data:
                            results = data["results"]
                        elif "data" in data and isinstance(data["data"], dict) and "results" in data["data"]:
                            results = data["data"]["results"]
                        elif "output" in data and isinstance(data["output"], dict) and "results" in data["output"]:
                            results = data["output"]["results"]
                    if results is None:
                        raise ValueError("Unexpected rerank response format")
                    pairs: list[tuple[int, float]] = []
                    for item in results:
                        idx = int(item.get("index"))
                        score = float(item.get("relevance_score", item.get("score", 0.0)))
                        pairs.append((idx, score))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    return pairs
                except Exception as e:
                    # Enrich error with response text if available
                    try:
                        import httpx as _hx
                        if isinstance(e, _hx.HTTPStatusError) and e.response is not None:
                            last_err = RuntimeError(f"{e} | body: {e.response.text[:500]}")
                        else:
                            last_err = e
                    except Exception:
                        last_err = e
                    continue
    raise RuntimeError(f"Rerank request failed: {last_err}")

