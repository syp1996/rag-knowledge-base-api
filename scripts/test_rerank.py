#!/usr/bin/env python3
"""
Quick local test for Aliyun BaiLian/DashScope rerank API

Usage:
  python scripts/test_rerank.py

Reads env via python-dotenv if available. Requires:
  - RERANK_PROVIDER=dashscope (default)
  - RERANK_MODEL (default: text-rerank-v1)
  - DASHSCOPE_API_KEY (required)
  - DASHSCOPE_BASE_URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
"""

import asyncio
import os
import sys
from pathlib import Path


def _load_env():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print("✅ .env loaded")
    except Exception:
        print("⚠️  Could not load .env; using process env only")


async def _run():
    # Ensure project root on sys.path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.rerank import rerank_texts

    provider = os.getenv("RERANK_PROVIDER", "dashscope")
    model = os.getenv("RERANK_MODEL", "text-rerank-v1")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("DASHSCOPE_API_KEY", "")

    print("RERANK_PROVIDER:", provider)
    print("RERANK_MODEL:", model)
    print("DASHSCOPE_BASE_URL:", base_url)
    print("DASHSCOPE_API_KEY set:", bool(api_key))

    # Minimal sample
    query = "测试：哪个颜色是冷色调？"
    docs = [
        "蓝色是一种冷色调，常用于表现安静和理性。",
        "红色属于暖色调，表现热情、危险或警示。",
        "绿色既可偏冷也可偏暖，取决于色相和饱和度。",
    ]

    print("\n== Test via app.rerank (HTTP/SDK auto) ==")
    try:
        order = await rerank_texts(query, docs, top_n=3)
        print("✅ Rerank success:", order)
        for idx, score in order:
            print(f"  - [{score:.4f}] {docs[idx]}")
    except Exception as e:
        print("❌ Rerank failed:", e)
        print("Hint: Ensure DASHSCOPE_API_KEY is valid and the Base URL matches your console settings.")

    print("\n== Test via dashscope SDK direct ==")
    try:
        import dashscope  # type: ignore
        dashscope.api_key = api_key
        resp = dashscope.TextReRank.call(
            model=model,
            query=query,
            documents=docs,
            top_n=3,
            return_documents=True,
        )
        sc = getattr(resp, 'status_code', None)
        print("status_code:", sc)
        # Try to extract results
        out = getattr(resp, 'output', None)
        if isinstance(out, dict) and 'results' in out:
            results = out['results']
        elif hasattr(resp, 'to_dict'):
            d = resp.to_dict()
            results = d.get('output', {}).get('results') or d.get('results')
        else:
            results = None
        if results:
            print("SDK results count:", len(results))
            for item in results:
                idx = int(item.get('index'))
                score = float(item.get('relevance_score', item.get('score', 0.0)))
                print(f"  - [{score:.4f}] {docs[idx]}")
        else:
            print("SDK raw response:", resp)
    except Exception as e:
        print("SDK call failed:", e)


if __name__ == "__main__":
    _load_env()
    asyncio.run(_run())
