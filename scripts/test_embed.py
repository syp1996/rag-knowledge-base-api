#!/usr/bin/env python3
"""
Quick test for embedding API connectivity.
It avoids TEST_MODE by forcing TEST_MODE=false before importing the module.

Usage:
  python scripts/test_embed.py

It will:
  1) Load .env
  2) Force TEST_MODE=false
  3) Try current provider from .env (EMBED_PROVIDER)
  4) If DashScope creds exist, try DashScope explicitly
"""

import os
import asyncio
from pathlib import Path


def load_env():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print("✅ .env loaded")
    except Exception:
        print("⚠️  Could not load .env; continue with process env")


async def try_provider(name: str, sample_texts: list[str]) -> None:
    # Set provider and ensure TEST_MODE disabled BEFORE import
    os.environ['TEST_MODE'] = 'false'
    os.environ['EMBED_PROVIDER'] = name
    # Re-import module to pick up new env
    import importlib
    if 'app.embedding' in list(importlib.sys.modules.keys()):
        del importlib.sys.modules['app.embedding']
    emb = importlib.import_module('app.embedding')
    print(f"\n== Provider: {name} ==")
    try:
        vectors = await emb.embed_texts(sample_texts)
        dims = len(vectors[0]) if vectors and vectors[0] is not None else 0
        print(f"✅ Success. Batch={len(vectors)}, dim={dims}")
    except Exception as e:
        print(f"❌ Failed: {e}")


async def main():
    # Ensure project root on path
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    load_env()

    sample_texts = [
        "今天天气不错，适合出去散步。",
        "Milvus 是一个高性能向量数据库。",
    ]

    # 1) Try current provider
    current = os.getenv('EMBED_PROVIDER', 'dashscope')
    await try_provider(current, sample_texts)

    # 2) Try DashScope explicitly if api key exists
    if os.getenv('DASHSCOPE_API_KEY'):
        os.environ['DASHSCOPE_BASE_URL'] = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        os.environ['DASHSCOPE_EMBED_MODEL'] = os.getenv('DASHSCOPE_EMBED_MODEL', 'text-embedding-v4')
        await try_provider('dashscope', sample_texts)


if __name__ == '__main__':
    asyncio.run(main())

