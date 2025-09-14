#!/usr/bin/env python3
"""
Call /api/v1/ask (non-stream) with adjustable retrieval params and print full JSON.

Usage:
  python scripts/call_ask.py \
    --query "交通大模型的应用发展" \
    --top-k 15 --nprobe 96 --score-threshold 0.05 \
    --per-doc-max 2 --mmr --min-unique-docs 4 --rerank \
    --llm-model deepseek-chat
"""

import argparse
import asyncio
import json
import httpx
from pathlib import Path
import sys


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='交通大模型的应用发展')
    parser.add_argument('--top-k', type=int, default=15)
    parser.add_argument('--nprobe', type=int, default=96)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--per-doc-max', type=int, default=2)
    parser.add_argument('--mmr', action='store_true', default=True)
    parser.add_argument('--min-unique-docs', type=int, default=4)
    parser.add_argument('--rerank', action='store_true', default=True)
    parser.add_argument('--llm-model', default='deepseek-chat')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    load_env()

    from app.main import app

    payload = {
        'query': args.query,
        'top_k': args.top_k,
        'nprobe': args.nprobe,
        'score_threshold': args.score_threshold,
        'per_doc_max': args.per_doc_max,
        'mmr': args.mmr,
        'min_unique_docs': args.min_unique_docs,
        'rerank': args.rerank,
        'llm_model': args.llm_model,
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test', timeout=300) as client:
        r = await client.post('/api/v1/ask', json=payload)
        print('status:', r.status_code)
        print(r.text)


if __name__ == '__main__':
    asyncio.run(main())

