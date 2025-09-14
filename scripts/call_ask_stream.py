#!/usr/bin/env python3
"""
Call /api/v1/ask/stream with given payload and print full SSE output.

Usage:
  python scripts/call_ask_stream.py
  python scripts/call_ask_stream.py --message "你的问题" [--top-k 15 --nprobe 96 ...]
"""

import asyncio
import argparse
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
    parser.add_argument('--message', default='交通大模型的应用发展')
    parser.add_argument('--top-k', type=int, default=15)
    parser.add_argument('--nprobe', type=int, default=96)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--per-doc-max', type=int, default=2)
    parser.add_argument('--mmr', action='store_true', default=True)
    parser.add_argument('--min-unique-docs', type=int, default=4)
    parser.add_argument('--rerank', action='store_true', default=True)
    parser.add_argument('--llm-model', default='deepseek-chat')
    args = parser.parse_args()

    # Ensure project root on path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    load_env()

    from app.main import app

    payload = {
        'message': args.message,
        'top_k': args.top_k,
        'nprobe': args.nprobe,
        'score_threshold': args.score_threshold,
        'per_doc_max': args.per_doc_max,
        'mmr': args.mmr,
        'min_unique_docs': args.min_unique_docs,
        'rerank': args.rerank,
        'llm_model': args.llm_model,
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test', timeout=None) as client:
        async with client.stream('POST', '/api/v1/ask/stream', json=payload) as r:
            print('status:', r.status_code)
            if r.status_code != 200:
                text = await r.aread()
                print(text.decode('utf-8', 'ignore'))
                return
            async for line in r.aiter_lines():
                if not line:
                    continue
                print(line)
                if line.strip() == 'data: [DONE]':
                    break


if __name__ == '__main__':
    asyncio.run(main())

