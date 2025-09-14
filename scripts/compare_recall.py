#!/usr/bin/env python3
import asyncio
import httpx
from collections import Counter
from pathlib import Path
import sys

def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env loaded")
    except Exception:
        pass

async def run(query: str):
    from app.main import app
    base_payload = {"query": query, "top_k": 5, "nprobe": 16, "rerank": True}
    wide_payload = {
        "query": query,
        "top_k": 15,
        "nprobe": 96,
        "rerank": True,
        "score_threshold": 0.05,
        "per_doc_max": 2,
        "mmr": True,
        "min_unique_docs": 4,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test', timeout=180) as client:
        for path in ('/api/v1/search','/api/v1/search/hybrid'):
            for name, payload in (("BASE", base_payload), ("WIDE", wide_payload)):
                r = await client.post(path, json=payload)
                d = r.json(); items = d.get('results', [])
                doc_counts = Counter([it['doc_id'] for it in items])
                print(f"\n[{name}] {path} status={r.status_code} total={d.get('total_hits')} uniq_docs={len(doc_counts)}")
                print(' per-doc:', doc_counts.most_common())
                for i, it in enumerate(items[:8], start=1):
                    print(f"  #{i} score={it['score']:.4f} doc_id={it['doc_id']} chunk={it['chunk_index']} title={it['title'][:36]}")

def main():
    load_env()
    q = sys.argv[1] if len(sys.argv) > 1 else '交通大模型的应用发展'
    asyncio.run(run(q))

if __name__ == '__main__':
    main()

