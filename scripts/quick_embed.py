#!/usr/bin/env python3
import os, sys
from pathlib import Path

# ensure project root on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
    print(".env loaded")
except Exception as e:
    print(f"load .env failed: {e}")

os.environ['TEST_MODE'] = 'false'

from app.embedding import embed_texts
import asyncio

async def run():
    vecs = await embed_texts(["测试一句话", "Milvus 向量检索"])
    print(len(vecs), len(vecs[0]))

asyncio.run(run())

