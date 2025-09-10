#!/usr/bin/env python3
"""
测试用嵌入函数 - 返回模拟向量避免API调用
"""
import random
from typing import List

async def mock_embed_texts(texts: List[str]) -> List[List[float]]:
    """
    返回模拟的1024维向量用于测试
    """
    vectors = []
    for text in texts:
        # 基于文本内容生成确定性的随机向量
        random.seed(hash(text) % 2**32)
        vector = [random.uniform(-1, 1) for _ in range(1024)]
        vectors.append(vector)
    return vectors

if __name__ == "__main__":
    import asyncio
    
    # 测试
    texts = ["这是测试文本1", "这是测试文本2"]
    vectors = asyncio.run(mock_embed_texts(texts))
    print(f"Generated {len(vectors)} vectors, each with {len(vectors[0])} dimensions")