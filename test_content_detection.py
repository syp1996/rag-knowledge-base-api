#!/usr/bin/env python3
"""
测试RAG更新接口的内容检测机制
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_content_detection():
    print("🧪 测试RAG更新接口的内容检测机制\n")
    
    # 1. 先创建一个测试文档
    print("1️⃣ 创建测试文档...")
    create_payload = {
        "title": "内容检测测试文档",
        "content": "这是原始内容，用于测试内容检测机制。"
    }
    
    response = requests.post(f"{BASE_URL}/ingest/text", json=create_payload)
    if response.status_code != 200:
        print(f"❌ 创建文档失败: {response.status_code} - {response.text}")
        return
        
    doc_data = response.json()
    doc_id = doc_data["document_id"]
    print(f"✅ 文档创建成功，ID: {doc_id}")
    print(f"   切块数: {doc_data['chunks_count']}, tokens: {doc_data['total_tokens']}")
    
    # 2. 测试相同内容更新（应该跳过重新索引）
    print("\n2️⃣ 测试相同内容更新...")
    update_payload = {
        "title": "内容检测测试文档",
        "content": "这是原始内容，用于测试内容检测机制。"  # 完全相同的内容
    }
    
    start_time = time.time()
    response = requests.put(f"{BASE_URL}/ingest/text/{doc_id}", json=update_payload)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 更新成功 (耗时: {end_time - start_time:.2f}秒)")
        print(f"   消息: {result['message']}")
        print(f"   内容是否变化: {result.get('content_changed', 'N/A')}")
        print(f"   是否重新索引: {result.get('reindexed', 'N/A')}")
        print(f"   切块数: {result['chunks_count']}")
    else:
        print(f"❌ 更新失败: {response.status_code} - {response.text}")
    
    # 3. 测试不同内容更新（应该重新索引）
    print("\n3️⃣ 测试不同内容更新...")
    update_payload = {
        "title": "内容检测测试文档（已修改）",
        "content": "这是修改后的内容，应该触发重新索引和向量化。添加了更多的文字来测试切块效果。"
    }
    
    start_time = time.time()
    response = requests.put(f"{BASE_URL}/ingest/text/{doc_id}", json=update_payload)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 更新成功 (耗时: {end_time - start_time:.2f}秒)")
        print(f"   消息: {result['message']}")
        print(f"   内容是否变化: {result.get('content_changed', 'N/A')}")
        print(f"   是否重新索引: {result.get('reindexed', 'N/A')}")
        print(f"   切块数: {result['chunks_count']}, tokens: {result['total_tokens']}")
    else:
        print(f"❌ 更新失败: {response.status_code} - {response.text}")
    
    # 4. 测试强制重新索引（相同内容但强制重新索引）
    print("\n4️⃣ 测试强制重新索引...")
    update_payload = {
        "content": "这是修改后的内容，应该触发重新索引和向量化。添加了更多的文字来测试切块效果。",  # 相同内容
        "force_reindex": True  # 强制重新索引
    }
    
    start_time = time.time()
    response = requests.put(f"{BASE_URL}/ingest/text/{doc_id}", json=update_payload)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 强制重新索引成功 (耗时: {end_time - start_time:.2f}秒)")
        print(f"   消息: {result['message']}")
        print(f"   内容是否变化: {result.get('content_changed', 'N/A')}")
        print(f"   是否重新索引: {result.get('reindexed', 'N/A')}")
        print(f"   切块数: {result['chunks_count']}")
    else:
        print(f"❌ 强制重新索引失败: {response.status_code} - {response.text}")
    
    # 5. 测试仅标题更新
    print("\n5️⃣ 测试仅标题更新...")
    update_payload = {
        "title": "仅标题更新测试"
        # 不提供content
    }
    
    start_time = time.time()
    response = requests.put(f"{BASE_URL}/ingest/text/{doc_id}", json=update_payload)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 仅标题更新成功 (耗时: {end_time - start_time:.2f}秒)")
        print(f"   消息: {result['message']}")
        print(f"   新标题: {result['title']}")
        print(f"   切块数: {result['chunks_count']}")
    else:
        print(f"❌ 仅标题更新失败: {response.status_code} - {response.text}")
    
    print(f"\n🏁 测试完成！测试文档ID: {doc_id}")

if __name__ == "__main__":
    test_content_detection()