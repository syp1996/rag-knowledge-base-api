# app/search.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from .deps import get_db, get_milvus
from .embedding import embed_query
from .rerank import rerank_texts
from typing import List, Optional, Dict, Any
import json

router = APIRouter()

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询文本")
    top_k: int = Field(default=8, ge=1, le=50, description="返回结果数量")
    nprobe: int = Field(default=16, ge=1, le=256, description="IVF_FLAT 搜索参数")
    rerank: bool = Field(default=True, description="是否启用重排")
    doc_ids: Optional[List[int]] = Field(default=None, description="限制搜索的文档ID列表")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度阈值")
    per_doc_max: Optional[int] = Field(default=None, ge=1, description="每个文档最多返回的片段数，用于结果多样化")
    mmr: bool = Field(default=False, description="启用简单的MMR式去冗与多样化（优先不同文档）")
    min_unique_docs: Optional[int] = Field(default=None, ge=1, description="保证至少覆盖的不同文档数量（两阶段：先保底覆盖，再按分数补齐）")

class SearchResult(BaseModel):
    doc_id: int
    chunk_index: int
    score: float
    title: str
    content: str
    preview: str
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    total_hits: int
    results: List[SearchResult]
    search_time_ms: Optional[float] = None

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    req: SearchRequest,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    语义检索：使用IVF_FLAT索引和COSINE距离进行向量搜索
    """
    try:
        # 根据多样化需求，获取更多候选以便后续筛选
        import time
        start_time = time.time()
        
        # 1. 生成查询向量
        query_vector = await embed_query(req.query)
        
        # 2. 构建Milvus搜索表达式
        search_filter = None
        if req.doc_ids:
            doc_ids_str = ",".join(map(str, req.doc_ids))
            search_filter = f"doc_id in [{doc_ids_str}]"
        
        # 3. 执行向量搜索
        search_params = {
            "metric_type": "COSINE", 
            "params": {"nprobe": req.nprobe}
        }
        
        # 如果需要多样化/重排，获取更多候选结果
        multiplier = 1
        if req.rerank:
            multiplier = max(multiplier, 3)
        if req.mmr:
            multiplier = max(multiplier, 3)
        search_limit = req.top_k * multiplier
        
        hits = milvus_client.search(
            collection_name="kb_chunks",
            data=[query_vector],
            anns_field="vector",
            limit=search_limit,
            search_params=search_params,
            output_fields=["doc_id", "chunk_index", "text"],
            filter=search_filter
        )[0]
        
        # 4. 过滤低分结果
        filtered_hits = [
            h for h in hits 
            if h["distance"] >= req.score_threshold
        ]
        
        if not filtered_hits:
            return SearchResponse(
                query=req.query,
                total_hits=0,
                results=[],
                search_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # 5. 获取对应的MySQL完整内容
        doc_chunk_pairs = [
            (h["entity"]["doc_id"], h["entity"]["chunk_index"]) 
            for h in filtered_hits
        ]
        
        # 构建SQL IN子句
        pairs_str = ",".join([f"({doc_id},{chunk_idx})" for doc_id, chunk_idx in doc_chunk_pairs])
        
        mysql_result = db.execute(sql_text(f"""
            SELECT d.id, d.title, d.tags_json,
                   c.document_id, c.chunk_index, c.content, c.metadata
            FROM doc_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE (c.document_id, c.chunk_index) IN ({pairs_str})
        """))
        
        # 创建MySQL数据的索引
        mysql_data = {}
        for row in mysql_result.fetchall():
            key = (row.document_id, row.chunk_index)
            mysql_data[key] = {
                "title": row.title,
                "content": row.content,
                "tags_json": row.tags_json,
                "metadata": json.loads(row.metadata) if row.metadata else None
            }
        
        # 6. 组合结果
        results = []
        for hit in filtered_hits:
            doc_id = hit["entity"]["doc_id"]
            chunk_index = hit["entity"]["chunk_index"]
            key = (doc_id, chunk_index)
            
            mysql_info = mysql_data.get(key, {})
            
            result = SearchResult(
                doc_id=doc_id,
                chunk_index=chunk_index,
                score=round(float(hit["distance"]), 4),
                title=mysql_info.get("title", "Unknown"),
                content=mysql_info.get("content", hit["entity"]["text"]),
                preview=hit["entity"]["text"][:200] + "..." if len(hit["entity"]["text"]) > 200 else hit["entity"]["text"],
                metadata=mysql_info.get("metadata")
            )
            results.append(result)
        
        # 7. 可选：重排序（基于外部Rerank服务，如阿里云百炼/DashScope）
        if req.rerank and results:
            try:
                texts = [ (r.content or r.preview or "")[:2048] for r in results ]
                order = await rerank_texts(req.query, texts, top_n=len(texts))
                # order: list[(index, score)] over original results
                ordered = [results[idx] for idx, _ in order if 0 <= idx < len(results)]
                # 保留重排分数到metadata
                for (idx, score), item in zip(order, ordered):
                    item.metadata = item.metadata or {}
                    item.metadata["rerank_score"] = score
                results = ordered
            except Exception as e:
                # 保底不影响主流程
                print(f"Rerank failed: {e}")
        
        # 8. 结果多样化与去冗处理（两阶段：先保底覆盖不同文档，再按分数补齐）
        def diversify(results: List[SearchResult], top_k: int, per_doc_max: Optional[int], mmr: bool, min_unique_docs: Optional[int]) -> List[SearchResult]:
            if not results:
                return []
            ranked = results
            picked: List[SearchResult] = []
            picked_by_doc: Dict[int, int] = {}
            # 第一阶段：保证至少覆盖 min_unique_docs 个不同文档
            if min_unique_docs:
                seen_docs = set()
                for r in ranked:
                    if len(picked) >= top_k:
                        break
                    if r.doc_id not in seen_docs:
                        if per_doc_max is None or picked_by_doc.get(r.doc_id, 0) < per_doc_max:
                            picked.append(r)
                            seen_docs.add(r.doc_id)
                            picked_by_doc[r.doc_id] = picked_by_doc.get(r.doc_id, 0) + 1
                    if len(seen_docs) >= min_unique_docs:
                        break
            # 构建剩余候选
            remaining = []
            picked_keys = {(p.doc_id, p.chunk_index) for p in picked}
            for r in ranked:
                if (r.doc_id, r.chunk_index) in picked_keys:
                    continue
                remaining.append(r)
            # 文档内限流（对剩余）
            if per_doc_max is not None and per_doc_max > 0:
                deduped = []
                for r in remaining:
                    cnt = picked_by_doc.get(r.doc_id, 0)
                    if cnt < per_doc_max:
                        deduped.append(r)
                        picked_by_doc[r.doc_id] = cnt + 1
                remaining = deduped
            # 如未启用MMR或已做保底覆盖，则按分数补齐
            if not mmr or min_unique_docs:
                for r in remaining:
                    if len(picked) >= top_k:
                        break
                    picked.append(r)
                return picked[:top_k]
            # 启用MMR：优先覆盖更多doc_id，再按分数补齐
            from collections import defaultdict, deque
            perdoc = defaultdict(list)
            for r in remaining:
                perdoc[r.doc_id].append(r)
            for d in perdoc:
                perdoc[d] = deque(perdoc[d])
            doc_order = sorted(perdoc.keys(), key=lambda d: perdoc[d][0].score if perdoc[d] else 0.0, reverse=True)
            while len(picked) < top_k and any(perdoc[d] for d in doc_order):
                for d in doc_order:
                    if len(picked) >= top_k:
                        break
                    if perdoc[d]:
                        picked.append(perdoc[d].popleft())
            return picked[:top_k]

        final_results = diversify(results, req.top_k, req.per_doc_max, req.mmr, req.min_unique_docs)
        
        search_time = round((time.time() - start_time) * 1000, 2)
        
        return SearchResponse(
            query=req.query,
            total_hits=len(final_results),
            results=final_results,
            search_time_ms=search_time
        )
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/search/hybrid")
async def hybrid_search(
    req: SearchRequest,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    混合检索：结合向量搜索和全文搜索
    """
    try:
        # 1. 向量搜索
        vector_results = await search_documents(req, db, milvus_client)
        
        # 2. MySQL全文搜索（作为补充）
        # 放大 FULLTEXT 候选以利于多样化
        _fulltext_limit = req.top_k * (3 if req.mmr else 1)
        fulltext_result = db.execute(sql_text("""
            SELECT d.id, d.title, c.document_id, c.chunk_index, c.content,
                   MATCH(c.content) AGAINST(:query IN NATURAL LANGUAGE MODE) as relevance
            FROM doc_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE MATCH(c.content) AGAINST(:query IN NATURAL LANGUAGE MODE)
            ORDER BY relevance DESC
            LIMIT :limit
        """), {"query": req.query, "limit": _fulltext_limit})
        
        fulltext_hits = []
        for row in fulltext_result.fetchall():
            if row.relevance > 0:  # MySQL fulltext相关性过滤
                result = SearchResult(
                    doc_id=row.document_id,
                    chunk_index=row.chunk_index,
                    score=float(row.relevance),
                    title=row.title,
                    content=row.content,
                    preview=row.content[:200] + "..." if len(row.content) > 200 else row.content,
                    metadata={"search_type": "fulltext", "relevance": row.relevance}
                )
                fulltext_hits.append(result)
        
        # 3. 合并和去重（简单策略：向量搜索优先，全文搜索补充）
        seen_chunks = set()
        combined_results = []
        
        # 先添加向量搜索结果
        for result in vector_results.results:
            chunk_key = (result.doc_id, result.chunk_index)
            if chunk_key not in seen_chunks:
                result.metadata = result.metadata or {}
                result.metadata["search_type"] = "vector"
                combined_results.append(result)
                seen_chunks.add(chunk_key)
        
        # 再添加全文搜索的补充结果
        for result in fulltext_hits:
            chunk_key = (result.doc_id, result.chunk_index)
            if chunk_key not in seen_chunks:
                combined_results.append(result)
                seen_chunks.add(chunk_key)
        
        # 3.5 可选重排序（在合并后统一重排）
        if req.rerank and combined_results:
            try:
                texts = [ (r.content or r.preview or "")[:2048] for r in combined_results ]
                order = await rerank_texts(req.query, texts, top_n=len(texts))
                ordered = [combined_results[idx] for idx, _ in order if 0 <= idx < len(combined_results)]
                for (idx, score), item in zip(order, ordered):
                    item.metadata = item.metadata or {}
                    item.metadata["rerank_score"] = score
                combined_results = ordered
            except Exception as e:
                print(f"Hybrid rerank failed: {e}")

        # 多样化与去冗
        def diversify(results: List[SearchResult], top_k: int, per_doc_max: Optional[int], mmr: bool, min_unique_docs: Optional[int]) -> List[SearchResult]:
            if not results:
                return []
            ranked = results
            picked: List[SearchResult] = []
            picked_by_doc: Dict[int, int] = {}
            if min_unique_docs:
                seen_docs = set()
                for r in ranked:
                    if len(picked) >= top_k:
                        break
                    if r.doc_id not in seen_docs:
                        if per_doc_max is None or picked_by_doc.get(r.doc_id, 0) < per_doc_max:
                            picked.append(r)
                            seen_docs.add(r.doc_id)
                            picked_by_doc[r.doc_id] = picked_by_doc.get(r.doc_id, 0) + 1
                    if len(seen_docs) >= min_unique_docs:
                        break
            remaining = []
            picked_keys = {(p.doc_id, p.chunk_index) for p in picked}
            for r in ranked:
                if (r.doc_id, r.chunk_index) in picked_keys:
                    continue
                remaining.append(r)
            if per_doc_max is not None and per_doc_max > 0:
                deduped = []
                for r in remaining:
                    cnt = picked_by_doc.get(r.doc_id, 0)
                    if cnt < per_doc_max:
                        deduped.append(r)
                        picked_by_doc[r.doc_id] = cnt + 1
                remaining = deduped
            if not mmr or min_unique_docs:
                for r in remaining:
                    if len(picked) >= top_k:
                        break
                    picked.append(r)
                return picked[:top_k]
            from collections import defaultdict, deque
            perdoc = defaultdict(list)
            for r in remaining:
                perdoc[r.doc_id].append(r)
            for d in perdoc:
                perdoc[d] = deque(perdoc[d])
            doc_order = sorted(perdoc.keys(), key=lambda d: perdoc[d][0].score if perdoc[d] else 0.0, reverse=True)
            while len(picked) < top_k and any(perdoc[d] for d in doc_order):
                for d in doc_order:
                    if len(picked) >= top_k:
                        break
                    if perdoc[d]:
                        picked.append(perdoc[d].popleft())
            return picked[:top_k]

        final_results = diversify(combined_results, req.top_k, req.per_doc_max, req.mmr, req.min_unique_docs)
        
        return SearchResponse(
            query=req.query,
            total_hits=len(final_results),
            results=final_results,
            search_time_ms=vector_results.search_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

@router.get("/collections/stats")
async def get_collection_stats(milvus_client = Depends(get_milvus)):
    """获取Milvus集合统计信息"""
    try:
        stats = milvus_client.get_collection_stats("kb_chunks")
        return {"collection_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/collections/compact")
async def compact_collection(milvus_client = Depends(get_milvus)):
    """压缩Milvus集合"""
    try:
        milvus_client.compact("kb_chunks")
        return {"success": True, "message": "Collection compacted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compact: {str(e)}")
