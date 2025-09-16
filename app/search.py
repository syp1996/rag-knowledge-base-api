# app/search.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from .deps import get_db, get_milvus
from .utils import highlight_search_text
from .embedding import embed_query
from .rerank import rerank_texts
from typing import List, Optional, Dict, Any
import json

router = APIRouter()

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询文本")  # 必填查询词
    engine: str = Field(
        default="keyword",
        description="检索引擎：keyword（Notion式全文）、vector、hybrid"
    )
    top_k: int = Field(
        default=8, ge=1, le=50,
        description="返回结果数量（更大→覆盖更广但可能包含低质结果）"
    )
    nprobe: int = Field(
        default=16, ge=1, le=256,
        description="Milvus IVF nprobe（更大→召回更广但耗时更高，常用64/96/128）"
    )
    rerank: bool = Field(
        default=True,
        description="是否启用外部重排（提升排序质量，但会放大候选并增加一次外部调用）"
    )
    doc_ids: Optional[List[int]] = Field(
        default=None,
        description="限制只在指定文档ID集合内检索（可选）"
    )
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="相似度分数阈值（COSINE，越大越相似；设为0以不过滤）"
    )
    per_doc_max: Optional[int] = Field(
        default=None, ge=1,
        description="每文档最多返回片段数（用于提升跨文档多样性）"
    )
    mmr: bool = Field(
        default=False,
        description="启用简单多样化（跨文档轮转）；开启后候选池会放大（top_k×3）"
    )
    min_unique_docs: Optional[int] = Field(
        default=None, ge=1,
        description="至少覆盖的不同文档数量（两阶段：先保底覆盖，再按分数补齐）"
    )

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
    支持三种检索：
      - keyword（默认，Notion式全文检索，倒排索引、多字段加权、无向量依赖）
      - vector（语义检索，Milvus 向量召回）
      - hybrid（混合检索，向量 + FULLTEXT 补充）
    """
    try:
        # 根据多样化需求，获取更多候选以便后续筛选
        import time
        start_time = time.time()
        # Engine dispatch
        if (req.engine or "keyword").lower() == "keyword":
            # Notion式全文检索（无向量依赖）
            # - 多字段加权：title / excerpt / content_text（documents）+ content（doc_chunks）
            # - 仅搜索已发布且未删除文档
            # - 可按 doc_ids 限定范围
            params = {"q": req.query, "limit": req.top_k}
            where_doc_ids = ""
            if req.doc_ids:
                ids = ",".join(map(str, req.doc_ids))
                where_doc_ids = f" AND c.document_id IN ({ids}) "

            sql = sql_text(
                f"""
                SELECT 
                  d.id AS doc_id,
                  d.title AS title,
                  c.chunk_index AS chunk_index,
                  c.content AS content,
                  (
                    2.0 * MATCH(d.title) AGAINST(:q IN NATURAL LANGUAGE MODE) +
                    1.0 * MATCH(d.excerpt) AGAINST(:q IN NATURAL LANGUAGE MODE) +
                    0.75 * MATCH(d.content_text) AGAINST(:q IN NATURAL LANGUAGE MODE) +
                    0.5 * MATCH(c.content) AGAINST(:q IN NATURAL LANGUAGE MODE)
                  ) AS score
                FROM doc_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE 1=1
                  {where_doc_ids}
                  AND (
                    MATCH(d.title) AGAINST(:q IN NATURAL LANGUAGE MODE) OR
                    MATCH(d.excerpt) AGAINST(:q IN NATURAL LANGUAGE MODE) OR
                    MATCH(d.content_text) AGAINST(:q IN NATURAL LANGUAGE MODE) OR
                    MATCH(c.content) AGAINST(:q IN NATURAL LANGUAGE MODE)
                  )
                ORDER BY score DESC, d.is_pinned DESC, d.created_at DESC
                LIMIT :limit
                """
            )
            try:
                rows = db.execute(sql, params).fetchall()
            except Exception as e:
                # 兼容性降级：部分环境 documents 上未建 FULLTEXT，回退为 chunks-only 的全文检索
                print(f"[keyword-search] falling back to chunks-only FULLTEXT due to: {e}")
                fallback_sql = f"""
                    SELECT 
                      d.id AS doc_id,
                      d.title AS title,
                      c.chunk_index AS chunk_index,
                      c.content AS content,
                      MATCH(c.content) AGAINST(:q IN NATURAL LANGUAGE MODE) AS score
                    FROM doc_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE 1=1
                      {where_doc_ids}
                      AND MATCH(c.content) AGAINST(:q IN NATURAL LANGUAGE MODE)
                    ORDER BY score DESC, d.is_pinned DESC, d.created_at DESC
                    LIMIT :limit
                """
                rows = db.execute(sql_text(fallback_sql), params).fetchall()

            results: List[SearchResult] = []
            for r in rows:
                txt = r.content or ""
                preview = txt[:200] + "..." if len(txt) > 200 else txt
                # 高亮仅作用于预览（不修改原文）
                try:
                    preview = highlight_search_text(preview, req.query)
                except Exception:
                    pass
                results.append(SearchResult(
                    doc_id=int(r.doc_id),
                    chunk_index=int(r.chunk_index),
                    score=float(r.score) if r.score is not None else 0.0,
                    title=r.title or "",
                    content=txt,
                    preview=preview,
                    metadata={"search_type": "keyword"}
                ))

            return SearchResponse(
                query=req.query,
                total_hits=len(results),
                results=results,
                search_time_ms=round((time.time() - start_time) * 1000, 2)
            )

        # ===== Vector engine (original flow) =====
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
        """)).fetchall()

        # 创建MySQL数据的索引
        mysql_data = {}
        for row in mysql_result:
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
        # 1. 向量搜索（强制使用 vector 引擎）
        try:
            new_req_data = req.model_dump()
        except Exception:
            new_req_data = req.dict()  # pydantic v1 fallback
        new_req_data["engine"] = "vector"
        vector_results = await search_documents(SearchRequest(**new_req_data), db, milvus_client)
        
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
