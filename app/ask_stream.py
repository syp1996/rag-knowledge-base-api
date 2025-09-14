import json
import os
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .ask import MAX_CONTEXT_TOKENS, build_context, get_rag_prompts
from .deps import milvus
from .embedding import embed_texts
from .rerank import rerank_texts

router = APIRouter()

class AskStreamReq(BaseModel):
    message: str = Field(..., description="用户问题文本（SSE流式生成）")  # 前端使用message而非query
    top_k: int = Field(20, ge=1, le=50, description="返回片段数量（更大→覆盖更广，可能稍降质量）")
    nprobe: int = Field(128, ge=1, le=4096, description="Milvus IVF nprobe（更大→召回更广但更慢，常用64/96/128）")
    # 二选一使用：保留向后兼容
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="相似度阈值=1-score；一般不推荐，建议用 score_threshold")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="相似度分数阈值（COSINE，越大越相似；设0不过滤）")
    per_doc_max: Optional[int] = Field(default=None, ge=1, description="每文档最多片段数（提升跨文档多样性）")
    mmr: bool = Field(False, description="启用简单多样化（跨文档轮转）；开启后候选池会放大（top_k×3）")
    min_unique_docs: Optional[int] = Field(default=None, ge=1, description="至少覆盖的不同文档数量（两阶段保底+补齐）")
    rerank: Optional[bool] = Field(True, description="是否启用外部重排（不传→按环境变量 ASK_USE_RERANK）")
    use_knowledge_base: bool = Field(True, description="是否启用知识库增强（关闭将直接报错）")
    # 可从请求透传；若不传，则回退到环境变量 DEEPSEEK_API_KEY
    user_llm_api_key: Optional[str] = Field(default=None, description="可选：覆盖默认 DEEPSEEK_API_KEY，仅用于本次请求")
    llm_model: str = Field(default=os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-reasoner"), description="LLM 模型名（默认读取 DEEPSEEK_LLM_MODEL）")

@router.post("/ask/stream")
async def ask_stream(req: AskStreamReq):
    if not req.use_knowledge_base:
        raise HTTPException(status_code=400, detail="知识库模式未启用")
    
    qv = (await embed_texts([req.message]))[0]
    # 根据多样化/重排放大候选
    multiplier = 1
    _do_rerank = (os.getenv("ASK_USE_RERANK", "false").lower() == "true") if (req.rerank is None) else bool(req.rerank)
    if _do_rerank:
        multiplier = max(multiplier, 3)
    if req.mmr:
        multiplier = max(multiplier, 3)
    search_limit = req.top_k * multiplier

    hits = milvus.search(
        collection_name="kb_chunks",
        data=[qv],
        anns_field="vector",
        limit=search_limit,
        search_params={"metric_type":"COSINE","params":{"nprobe": req.nprobe}},
        output_fields=["doc_id","chunk_index","text"]
    )[0]
    # 过滤（支持 similarity 或 score 两种阈值）
    candidates = []
    for h in hits:
        score = float(h["distance"])         # COSINE 分数（越大越相似）
        similarity = 1.0 - score              # 兼容旧字段
        pass_score = (req.score_threshold > 0 and score >= req.score_threshold)
        pass_sim = (req.similarity_threshold > 0 and similarity >= req.similarity_threshold)
        if (req.score_threshold > 0 and pass_score) or (req.score_threshold <= 0 and (req.similarity_threshold <= 0 or pass_sim)):
            candidates.append({
                "doc_id": int(h["entity"]["doc_id"]),
                "chunk_index": int(h["entity"]["chunk_index"]),
                "text": str(h["entity"]["text"]),
                "score": score,
                "similarity": similarity
            })

    if not candidates:
        raise HTTPException(status_code=404, detail=f"未找到相似度高于{req.similarity_threshold}的相关片段")

    # 可选：重排候选（通过环境变量 ASK_USE_RERANK 控制）
    try:
        import os as _os
        if _do_rerank and candidates:
            texts = [c["text"][:2048] for c in candidates]
            order = await rerank_texts(req.message, texts, top_n=len(texts))
            candidates = [candidates[idx] for idx, _ in order if 0 <= idx < len(candidates)]
    except Exception as e:
        print(f"Ask stream rerank failed: {e}")

    # 多样化处理
    def diversify(results, top_k, per_doc_max, mmr, min_unique_docs):
        if not results:
            return []
        ranked = results
        picked = []
        picked_by_doc = {}
        if min_unique_docs:
            seen_docs = set()
            for r in ranked:
                if len(picked) >= top_k:
                    break
                if r["doc_id"] not in seen_docs:
                    if per_doc_max is None or picked_by_doc.get(r["doc_id"], 0) < per_doc_max:
                        picked.append(r)
                        seen_docs.add(r["doc_id"])
                        picked_by_doc[r["doc_id"]] = picked_by_doc.get(r["doc_id"], 0) + 1
                if len(seen_docs) >= min_unique_docs:
                    break
        remaining = []
        picked_keys = {(p["doc_id"], p["chunk_index"]) for p in picked}
        for r in ranked:
            if (r["doc_id"], r["chunk_index"]) in picked_keys:
                continue
            remaining.append(r)
        if per_doc_max is not None and per_doc_max > 0:
            deduped = []
            for r in remaining:
                cnt = picked_by_doc.get(r["doc_id"], 0)
                if cnt < per_doc_max:
                    deduped.append(r)
                    picked_by_doc[r["doc_id"]] = cnt + 1
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
            perdoc[r["doc_id"]].append(r)
        for d in perdoc:
            perdoc[d] = deque(perdoc[d])
        doc_order = sorted(perdoc.keys(), key=lambda d: perdoc[d][0]["score"] if perdoc[d] else 0.0, reverse=True)
        while len(picked) < top_k and any(perdoc[d] for d in doc_order):
            for d in doc_order:
                if len(picked) >= top_k:
                    break
                if perdoc[d]:
                    picked.append(perdoc[d].popleft())
        return picked[:top_k]

    final_candidates = diversify(candidates, req.top_k, req.per_doc_max, req.mmr, req.min_unique_docs)

    context = build_context(final_candidates, budget_tokens=MAX_CONTEXT_TOKENS)
    system_prompt, user_prompt = get_rag_prompts(req.message, context)

    _api_key = req.user_llm_api_key or os.getenv("DEEPSEEK_API_KEY")
    if not _api_key or len(_api_key) < 10:
        raise HTTPException(status_code=400, detail="DeepSeek API Key 未提供或不合法（请在请求中传 user_llm_api_key 或配置环境变量 DEEPSEEK_API_KEY）")

    headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": req.llm_model,
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        "temperature": 0.2,
        "stream": True               # 关键：开启流式
    }

    async def event_source():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", "https://api.deepseek.com/chat/completions",
                                     headers=headers, json=payload) as r:
                if r.status_code in (401, 403):
                    yield f"data: {json.dumps({'error':'DeepSeek API Key 无效或权限不足'})}\n\n"
                    return
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # DeepSeek SSE 行通常以 "data: {json}" 形式
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        yield f"{line}\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
