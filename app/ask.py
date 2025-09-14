from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import os, httpx
from .deps import milvus
from .embedding import embed_texts
from .rerank import rerank_texts
import tiktoken

router = APIRouter()

MAX_CONTEXT_TOKENS = 6000
MAX_CHUNK_TOKENS   = 450
DEFAULT_TOP_K      = 8

def count_tokens(text: str, enc_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(enc_name)
    return len(enc.encode(text))

def build_context(chunks: List[dict], budget_tokens=MAX_CONTEXT_TOKENS) -> str:
    pieces, used = [], 0
    for c in chunks:
        txt = c["text"]
        # 单块长度控制
        if count_tokens(txt) > MAX_CHUNK_TOKENS:
            enc = tiktoken.get_encoding("cl100k_base")
            txt = enc.decode(enc.encode(txt)[:MAX_CHUNK_TOKENS])
        t = count_tokens(txt)
        if used + t > budget_tokens:
            break
        pieces.append(f"[doc_id={c['doc_id']}, chunk_index={c['chunk_index']}]\n{txt}")
        used += t
    return "\n\n---\n\n".join(pieces)

class AskReq(BaseModel):
    query: str = Field(..., description="用户问题文本（用于生成答案与向量检索）")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50, description="返回片段数量（更大→覆盖更广，可能稍降质量）")
    nprobe: int = Field(16, ge=1, le=4096, description="Milvus IVF nprobe（更大→召回更广但更慢，常用64/96/128）")
    # 可调检索控制
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="相似度/分数阈值（COSINE距离，越大越相似）")
    per_doc_max: Optional[int] = Field(default=None, ge=1, description="每文档最多片段数（提升多样性）")
    mmr: bool = Field(default=False, description="启用简单多样化（跨文档轮转）")
    min_unique_docs: Optional[int] = Field(default=None, ge=1, description="至少覆盖的不同文档数")
    rerank: Optional[bool] = Field(default=None, description="是否启用外部重排（不传→按环境变量 ASK_USE_RERANK）")
    # 由前端传入的 DeepSeek API Key（只用于本次请求，不落库）；不传则回退到环境变量
    user_llm_api_key: Optional[str] = Field(default=None, description="可选：覆盖默认 DEEPSEEK_API_KEY，仅用于本次请求")
    # DeepSeek 模型名：常见 deepseek-chat（对话）或 deepseek-reasoner（推理）
    llm_model: str = Field(default=os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-reasoner"), description="LLM 模型名（默认读取 DEEPSEEK_LLM_MODEL）")

@router.post("/ask")
async def ask(req: AskReq):
    # 1) embedding（仍用你既定的供应商）→ 查询向量
    qv = (await embed_texts([req.query]))[0]

    # 2) Milvus 相似检索（根据多样化/重排放大候选）
    multiplier = 1
    do_rerank = (os.getenv("ASK_USE_RERANK", "false").lower() == "true") if (req.rerank is None) else bool(req.rerank)
    if do_rerank:
        multiplier = max(multiplier, 3)
    if req.mmr:
        multiplier = max(multiplier, 3)
    search_limit = req.top_k * multiplier

    hits = milvus.search(
        collection_name="kb_chunks",
        data=[qv],
        anns_field="vector",
        limit=search_limit,
        search_params={"metric_type": "COSINE", "params": {"nprobe": req.nprobe}},
        output_fields=["doc_id", "chunk_index", "text"]
    )[0]

    # 初筛（阈值过滤）
    candidates = []
    for h in hits:
        score = float(h["distance"])  # COSINE 相似度分数（越大越相似）
        if score >= req.score_threshold:
            candidates.append({
                "doc_id": int(h["entity"]["doc_id"]),
                "chunk_index": int(h["entity"]["chunk_index"]),
                "text": str(h["entity"]["text"]),
                "score": score,
            })

    if not candidates:
        raise HTTPException(status_code=404, detail="未检索到相关片段")

    # 3) 可选重排候选（基于外部Rerank服务），通过环境变量 ASK_USE_RERANK 控制
    try:
        if do_rerank and candidates:
            texts = [c["text"][:2048] for c in candidates]
            order = await rerank_texts(req.query, texts, top_n=len(texts))
            candidates = [candidates[idx] for idx, _ in order if 0 <= idx < len(candidates)]
    except Exception as e:
        # 保底不影响主流程
        print(f"Ask rerank failed: {e}")

    # 3.5) 多样化与去冗（与搜索接口一致的简化实现）
    def diversify(results: List[dict], top_k: int, per_doc_max: Optional[int], mmr: bool, min_unique_docs: Optional[int]) -> List[dict]:
        if not results:
            return []
        ranked = results
        picked: List[dict] = []
        picked_by_doc: dict[int, int] = {}
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
        remaining: List[dict] = []
        picked_keys = {(p["doc_id"], p["chunk_index"]) for p in picked}
        for r in ranked:
            if (r["doc_id"], r["chunk_index"]) in picked_keys:
                continue
            remaining.append(r)
        if per_doc_max is not None and per_doc_max > 0:
            deduped: List[dict] = []
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

    # 4) 组装上下文
    context = build_context(final_candidates, budget_tokens=MAX_CONTEXT_TOKENS)

    # 5) DeepSeek Prompt
    system_prompt = (
        "你是专业的知识库助理。仅依据'上下文'回答问题；"
        "若上下文没有答案，请明确说明'未在知识库中找到'。"
        "回答末尾用'参考片段: [doc_id,chunk_index,...]'列出用到的片段标识。"
    )
    user_prompt = (
        f"问题：{req.query}\n\n"
        f"上下文：\n{context}\n\n"
        "要求：\n- 使用上下文事实作答，必要时概括归纳\n- 不要编造。"
    )

    # 6) 调用 DeepSeek Chat Completions
    _api_key = req.user_llm_api_key or os.getenv("DEEPSEEK_API_KEY")
    if not _api_key or len(_api_key) < 10:
        raise HTTPException(status_code=400, detail="DeepSeek API Key 未提供或不合法（请在请求中传 user_llm_api_key 或配置环境变量 DEEPSEEK_API_KEY）")

    headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": req.llm_model,          # "deepseek-chat" 或 "deepseek-reasoner"
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": 0.2,
        "stream": False                  # 非流式
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.deepseek.com/chat/completions",
                              headers=headers, json=payload)
        if r.status_code in (401, 403):
            raise HTTPException(status_code=r.status_code, detail="DeepSeek API Key 无效或权限不足")
        r.raise_for_status()
        data = r.json()
        answer = data["choices"][0]["message"]["content"]

    used_refs = [(c["doc_id"], c["chunk_index"]) for c in final_candidates]
    return {
        "answer": answer,
        "used_chunks": used_refs,
        "retrieval_count": len(final_candidates),
        "model": req.llm_model
    }
