from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
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
    query: str
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50)
    nprobe: int = Field(16, ge=1, le=4096)
    # 由前端传入的 DeepSeek API Key（只用于本次请求，不落库）
    user_llm_api_key: str = Field(..., min_length=10)
    # DeepSeek 模型名：常见 deepseek-chat（对话）或 deepseek-reasoner（推理）
    llm_model: str = Field("deepseek-chat")

@router.post("/ask")
async def ask(req: AskReq):
    # 1) embedding（仍用你既定的供应商）→ 查询向量
    qv = (await embed_texts([req.query]))[0]

    # 2) Milvus 相似检索
    hits = milvus.search(
        collection_name="kb_chunks",
        data=[qv],
        anns_field="vector",
        limit=req.top_k,
        search_params={"metric_type": "COSINE", "params": {"nprobe": req.nprobe}},
        output_fields=["doc_id", "chunk_index", "text"]
    )[0]

    candidates = [{
        "doc_id": int(h["entity"]["doc_id"]),
        "chunk_index": int(h["entity"]["chunk_index"]),
        "text": str(h["entity"]["text"]),
        "score": float(h["distance"])
    } for h in hits]

    if not candidates:
        raise HTTPException(status_code=404, detail="未检索到相关片段")

    # 3) 可选重排候选（基于外部Rerank服务），通过环境变量 ASK_USE_RERANK 控制
    try:
        if os.getenv("ASK_USE_RERANK", "false").lower() == "true" and candidates:
            texts = [c["text"][:2048] for c in candidates]
            order = await rerank_texts(req.query, texts, top_n=len(texts))
            candidates = [candidates[idx] for idx, _ in order if 0 <= idx < len(candidates)]
    except Exception as e:
        # 保底不影响主流程
        print(f"Ask rerank failed: {e}")

    # 4) 组装上下文
    context = build_context(candidates, budget_tokens=MAX_CONTEXT_TOKENS)

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
    headers = {
        "Authorization": f"Bearer {req.user_llm_api_key}",
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

    used_refs = [(c["doc_id"], c["chunk_index"]) for c in candidates]
    return {
        "answer": answer,
        "used_chunks": used_refs,
        "retrieval_count": len(candidates),
        "model": req.llm_model
    }
