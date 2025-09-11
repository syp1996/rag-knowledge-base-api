from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx, json, os
from .deps import milvus
from .embedding import embed_texts
from .ask import build_context, MAX_CONTEXT_TOKENS

router = APIRouter()

class AskStreamReq(BaseModel):
    message: str  # 前端使用message而不是query
    top_k: int = 5
    similarity_threshold: float = 0.3
    use_knowledge_base: bool = True
    user_llm_api_key: str = Field(default="sk-d946f7d6c6a54aa198a066623f68f8a4", min_length=10)
    llm_model: str = "deepseek-chat"

@router.post("/ask/stream")
async def ask_stream(req: AskStreamReq):
    if not req.use_knowledge_base:
        raise HTTPException(status_code=400, detail="知识库模式未启用")
    
    qv = (await embed_texts([req.message]))[0]
    hits = milvus.search(
        collection_name="kb_chunks",
        data=[qv],
        anns_field="vector",
        limit=req.top_k,
        search_params={"metric_type":"COSINE","params":{"nprobe": 16}},
        output_fields=["doc_id","chunk_index","text"]
    )[0]
    # 过滤相似度阈值并转换为相似度（1 - distance）
    candidates = []
    for h in hits:
        similarity = 1 - h["distance"]  # Milvus返回的是distance，转换为similarity
        if similarity >= req.similarity_threshold:
            candidates.append({
                "doc_id": int(h["entity"]["doc_id"]),
                "chunk_index": int(h["entity"]["chunk_index"]),
                "text": str(h["entity"]["text"]),
                "similarity": similarity
            })
    
    if not candidates:
        raise HTTPException(status_code=404, detail=f"未找到相似度高于{req.similarity_threshold}的相关片段")

    context = build_context(candidates, budget_tokens=MAX_CONTEXT_TOKENS)

    system_prompt = "你是专业的知识库助理。仅依据'上下文'回答问题；不足则直说。"
    user_prompt = f"问题：{req.message}\n\n上下文：\n{context}"

    headers = {
        "Authorization": f"Bearer {req.user_llm_api_key}",
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