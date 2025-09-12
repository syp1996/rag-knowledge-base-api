# app/ingest.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .deps import get_db, get_milvus
from .embedding import embed_texts
import tiktoken
import json
import asyncio
from typing import Optional, List, Union, Any, Tuple
from pydantic import BaseModel
from .models import Document
from .utils import get_or_create_default_user, generate_unique_slug

router = APIRouter()

def token_len(s: str) -> int:
    """计算文本的token长度"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))

def truncate_utf8_bytes(s: str, max_bytes: int = 1000) -> str:
    if s is None:
        return ""
    b = s.encode('utf-8')
    if len(b) <= max_bytes:
        return s
    # truncate safely
    # start with rough cut
    lo, hi = 0, len(s)
    res = s
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(s[:mid].encode('utf-8')) <= max_bytes:
            res = s[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return res

async def embed_in_batches(texts, batch_size: int = 10, delay: float = 0.2):
    """按批嵌入，遵守 DashScope batch<=10 的限制。"""
    vectors_all = []
    for i in range(0, len(texts), batch_size):
        part = texts[i:i+batch_size]
        vecs = await embed_texts(part)
        vectors_all.extend(vecs)
        if delay:
            await asyncio.sleep(delay)
    return vectors_all

# ========= 新增：基于纯文本的入库与更新接口 =========

class ContentPayload(BaseModel):
    text: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    type: Optional[str] = None


class IngestTextRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[Union[str, ContentPayload]] = None
    tags: Optional[List[str]] = None


@router.post("/ingest/text")
async def ingest_text_document(
    payload: IngestTextRequest,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    文本入库：前端直接提交纯文本，后端保存到 documents.content，并完成切分、嵌入、写入 Milvus & doc_chunks。
    - content 保存到 documents.content 字段（JSON: {"text": content}）
    - 自动生成/去重 slug
    - 生成 excerpt
    - 将向量写入 Milvus（collection: kb_chunks），并记录到 doc_chunks
    """
    try:
        # 解析 content（支持字符串或对象）
        def normalize_content(content: Optional[Union[str, ContentPayload]]) -> Tuple[str, dict]:
            if content is None:
                return "", {"text": ""}
            # 字符串
            if isinstance(content, str):
                # 默认将字符串视为markdown保存
                return content.strip(), {"markdown": content}
            # 对象
            # pydantic模型 -> dict
            data = content.model_dump() if hasattr(content, "model_dump") else dict(content)
            text = ""
            if data.get("markdown"):
                text = data["markdown"]
            elif data.get("text"):
                text = data["text"]
            elif data.get("html"):
                # 粗略去除HTML标签
                import re
                text = re.sub(r"<[^>]+>", "", data["html"]) or ""
            return (text.strip(), data)

        raw_text, content_json = normalize_content(payload.content)

        # 若内容为空，则先创建“草稿”文档，跳过切块与向量化，由后续更新接口完成
        do_embedding = len(raw_text) > 0
        if do_embedding:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=120,
                separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
            )
            chunks = splitter.split_text(raw_text)
            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks generated from content")
            vectors = await embed_in_batches(chunks, batch_size=10, delay=0.3)
        else:
            chunks = []
            vectors = []

        # 创建/获取默认用户
        default_user = get_or_create_default_user(db)

        # 组装标题与 slug
        title = payload.title or (raw_text.split("\n", 1)[0].strip()[:100] if raw_text else "Untitled Document")
        slug = generate_unique_slug(db, title)

        # 提取摘要（避免触发生成列写入，使用原生SQL插入）
        temp_doc = Document(user_id=default_user.id, title=title, content={"text": raw_text}, slug=slug, status=0)
        excerpt = temp_doc.extract_excerpt() if hasattr(temp_doc, 'extract_excerpt') else None

        result = db.execute(sql_text("""
            INSERT INTO documents(user_id, title, content, slug, status, excerpt)
            VALUES (:user_id, :title, :content, :slug, :status, :excerpt)
        """), {
            "user_id": int(default_user.id),
            "title": title,
            "content": json.dumps(content_json),
            "slug": slug,
            "status": 0,
            "excerpt": excerpt
        })
        db.commit()
        doc_id = result.lastrowid

        # 写入 Milvus
        if do_embedding:
            milvus_rows = []
            for i, (content, vec) in enumerate(zip(chunks, vectors)):
                milvus_rows.append({
                    "doc_id": int(doc_id),
                    "chunk_index": i,
                    "text": truncate_utf8_bytes(content, 1000),
                    "vector": vec
                })
            insert_result = milvus_client.insert(collection_name="kb_chunks", data=milvus_rows)
            milvus_client.flush("kb_chunks")
            milvus_pks = insert_result.primary_keys if hasattr(insert_result, 'primary_keys') else []

            # 写入 doc_chunks
            for i, content in enumerate(chunks):
                milvus_pk = milvus_pks[i] if i < len(milvus_pks) else None
                db.execute(sql_text("""
                    INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk)
                    VALUES (:doc_id, :chunk_index, :content, :token_count, :milvus_pk)
                """), {
                    "doc_id": int(doc_id),
                    "chunk_index": i,
                    "content": content,
                    "token_count": token_len(content),
                    "milvus_pk": milvus_pk
                })

        db.commit()

        return {
            "success": True,
            "message": "Text document ingested successfully" if do_embedding else "Empty draft created successfully",
            "document_id": int(doc_id),
            "title": title,
            "chunks_count": len(chunks),
            "total_tokens": sum(token_len(chunk) for chunk in chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to ingest text: {str(e)}")


class UpdateTextRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[Union[str, ContentPayload]] = None


@router.put("/ingest/text/{document_id}")
async def update_text_document(
    document_id: int,
    payload: UpdateTextRequest,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    文本更新：更新 documents.content 与标题，并重新切分、嵌入，替换 Milvus 与 doc_chunks 中的数据。
    """
    try:
        # 确认文档存在
        doc_exist = db.execute(sql_text("SELECT id, title FROM documents WHERE id = :id"), {"id": int(document_id)}).fetchone()
        if not doc_exist:
            raise HTTPException(status_code=404, detail="Document not found")

        # 如果仅更新标题（未提供content），则不做重切块与嵌入
        if payload.content is None:
            title_changed = payload.title is not None
            if not title_changed:
                raise HTTPException(status_code=400, detail="Nothing to update")
            new_title = payload.title
            new_slug = generate_unique_slug(db, new_title, document_id=document_id)

            db.execute(sql_text("""
                UPDATE documents
                SET title = :title,
                    slug = :slug,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """), {
                "title": new_title,
                "slug": new_slug,
                "id": int(document_id)
            })
            db.commit()

            # 读取现有 chunk 统计
            row = db.execute(sql_text(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(token_count),0) AS total FROM doc_chunks WHERE document_id = :id"
            ), {"id": int(document_id)}).fetchone()

            return {
                "success": True,
                "message": "Title updated successfully",
                "document_id": int(document_id),
                "title": new_title,
                "chunks_count": int(row.cnt) if row and hasattr(row, 'cnt') else 0,
                "total_tokens": int(row.total) if row and hasattr(row, 'total') else 0
            }

        # 解析 content（支持字符串或对象）
        def normalize_content_update(content: Union[str, ContentPayload]) -> Tuple[str, dict]:
            if isinstance(content, str):
                # 默认将字符串视为markdown保存
                return content.strip(), {"markdown": content}
            data = content.model_dump() if hasattr(content, "model_dump") else dict(content)
            text = ""
            if data.get("markdown"):
                text = data["markdown"]
            elif data.get("text"):
                text = data["text"]
            elif data.get("html"):
                import re
                text = re.sub(r"<[^>]+>", "", data["html"]) or ""
            return (text.strip(), data)

        raw_text, content_json = normalize_content_update(payload.content)
        if not raw_text:
            raise HTTPException(status_code=400, detail="Content is empty")

        # 更新标题与内容
        # 计算标题与 slug（不主动覆盖旧标题，除非显式传入）
        title_changed = payload.title is not None
        new_title = payload.title if title_changed else doc_exist.title
        new_slug = generate_unique_slug(db, new_title, document_id=document_id) if title_changed else None

        # 计算 excerpt（使用临时模型实例）
        temp_doc = Document(title=new_title, content=content_json)
        new_excerpt = temp_doc.extract_excerpt() if hasattr(temp_doc, 'extract_excerpt') else None

        # 使用原生 SQL 更新（避免写入生成列）
        db.execute(sql_text("""
            UPDATE documents
            SET title = :title,
                content = :content,
                slug = CASE WHEN :slug_update = 1 THEN :slug ELSE slug END,
                excerpt = :excerpt,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :id
        """), {
            "title": new_title,
            "content": json.dumps(content_json),
            "slug": new_slug,
            "slug_update": 1 if title_changed else 0,
            "excerpt": new_excerpt,
            "id": int(document_id)
        })
        db.commit()

        # 删除旧的向量与 chunks
        try:
            milvus_client.delete(collection_name="kb_chunks", filter=f"doc_id == {int(document_id)}")
            milvus_client.flush("kb_chunks")
        except Exception:
            # 即使 Milvus 删除异常也继续，稍后用新的数据覆盖
            pass
        db.execute(sql_text("DELETE FROM doc_chunks WHERE document_id = :doc_id"), {"doc_id": int(document_id)})
        db.commit()

        # 重新切分与嵌入
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from content")
        vectors = await embed_in_batches(chunks, batch_size=10, delay=0.3)

        # 写入新的向量与 doc_chunks
        milvus_rows = []
        for i, (content, vec) in enumerate(zip(chunks, vectors)):
            milvus_rows.append({
                "doc_id": int(document_id),
                "chunk_index": i,
                "text": truncate_utf8_bytes(content, 1000),
                "vector": vec
            })
        insert_result = milvus_client.insert(collection_name="kb_chunks", data=milvus_rows)
        milvus_client.flush("kb_chunks")
        milvus_pks = insert_result.primary_keys if hasattr(insert_result, 'primary_keys') else []

        for i, content in enumerate(chunks):
            milvus_pk = milvus_pks[i] if i < len(milvus_pks) else None
            db.execute(sql_text("""
                INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk)
                VALUES (:doc_id, :chunk_index, :content, :token_count, :milvus_pk)
            """), {
                "doc_id": int(document_id),
                "chunk_index": i,
                "content": content,
                "token_count": token_len(content),
                "milvus_pk": milvus_pk
            })

        db.commit()

        return {
            "success": True,
            "message": "Text document updated and re-indexed successfully",
            "document_id": int(document_id),
            "title": new_title,
            "chunks_count": len(chunks),
            "total_tokens": sum(token_len(chunk) for chunk in chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update text: {str(e)}")


@router.get("/ingest/status")
async def ingest_status(
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    返回每个文档的切块与向量化状态，用于快速自检。
    - MySQL: 统计 doc_chunks 行数与总 token 数
    - Milvus: 按 doc_id 统计向量条数
    """
    try:
        # 获取文档列表
        docs = db.execute(sql_text("""
            SELECT id, title FROM documents WHERE deleted_at IS NULL ORDER BY created_at DESC
        """)).fetchall()

        items = []
        total_docs = len(docs)
        mysql_chunks_total = 0
        milvus_vectors_total = 0
        mismatches = 0
        zeros = 0

        for row in docs:
            doc_id = int(row.id)
            title = row.title

            # MySQL 统计
            c = db.execute(sql_text(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(token_count),0) AS tokens FROM doc_chunks WHERE document_id = :id"
            ), {"id": doc_id}).fetchone()
            chunks_cnt = int(c.cnt) if hasattr(c, 'cnt') else int(c[0])
            tokens_sum = int(c.tokens) if hasattr(c, 'tokens') else int(c[1])

            # Milvus 统计（尽量避免超大limit，常规场景足够）
            vectors_cnt = 0
            try:
                res = milvus_client.query(
                    collection_name="kb_chunks",
                    filter=f"doc_id == {doc_id}",
                    output_fields=["doc_id"],
                    # Milvus has a max (offset+limit) window of 16384
                    limit=16384
                )
                vectors_cnt = len(res) if isinstance(res, list) else 0
            except Exception:
                vectors_cnt = -1

            status = "ok"
            if vectors_cnt == -1:
                status = "unknown"
            elif chunks_cnt == 0 or vectors_cnt == 0:
                status = "missing"
                zeros += 1
            elif vectors_cnt != chunks_cnt:
                status = "mismatch"
                mismatches += 1

            mysql_chunks_total += chunks_cnt
            milvus_vectors_total += max(0, vectors_cnt)

            items.append({
                "document_id": doc_id,
                "title": title,
                "chunks_mysql": chunks_cnt,
                "tokens_sum": tokens_sum,
                "vectors_milvus": vectors_cnt,
                "status": status
            })

        return {
            "summary": {
                "total_documents": total_docs,
                "mysql_chunks_total": mysql_chunks_total,
                "milvus_vectors_total": milvus_vectors_total,
                "documents_zero_or_missing": zeros,
                "documents_mismatch": mismatches
            },
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ingest status: {str(e)}")


class ReindexRequest(BaseModel):
    only_missing: bool = True  # 仅处理 chunks==0 的文档
    include_mismatch: bool = True  # 同时处理 chunks!=vectors 的文档（如可统计）
    limit: Optional[int] = None  # 限制处理数量
    dry_run: bool = False  # 仅统计不执行


@router.post("/ingest/reindex_missing")
async def reindex_missing(
    req: ReindexRequest = ReindexRequest(),
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    批量修复：为未切块/未向量化的文档执行切块+嵌入并写入 Milvus 与 doc_chunks。
    - 依据条件：默认仅处理 MySQL 中 doc_chunks 为 0 的文档；可选包含 mismatch。
    - 支持 dry_run 仅统计。
    """
    try:
        # 获取候选文档（按创建时间倒序）
        rows = db.execute(sql_text(
            "SELECT id, title, content FROM documents WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )).fetchall()

        candidates = []
        for r in rows:
            doc_id = int(r.id)
            # 统计 MySQL 切块数
            c = db.execute(sql_text(
                "SELECT COUNT(*) AS cnt FROM doc_chunks WHERE document_id = :id"
            ), {"id": doc_id}).fetchone()
            chunks_cnt = int(c.cnt) if hasattr(c, 'cnt') else int(c[0])

            need = False
            reason = []
            if req.only_missing and chunks_cnt == 0:
                need = True
                reason.append("chunks_missing")
            elif not req.only_missing:
                need = True

            # 可选：检查 mismatch（需要能统计向量条数时）
            if req.include_mismatch and chunks_cnt > 0:
                try:
                    res = milvus_client.query(
                        collection_name="kb_chunks",
                        filter=f"doc_id == {doc_id}",
                        output_fields=["doc_id"],
                        # Milvus has a max (offset+limit) window of 16384
                        limit=16384
                    )
                    vectors_cnt = len(res) if isinstance(res, list) else 0
                    if vectors_cnt != chunks_cnt:
                        need = True
                        reason.append(f"mismatch:{chunks_cnt}!={vectors_cnt}")
                except Exception:
                    # 统计失败，忽略 mismatch 判定
                    pass

            if need:
                candidates.append({
                    "id": doc_id,
                    "title": r.title,
                    "content": r.content,
                    "chunks_cnt": chunks_cnt,
                    "reason": ",".join(reason) if reason else ("all" if not req.only_missing else "missing")
                })

        if req.limit:
            candidates = candidates[: int(req.limit)]

        if req.dry_run:
            return {"to_process": len(candidates), "candidates": candidates}

        processed = 0
        successes = 0
        errors = []
        items = []

        # 处理函数：从存储的 content 中提取纯文本
        import re as _re
        def extract_raw_text(content_obj) -> tuple[str, dict]:
            if not content_obj:
                return "", {"text": ""}
            if isinstance(content_obj, str):
                return content_obj.strip(), {"markdown": content_obj}
            # 尝试解析 JSON 字符串
            if isinstance(content_obj, (bytes,)):
                try:
                    content_obj = content_obj.decode("utf-8", "ignore")
                except Exception:
                    content_obj = str(content_obj)
            if isinstance(content_obj, str):
                try:
                    import json as _json
                    content_obj = _json.loads(content_obj)
                except Exception:
                    return content_obj.strip(), {"markdown": content_obj}
            if isinstance(content_obj, dict):
                if content_obj.get("markdown"):
                    return content_obj["markdown"].strip(), content_obj
                if content_obj.get("text"):
                    return content_obj["text"].strip(), content_obj
                if content_obj.get("html"):
                    txt = _re.sub(r"<[^>]+>", "", content_obj["html"]) or ""
                    return txt.strip(), content_obj
            # 其他情况，尽力转字符串
            s = str(content_obj)
            return s.strip(), {"markdown": s}

        # 切块器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        )

        async def embed_in_batches(texts, batch_size: int = 10, delay: float = 0.3):
            vectors_all = []
            for i in range(0, len(texts), batch_size):
                part = texts[i:i+batch_size]
                vecs = await embed_texts(part)
                vectors_all.extend(vecs)
                if delay:
                    await asyncio.sleep(delay)
            return vectors_all

        for c in candidates:
            doc_id = c["id"]
            processed += 1
            try:
                raw_text, content_json = extract_raw_text(c["content"])
                if not raw_text:
                    items.append({"id": doc_id, "title": c["title"], "status": "skipped_no_content"})
                    continue

                chunks = splitter.split_text(raw_text)
                if not chunks:
                    items.append({"id": doc_id, "title": c["title"], "status": "skipped_no_chunks"})
                    continue

                # 嵌入
                # 分批嵌入，避免提供商的批量/速率限制（DashScope <=10）
                vectors = await embed_in_batches(chunks, batch_size=10, delay=0.3)

                # 清理旧数据
                try:
                    milvus_client.delete(collection_name="kb_chunks", filter=f"doc_id == {int(doc_id)}")
                    milvus_client.flush("kb_chunks")
                except Exception:
                    pass
                db.execute(sql_text("DELETE FROM doc_chunks WHERE document_id = :doc_id"), {"doc_id": int(doc_id)})
                db.commit()

                # 写 Milvus
                milvus_rows = []
                for i, (t, vec) in enumerate(zip(chunks, vectors)):
                    milvus_rows.append({
                        "doc_id": int(doc_id),
                        "chunk_index": i,
                        "text": truncate_utf8_bytes(t, 1000),
                        "vector": vec
                    })
                insert_result = milvus_client.insert(collection_name="kb_chunks", data=milvus_rows)
                milvus_client.flush("kb_chunks")
                milvus_pks = insert_result.primary_keys if hasattr(insert_result, 'primary_keys') else []

                # 写 doc_chunks
                for i, t in enumerate(chunks):
                    milvus_pk = milvus_pks[i] if i < len(milvus_pks) else None
                    db.execute(sql_text("""
                        INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk)
                        VALUES (:doc_id, :chunk_index, :content, :token_count, :milvus_pk)
                    """), {
                        "doc_id": int(doc_id),
                        "chunk_index": i,
                        "content": t,
                        "token_count": token_len(t),
                        "milvus_pk": milvus_pk
                    })
                db.commit()

                successes += 1
                items.append({
                    "id": doc_id,
                    "title": c["title"],
                    "status": "reindexed",
                    "chunks": len(chunks)
                })
            except Exception as e:
                db.rollback()
                # 展开 tenacity RetryError 的内部 HTTP 错误信息（如有）
                msg = str(e)
                try:
                    from tenacity import RetryError
                    import httpx
                    if isinstance(e, RetryError) and e.last_attempt and e.last_attempt.failed:
                        inner = e.last_attempt.exception()
                        if isinstance(inner, httpx.HTTPStatusError):
                            resp = inner.response
                            body = None
                            try:
                                body = resp.text[:500]
                            except Exception:
                                body = None
                            msg = f"HTTP {resp.status_code} at {resp.request.url}. Body: {body}"
                except Exception:
                    pass
                errors.append({"id": doc_id, "title": c["title"], "error": msg})
                items.append({"id": doc_id, "title": c["title"], "status": "error"})

        return {
            "total_candidates": len(candidates),
            "processed": processed,
            "successes": successes,
            "errors": errors,
            "items": items[:200]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reindex: {str(e)}")

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(None),
    tags: str = Form(None),
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """
    文档入库：切分 → 嵌入 → 写入 Milvus & MySQL
    """
    try:
        # 读取文件内容
        content = await file.read()
        raw_text = content.decode("utf-8", "ignore")
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="File is empty or cannot be decoded")
        
        # 文本切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from file")
        
        # 生成嵌入向量
        print(f"Generating embeddings for {len(chunks)} chunks...")
        vectors = await embed_in_batches(chunks, batch_size=10, delay=0.3)
        
        # 开始数据库事务
        # 1. 插入文档记录到documents表
        doc_result = db.execute(sql_text("""
            INSERT INTO documents(title, source, uri, tags_json)
            VALUES (:title, :source, :uri, :tags)
        """), {
            "title": title or file.filename,
            "source": "upload",
            "uri": file.filename,
            "tags": tags if tags else None
        })
        doc_id = doc_result.lastrowid
        
        # 2. 准备Milvus数据
        milvus_rows = []
        for i, (content, vec) in enumerate(zip(chunks, vectors)):
            milvus_rows.append({
                "doc_id": int(doc_id),
                "chunk_index": i,
                "text": content[:1000],  # VARCHAR长度限制
                "vector": vec
            })
        
        # 3. 插入到Milvus
        print(f"Inserting {len(milvus_rows)} vectors to Milvus...")
        insert_result = milvus_client.insert(
            collection_name="kb_chunks", 
            data=milvus_rows
        )
        milvus_client.flush("kb_chunks")
        
        # 4. 获取Milvus主键（可选，用于回填MySQL）
        milvus_pks = insert_result.primary_keys if hasattr(insert_result, 'primary_keys') else []
        
        # 5. 插入chunk记录到doc_chunks表
        for i, content in enumerate(chunks):
            milvus_pk = milvus_pks[i] if i < len(milvus_pks) else None
            db.execute(sql_text("""
                INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk)
                VALUES (:doc_id, :chunk_index, :content, :token_count, :milvus_pk)
            """), {
                "doc_id": doc_id,
                "chunk_index": i,
                "content": content,
                "token_count": token_len(content),
                "milvus_pk": milvus_pk
            })
        
        # 提交事务
        db.commit()
        
        return {
            "success": True,
            "message": "Document ingested successfully",
            "document_id": doc_id,
            "chunks_count": len(chunks),
            "total_tokens": sum(token_len(chunk) for chunk in chunks)
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@router.get("/documents")
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """列出已入库的文档"""
    try:
        # 查询文档列表
        result = db.execute(sql_text("""
            SELECT d.id, d.title, d.source, d.uri, d.tags_json, d.created_at,
                   COUNT(c.id) as chunks_count,
                   SUM(c.token_count) as total_tokens
            FROM documents d
            LEFT JOIN doc_chunks c ON d.id = c.document_id
            GROUP BY d.id, d.title, d.source, d.uri, d.tags_json, d.created_at
            ORDER BY d.created_at DESC
            LIMIT :limit OFFSET :offset
        """), {"limit": limit, "offset": offset})
        
        documents = []
        for row in result.fetchall():
            doc = {
                "id": row.id,
                "title": row.title,
                "source": row.source,
                "uri": row.uri,
                "tags": json.loads(row.tags_json) if row.tags_json else [],
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "chunks_count": row.chunks_count or 0,
                "total_tokens": row.total_tokens or 0
            }
            documents.append(doc)
        
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """删除文档及其相关数据"""
    try:
        # 检查文档是否存在
        doc_result = db.execute(sql_text("""
            SELECT id FROM documents WHERE id = :doc_id
        """), {"doc_id": doc_id})
        
        if not doc_result.fetchone():
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 从Milvus删除向量（按doc_id过滤）
        milvus_client.delete(
            collection_name="kb_chunks",
            filter=f"doc_id == {doc_id}"
        )
        
        # 从MySQL删除（CASCADE会自动删除chunks）
        db.execute(sql_text("""
            DELETE FROM documents WHERE id = :doc_id
        """), {"doc_id": doc_id})
        
        db.commit()
        
        return {"success": True, "message": f"Document {doc_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
