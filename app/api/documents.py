# app/api/documents.py
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func, text
from typing import List, Optional
from datetime import datetime
import json
import re

from ..deps import get_db, get_milvus
from ..models import Document, User, Category
from ..schemas import (
    Document as DocumentSchema,
    DocumentCreate,
    DocumentUpdate,
    DocumentUpload,
    PluginDocumentCreate,
    DocumentList,
    SearchResult,
    MessageResponse
)
from ..auth import get_current_user_optional, get_current_active_user
from ..utils import (
    generate_unique_slug,
    get_or_create_default_user,
    get_or_create_chrome_plugin_user,
    highlight_search_text,
    extract_title_from_content
)
from ..ingest import _env_chunk_params, _make_splitter, embed_in_batches, token_len, truncate_utf8_bytes
from ..embedding import embed_texts

router = APIRouter()


# --- Helpers: reindex a document into doc_chunks + Milvus ---
async def _reindex_document(
    *,
    db: Session,
    milvus_client,
    document_id: int,
    content_obj: dict | None,
):
    """Split, embed and (re)index a document's content into Milvus and doc_chunks.

    - Deletes previous vectors/chunks for the doc
    - Splits using configured chunk params
    - Embeds in batches
    - Inserts to Milvus and doc_chunks
    """
    # Extract raw text from content_obj
    import re
    raw_text = ""
    if content_obj:
        if isinstance(content_obj, dict):
            if content_obj.get("markdown"):
                raw_text = str(content_obj.get("markdown") or "").strip()
            elif content_obj.get("text"):
                raw_text = str(content_obj.get("text") or "").strip()
            elif content_obj.get("html"):
                raw_text = re.sub(r"<[^>]+>", "", str(content_obj.get("html") or "")) or ""
                raw_text = raw_text.strip()
        else:
            raw_text = str(content_obj).strip()

    if not raw_text:
        # Nothing to index; clean existing if any
        try:
            milvus_client.delete(collection_name="kb_chunks", filter=f"doc_id == {int(document_id)}")
        except Exception:
            pass
        db.execute(text("DELETE FROM doc_chunks WHERE document_id = :id"), {"id": int(document_id)})
        db.commit()
        return {"chunks": 0, "tokens": 0}

    # Chunking
    size, overlap = _env_chunk_params()
    splitter = _make_splitter(size, overlap)
    chunks = splitter.split_text(raw_text)
    if not chunks:
        # Clean existing and return
        try:
            milvus_client.delete(collection_name="kb_chunks", filter=f"doc_id == {int(document_id)}")
        except Exception:
            pass
        db.execute(text("DELETE FROM doc_chunks WHERE document_id = :id"), {"id": int(document_id)})
        db.commit()
        return {"chunks": 0, "tokens": 0}

    # Embeddings
    vectors = await embed_in_batches(chunks, batch_size=10, delay=0.3)

    # Remove previous entries
    try:
        milvus_client.delete(collection_name="kb_chunks", filter=f"doc_id == {int(document_id)}")
        milvus_client.flush("kb_chunks")
    except Exception:
        pass
    db.execute(text("DELETE FROM doc_chunks WHERE document_id = :id"), {"id": int(document_id)})

    # Insert new vectors
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

    # Insert doc_chunks
    for i, content in enumerate(chunks):
        milvus_pk = milvus_pks[i] if i < len(milvus_pks) else None
        db.execute(text(
            """
            INSERT INTO doc_chunks(document_id, chunk_index, content, token_count, milvus_pk)
            VALUES (:doc_id, :chunk_index, :content, :token_count, :milvus_pk)
            """
        ), {
            "doc_id": int(document_id),
            "chunk_index": i,
            "content": content,
            "token_count": token_len(content),
            "milvus_pk": milvus_pk
        })

    db.commit()
    return {"chunks": len(chunks), "tokens": sum(token_len(c) for c in chunks)}

@router.get("/", response_model=DocumentList)
async def get_documents(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    category_id: Optional[int] = Query(None, description="分类过滤"),
    db: Session = Depends(get_db)
):
    """获取文档列表（不分页，返回全部）"""
    # 基础查询（返回全部未物理删除的文档）
    query = db.query(Document)
    
    # 分类过滤
    if category_id is not None:
        query = query.filter(Document.category_id == category_id)
    
    # 排序：置顶文档在前，然后按创建时间倒序
    query = query.order_by(Document.is_pinned.desc(), Document.created_at.desc())
    
    # 不分页：直接取全部
    documents = query.all()
    total = len(documents)
    
    return DocumentList(
        documents=documents,
        total=total,
        page=1,
        per_page=total,
        pages=1
    )

@router.post("/", response_model=DocumentSchema)
async def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """创建文档（无需认证，使用默认用户）"""
    # 获取或创建默认用户
    default_user = get_or_create_default_user(db)
    
    # 从content中提取标题（如果没有提供标题）
    title = document.title
    if not title and document.content:
        title = extract_title_from_content(document.content)
    
    # 生成唯一slug
    slug = generate_unique_slug(db, title)
    
    # 创建文档
    db_document = Document(
        user_id=default_user.id,
        category_id=document.category_id,
        title=title,
        content=document.content,
        slug=slug
    )
    
    # 提取摘要
    if document.content:
        db_document.excerpt = db_document.extract_excerpt()
    
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Reindex content so it’s searchable immediately
    try:
        await _reindex_document(db=db, milvus_client=milvus_client, document_id=int(db_document.id), content_obj=db_document.content)
    except Exception as e:
        # Do not fail creation due to indexing
        print(f"[create_document] Reindex failed for doc {db_document.id}: {e}")
    
    return db_document

@router.get("/search", response_model=SearchResult)
async def search_documents(
    keyword: str = Query(..., description="搜索关键词"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    search_mode: str = Query("fulltext", description="搜索模式: fulltext 或 basic"),
    highlight: bool = Query(True, description="是否高亮显示"),
    db: Session = Depends(get_db)
):
    """文档搜索（未删除的全部文档；默认 FULLTEXT，索引缺失自动回退 LIKE）。"""
    offset = (page - 1) * per_page

    # 搜索全部文档
    base_query = db.query(Document)

    used_mode = search_mode
    search_query = None
    total = 0

    if used_mode == "fulltext":
        try:
            q = base_query.filter(
                text("MATCH(content_text) AGAINST(:keyword IN NATURAL LANGUAGE MODE)")
            ).params(keyword=keyword)
            q = q.order_by(
                text("MATCH(content_text) AGAINST(:keyword IN NATURAL LANGUAGE MODE) DESC")
            ).params(keyword=keyword)
            # 触发执行以便检测索引缺失问题
            total = q.count()
            search_query = q
        except Exception as e:
            # 索引缺失或不支持，回退到 basic
            print(f"[/api/documents/search] FULLTEXT unavailable, fallback to LIKE: {e}")
            used_mode = "basic"

    if used_mode == "basic":
        pattern = f"%{keyword}%"
        q = base_query.filter(
            or_(
                Document.title.like(pattern),
                Document.content_text.like(pattern),
                Document.excerpt.like(pattern)
            )
        ).order_by(Document.is_pinned.desc(), Document.created_at.desc())
        total = q.count()
        search_query = q

    documents = search_query.offset(offset).limit(per_page).all() if search_query else []

    if highlight:
        for doc in documents:
            if getattr(doc, "excerpt", None):
                doc.excerpt = highlight_search_text(doc.excerpt, keyword)

    return SearchResult(
        documents=documents,
        total=total,
        page=page,
        per_page=per_page,
        keyword=keyword,
        search_mode=used_mode
    )

@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """获取单个文档"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return document

@router.put("/{document_id}", response_model=DocumentSchema)
async def update_document(
    document_id: int,
    document_update: DocumentUpdate,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """更新文档（无需认证）"""
    # 检查文档存在性和软删除状态
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # 更新标题
    if document_update.title:
        document.title = document_update.title
        # 标题更新时自动更新slug
        document.slug = generate_unique_slug(db, document_update.title, document_id)
    
    # 更新内容
    content_changed = False
    if document_update.content is not None:
        document.content = document_update.content
        document.excerpt = document.extract_excerpt()
        content_changed = True
    
    # 更新分类
    if document_update.category_id is not None:
        document.category_id = document_update.category_id
    
    db.commit()
    db.refresh(document)

    # If content changed, reindex into Milvus + doc_chunks
    if content_changed:
        try:
            await _reindex_document(db=db, milvus_client=milvus_client, document_id=int(document.id), content_obj=document.content)
        except Exception as e:
            print(f"[update_document] Reindex failed for doc {document.id}: {e}")
            # Keep returning the updated document even if indexing fails
    
    return document

@router.delete("/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """删除文档（硬删除：同时删除 Milvus 向量与数据库记录）"""
    # 检查文档是否存在
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # 从Milvus删除向量（按doc_id过滤）
    try:
        milvus_client.delete(collection_name="kb_chunks", filter=f"doc_id == {int(document_id)}")
    except Exception as e:
        # 不中断删除流程，但记录日志
        print(f"[documents.delete] Milvus delete failed for doc {document_id}: {e}")

    # 从MySQL删除（CASCADE自动删除 doc_chunks）
    db.execute(text("""
        DELETE FROM documents WHERE id = :doc_id
    """), {"doc_id": int(document_id)})
    db.commit()

    return {"message": f"Document {document_id} deleted successfully"}


@router.post("/{document_id}/pin", response_model=DocumentSchema)
async def pin_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """置顶/取消置顶文档"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # 切换置顶状态
    document.is_pinned = not document.is_pinned
    db.commit()
    db.refresh(document)
    
    return document

@router.post("/upload", response_model=DocumentSchema)
async def upload_document(
    file: UploadFile = File(...),
    category_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """上传文档"""
    # 检查文件类型
    if not file.filename.endswith('.md'):
        raise HTTPException(
            status_code=400,
            detail="Only markdown files (.md) are supported"
        )
    
    # 读取文件内容
    content = await file.read()
    markdown_content = content.decode('utf-8')
    
    # 从文件名提取标题
    title = file.filename.replace('.md', '').replace('_', ' ').replace('-', ' ')
    
    # 生成唯一slug
    slug = generate_unique_slug(db, title)
    
    # 获取默认用户
    default_user = get_or_create_default_user(db)
    
    # 创建文档
    db_document = Document(
        user_id=default_user.id,
        category_id=category_id,
        title=title,
        content={"markdown": markdown_content},
        slug=slug
    )
    
    # 提取摘要
    db_document.excerpt = db_document.extract_excerpt()
    
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Index uploaded content
    try:
        await _reindex_document(db=db, milvus_client=milvus_client, document_id=int(db_document.id), content_obj=db_document.content)
    except Exception as e:
        print(f"[upload_document] Reindex failed for doc {db_document.id}: {e}")
    
    return db_document

@router.post("/plugin", response_model=DocumentSchema)
async def create_plugin_document(
    plugin_doc: PluginDocumentCreate,
    db: Session = Depends(get_db),
    milvus_client = Depends(get_milvus)
):
    """Chrome插件创建文档（无需认证）"""
    # 获取Chrome插件专用用户
    plugin_user = get_or_create_chrome_plugin_user(db)
    
    # 生成唯一slug
    slug = generate_unique_slug(db, plugin_doc.title)
    
    # 创建文档
    db_document = Document(
        user_id=plugin_user.id,
        category_id=plugin_doc.category_id,
        title=plugin_doc.title,
        content={
            "html": plugin_doc.content,
            "url": plugin_doc.url
        },
        slug=slug
    )
    
    # 提取摘要
    db_document.excerpt = db_document.extract_excerpt()
    
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Index captured page content
    try:
        await _reindex_document(db=db, milvus_client=milvus_client, document_id=int(db_document.id), content_obj=db_document.content)
    except Exception as e:
        print(f"[create_plugin_document] Reindex failed for doc {db_document.id}: {e}")
    
    return db_document
