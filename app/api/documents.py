# app/api/documents.py
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func, text
from typing import List, Optional
from datetime import datetime
import json
import re

from ..deps import get_db
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

router = APIRouter()

@router.get("/", response_model=DocumentList)
async def get_documents(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    status: Optional[int] = Query(None, description="文档状态过滤"),
    category_id: Optional[int] = Query(None, description="分类过滤"),
    db: Session = Depends(get_db)
):
    """获取文档列表（不分页，返回全部）"""
    # 基础查询（过滤软删除）
    query = db.query(Document).filter(Document.deleted_at.is_(None))
    
    # 状态过滤
    if status is not None:
        query = query.filter(Document.status == status)
    
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
    db: Session = Depends(get_db)
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
        slug=slug,
        status=document.status or 0
    )
    
    # 提取摘要
    if document.content:
        db_document.excerpt = db_document.extract_excerpt()
    
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    return db_document

@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """获取单个文档"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.deleted_at.is_(None)  # 检查软删除
    ).first()
    
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
    db: Session = Depends(get_db)
):
    """更新文档（无需认证）"""
    # 检查文档存在性和软删除状态
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.deleted_at.is_(None)
    ).first()
    
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
    if document_update.content is not None:
        document.content = document_update.content
        document.excerpt = document.extract_excerpt()
    
    # 更新分类
    if document_update.category_id is not None:
        document.category_id = document_update.category_id
    
    # 更新状态
    if document_update.status is not None:
        document.status = document_update.status
    
    db.commit()
    db.refresh(document)
    
    return document

@router.delete("/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """删除文档（软删除，无需认证）"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.deleted_at.is_(None)
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # 软删除
    document.deleted_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Document deleted successfully"}

@router.post("/{document_id}/publish", response_model=DocumentSchema)
async def publish_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """发布文档"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.deleted_at.is_(None)
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    document.status = 1  # 发布状态
    db.commit()
    db.refresh(document)
    
    return document

@router.post("/{document_id}/pin", response_model=DocumentSchema)
async def pin_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """置顶/取消置顶文档"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.deleted_at.is_(None)
    ).first()
    
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

@router.get("/search", response_model=SearchResult)
async def search_documents(
    keyword: str = Query(..., description="搜索关键词"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    search_mode: str = Query("fulltext", description="搜索模式: fulltext 或 basic"),
    highlight: bool = Query(True, description="是否高亮显示"),
    db: Session = Depends(get_db)
):
    """文档搜索"""
    offset = (page - 1) * per_page
    
    # 基础查询（只搜索已发布且未删除的文档）
    base_query = db.query(Document).filter(
        Document.status == 1,  # 已发布
        Document.deleted_at.is_(None)  # 未删除
    )
    
    if search_mode == "fulltext":
        # 全文搜索（MySQL FULLTEXT，需要先创建索引）
        try:
            # 使用MATCH AGAINST进行全文搜索
            search_query = base_query.filter(
                text("MATCH(content_text) AGAINST(:keyword IN NATURAL LANGUAGE MODE)")
            ).params(keyword=keyword)
            
            # 按相关度排序
            search_query = search_query.order_by(
                text("MATCH(content_text) AGAINST(:keyword IN NATURAL LANGUAGE MODE) DESC")
            ).params(keyword=keyword)
            
        except Exception:
            # 如果全文搜索失败，回退到基础搜索
            search_mode = "basic"
    
    if search_mode == "basic":
        # 基础LIKE搜索
        search_pattern = f"%{keyword}%"
        search_query = base_query.filter(
            or_(
                Document.title.like(search_pattern),
                Document.content_text.like(search_pattern),
                Document.excerpt.like(search_pattern)
            )
        ).order_by(Document.is_pinned.desc(), Document.created_at.desc())
    
    # 获取总数
    total = search_query.count()
    
    # 分页
    documents = search_query.offset(offset).limit(per_page).all()
    
    # 高亮处理
    if highlight:
        for doc in documents:
            if doc.excerpt:
                doc.excerpt = highlight_search_text(doc.excerpt, keyword)
    
    return SearchResult(
        documents=documents,
        total=total,
        page=page,
        per_page=per_page,
        keyword=keyword,
        search_mode=search_mode
    )

@router.post("/upload", response_model=DocumentSchema)
async def upload_document(
    file: UploadFile = File(...),
    category_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
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
        slug=slug,
        status=0  # 草稿状态
    )
    
    # 提取摘要
    db_document.excerpt = db_document.extract_excerpt()
    
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    return db_document

@router.post("/plugin", response_model=DocumentSchema)
async def create_plugin_document(
    plugin_doc: PluginDocumentCreate,
    db: Session = Depends(get_db)
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
        slug=slug,
        status=0  # 草稿状态
    )
    
    # 提取摘要
    db_document.excerpt = db_document.extract_excerpt()
    
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    return db_document
