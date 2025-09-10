# app/api/categories.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from ..deps import get_db
from ..models import Category, Document, User
from ..schemas import (
    Category as CategorySchema, 
    CategoryCreate, 
    CategoryUpdate, 
    MessageResponse,
    Document as DocumentSchema
)
from ..auth import get_current_admin_user

router = APIRouter()

@router.get("/", response_model=List[CategorySchema])
async def get_categories(db: Session = Depends(get_db)):
    """获取分类列表（无需认证）"""
    categories = db.query(Category).all()
    return categories

@router.post("/", response_model=CategorySchema)
async def create_category(
    category: CategoryCreate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """创建分类（仅管理员）"""
    # 检查分类名唯一性
    existing_category = db.query(Category).filter(Category.name == category.name).first()
    if existing_category:
        raise HTTPException(
            status_code=400,
            detail="Category name already exists"
        )
    
    db_category = Category(
        name=category.name,
        description=category.description
    )
    
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    
    return db_category

@router.get("/{category_id}", response_model=CategorySchema)
async def get_category(
    category_id: int,
    db: Session = Depends(get_db)
):
    """获取单个分类"""
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    return category

@router.put("/{category_id}", response_model=CategorySchema)
async def update_category(
    category_id: int,
    category_update: CategoryUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """更新分类（仅管理员）"""
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    # 检查分类名唯一性（如果要更新名称）
    if category_update.name and category_update.name != category.name:
        existing_category = db.query(Category).filter(Category.name == category_update.name).first()
        if existing_category:
            raise HTTPException(
                status_code=400,
                detail="Category name already exists"
            )
        category.name = category_update.name
    
    # 更新描述
    if category_update.description is not None:
        category.description = category_update.description
    
    db.commit()
    db.refresh(category)
    
    return category

@router.delete("/{category_id}", response_model=MessageResponse)
async def delete_category(
    category_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """删除分类（仅管理员）"""
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    # 检查是否有文档使用此分类
    documents_count = db.query(Document).filter(
        Document.category_id == category_id,
        Document.deleted_at.is_(None)
    ).count()
    
    if documents_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete category with {documents_count} documents. Please move or delete the documents first."
        )
    
    db.delete(category)
    db.commit()
    
    return {"message": "Category deleted successfully"}

@router.get("/{category_id}/documents", response_model=List[DocumentSchema])
async def get_category_documents(
    category_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """获取分类下的文档（只返回已发布的文档）"""
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    offset = (page - 1) * per_page
    
    # 只返回已发布的文档
    documents = db.query(Document).filter(
        Document.category_id == category_id,
        Document.status == 1,  # 已发布
        Document.deleted_at.is_(None)  # 未删除
    ).order_by(Document.is_pinned.desc(), Document.created_at.desc()).offset(offset).limit(per_page).all()
    
    return documents