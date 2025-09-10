# app/schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Any
from datetime import datetime

# 用户相关schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class User(UserBase):
    id: int
    is_admin: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserProfile(User):
    pass

# Token相关schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# 分类相关schemas
class CategoryBase(BaseModel):
    name: str
    description: Optional[str] = None

class CategoryCreate(CategoryBase):
    pass

class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class Category(CategoryBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# 文档相关schemas
class DocumentBase(BaseModel):
    title: str
    content: Optional[dict] = None
    category_id: Optional[int] = None
    status: Optional[int] = 0

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[dict] = None
    category_id: Optional[int] = None
    status: Optional[int] = None

class DocumentUpload(BaseModel):
    title: str
    content: str
    category_id: Optional[int] = None

class Document(DocumentBase):
    id: int
    user_id: int
    excerpt: Optional[str] = None
    slug: Optional[str] = None
    is_pinned: bool
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None
    
    # 关联对象
    user: Optional[User] = None
    category: Optional[Category] = None
    
    class Config:
        from_attributes = True

class DocumentList(BaseModel):
    documents: List[Document]
    total: int
    page: int
    per_page: int
    pages: int

# 搜索相关schemas
class SearchResult(BaseModel):
    documents: List[Document]
    total: int
    page: int
    per_page: int
    keyword: str
    search_mode: str

# Chrome插件相关schemas
class PluginDocumentCreate(BaseModel):
    title: str
    url: str
    content: str
    category_id: Optional[int] = None

# 响应schemas
class MessageResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None