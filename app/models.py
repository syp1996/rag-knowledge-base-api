# app/models.py
from sqlalchemy import Column, Integer, BigInteger, String, Boolean, DateTime, Text, JSON, ForeignKey, Index, Computed
from sqlalchemy.sql import func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, deferred
from passlib.context import CryptContext
import re

Base = declarative_base()

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # 关系
    documents = relationship("Document", back_populates="user")
    
    def set_password(self, password):
        self.password_hash = pwd_context.hash(password)
    
    def check_password(self, password):
        return pwd_context.verify(password, self.password_hash)

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # 关系
    documents = relationship("Document", back_populates="category")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"))
    title = Column(String(255), nullable=False, comment="冗余字段，需与content中的H1标题同步")
    excerpt = Column(String(500), comment="冗余字段，自动从content中提取的文本摘要")
    content = Column(JSON, comment="存储文档内容的块结构JSON对象")
    content_text = deferred(Column(Text, Computed("CASE WHEN JSON_VALID(content) AND JSON_EXTRACT(content, '$.markdown') IS NOT NULL THEN JSON_UNQUOTE(JSON_EXTRACT(content, '$.markdown')) WHEN JSON_VALID(content) AND JSON_EXTRACT(content, '$.html') IS NOT NULL THEN JSON_UNQUOTE(JSON_EXTRACT(content, '$.html')) ELSE NULL END"), comment="从content JSON中提取的文本内容，用于全文搜索（生成列）"))
    slug = Column(String(255), unique=True)
    is_pinned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # 关系
    user = relationship("User", back_populates="documents")
    category = relationship("Category", back_populates="documents")
    
    # 索引
    __table_args__ = (
        Index('idx_user_id', 'user_id'),
        Index('idx_category_id', 'category_id'),
        Index('idx_slug', 'slug'),
        Index('idx_created_at', 'created_at'),
        Index('idx_is_pinned', 'is_pinned'),
    )
    
    def generate_slug(self, title):
        """生成唯一的slug"""
        # 基础slug生成
        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower())
        slug = re.sub(r'\s+', '-', slug)
        slug = slug.strip('-')
        
        if not slug:
            slug = "document"
        
        return slug
    
    def extract_content_text(self):
        """从content JSON中提取文本内容"""
        if not self.content:
            return ""
        
        if isinstance(self.content, dict):
            if 'markdown' in self.content:
                return self.content['markdown']
            elif 'html' in self.content:
                return self.content['html']
            elif 'text' in self.content:
                return self.content['text']
        
        return ""
    
    def extract_excerpt(self, length=200):
        """从content中提取摘要"""
        # 直接从content中提取文本，不使用content_text字段
        if not self.content:
            return ""
        
        text = ""
        if isinstance(self.content, dict):
            if 'markdown' in self.content:
                text = self.content['markdown']
            elif 'html' in self.content:
                text = self.content['html']
            elif 'text' in self.content:
                text = self.content['text']
        
        if not text:
            return ""
        
        # 移除markdown语法
        import re
        text = re.sub(r'[#*`_\[\]()]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if len(text) <= length:
            return text
        
        return text[:length].rsplit(' ', 1)[0] + "..."
