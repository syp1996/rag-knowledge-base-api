# app/utils.py
import re
import uuid
from typing import Optional
from sqlalchemy.orm import Session
from .models import User, Document

def generate_unique_slug(db: Session, title: str, document_id: Optional[int] = None) -> str:
    """生成唯一的slug"""
    # 基础slug生成
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower())
    slug = re.sub(r'\s+', '-', slug)
    slug = slug.strip('-')
    
    if not slug:
        slug = "document"
    
    # 检查唯一性
    base_slug = slug
    counter = 1
    
    while True:
        # 查询是否存在相同的slug（排除当前文档）
        query = db.query(Document).filter(Document.slug == slug)
        if document_id:
            query = query.filter(Document.id != document_id)
        
        existing = query.first()
        if not existing:
            break
        
        # 如果存在，添加数字后缀
        slug = f"{base_slug}-{counter}"
        counter += 1
    
    return slug

def get_or_create_default_user(db: Session) -> User:
    """获取或创建默认用户"""
    # 查找用户名为 'default' 的用户
    default_user = db.query(User).filter(User.username == "default").first()
    
    if not default_user:
        # 检查是否已有同样邮箱的用户
        existing_email_user = db.query(User).filter(User.email == "default@example.com").first()
        if existing_email_user:
            # 如果有同样邮箱的用户，就返回那个用户
            return existing_email_user
        
        # 创建默认用户
        default_user = User(
            username="default",
            email="default@example.com",
            is_admin=False
        )
        default_user.set_password("default123")
        
        db.add(default_user)
        db.commit()
        db.refresh(default_user)
    
    return default_user

def get_or_create_chrome_plugin_user(db: Session) -> User:
    """获取或创建Chrome插件专用用户"""
    plugin_user = db.query(User).filter(User.username == "chrome_plugin_user").first()
    
    if not plugin_user:
        # 创建Chrome插件用户
        plugin_user = User(
            username="chrome_plugin_user",
            email="chrome_plugin@example.com",
            is_admin=False
        )
        plugin_user.set_password(str(uuid.uuid4()))  # 随机密码
        
        db.add(plugin_user)
        db.commit()
        db.refresh(plugin_user)
    
    return plugin_user

def highlight_search_text(text: str, keyword: str, max_length: int = 300) -> str:
    """高亮搜索关键词"""
    if not keyword or not text:
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # 查找关键词位置
    keyword_lower = keyword.lower()
    text_lower = text.lower()
    
    start_pos = text_lower.find(keyword_lower)
    if start_pos == -1:
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # 计算摘要范围
    keyword_len = len(keyword)
    before_len = (max_length - keyword_len) // 2
    after_len = max_length - keyword_len - before_len
    
    start = max(0, start_pos - before_len)
    end = min(len(text), start_pos + keyword_len + after_len)
    
    # 调整到词边界
    if start > 0:
        space_pos = text.find(' ', start)
        if space_pos != -1 and space_pos < start + 20:
            start = space_pos + 1
    
    if end < len(text):
        space_pos = text.rfind(' ', start, end)
        if space_pos != -1 and space_pos > end - 20:
            end = space_pos
    
    excerpt = text[start:end]
    
    # 高亮关键词
    highlighted = re.sub(
        f'({re.escape(keyword)})',
        r'<mark>\1</mark>',
        excerpt,
        flags=re.IGNORECASE
    )
    
    # 添加省略号
    if start > 0:
        highlighted = "..." + highlighted
    if end < len(text):
        highlighted = highlighted + "..."
    
    return highlighted

def extract_title_from_content(content: dict) -> str:
    """从content中提取标题"""
    if not content:
        return "Untitled Document"
    
    # 从markdown中提取H1标题
    if 'markdown' in content:
        markdown_text = content['markdown']
        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
    
    # 从HTML中提取H1标题
    if 'html' in content:
        html_text = content['html']
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_text, re.IGNORECASE | re.DOTALL)
        if h1_match:
            return re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()
    
    # 从纯文本中提取第一行作为标题
    if 'text' in content:
        text = content['text']
        first_line = text.split('\n')[0].strip()
        if first_line:
            return first_line[:100]  # 限制标题长度
    
    return "Untitled Document"