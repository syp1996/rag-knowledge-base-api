# RAG Knowledge Base with Document Management System

基于FastAPI的RAG知识库系统，现已集成完整的文档管理功能，支持用户认证、分类管理、文档CRUD操作等。

## 🚀 新增功能

### 1. 用户认证系统
- ✅ 用户注册/登录
- ✅ JWT Token认证  
- ✅ 权限管理（普通用户/管理员）
- ✅ 用户资料管理

### 2. 文档管理系统
- ✅ 文档增删查改（删除为硬删除，直接从数据库移除并同步清理向量）
- ✅ 文档置顶功能
- ✅ 文档搜索（全文搜索 + 基础搜索）
- ✅ Markdown文件上传
- ✅ Chrome插件支持

### 3. 分类管理
- ✅ 分类增删查改
- ✅ 分类下文档查询

### 4. 用户管理
- ✅ 用户列表查询
- ✅ 用户权限管理
- ✅ 管理员功能

## 📋 数据库表结构

### users表 (用户表)
```sql
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,  
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### categories表 (分类表)
```sql
CREATE TABLE categories (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL,
    description VARCHAR(200),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### documents表 (文档表)
```sql
CREATE TABLE documents (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    category_id INT,
    title VARCHAR(255) NOT NULL,
    excerpt VARCHAR(500),
    content JSON,
    content_text LONGTEXT,
    slug VARCHAR(255) UNIQUE,
    is_pinned BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    -- 外键和索引
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);
```

## 🔧 安装和配置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 初始化数据库
```bash
python init_database.py
```

这将：
- 创建所有数据库表
- 添加全文搜索索引
- 创建初始用户和分类数据

### 3. 启动服务
```bash
python run.py
```

## 📚 API接口文档

服务启动后访问: http://localhost:8000/docs

### 认证相关接口 (`/api/auth/`)

- **POST** `/api/auth/register` - 用户注册
- **POST** `/api/auth/login` - 用户登录  
- **GET** `/api/auth/profile` - 获取用户资料
- **PUT** `/api/auth/profile` - 更新用户资料

### 文档相关接口 (`/api/documents/`)

- **GET** `/api/documents/` - 获取文档列表
- **POST** `/api/documents/` - 创建文档
- **GET** `/api/documents/{id}` - 获取单个文档
- **PUT** `/api/documents/{id}` - 更新文档
- **DELETE** `/api/documents/{id}` - 删除文档（硬删除：移除数据库并清理 Milvus 向量）
- **POST** `/api/documents/{id}/pin` - 置顶文档
- **GET** `/api/documents/search` - 搜索文档
- **POST** `/api/documents/upload` - 上传Markdown文件
- **POST** `/api/documents/plugin` - Chrome插件创建文档

### 用户管理接口 (`/api/users/`)

- **GET** `/api/users/` - 获取用户列表（仅管理员）
- **GET** `/api/users/{id}` - 获取用户信息
- **PUT** `/api/users/{id}` - 更新用户信息
- **DELETE** `/api/users/{id}` - 删除用户（仅管理员）
- **POST** `/api/users/{id}/toggle-admin` - 切换管理员权限

### 分类管理接口 (`/api/categories/`)

- **GET** `/api/categories/` - 获取分类列表
- **POST** `/api/categories/` - 创建分类（仅管理员）
- **GET** `/api/categories/{id}` - 获取单个分类
- **PUT** `/api/categories/{id}` - 更新分类（仅管理员）
- **DELETE** `/api/categories/{id}` - 删除分类（仅管理员）
- **GET** `/api/categories/{id}/documents` - 获取分类下的文档

## 👤 默认用户账号

初始化完成后，系统会创建以下默认账号：

- **管理员**: `admin` / `admin123`
- **普通用户**: `default` / `default123`  
- **插件用户**: `chrome_plugin_user` / `plugin123`

## 🔍 搜索功能

支持两种搜索模式：

1. **全文搜索** (`search_mode=fulltext`)
   - 使用MySQL的FULLTEXT索引
   - 支持相关度排序
   - 性能更好

2. **基础搜索** (`search_mode=basic`)
   - 使用LIKE查询
   - 兼容性更好
   - 支持标题、内容、摘要搜索

### 搜索示例
```bash
GET /api/documents/search?keyword=python&search_mode=fulltext&highlight=true
```

## 📝 文档内容格式

文档的content字段支持JSON格式存储：

```json
{
  "markdown": "# 标题\n\n内容...",
  "html": "<h1>标题</h1><p>内容...</p>",
  "text": "纯文本内容",
  "url": "https://example.com"  // Chrome插件使用
}
```

## 🔐 权限控制

### 三级权限体系：

1. **无需认证**
   - 文档查看
   - 分类查看
   - 文档搜索
   - 文档创建（使用默认用户）

2. **用户认证**
   - 个人资料管理
   - 查看自己的信息

3. **管理员权限**
   - 用户管理
   - 分类管理
   - 权限管理

## 🛠️ 技术栈

- **Web框架**: FastAPI
- **数据库**: MySQL + SQLAlchemy ORM
- **向量数据库**: Milvus
- **认证**: JWT + bcrypt
- **文档**: Swagger/OpenAPI

## 📂 项目结构

```
F:\project\
├── app/
│   ├── api/                 # API路由
│   │   ├── auth.py         # 认证接口
│   │   ├── users.py        # 用户管理接口
│   │   ├── categories.py   # 分类管理接口
│   │   └── documents.py    # 文档管理接口
│   ├── models.py           # 数据库模型
│   ├── schemas.py          # Pydantic模型
│   ├── auth.py             # 认证逻辑
│   ├── utils.py            # 工具函数
│   ├── deps.py             # 依赖注入
│   └── main.py             # 主应用
├── init_database.py        # 数据库初始化脚本
├── run.py                  # 启动脚本
└── requirements.txt        # 依赖列表
```

## 🎯 使用示例

### 1. 用户注册和登录
```python
import requests

# 注册用户
response = requests.post("http://localhost:8000/api/auth/register", json={
    "username": "testuser",
    "email": "test@example.com", 
    "password": "password123"
})

# 登录获取token
response = requests.post("http://localhost:8000/api/auth/login", json={
    "username": "testuser",
    "password": "password123"
})
token = response.json()["access_token"]

# 使用token访问需要认证的接口
headers = {"Authorization": f"Bearer {token}"}
profile = requests.get("http://localhost:8000/api/auth/profile", headers=headers)
```

### 2. 文档操作
```python
# 创建文档
doc_data = {
    "title": "我的第一篇文档",
    "content": {
        "markdown": "# 标题\n\n这是文档内容..."
    },
    "category_id": 1
}

response = requests.post("http://localhost:8000/api/documents/", json=doc_data)
doc_id = response.json()["id"]

# 搜索文档（全文 / 基础）
search_result = requests.get(
    "http://localhost:8000/api/documents/search",
    params={"keyword": "标题", "search_mode": "fulltext"}
)
```

## 🤝 原有RAG功能

原有的RAG知识库功能完全保留：

- **文档入库**: `/api/v1/ingest/`
- **向量搜索**: `/api/v1/search/`  
- **健康检查**: `/health`

新的文档管理系统与RAG功能可以并行使用，为不同的业务场景提供支持。

---

🎉 现在你拥有一个功能完整的文档管理系统！可以开始测试各项功能了。
