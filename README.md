# RAG Knowledge Base API

基于 FastAPI + Milvus + MySQL 的向量检索知识库系统，支持文档入库、语义搜索和混合检索。

## 🚀 功能特性

- **文档入库**：支持文本文件上传、自动切分、向量化存储
- **语义检索**：基于 Milvus IVF_FLAT 索引的高效向量搜索
- **混合检索**：结合向量搜索和 MySQL 全文检索
- **多嵌入供应商**：支持 Cohere、OpenAI、Voyage AI 等
- **RESTful API**：完整的 API 文档和交互界面

## 📋 系统架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │    │   Milvus    │    │    MySQL    │
│   (API层)   │───▶│  (向量存储)  │    │ (元数据存储) │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └─────────────────────┼───────────────────┘
                            │
                    ┌─────────────┐
                    │ Embedding   │
                    │ Providers   │
                    └─────────────┘
```

## 🛠️ 安装部署

### 1. 环境准备

```bash
# Python 3.8+
pip install -r requirements.txt

# 复制环境变量配置
cp .env.example .env
```

### 2. 数据库初始化

```bash
# 初始化 MySQL 表结构
python setup_mysql.py

# 初始化 Milvus 集合和索引
python tools/init_milvus.py
```

### 3. 环境变量配置

编辑 `.env` 文件：

```env
# 数据库配置（已配置好）
DB_HOST=rm-bp1ljbjb34n55su6uko.mysql.rds.aliyuncs.com
DB_PORT=3306
DB_NAME=markdown_manager
DB_USER=markdown_user
DB_PASSWORD=Syp19960424

# Milvus配置
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=

# 嵌入模型配置
EMBED_PROVIDER=cohere
EMBED_DIM=1024
COHERE_API_KEY=your_cohere_api_key
COHERE_EMBED_MODEL=embed-multilingual-v3.0
```

### 4. 启动服务

```bash
# 开发模式
python run.py

# 或直接使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 📖 API 使用说明

### 文档入库

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.txt" \
  -F "title=测试文档" \
  -F "tags=测试,知识库"
```

### 语义检索

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用向量数据库？",
    "top_k": 5,
    "nprobe": 16,
    "score_threshold": 0.7
  }'
```

### 混合检索

```bash
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "机器学习算法",
    "top_k": 8,
    "rerank": false
  }'
```

## 🔧 技术细节

### 数据库表结构

**documents 表**：文档元数据
```sql
CREATE TABLE documents (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(255),
  source VARCHAR(64),
  uri TEXT,
  tags_json JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**doc_chunks 表**：文档片段
```sql
CREATE TABLE doc_chunks (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  document_id BIGINT UNSIGNED NOT NULL,
  chunk_index INT NOT NULL,
  content MEDIUMTEXT NOT NULL,
  token_count INT,
  milvus_pk BIGINT,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

### Milvus 集合结构

```python
schema.add_field(field_name="pk", datatype="INT64", is_primary=True)
schema.add_field(field_name="doc_id", datatype="INT64")
schema.add_field(field_name="chunk_index", datatype="INT32")
schema.add_field(field_name="text", datatype="VARCHAR", max_length=1000)
schema.add_field(field_name="vector", datatype="FLOAT_VECTOR", dim=1024)
```

### 索引配置

- **索引类型**：IVF_FLAT
- **距离度量**：COSINE
- **参数**：nlist=2048, nprobe=16（可调优）

## 🔍 性能优化

1. **分块策略**：chunk_size=900, chunk_overlap=150
2. **批处理**：支持批量文档入库和搜索
3. **连接池**：数据库连接复用
4. **索引优化**：合理的 nlist 和 nprobe 参数

## 🐛 问题排查

### 常见问题

1. **Milvus 连接失败**
   ```bash
   # 检查 Milvus 服务状态
   docker ps | grep milvus
   ```

2. **MySQL 连接失败**
   ```bash
   # 测试连接
   python -c "from app.deps import engine; print(engine.execute('SELECT 1').fetchone())"
   ```

3. **嵌入 API 调用失败**
   ```bash
   # 检查 API Key 和网络连接
   curl -H "Authorization: Bearer $COHERE_API_KEY" https://api.cohere.com/v1/models
   ```

### 日志查看

```bash
# 查看应用日志
tail -f app.log

# 调试模式启动
DEBUG=true python run.py
```

## 📈 后续扩展

- [ ] 支持更多文档格式（PDF、Word 等）
- [ ] 集成重排序模型（Cohere Rerank、BGE Reranker）
- [ ] 添加用户权限管理
- [ ] 实现实时流式搜索
- [ ] 支持多语言检索
- [ ] 添加检索结果缓存

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License