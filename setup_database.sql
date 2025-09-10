-- 文档和片段存储数据库 Schema 设计
-- 用于向量检索系统的元数据和文本存储

-- 1. 创建文档表
CREATE TABLE IF NOT EXISTS documents (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(255),
  source VARCHAR(64),
  uri TEXT,
  tags JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. 创建文档片段表
CREATE TABLE IF NOT EXISTS chunks (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  document_id BIGINT NOT NULL,
  chunk_index INT NOT NULL,
  content MEDIUMTEXT NOT NULL,
  token_count INT,
  milvus_pk BIGINT,        -- 对应 Milvus 的主键（或 UUID 映射）
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 3. 添加全文检索索引（用于混合检索，MySQL 提供 BM25 类似的 Fulltext 搜索）
ALTER TABLE chunks ADD FULLTEXT KEY ft_content (content);

-- 4. 创建一些有用的索引
CREATE INDEX idx_document_source ON documents(source);
CREATE INDEX idx_document_created ON documents(created_at);
CREATE INDEX idx_chunk_document_id ON chunks(document_id);
CREATE INDEX idx_chunk_milvus_pk ON chunks(milvus_pk);
CREATE INDEX idx_chunk_created ON chunks(created_at);

-- 5. 显示创建的表结构
SHOW CREATE TABLE documents;
SHOW CREATE TABLE chunks;