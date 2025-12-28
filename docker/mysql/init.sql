-- MySQL初始化脚本
-- 金融研报智能系统数据库初始化

-- 设置字符集
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS financial_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE financial_rag;

-- 用户表（如果需要认证功能）
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),
    bio TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    role ENUM('admin', 'user', 'viewer') DEFAULT 'user',
    preferences JSON,
    document_count INT DEFAULT 0,
    search_count INT DEFAULT 0,
    last_login_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_users_username (username),
    INDEX idx_users_email (email),
    INDEX idx_users_role (role),
    INDEX idx_users_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建默认管理员用户（可选）
INSERT IGNORE INTO users (username, email, hashed_password, full_name, role)
VALUES ('admin', 'admin@financial-rag.com', 'placeholder_hash', '系统管理员', 'admin');

-- 新增表结构：Phase 1 完整数据库表结构
-- 由后端SQLAlchemy自动创建，这里提供手动版本作为备份

-- 文档表
CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    file_size BIGINT,
    file_type VARCHAR(50),
    content_type VARCHAR(100),
    file_hash VARCHAR(64),
    content_hash VARCHAR(64),
    status ENUM('uploading', 'uploaded', 'processing', 'parsing', 'parsed', 'embedding', 'embedded', 'knowledge_graph_processing', 'knowledge_graph_processed', 'completed', 'duplicate', 'upload_failed', 'parsing_failed', 'embedding_failed', 'processing_failed', 'knowledge_graph_failed', 'permanently_failed') DEFAULT 'uploading',
    task_id VARCHAR(255),
    processing_mode VARCHAR(50),
    error_message TEXT,
    processing_result JSON,
    retry_count INT DEFAULT 0,
    next_retry_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    processed_at TIMESTAMP NULL,
    doc_metadata JSON,
    mime_type VARCHAR(100),
    storage_path VARCHAR(1000),
    parsed_content JSON,
    INDEX idx_documents_status (status),
    INDEX idx_documents_task_id (task_id),
    INDEX idx_documents_file_hash (file_hash),
    INDEX idx_documents_content_hash (content_hash),
    INDEX idx_documents_created_at (created_at),
    INDEX idx_documents_retry_count (retry_count),
    INDEX idx_documents_next_retry_at (next_retry_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 文档分块表
CREATE TABLE IF NOT EXISTS document_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding_id VARCHAR(255),
    chunk_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_chunks_document (document_id),
    INDEX idx_chunks_embedding (embedding_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 任务表
CREATE TABLE IF NOT EXISTS tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(255) UNIQUE NOT NULL,
    task_name VARCHAR(255) NOT NULL,
    task_type ENUM('document_parse', 'vision_analysis', 'vector_embedding', 'knowledge_graph', 'document_index', 'batch_process') NOT NULL,
    document_id INT NULL,
    user_id INT NULL,
    parent_task_id INT NULL,
    status ENUM('pending', 'running', 'success', 'failed', 'cancelled') DEFAULT 'pending',
    priority ENUM('low', 'medium', 'high', 'urgent') DEFAULT 'medium',
    progress FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    task_params JSON,
    task_result JSON,
    error_message TEXT,
    worker_name VARCHAR(255),
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    execution_time FLOAT,
    memory_usage BIGINT,
    INDEX idx_tasks_status (status),
    INDEX idx_tasks_type (task_type),
    INDEX idx_tasks_document (document_id),
    INDEX idx_tasks_user (user_id),
    INDEX idx_tasks_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 章节表
CREATE TABLE IF NOT EXISTS chapters (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    parent_id INT NULL,
    chapter_type ENUM('part', 'chapter', 'section', 'subsection', 'appendix') NOT NULL,
    level INT NOT NULL DEFAULT 1,
    order_index INT NOT NULL,
    title VARCHAR(1000) NOT NULL,
    subtitle VARCHAR(1000),
    content TEXT,
    summary TEXT,
    key_points JSON,
    tags JSON,
    page_start INT,
    page_end INT,
    page_count INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    confidence FLOAT,
    INDEX idx_chapters_document (document_id),
    INDEX idx_chapters_parent (parent_id),
    INDEX idx_chapters_level (level),
    INDEX idx_chapters_type (chapter_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 图像内容表
CREATE TABLE IF NOT EXISTS image_contents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chapter_id INT NULL,
    image_type ENUM('photograph', 'diagram', 'illustration', 'screenshot', 'scan', 'icon', 'logo') NOT NULL,
    title VARCHAR(1000),
    caption TEXT,
    alt_text VARCHAR(2000),
    width INT,
    height INT,
    resolution FLOAT,
    color_mode VARCHAR(20),
    file_size BIGINT,
    format VARCHAR(10),
    page_number INT,
    position_x FLOAT,
    position_y FLOAT,
    bbox JSON,
    storage_key VARCHAR(500),
    description TEXT,
    tags JSON,
    objects JSON,
    text_content TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    INDEX idx_image_contents_document (document_id),
    INDEX idx_image_contents_chapter (chapter_id),
    INDEX idx_image_contents_page (page_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 图表内容表
CREATE TABLE IF NOT EXISTS chart_contents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chapter_id INT NULL,
    image_id INT NULL,
    chart_type ENUM('bar', 'line', 'pie', 'scatter', 'area', 'histogram', 'box_plot', 'heatmap', 'radar', 'candlestick', 'waterfall') NOT NULL,
    title VARCHAR(1000) NOT NULL,
    subtitle VARCHAR(1000),
    caption TEXT,
    data_series JSON,
    axes JSON,
    legend JSON,
    grid JSON,
    page_number INT,
    position JSON,
    size JSON,
    insights JSON,
    key_trends JSON,
    anomalies JSON,
    interpretation TEXT,
    extracted_data JSON,
    data_quality_score FLOAT,
    extraction_confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    INDEX idx_chart_contents_document (document_id),
    INDEX idx_chart_contents_chapter (chapter_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 表格内容表
CREATE TABLE IF NOT EXISTS table_contents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chapter_id INT NULL,
    image_id INT NULL,
    table_type ENUM('data_table', 'financial_report', 'comparison_table', 'schedule_table', 'matrix_table') NOT NULL,
    title VARCHAR(1000),
    caption TEXT,
    description TEXT,
    row_count INT NOT NULL,
    column_count INT NOT NULL,
    has_header BOOLEAN DEFAULT TRUE,
    has_footer BOOLEAN DEFAULT FALSE,
    headers JSON,
    `rows` JSON,
    cells JSON,
    structure JSON,
    page_number INT,
    position JSON,
    size JSON,
    summary TEXT,
    key_findings JSON,
    data_insights JSON,
    confidence FLOAT,
    completeness_score FLOAT,
    accuracy_score FLOAT,
    consistency_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    INDEX idx_table_contents_document (document_id),
    INDEX idx_table_contents_chapter (chapter_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 公式内容表
CREATE TABLE IF NOT EXISTS formula_contents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chapter_id INT NULL,
    image_id INT NULL,
    formula_type ENUM('mathematical', 'financial', 'statistical', 'chemical', 'physics') NOT NULL,
    name VARCHAR(500),
    notation VARCHAR(500),
    description TEXT,
    latex TEXT,
    mathml TEXT,
    ascii_math TEXT,
    plain_text TEXT,
    variables JSON,
    parameters JSON,
    constants JSON,
    page_number INT,
    position JSON,
    size JSON,
    explanation TEXT,
    interpretation TEXT,
    applications JSON,
    related_formulas JSON,
    is_computable BOOLEAN DEFAULT FALSE,
    computation_result JSON,
    computation_method TEXT,
    financial_context TEXT,
    financial_meaning TEXT,
    industry_usage TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    confidence FLOAT,
    INDEX idx_formula_contents_document (document_id),
    INDEX idx_formula_contents_chapter (chapter_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 知识图谱节点表
CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    node_id VARCHAR(255) UNIQUE NOT NULL,
    neo4j_id VARCHAR(255),
    node_type ENUM('entity', 'concept', 'relation', 'event', 'organization', 'person', 'location', 'date', 'amount') NOT NULL,
    node_name VARCHAR(1000) NOT NULL,
    node_label VARCHAR(500),
    node_alias JSON,
    properties JSON,
    attributes JSON,
    confidence FLOAT,
    importance FLOAT,
    source_text TEXT,
    page_number INT,
    position JSON,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_kg_nodes_document (document_id),
    INDEX idx_kg_nodes_type (node_type),
    INDEX idx_kg_nodes_name (node_name(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 知识图谱关系表
CREATE TABLE IF NOT EXISTS knowledge_graph_relations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    relation_id VARCHAR(255) UNIQUE NOT NULL,
    neo4j_id VARCHAR(255),
    source_node_id VARCHAR(255) NOT NULL,
    target_node_id VARCHAR(255) NOT NULL,
    source_node_neo4j_id VARCHAR(255),
    target_node_neo4j_id VARCHAR(255),
    relation_type ENUM('owns', 'works_for', 'located_in', 'part_of', 'related_to', 'invests_in', 'acquires', 'merges_with', 'collaborates_with', 'reports_to', 'regulated_by') NOT NULL,
    relation_label VARCHAR(500),
    description TEXT,
    properties JSON,
    attributes JSON,
    weight FLOAT DEFAULT 1.0,
    confidence FLOAT,
    direction VARCHAR(20) DEFAULT 'directed',
    evidence TEXT,
    source_text TEXT,
    page_number INT,
    position JSON,
    context TEXT,
    is_verified BOOLEAN DEFAULT FALSE,
    verification_method VARCHAR(100),
    verification_confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_kg_relations_document (document_id),
    INDEX idx_kg_relations_source (source_node_id),
    INDEX idx_kg_relations_target (target_node_id),
    INDEX idx_kg_relations_type (relation_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 数据库表结构和初始数据将由后端应用通过SQLAlchemy自动创建和补充

-- ===================================================
-- 数据库迁移脚本整合部分
-- 根据migrations文件夹中的脚本添加额外字段
-- ===================================================

-- 添加文档处理自动化触发机制所需字段
-- 来自 add_document_processing_fields.py
-- 字段已在 CREATE TABLE documents 中定义,无需重复添加

-- 添加文档重试机制所需字段
-- 来自 add_retry_fields.py
-- 字段已在 CREATE TABLE documents 中定义,无需重复添加

-- 添加额外的文档管理字段
-- 来自 add_additional_fields.py
-- 字段已在 CREATE TABLE documents 中定义,无需重复添加

-- 设置时区和优化配置需要 SUPER 权限,跳过
-- SET GLOBAL time_zone = '+8:00';
-- SET GLOBAL innodb_buffer_pool_size = 2147483648;
-- SET GLOBAL innodb_log_file_size = 268435456;
-- SET GLOBAL innodb_flush_log_at_trx_commit = 2;
-- SET GLOBAL sync_binlog = 0;
-- SET GLOBAL innodb_flush_method = O_DIRECT;
-- SET GLOBAL sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO';
-- SET GLOBAL innodb_file_per_table = ON;
-- SET GLOBAL innodb_file_format = Barracuda;

-- 创建性能监控视图(可选)
-- CREATE OR REPLACE VIEW performance_summary AS
-- SELECT
--     SCHEMA_NAME as 'database',
--     ROUND(SUM(DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS 'size_mb',
--     COUNT(TABLE_NAME) AS 'tables'
-- FROM information_schema.TABLES
-- WHERE SCHEMA_NAME = 'financial_rag'
-- GROUP BY SCHEMA_NAME;

-- ===================================================
-- Qwen模型集成部分
-- ===================================================

-- 创建向量存储表（如果不存在）
CREATE TABLE IF NOT EXISTS vector_storage (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT NOT NULL,
    chunk_id INT,
    vector_id VARCHAR(255) UNIQUE NOT NULL,
    embedding_data JSON,
    model_provider VARCHAR(50) DEFAULT 'qwen',
    model_name VARCHAR(100) DEFAULT 'text-embedding-v4',
    embedding_dimension INT DEFAULT 1536,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_vector_storage_document (document_id),
    INDEX idx_vector_storage_chunk (chunk_id),
    INDEX idx_vector_storage_provider (model_provider),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建模型配置表
CREATE TABLE IF NOT EXISTS model_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    api_base VARCHAR(255),
    api_key VARCHAR(255),
    configuration JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_model_type (model_type),
    INDEX idx_provider (provider),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 插入Qwen模型配置（使用环境变量，如果未设置则使用默认值）
INSERT IGNORE INTO model_configurations (model_type, model_name, provider, api_base, configuration) VALUES
('embedding', 'text-embedding-v4', 'qwen', 'https://dashscope.aliyuncs.com/compatible-mode/v1', '{"dimension": 1536}'),
('reranking', 'bge-reranker-v2-m3', 'qwen', 'https://dashscope.aliyuncs.com/compatible-mode/v1', '{"top_k": 10}'),
('multimodal', 'qwen-vl-plus', 'qwen', 'https://dashscope.aliyuncs.com/compatible-mode/v1', '{"ocr_model": "qwen-vl-ocr"}');

-- 更新现有向量数据的模型信息（如果有数据）
UPDATE vector_storage
SET model_provider = 'qwen',
    model_name = 'text-embedding-v4',
    embedding_dimension = 1536
WHERE model_provider IS NULL OR model_provider = 'bge';

SET FOREIGN_KEY_CHECKS = 1;