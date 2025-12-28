"""
文档数据模型
"""

from sqlalchemy import Column, Integer, String, Text, BigInteger, DateTime, ForeignKey, Enum, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime

from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PARSING = "parsing"
    PARSED = "parsed"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    KNOWLEDGE_GRAPH_PROCESSING = "knowledge_graph_processing"
    KNOWLEDGE_GRAPH_PROCESSED = "knowledge_graph_processed"
    COMPLETED = "completed"
    DUPLICATE = "duplicate"
    FAILED = "failed"
    UPLOAD_FAILED = "upload_failed"
    PARSING_FAILED = "parsing_failed"
    EMBEDDING_FAILED = "embedding_failed"
    PROCESSING_FAILED = "processing_failed"
    KNOWLEDGE_GRAPH_FAILED = "knowledge_graph_failed"
    PERMANENTLY_FAILED = "permanently_failed"


class Document(Base):
    """文档表"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_size = Column(BigInteger)
    file_type = Column(String(50))
    content_type = Column(String(100))
    file_hash = Column(String(64), index=True)  # MD5 hash for file deduplication
    content_hash = Column(String(64), index=True)  # SHA-256 hash for content deduplication
    status = Column(Enum(DocumentStatus, values_callable=lambda obj: [e.value for e in obj]), default=DocumentStatus.UPLOADING, index=True)
    task_id = Column(String(255), index=True)  # Celery任务ID
    processing_mode = Column(String(50))  # 处理模式
    error_message = Column(Text)  # 错误信息
    processing_result = Column(JSON)  # 处理结果
    retry_count = Column(Integer, default=0, index=True)  # 重试次数
    next_retry_at = Column(DateTime(timezone=True), index=True)  # 下次重试时间
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))
    doc_metadata = Column(JSON)

    # Additional fields for enhanced document management
    mime_type = Column(String(100))  # MIME类型
    storage_path = Column(String(1000))  # 存储路径
    parsed_content = Column(JSON)  # 解析后的内容

    # 关系
    chunks = relationship("DocumentChunk", back_populates="document")

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}', status='{self.status}')>"


class DocumentChunk(Base):
    """文档分块表"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_id = Column(String(255), index=True)
    chunk_metadata = Column(JSON)  # 块级别的元数据（页码、位置等）
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class DocumentStorageIndex(Base):
    """文档存储索引表"""
    __tablename__ = "document_storage_index"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    storage_key = Column(String(500), nullable=False, index=True)
    storage_type = Column(String(50), nullable=False)  # memory, redis, mongodb, filesystem
    location = Column(String(1000))
    size_bytes = Column(BigInteger)
    storage_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))

    # 关系
    document = relationship("Document")

    def __repr__(self):
        return f"<DocumentStorageIndex(id={self.id}, document_id={self.document_id}, storage_type='{self.storage_type}')>"


class VectorStorage(Base):
    """向量存储记录表"""
    __tablename__ = "vector_storage"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), index=True)
    vector_id = Column(String(255), nullable=False, unique=True)  # Milvus中的向量ID
    embedding_data = Column(JSON, nullable=True)  # 向量数据
    model_provider = Column(String(50), index=True)  # 模型提供商
    model_name = Column(String(100))  # 模型名称
    embedding_dimension = Column(Integer, default=1024)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document")
    chunk = relationship("DocumentChunk")

    def __repr__(self):
        return f"<VectorStorage(id={self.id}, document_id={self.document_id}, vector_id='{self.vector_id}')>"



class DocumentTask(Base):
    """文档任务模型"""
    __tablename__ = "document_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    document = relationship("Document", backref="tasks")
