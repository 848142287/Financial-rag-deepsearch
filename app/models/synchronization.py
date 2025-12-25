"""
双库同步数据模型
"""

from sqlalchemy import Column, Integer, String, Text, BigInteger, DateTime, ForeignKey, Enum, JSON, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class SyncStatus(str, enum.Enum):
    """同步状态枚举"""
    INIT = "init"                    # 初始状态
    READY = "ready"                  # 准备同步
    VECTOR_ING = "vector_ing"        # 向量库同步中
    GRAPH_ING = "graph_ing"          # 图谱库同步中
    LINK_ING = "link_ing"            # 关联建立中
    COMPLETED = "completed"          # 同步完成
    FAILED = "failed"                # 同步失败
    CANCELLED = "cancelled"          # 已取消


class SyncPriority(str, enum.Enum):
    """同步优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DocumentSync(Base):
    """文档同步状态表"""
    __tablename__ = "document_syncs"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    document_version = Column(String(50), nullable=False)  # 文档版本号
    content_hash = Column(String(64), nullable=False, index=True)  # 内容哈希值

    # 同步状态
    sync_status = Column(Enum(SyncStatus), default=SyncStatus.INIT, index=True)
    priority = Column(Enum(SyncPriority), default=SyncPriority.NORMAL)

    # 进度信息
    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    vector_progress = Column(Float, default=0.0)  # 向量同步进度 (0-100)
    graph_progress = Column(Float, default=0.0)   # 图谱同步进度 (0-100)
    link_progress = Column(Float, default=0.0)    # 关联建立进度 (0-100)

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 错误信息
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # 配置信息
    sync_config = Column(JSON)  # 同步配置

    # 关系
    document = relationship("Document", backref="sync_records")
    vector_records = relationship("VectorSync", back_populates="document_sync", cascade="all, delete-orphan")
    graph_records = relationship("GraphSync", back_populates="document_sync", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DocumentSync(id={self.id}, document_id={self.document_id}, status='{self.sync_status}')>"


class VectorSync(Base):
    """向量库同步记录表"""
    __tablename__ = "vector_syncs"

    id = Column(Integer, primary_key=True, index=True)
    document_sync_id = Column(Integer, ForeignKey("document_syncs.id"), nullable=False, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=False)
    vector_id = Column(String(255), unique=True, index=True)  # 向量数据库中的ID
    embedding_model = Column(String(100))  # 使用的嵌入模型
    vector_dimension = Column(Integer)     # 向量维度

    # 同步状态
    sync_status = Column(Enum(SyncStatus), default=SyncStatus.INIT, index=True)
    batch_id = Column(String(100))         # 批次ID

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    synced_at = Column(DateTime(timezone=True))

    # 元数据
    vector_metadata = Column(JSON)  # 向量相关的元数据

    # 关系
    document_sync = relationship("DocumentSync", back_populates="vector_records")
    chunk = relationship("DocumentChunk")

    def __repr__(self):
        return f"<VectorSync(id={self.id}, vector_id='{self.vector_id}', status='{self.sync_status}')>"


class GraphSync(Base):
    """图谱库同步记录表"""
    __tablename__ = "graph_syncs"

    id = Column(Integer, primary_key=True, index=True)
    document_sync_id = Column(Integer, ForeignKey("document_syncs.id"), nullable=False, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=False)
    entity_id = Column(String(255), index=True)   # 实体ID
    entity_type = Column(String(100))             # 实体类型
    entity_name = Column(String(500))             # 实体名称

    # 关系信息
    relation_id = Column(String(255), index=True) # 关系ID
    relation_type = Column(String(100))           # 关系类型
    source_entity = Column(String(255))           # 源实体
    target_entity = Column(String(255))           # 目标实体

    # 同步状态
    sync_status = Column(Enum(SyncStatus), default=SyncStatus.INIT, index=True)
    batch_id = Column(String(100))                # 批次ID

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    synced_at = Column(DateTime(timezone=True))

    # 元数据
    graph_metadata = Column(JSON)  # 图谱相关的元数据

    # 关系
    document_sync = relationship("DocumentSync", back_populates="graph_records")
    chunk = relationship("DocumentChunk")

    def __repr__(self):
        return f"<GraphSync(id={self.id}, entity='{self.entity_name}', type='{self.entity_type}')>"


class EntityLink(Base):
    """实体关联表"""
    __tablename__ = "entity_links"

    id = Column(Integer, primary_key=True, index=True)
    document_sync_id = Column(Integer, ForeignKey("document_syncs.id"), nullable=False, index=True)
    vector_entity_id = Column(String(255))        # 向量库中的实体ID
    graph_entity_id = Column(String(255))         # 图谱库中的实体ID

    # 关联类型和强度
    link_type = Column(String(100))               # 关联类型
    confidence_score = Column(Float)              # 置信度分数 (0-1)
    link_strength = Column(Float, default=1.0)    # 关联强度

    # 同步状态
    sync_status = Column(Enum(SyncStatus), default=SyncStatus.INIT, index=True)

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    linked_at = Column(DateTime(timezone=True))

    # 元数据
    link_metadata = Column(JSON)

    def __repr__(self):
        return f"<EntityLink(id={self.id}, vector_id='{self.vector_entity_id}', graph_id='{self.graph_entity_id}')>"


class SyncConfiguration(Base):
    """同步配置表"""
    __tablename__ = "sync_configurations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)  # 配置名称
    description = Column(Text)                                # 配置描述

    # 功能开关
    enable_vision_model = Column(Boolean, default=False)      # 是否启用视觉模型
    enable_entity_filter = Column(Boolean, default=True)      # 是否启用实体过滤
    enable_incremental_sync = Column(Boolean, default=True)   # 是否启用增量同步

    # 处理参数
    chunk_size = Column(Integer, default=1000)               # 文本切块大小
    chunk_overlap = Column(Integer, default=200)             # 切块重叠大小
    vector_dimension = Column(Integer, default=1536)         # 向量维度
    embedding_model = Column(String(100), default="text-embedding-ada-002")

    # 实体抽取配置
    entity_types = Column(JSON)  # 实体类型配置
    extraction_rules = Column(JSON)  # 抽取规则

    # 同步策略
    sync_batch_size = Column(Integer, default=100)           # 同步批次大小
    retry_attempts = Column(Integer, default=3)              # 重试次数
    timeout_seconds = Column(Integer, default=300)           # 超时时间

    # 适用范围
    applicable_document_types = Column(JSON)                 # 适用的文档类型

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)                # 是否启用

    def __repr__(self):
        return f"<SyncConfiguration(id={self.id}, name='{self.name}', active={self.is_active})>"


class SyncLog(Base):
    """同步日志表"""
    __tablename__ = "sync_logs"

    id = Column(Integer, primary_key=True, index=True)
    document_sync_id = Column(Integer, ForeignKey("document_syncs.id"), nullable=False, index=True)

    # 日志信息
    log_level = Column(String(20), index=True)               # 日志级别 (INFO, WARNING, ERROR)
    component = Column(String(100), index=True)              # 组件名称 (vector, graph, link)
    message = Column(Text, nullable=False)                   # 日志消息

    # 详细信息
    details = Column(JSON)                                   # 详细信息
    stack_trace = Column(Text)                               # 错误堆栈

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # 关系
    document_sync = relationship("DocumentSync", backref="logs")

    def __repr__(self):
        return f"<SyncLog(id={self.id}, level='{self.log_level}', component='{self.component}')>"