"""
任务数据模型
"""

from sqlalchemy import Column, Integer, String, Text, BigInteger, DateTime, ForeignKey, Enum, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base
from app.core.enum_utils import CaseInsensitiveEnum


# TODO: TaskStatus → core.TaskStatus
class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskType(str, enum.Enum):
    DOCUMENT_PARSE = "document_parse"
    VISION_ANALYSIS = "vision_analysis"
    VECTOR_EMBEDDING = "vector_embedding"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DOCUMENT_INDEX = "document_index"
    BATCH_PROCESS = "batch_process"


class Task(Base):
    """任务表"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)  # Celery任务ID
    task_name = Column(String(255), nullable=False)
    task_type = Column(Enum(TaskType), nullable=False, index=True)

    # 任务关联
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    parent_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True, index=True)

    # 任务状态
    status = Column(CaseInsensitiveEnum(TaskStatus, 50), default=TaskStatus.PENDING, index=True)
    priority = Column(CaseInsensitiveEnum(TaskPriority, 50), default=TaskPriority.MEDIUM, index=True)
    progress = Column(Float, default=0.0)  # 0.0-100.0

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 任务参数和结果
    task_params = Column(JSON)  # 任务输入参数
    task_result = Column(JSON)   # 任务执行结果
    error_message = Column(Text)  # 错误信息

    # 执行信息
    worker_name = Column(String(255))  # 执行的worker名称
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # 资源使用
    execution_time = Column(Float)  # 执行时间（秒）
    memory_usage = Column(BigInteger)  # 内存使用（字节）

    def __repr__(self):
        return f"<Task(id={self.id}, task_id='{self.task_id}', status='{self.status}', type='{self.task_type}')>"


class TaskDependency(Base):
    """任务依赖关系表"""
    __tablename__ = "task_dependencies"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    depends_on_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    dependency_type = Column(String(50), default="sequential")  # sequential, parallel, conditional
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    task = relationship("Task", foreign_keys=[task_id], backref="dependencies")
    depends_on_task = relationship("Task", foreign_keys=[depends_on_task_id])

    def __repr__(self):
        return f"<TaskDependency(task_id={self.task_id}, depends_on={self.depends_on_task_id})>"


class TaskExecutionLog(Base):
    """任务执行日志表"""
    __tablename__ = "task_execution_logs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    details = Column(JSON)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # 关系
    task = relationship("Task", backref="execution_logs")

    def __repr__(self):
        return f"<TaskExecutionLog(task_id={self.task_id}, level='{self.level}', timestamp='{self.timestamp}')>"