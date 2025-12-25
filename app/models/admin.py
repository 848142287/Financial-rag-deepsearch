"""
系统管理相关数据模型
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.sql import func
from app.core.database import Base


class SystemConfig(Base):
    """系统配置表"""
    __tablename__ = "system_configs"

    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_value = Column(Text)
    description = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<SystemConfig(id={self.id}, key='{self.config_key}')>"


# RetrievalLog 已移动到 evaluation.py 以避免循环导入


class TaskQueue(Base):
    """任务队列表"""
    __tablename__ = "task_queue"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    task_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), default='pending', index=True)
    priority = Column(Integer, default=0, index=True)
    payload = Column(Text)  # JSON格式
    result = Column(Text)   # JSON格式
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<TaskQueue(id={self.id}, task_id='{self.task_id}', status='{self.status}')>"


# SLA相关模型
class SLARule(Base):
    """SLA规则表"""
    __tablename__ = "sla_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    metric_type = Column(String(50), nullable=False, index=True)  # 'response_time', 'availability', 'throughput'
    threshold = Column(Float, nullable=False)
    operator = Column(String(10), nullable=False)  # 'lt', 'gt', 'eq', 'lte', 'gte'
    time_window = Column(Integer, nullable=False)  # 时间窗口（分钟）
    severity = Column(String(20), default='medium')  # 'low', 'medium', 'high', 'critical'
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<SLARule(id={self.id}, name='{self.name}', type='{self.metric_type}')>"


class SLAMetric(Base):
    """SLA指标表"""
    __tablename__ = "sla_metrics"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(Integer, nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    current_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    status = Column(String(20), default='pass')  # 'pass', 'fail', 'warning'
    measured_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    additional_data = Column(Text)  # JSON格式

    def __repr__(self):
        return f"<SLAMetric(id={self.id}, rule_id={self.rule_id}, status='{self.status}')>"


class SLAViolation(Base):
    """SLA违规表"""
    __tablename__ = "sla_violations"

    id = Column(Integer, primary_key=True, index=True)
    rule_id = Column(Integer, nullable=False, index=True)
    rule_name = Column(String(255), nullable=False)
    violation_type = Column(String(50), nullable=False)
    actual_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    deviation = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    resolved_at = Column(DateTime(timezone=True))
    is_resolved = Column(Boolean, default=False, index=True)
    description = Column(Text)

    def __repr__(self):
        return f"<SLAViolation(id={self.id}, rule_id={self.rule_id}, resolved={self.is_resolved})>"