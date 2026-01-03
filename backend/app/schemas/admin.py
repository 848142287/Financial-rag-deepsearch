"""
系统管理相关的Pydantic模式
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class SystemStatus(BaseModel):
    """系统状态模式"""
    status: str  # 'healthy', 'degraded', 'down'
    services: Dict[str, str]  # service_name -> status
    uptime: Optional[str] = None
    version: str = "1.0.0"


class SystemConfig(BaseModel):
    """系统配置模式"""
    config_key: str
    config_value: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SystemLog(BaseModel):
    """系统日志模式"""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None


class SystemStatistics(BaseModel):
    """系统统计模式"""
    total_documents: int
    total_users: int
    total_conversations: int
    total_messages: int
    total_queries: int
    average_response_time: float
    storage_usage: Dict[str, Any]  # storage_type -> usage_info
    recent_activity: List[Dict[str, Any]]


# SLA相关模式
class SLARule(BaseModel):
    """SLA规则模式"""
    id: Optional[str] = None
    name: str
    description: str
    metric_type: str  # 'response_time', 'availability', 'throughput'
    threshold: float
    operator: str  # 'lt', 'gt', 'eq', 'lte', 'gte'
    time_window: int  # 时间窗口（分钟）
    severity: str  # 'low', 'medium', 'high', 'critical'
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SLAMetric(BaseModel):
    """SLA指标模式"""
    id: Optional[str] = None
    rule_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    status: str  # 'pass', 'fail', 'warning'
    measured_at: datetime
    additional_data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class SLAViolation(BaseModel):
    """SLA违规模式"""
    id: Optional[str] = None
    rule_id: str
    rule_name: str
    violation_type: str
    actual_value: float
    expected_value: float
    deviation: float
    severity: str
    started_at: datetime
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False
    description: Optional[str] = None

    class Config:
        from_attributes = True


class SLAReport(BaseModel):
    """SLA报告模式"""
    period_start: datetime
    period_end: datetime
    total_rules: int
    passed_rules: int
    failed_rules: int
    violations_count: int
    availability_percentage: float
    average_response_time: float
    rules: List[Dict[str, Any]]


class HealthCheck(BaseModel):
    """健康检查模式"""
    service_name: str
    status: str
    response_time_ms: Optional[int] = None
    last_check: datetime
    details: Optional[Dict[str, Any]] = None


class PerformanceMetrics(BaseModel):
    """性能指标模式"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    request_rate: float
    error_rate: float
    timestamp: datetime


# 任务进度相关模式
class TaskProgress(BaseModel):
    """任务进度模式"""
    task_id: str
    task_type: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    completed_steps: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None

    class Config:
        from_attributes = True


# TODO: TaskStatus → core.TaskStatus
class TaskStatus(BaseModel):
    """任务状态模式"""
    task_id: str
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime


class ProgressUpdate(BaseModel):
    """进度更新模式"""
    task_id: str
    progress: float
    message: Optional[str] = None
    current_step: Optional[str] = None
    timestamp: datetime