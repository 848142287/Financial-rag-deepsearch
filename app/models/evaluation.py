"""
评估相关数据模型
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.sql import func
from app.core.database import Base


class RetrievalLog(Base):
    """检索记录表"""
    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    retrieval_type = Column(String(50), nullable=False, index=True)
    retrieved_documents = Column(Text)  # JSON格式
    response_time_ms = Column(Integer)
    relevance_score = Column(Integer)  # 使用整数存储分数
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self):
        return f"<RetrievalLog(id={self.id}, type='{self.retrieval_type}')>"


class EvaluationResult(Base):
    """评估结果表"""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    contexts = Column(Text)  # JSON格式
    ground_truth = Column(Text)
    metrics = Column(Text)  # JSON格式
    overall_score = Column(Float)
    evaluation_type = Column(String(50), default="ragas")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<EvaluationResult(id={self.id}, evaluation_id='{self.evaluation_id}', score={self.overall_score})>"


class EvaluationMetric(Base):
    """评估指标表"""
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)
    reasoning = Column(Text)
    confidence = Column(Float)
    additional_info = Column(Text)  # JSON格式
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<EvaluationMetric(id={self.id}, name='{self.metric_name}', score={self.score})>"


class EvaluationSession(Base):
    """评估会话表"""
    __tablename__ = "evaluation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    description = Column(Text)
    status = Column(String(20), default="active")  # active, completed, cancelled
    total_questions = Column(Integer, default=0)
    completed_questions = Column(Integer, default=0)
    average_score = Column(Float)
    created_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<EvaluationSession(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"


# 兼容性：将RetrievalLog从admin.py移到这里以避免循环导入
# 同时在app.models.__init__.py中导入