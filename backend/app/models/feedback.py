"""
反馈数据模型
将反馈数据从文件系统迁移到MySQL数据库
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, Index

from app.core.database import Base

class FeedbackRecord(Base):
    """用户反馈记录表"""
    __tablename__ = 'feedback_records'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')

    # 查询信息
    query_id = Column(String(64), nullable=False, index=True, comment='查询ID')
    query = Column(Text, nullable=False, comment='查询文本')

    # 检索信息
    retrieval_method = Column(String(50), nullable=False, comment='检索方法(lightrag/graphrag/deepsearch)')
    results = Column(JSON, comment='检索结果列表')

    # 用户反馈
    rating = Column(Integer, comment='用户评分(1-5)')
    clicked_docs = Column(JSON, comment='点击的文档ID列表')
    relevance = Column(String(50), comment='相关性评价(relevant/partially_relevant/not_relevant)')
    feedback_text = Column(Text, comment='用户反馈文本')

    # 性能指标
    latency = Column(Float, comment='检索延迟(秒)')

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True, comment='创建时间')

    # 索引
    __table_args__ = (
        Index('idx_query_method', 'query_id', 'retrieval_method'),
        Index('idx_created_relevance', 'created_at', 'relevance'),
    )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'query_id': self.query_id,
            'query': self.query,
            'retrieval_method': self.retrieval_method,
            'results': self.results,
            'rating': self.rating,
            'clicked_docs': self.clicked_docs,
            'relevance': self.relevance,
            'feedback_text': self.feedback_text,
            'latency': self.latency,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class FeedbackMetrics(Base):
    """反馈统计指标表"""
    __tablename__ = 'feedback_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')

    # 时间戳
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, comment='统计时间')

    # 检索方法性能
    retrieval_method = Column(String(50), nullable=False, comment='检索方法')
    avg_rating = Column(Float, comment='平均评分')
    avg_latency = Column(Float, comment='平均延迟')
    relevance_rate = Column(Float, comment='相关性率')
    click_rate = Column(Float, comment='点击率')
    total_queries = Column(Integer, comment='总查询数')

    # 优化建议
    suggestions = Column(JSON, comment='优化建议列表')

    # 索引
    __table_args__ = (
        Index('idx_method_timestamp', 'retrieval_method', 'timestamp'),
    )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'retrieval_method': self.retrieval_method,
            'avg_rating': self.avg_rating,
            'avg_latency': self.avg_latency,
            'relevance_rate': self.relevance_rate,
            'click_rate': self.click_rate,
            'total_queries': self.total_queries,
            'suggestions': self.suggestions
        }

class FeedbackSummary(Base):
    """反馈汇总表（按时间段）"""
    __tablename__ = 'feedback_summaries'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')

    # 时间范围
    period_start = Column(DateTime, nullable=False, comment='统计周期开始时间')
    period_end = Column(DateTime, nullable=False, comment='统计周期结束时间')
    period_type = Column(String(20), nullable=False, comment='周期类型(hourly/daily/weekly)')

    # 总体统计
    total_feedbacks = Column(Integer, comment='总反馈数')
    avg_rating = Column(Float, comment='总体平均评分')
    avg_latency = Column(Float, comment='总体平均延迟')

    # 各方法统计
    method_stats = Column(JSON, comment='各检索方法统计')

    created_at = Column(DateTime, default=datetime.utcnow, comment='记录创建时间')

    # 索引
    __table_args__ = (
        Index('idx_period', 'period_start', 'period_end', 'period_type'),
    )
