"""
RAGAS检索评估数据模型
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Enum
from sqlalchemy.sql import func
import enum

from app.core.database import Base

class RetrievalEvalStatus(str, enum.Enum):
    """检索评估状态枚举"""
    PENDING = "pending"                    # 待处理
    GENERATING_QUESTIONS = "generating_questions"  # 生成检索问题中
    SEARCHING = "searching"                # 问题检索中
    GENERATING_REPORT = "generating_report"  # 生成报告
    COMPLETED = "completed"                # 已完成
    FAILED = "failed"                      # 失败
    TIMEOUT = "timeout"                    # 超时

class RetrievalEvalQuestionStatus(str, enum.Enum):
    """单个问题检索状态枚举"""
    PENDING = "pending"                    # 待检索
    SEARCHING = "searching"                # 检索中
    SUCCESS = "success"                    # 成功
    FAILED = "failed"                      # 失败
    TIMEOUT = "timeout"                    # 超时

class RetrievalEvalTask(Base):
    """RAGAS检索评估任务表"""
    __tablename__ = "ragas_retrieval_eval_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)

    # 配置参数
    question_count = Column(Integer, default=300)  # 检索问题数量
    selected_documents = Column(JSON)  # 选中的文档列表 [file_path, ...]
    worker_count = Column(Integer, default=2)  # Worker数量
    batch_size = Column(Integer, default=30)  # 每批处理数量
    max_retries = Column(Integer, default=3)  # 最大重试次数

    # 状态信息
    status = Column(Enum(RetrievalEvalStatus), default=RetrievalEvalStatus.PENDING, index=True)
    current_node = Column(String(100))  # 当前执行节点：开始/生成检索问题/进行问题检索/生成检索报告/完成
    progress = Column(Float, default=0.0)  # 进度 0-100

    # 统计信息
    total_questions = Column(Integer, default=0)  # 总问题数
    generated_questions = Column(Integer, default=0)  # 已生成问题数
    searched_questions = Column(Integer, default=0)  # 已检索问题数
    success_count = Column(Integer, default=0)  # 成功数
    failed_count = Column(Integer, default=0)  # 失败数
    timeout_count = Column(Integer, default=0)  # 超时数

    # 性能指标
    total_duration = Column(Float, default=0.0)  # 总耗时（秒）
    average_accuracy = Column(Float)  # 平均准确率
    average_recall = Column(Float)  # 平均召回率
    average_top_k = Column(Float)  # 平均Top-K
    average_mrr = Column(Float)  # 平均MRR
    average_ndcg = Column(Float)  # 平均NDCG

    # 报告信息
    report_html = Column(Text)  # HTML格式报告
    report_pdf_path = Column(String(500))  # PDF报告路径

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # 错误信息
    error_message = Column(Text)

    def __repr__(self):
        return f"<RetrievalEvalTask(id={self.id}, task_id='{self.task_id}', status='{self.status}')>"

class RetrievalEvalQuestion(Base):
    """RAGAS检索评估问题表"""
    __tablename__ = "ragas_retrieval_eval_questions"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), nullable=False, index=True)
    question_id = Column(String(255), unique=True, nullable=False, index=True)

    # 问题信息
    question_text = Column(Text, nullable=False)  # 问题内容
    source_document = Column(String(500))  # 来源文档路径

    # 检索状态
    status = Column(Enum(RetrievalEvalQuestionStatus), default=RetrievalEvalQuestionStatus.PENDING, index=True)
    retry_count = Column(Integer, default=0)  # 已重试次数

    # 检索结果
    retrieved_fragments = Column(JSON)  # 检索到的文档片段列表
    retrieval_path = Column(String(100))  # 检索路径（使用的检索策略）
    duration_ms = Column(Float)  # 检索耗时（毫秒）

    # GLM-4.7生成的答案
    generated_answer = Column(Text)  # GLM-4.7生成的答案
    answer_generation_time = Column(Float)  # 答案生成耗时（毫秒）
    answer_quality_score = Column(Float)  # 答案质量评分（使用GLM-4.7评估）
    answer_relevance_score = Column(Float)  # 答案相关性评分（使用GLM-4.7评估）

    # 评估指标
    accuracy = Column(Float)  # 准确率
    recall = Column(Float)  # 召回率
    top_k = Column(Float)  # Top-K
    mrr = Column(Float)  # MRR
    ndcg = Column(Float)  # NDCG

    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    searched_at = Column(DateTime(timezone=True))

    # 错误信息
    error_message = Column(Text)

    def __repr__(self):
        return f"<RetrievalEvalQuestion(id={self.id}, question_id='{self.question_id}', status='{self.status}')>"
