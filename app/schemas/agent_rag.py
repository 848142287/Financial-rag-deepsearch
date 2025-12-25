"""
AgentRAG相关的Pydantic模式
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class AgentRAGQueryRequest(BaseModel):
    """AgentRAG查询请求"""
    query: str = Field(..., description="用户查询")
    context: Optional[Dict[str, Any]] = Field(None, description="查询上下文")
    options: Optional[Dict[str, Any]] = Field(None, description="查询选项")


class AgentRAGQueryResponse(BaseModel):
    """AgentRAG查询响应"""
    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="生成的答案")
    confidence_score: float = Field(..., description="置信度分数")
    sources: List[Dict[str, Any]] = Field(default=[], description="答案来源")
    ragas_metrics: Dict[str, float] = Field(default={}, description="RAGAS评估指标")
    execution_trace: List[Dict[str, Any]] = Field(default=[], description="执行轨迹")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class BatchProcessRequest(BaseModel):
    """批量处理请求"""
    file_paths: List[str] = Field(..., description="文件路径列表")
    config: Optional[Dict[str, Any]] = Field(None, description="处理配置")


class BatchProcessResponse(BaseModel):
    """批量处理响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    total_files: int = Field(..., description="总文件数")
    message: str = Field(..., description="状态消息")


class DeepSearchRequest(BaseModel):
    """DeepSearch请求"""
    query: str = Field(..., description="搜索查询")
    max_iterations: Optional[int] = Field(3, description="最大迭代次数")
    strategies: Optional[List[str]] = Field(['vector', 'graph', 'keyword'], description="检索策略")


class TableMergeRequest(BaseModel):
    """表格合并请求"""
    tables: List[Dict[str, Any]] = Field(..., description="待合并的表格列表")


class TableMergeResponse(BaseModel):
    """表格合并响应"""
    merged_tables: List[Dict[str, Any]] = Field(..., description="合并后的表格")
    validation_results: List[Dict[str, Any]] = Field(..., description="验证结果")


class QueryHistoryItem(BaseModel):
    """查询历史项"""
    id: str = Field(..., description="历史记录ID")
    query: str = Field(..., description="查询内容")
    answer: str = Field(..., description="答案内容")
    confidence_score: float = Field(..., description="置信度")
    timestamp: datetime = Field(..., description="查询时间")
    ragas_metrics: Optional[Dict[str, float]] = Field(None, description="评估指标")


class EvaluationRequest(BaseModel):
    """评估请求"""
    query: str = Field(..., description="查询")
    answer: str = Field(..., description="答案")
    retrieved_docs: List[Dict[str, Any]] = Field(..., description="检索文档")
    ground_truth: Optional[str] = Field(None, description="标准答案")


class EvaluationResponse(BaseModel):
    """评估响应"""
    metrics: Dict[str, float] = Field(..., description="评估指标")