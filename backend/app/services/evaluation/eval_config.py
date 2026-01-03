"""
评估系统配置管理
集中管理所有评估相关的配置
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EvaluationMode(str, Enum):
    """评估模式"""
    RETRIEVAL = "retrieval"  # 仅检索评估
    QUALITY = "quality"  # 仅质量评估
    FULL = "full"  # 完整评估(检索+质量)


@dataclass
class EvaluatorConfig:
    """评估器配置"""

    # 基础配置
    mode: EvaluationMode = EvaluationMode.FULL
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 缓存存活时间(秒)

    # 检索评估配置
    retrieval_top_k_list: list = field(default_factory=lambda: [1, 3, 5, 10])
    retrieval_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "precision": 0.85,
        "recall": 0.85,
        "f1_score": 0.80,
        "top_k_accuracy": 0.85,
        "mrr": 0.80,
        "ndcg": 0.80,
    })

    # 质量评估配置
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
        "context_precision": 0.85,
        "context_recall": 0.85,
    })

    # Worker配置
    worker_batch_size: int = 500  # 批量大小
    worker_max_workers: int = 10  # 最大并发数
    worker_commit_threshold: int = 1000  # 提交阈值
    worker_timeout: int = 300  # 任务超时(秒)

    # 数据库配置
    db_pool_size: int = 20
    db_max_overflow: int = 40
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600

    # 报告配置
    report_template_dir: str = "app/templates/evaluation"
    report_output_dir: str = "reports/ragas_retrieval_eval"
    report_include_charts: bool = True
    report_include_details: bool = True

    # 问题生成配置
    question_templates: Dict[str, list] = field(default_factory=lambda: {
        "factual": [
            "什么是{topic}？",
            "{topic}的定义是什么？",
            "请解释{topic}。",
        ],
        "conceptual": [
            "{topic}的特点有哪些？",
            "如何理解{topic}？",
            "{topic}的核心内容是什么？",
        ],
        "analytical": [
            "如何分析{topic}？",
            "{topic}的发展趋势如何？",
            "评估{topic}的效果。",
        ],
    })

    # 检索服务配置
    retrieval_service_type: str = "unified"  # unified, rag, vector, graph
    retrieval_default_top_k: int = 5
    retrieval_search_type: str = "hybrid"  # vector, graph, hybrid, fulltext

    # 性能优化配置
    enable_async: bool = True
    enable_batch_processing: bool = True
    enable_result_cache: bool = True
    max_concurrent_evaluations: int = 5

    # 日志配置
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/evaluation.log"

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0  # 重试延迟(秒)
    retry_backoff: float = 2.0  # 退避系数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "mode": self.mode.value,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "retrieval_top_k_list": self.retrieval_top_k_list,
            "retrieval_thresholds": self.retrieval_thresholds,
            "quality_thresholds": self.quality_thresholds,
            "worker_batch_size": self.worker_batch_size,
            "worker_max_workers": self.worker_max_workers,
            "worker_commit_threshold": self.worker_commit_threshold,
            "worker_timeout": self.worker_timeout,
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "db_pool_timeout": self.db_pool_timeout,
            "db_pool_recycle": self.db_pool_recycle,
            "report_template_dir": self.report_template_dir,
            "report_output_dir": self.report_output_dir,
            "report_include_charts": self.report_include_charts,
            "report_include_details": self.report_include_details,
            "retrieval_service_type": self.retrieval_service_type,
            "retrieval_default_top_k": self.retrieval_default_top_k,
            "retrieval_search_type": self.retrieval_search_type,
            "enable_async": self.enable_async,
            "enable_batch_processing": self.enable_batch_processing,
            "enable_result_cache": self.enable_result_cache,
            "max_concurrent_evaluations": self.max_concurrent_evaluations,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_file_path": self.log_file_path,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "retry_backoff": self.retry_backoff,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluatorConfig":
        """从字典创建配置"""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })

    @classmethod
    def from_env(cls) -> "EvaluatorConfig":
        """从环境变量创建配置"""
        import os

        config = cls()

        # 从环境变量覆盖配置
        if os.getenv("EVAL_MODE"):
            config.mode = EvaluationMode(os.getenv("EVAL_MODE"))
        if os.getenv("EVAL_CACHE_ENABLED"):
            config.cache_enabled = os.getenv("EVAL_CACHE_ENABLED").lower() == "true"
        if os.getenv("EVAL_WORKER_MAX_WORKERS"):
            config.worker_max_workers = int(os.getenv("EVAL_WORKER_MAX_WORKERS"))
        if os.getenv("EVAL_WORKER_BATCH_SIZE"):
            config.worker_batch_size = int(os.getenv("EVAL_WORKER_BATCH_SIZE"))
        if os.getenv("EVAL_LOG_LEVEL"):
            config.log_level = os.getenv("EVAL_LOG_LEVEL")

        return config

    def validate(self) -> bool:
        """验证配置"""
        if self.worker_batch_size <= 0:
            raise ValueError("worker_batch_size must be positive")
        if self.worker_max_workers <= 0:
            raise ValueError("worker_max_workers must be positive")
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        return True


# 默认配置实例
DEFAULT_CONFIG = EvaluatorConfig()


def get_config(config: Optional[Dict[str, Any]] = None) -> EvaluatorConfig:
    """
    获取配置

    Args:
        config: 配置字典(可选)

    Returns:
        EvaluatorConfig实例
    """
    if config:
        return EvaluatorConfig.from_dict(config)
    return DEFAULT_CONFIG


def merge_configs(base: EvaluatorConfig, override: Dict[str, Any]) -> EvaluatorConfig:
    """
    合并配置

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    base_dict = base.to_dict()
    base_dict.update(override)
    return EvaluatorConfig.from_dict(base_dict)
