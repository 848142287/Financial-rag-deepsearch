"""
反馈驱动迭代检索优化模块
提供用户反馈驱动的智能检索优化功能
"""

from .session_state import (
    session_manager, SessionState, FeedbackType, FeedbackData,
    OptimizationRecord, SearchSession
)
from .feedback_analyzer import FeedbackAnalyzer, OptimizationNeed
from .optimization_strategies import (
    OptimizationStrategySelector, OptimizationParams,
    QueryRewriteStrategy, RangeExpansionStrategy,
    AuthorityBoostStrategy, SortOptimizationStrategy,
    ComprehensiveStrategy
)
from .feedback_optimizer import FeedbackDrivenOptimizer, feedback_optimizer

__all__ = [
    # 核心类
    "FeedbackDrivenOptimizer",
    "feedback_optimizer",

    # 会话管理
    "session_manager",
    "SessionState",
    "FeedbackType",
    "FeedbackData",
    "OptimizationRecord",
    "SearchSession",

    # 反馈分析
    "FeedbackAnalyzer",
    "OptimizationNeed",

    # 优化策略
    "OptimizationStrategySelector",
    "OptimizationParams",
    "QueryRewriteStrategy",
    "RangeExpansionStrategy",
    "AuthorityBoostStrategy",
    "SortOptimizationStrategy",
    "ComprehensiveStrategy"
]