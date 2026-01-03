"""
自适应检索模块

自动选择最优检索方法和参数的智能检索系统
"""

from .query_classifier import (
    QueryClassifier,
    QueryFeatures,
    QueryType,
    QueryComplexity,
    get_query_classifier
)

from .adaptive_optimizer import (
    AdaptiveParameterOptimizer,
    get_adaptive_parameter_optimizer
)

# 创建别名方便使用
def get_adaptive_optimizer():
    """获取自适应参数优化器实例的别名"""
    return get_adaptive_parameter_optimizer()

from .bandit_selector import (
    RetrievalArm,
    BanditRetrievalSelector,
    get_bandit_selector
)

from .adaptive_feedback import (
    AdaptiveFeedbackProcessor,
    get_adaptive_feedback_processor
)

__all__ = [
    # Query Classifier
    "QueryClassifier",
    "QueryFeatures",
    "QueryType",
    "QueryComplexity",
    "get_query_classifier",

    # Adaptive Optimizer
    "AdaptiveParameterOptimizer",
    "get_adaptive_parameter_optimizer",
    "get_adaptive_optimizer",

    # Bandit Selector
    "RetrievalArm",
    "BanditRetrievalSelector",
    "get_bandit_selector",

    # Adaptive Feedback
    "AdaptiveFeedbackProcessor",
    "get_adaptive_feedback_processor",
]
