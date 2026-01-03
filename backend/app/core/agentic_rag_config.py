"""
Agentic RAG 系统配置
定义不同检索级别的参数配置
"""

from typing import Dict, Any


class RetrievalLevelConfig:
    """检索级别配置"""

    # 快速检索：P95 ≤ 3秒
    FAST = {
        "max_plan_time": 1.0,
        "max_execution_time": 5.0,
        "max_generation_time": 2.0,
        "max_results": 5,
        "quality_threshold": 0.6,
        "description": "快速检索，适合实时交互"
    }

    # 增强检索：P95 ≤ 8秒
    ENHANCED = {
        "max_plan_time": 2.0,
        "max_execution_time": 10.0,
        "max_generation_time": 4.0,
        "max_results": 15,
        "quality_threshold": 0.75,
        "description": "增强检索，平衡速度和质量"
    }

    # 深度检索：支持异步，无严格时间限制
    DEEP_SEARCH = {
        "max_plan_time": 3.0,
        "max_execution_time": 20.0,
        "max_generation_time": 8.0,
        "max_results": 30,
        "quality_threshold": 0.8,
        "description": "深度检索，最高质量"
    }


# 任务清理配置
CLEANUP_CONFIG = {
    "max_age_hours": 24,          # 保留最近24小时的任务
    "max_completed": 1000,        # 最多保留1000个已完成任务
    "cleanup_interval_hours": 6,  # 每6小时清理一次
}


# 缓存配置
CACHE_CONFIG = {
    "query_ttl": 3600,            # 查询结果缓存1小时
    "plan_ttl": 7200,             # 检索计划缓存2小时
    "max_cache_size": 10000,      # 最大缓存条目数
}


# 重试配置
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,
    "max_delay": 10.0,
    "exponential_base": 2.0,
}


def get_level_config(level: str) -> Dict[str, Any]:
    """
    获取指定级别的配置

    Args:
        level: 检索级别 (fast, enhanced, deep_search)

    Returns:
        配置字典
    """
    level_upper = level.upper()
    if hasattr(RetrievalLevelConfig, level_upper):
        return getattr(RetrievalLevelConfig, level_upper)
    else:
        # 默认返回FAST配置
        return RetrievalLevelConfig.FAST


# 级别映射
LEVEL_MAP = {
    "fast": RetrievalLevelConfig.FAST,
    "enhanced": RetrievalLevelConfig.ENHANCED,
    "deep_search": RetrievalLevelConfig.DEEP_SEARCH,
}
