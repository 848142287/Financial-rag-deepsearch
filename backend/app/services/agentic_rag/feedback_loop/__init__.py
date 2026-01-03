"""
反馈回路模块

实现多层反馈机制，持续优化检索质量

架构:
- L1: RealTimeFeedbackProcessor - 实时反馈 (秒级)
- L2: ShortTermFeedbackLoop - 中期反馈 (小时级) - 待实现
- L3: LongTermFeedbackLoop - 长期反馈 (周级) - 待实现
"""

from .realtime_feedback import (
    RealTimeFeedbackProcessor,
    get_realtime_feedback_processor,
    QueryPattern,
    UserPreference,
    SessionHistory
)

__all__ = [
    "RealTimeFeedbackProcessor",
    "get_realtime_feedback_processor",
    "QueryPattern",
    "UserPreference",
    "SessionHistory"
]
