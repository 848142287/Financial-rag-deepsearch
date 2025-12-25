"""
会话状态管理器
管理反馈驱动检索优化的会话状态和轮次控制
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """会话状态枚举"""
    INIT = "init"                    # 初始状态
    SEARCHING = "searching"          # 检索执行中
    FEEDBACK = "feedback"            # 等待用户反馈
    PROCESSING = "processing"        # 处理反馈中
    OPTIMIZING = "optimizing"        # 优化检索中
    SHOWING = "showing"              # 显示优化结果
    COMPLETED = "completed"          # 会话完成
    MAX_ROUNDS = "max_rounds"        # 达到最大轮次


class FeedbackType(Enum):
    """反馈类型枚举"""
    RELEVANCE_LOW = "relevance_low"          # 相关性不足
    INCOMPLETE = "incomplete"                # 信息不全
    ACCURACY_ISSUE = "accuracy_issue"        # 准确性问题
    SORTING_ISSUE = "sorting_issue"          # 排序问题
    GENERAL_DISSATISFACTION = "general"      # 一般不满意
    SPECIFIC_REQUIREMENT = "specific"        # 特定需求
    UNCLEAR = "unclear"                      # 不清楚


@dataclass
class FeedbackData:
    """反馈数据结构"""
    feedback_type: FeedbackType
    rating: Optional[int] = None              # 评分 1-5
    comments: Optional[str] = None            # 文字反馈
    rewritten_query: Optional[str] = None     # 用户改写的查询
    highlighted_items: List[int] = field(default_factory=list)  # 高亮的结果项
    specific_requirements: List[str] = field(default_factory=list)  # 特定需求
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationRecord:
    """优化记录"""
    round_number: int
    feedback_data: FeedbackData
    optimization_strategy: str
    adjusted_params: Dict[str, Any]
    original_query: str
    optimized_query: str
    results_count: int
    optimization_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchSession:
    """搜索会话"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[int] = None
    initial_query: str = ""
    current_query: str = ""
    state: SessionState = SessionState.INIT
    current_round: int = 0
    max_rounds: int = 5
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # 检索历史
    search_history: List[Dict[str, Any]] = field(default_factory=list)

    # 反馈历史
    feedback_history: List[FeedbackData] = field(default_factory=list)

    # 优化历史
    optimization_history: List[OptimizationRecord] = field(default_factory=list)

    # 当前结果
    current_results: List[Dict[str, Any]] = field(default_factory=list)

    # 会话统计
    total_searches: int = 0
    total_optimizations: int = 0
    satisfaction_score: Optional[float] = None
    session_outcome: Optional[str] = None  # success, max_rounds, abandoned


class SessionStateManager:
    """会话状态管理器"""

    def __init__(self):
        self.active_sessions: Dict[str, SearchSession] = {}
        self.session_timeout = 3600  # 1小时超时

    def create_session(self, user_id: Optional[int], initial_query: str) -> SearchSession:
        """创建新的搜索会话"""
        session = SearchSession(
            user_id=user_id,
            initial_query=initial_query,
            current_query=initial_query,
            state=SessionState.INIT
        )

        self.active_sessions[session.session_id] = session
        logger.info(f"创建新会话 {session.session_id} for user {user_id}")

        return session

    def get_session(self, session_id: str) -> Optional[SearchSession]:
        """获取会话"""
        return self.active_sessions.get(session_id)

    def update_state(self, session_id: str, new_state: SessionState, **kwargs) -> bool:
        """更新会话状态"""
        session = self.get_session(session_id)
        if not session:
            return False

        old_state = session.state
        session.state = new_state

        # 根据状态更新执行特定操作
        if new_state == SessionState.SEARCHING:
            session.current_round += 1
            session.total_searches += 1

        elif new_state == SessionState.OPTIMIZING:
            session.total_optimizations += 1

        elif new_state == SessionState.COMPLETED:
            session.end_time = datetime.now()
            session.session_outcome = kwargs.get('outcome', 'success')
            if 'satisfaction_score' in kwargs:
                session.satisfaction_score = kwargs['satisfaction_score']

        elif new_state == SessionState.MAX_ROUNDS:
            session.end_time = datetime.now()
            session.session_outcome = 'max_rounds'

        logger.info(f"会话 {session_id} 状态从 {old_state.value} 更新为 {new_state.value}")
        return True

    def add_feedback(self, session_id: str, feedback_data: FeedbackData) -> bool:
        """添加用户反馈"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.feedback_history.append(feedback_data)
        return True

    def add_optimization_record(self, session_id: str, record: OptimizationRecord) -> bool:
        """添加优化记录"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.optimization_history.append(record)
        return True

    def update_current_results(self, session_id: str, results: List[Dict[str, Any]]) -> bool:
        """更新当前结果"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.current_results = results
        return True

    def can_continue(self, session_id: str) -> bool:
        """检查是否可以继续优化"""
        session = self.get_session(session_id)
        if not session:
            return False

        return session.current_round < session.max_rounds

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话统计信息"""
        session = self.get_session(session_id)
        if not session:
            return None

        duration = None
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()

        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'duration': duration,
            'total_rounds': session.current_round,
            'total_searches': session.total_searches,
            'total_optimizations': session.total_optimizations,
            'feedback_count': len(session.feedback_history),
            'satisfaction_score': session.satisfaction_score,
            'session_outcome': session.session_outcome,
            'optimization_success_rate': self._calculate_success_rate(session)
        }

    def _calculate_success_rate(self, session: SearchSession) -> float:
        """计算优化成功率"""
        if not session.optimization_history:
            return 0.0

        # 简单的成功率计算：如果会话成功完成，认为优化成功
        if session.session_outcome == 'success':
            return 1.0
        elif session.session_outcome == 'max_rounds':
            return 0.3  # 达到最大轮次算部分成功
        else:
            return 0.0

    def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            if (current_time - session.start_time).total_seconds() > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            session = self.active_sessions.pop(session_id, None)
            if session:
                session.end_time = current_time
                if not session.session_outcome:
                    session.session_outcome = 'abandoned'
                logger.info(f"清理过期会话 {session_id}")

        return len(expired_sessions)

    def get_active_sessions_count(self) -> int:
        """获取活跃会话数"""
        return len(self.active_sessions)

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话摘要"""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            'session_id': session.session_id,
            'initial_query': session.initial_query,
            'current_round': session.current_round,
            'max_rounds': session.max_rounds,
            'current_state': session.state.value,
            'can_continue': self.can_continue(session_id),
            'current_results_count': len(session.current_results),
            'feedback_count': len(session.feedback_history),
            'optimization_count': len(session.optimization_history)
        }


# 全局会话管理器实例
session_manager = SessionStateManager()