"""
实时反馈处理器 - L1快速反馈回路

在查询执行过程中实时利用用户反馈，优化检索结果
延迟: < 100ms
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class QueryPattern:
    """查询模式统计"""

    def __init__(self):
        self.count = 0
        self.avg_rating = 0.0
        self.ratings = []
        self.common_results = []  # 经常被点击的文档
        self.avg_click_position = 0.0
        self.preferred_compression = 0.5


class UserPreference:
    """用户偏好"""

    def __init__(self):
        self.preferred_terms = []  # 偏好的术语
        self.avg_rating_given = 0.0
        self.interaction_patterns = {
            "avg_click_position": 5.0,
            "result_utilization": 0.5,
            "preferred_compression": 0.5
        }
        self.query_history = []


class SessionHistory:
    """会话历史"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.events = []
        self.created_at = datetime.now()

    def add_event(self, event: Dict):
        """添加事件"""
        self.events.append({
            "timestamp": datetime.now(),
            "data": event
        })

    def get_recent_clicks(self, limit: int = 10) -> List[Dict]:
        """获取最近的点击事件"""
        clicks = [
            e["data"] for e in self.events
            if e["data"].get("event_type") == "click"
        ]
        return clicks[-limit:]

    def get_avg_click_position(self) -> float:
        """计算平均点击位置"""
        clicks = self.get_recent_clicks()
        if not clicks:
            return 5.0  # 默认值

        positions = [c.get("position", 5) for c in clicks]
        return sum(positions) / len(positions)


class RealTimeFeedbackProcessor:
    """
    实时反馈处理器

    功能:
    1. 查询重写 - 基于用户偏好
    2. 动态参数调整 - top_k, 压缩率
    3. 结果重排 - 基于历史反馈
    4. 反馈收集 - 记录用户交互
    """

    def __init__(self):
        # 会话历史 (session_id -> SessionHistory)
        self.sessions: Dict[str, SessionHistory] = {}

        # 用户偏好 (user_id -> UserPreference)
        self.user_preferences: Dict[str, UserPreference] = {}

        # 查询模式统计 (query -> QueryPattern)
        self.query_patterns: Dict[str, QueryPattern] = {}

        # 术语标准化映射
        self.term_normalizations = {
            "盈利": "净利润",
            "营收": "营业收入",
            "财报": "财务报表",
            "ROE": "净资产收益率",
            "ROA": "总资产收益率",
            "EPS": "每股收益",
            "PB": "市净率",
            "PE": "市盈率"
        }

        logger.info("✅ 实时反馈处理器初始化完成")

    async def enhance_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retrieval_level: str = "enhanced"
    ) -> Dict[str, Any]:
        """
        增强查询 - 应用反馈优化

        Args:
            query: 原始查询
            user_id: 用户ID
            session_id: 会话ID
            retrieval_level: 检索级别

        Returns:
            增强后的查询配置
        """
        start_time = datetime.now()

        # 1. 获取反馈上下文
        feedback_context = await self._get_feedback_context(
            query, user_id, session_id
        )

        # 2. 查询重写
        optimized_query = self._rewrite_query(query, feedback_context)

        # 3. 动态参数调整
        params = self._adjust_parameters(
            retrieval_level,
            feedback_context
        )

        # 4. 记录查询事件
        self._log_query_event(
            query,
            optimized_query,
            user_id,
            session_id,
            params,
            feedback_context
        )

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"查询增强完成: {query[:30]} → {optimized_query[:30]}, "
            f"参数={params}, 耗时={elapsed:.2f}ms"
        )

        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "params": params,
            "feedback_context": feedback_context,
            "optimization_time_ms": elapsed
        }

    async def collect_feedback(
        self,
        query: str,
        results: List[Dict],
        user_interactions: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        收集用户反馈

        Args:
            query: 查询
            results: 检索结果
            user_interactions: 用户交互数据
                - clicks: 点击事件列表 [{"doc_id": 1, "position": 1}, ...]
                - dwell_times: 停留时间 {doc_id: seconds}
                - rating: 评分 (1-5)
                - skipped: 是否跳过
            user_id: 用户ID
            session_id: 会话ID
        """
        try:
            # 1. 解析交互数据
            events = self._parse_interactions(
                query, results, user_interactions, user_id, session_id
            )

            # 2. 更新会话历史
            if session_id:
                self._update_session_history(session_id, events)

            # 3. 更新用户偏好
            if user_id:
                self._update_user_preferences(user_id, events)

            # 4. 更新查询模式
            self._update_query_patterns(query, results, events)

            # 5. 持久化反馈（异步）
            await self._persist_feedback(query, results, events, user_id, session_id)

            logger.info(
                f"收集反馈: query={query[:30]}, events={len(events)}, "
                f"user={user_id}, session={session_id}"
            )

        except Exception as e:
            logger.error(f"收集反馈失败: {e}")

    def _rewrite_query(
        self,
        query: str,
        feedback_context: Dict
    ) -> str:
        """
        基于反馈重写查询

        优先级:
        1. 用户偏好术语
        2. 术语标准化
        3. 保持原样
        """
        # 1. 检查用户偏好
        if feedback_context.get("user_preferences"):
            user_prefs = feedback_context["user_preferences"]
            if user_prefs.get("preferred_terms"):
                # 使用用户最偏好的术语
                preferred = user_prefs["preferred_terms"][0]
                if preferred.get("confidence", 0) > 0.7:
                    logger.info(f"应用用户偏好: {query} → {preferred['term']}")
                    return preferred["term"]

        # 2. 术语标准化
        normalized = self._normalize_query(query)
        if normalized != query:
            logger.info(f"术语标准化: {query} → {normalized}")
            return normalized

        # 3. 保持原样
        return query

    def _normalize_query(self, query: str) -> str:
        """查询术语标准化"""
        for term, standard in self.term_normalizations.items():
            if term in query:
                return query.replace(term, standard)
        return query

    def _adjust_parameters(
        self,
        retrieval_level: str,
        feedback_context: Dict
    ) -> Dict[str, Any]:
        """
        基于反馈动态调整检索参数

        调整:
        1. top_k - 结果数量
        2. compression_rate - 压缩率
        3. similarity_threshold - 相似度阈值
        """
        params = {
            "retrieval_level": retrieval_level,
            "top_k": 10,
            "compression_rate": 0.5,
            "similarity_threshold": 0.6
        }

        # 1. 根据用户点击行为调整
        if feedback_context.get("session_history"):
            session_stats = feedback_context["session_history"]
            avg_pos = session_stats.get("avg_click_position", 5)

            # 用户倾向于点击前面的结果 → 可以减少返回数量，增加压缩
            if avg_pos < 3:
                params["top_k"] = 5
                params["compression_rate"] = 0.4
                logger.info(f"激进策略: top_k=5, compression=0.4 (avg_pos={avg_pos:.1f})")

            # 用户经常翻页 → 需要更多结果，减少压缩
            elif avg_pos > 7:
                params["top_k"] = 15
                params["compression_rate"] = 0.7
                logger.info(f"保守策略: top_k=15, compression=0.7 (avg_pos={avg_pos:.1f})")

            else:
                params["top_k"] = 10
                params["compression_rate"] = 0.5

        # 2. 根据用户偏好调整
        if feedback_context.get("user_preferences"):
            user_prefs = feedback_context["user_preferences"]
            if user_prefs.get("preferred_compression"):
                params["compression_rate"] = user_prefs["preferred_compression"]

        # 3. 根据查询历史调整
        if feedback_context.get("query_stats"):
            query_stats = feedback_context["query_stats"]
            if query_stats.get("avg_rating", 0) < 3.0:
                # 历史评分低，降低阈值增加召回
                params["similarity_threshold"] = 0.5
                logger.info(f"降低相似度阈值: 0.5 (历史评分低)")

        return params

    def _parse_interactions(
        self,
        query: str,
        results: List[Dict],
        interactions: Dict,
        user_id: Optional[str],
        session_id: Optional[str]
    ) -> List[Dict]:
        """解析用户交互数据"""
        events = []
        timestamp = datetime.now()

        # 1. 点击事件
        if "clicks" in interactions:
            for click in interactions["clicks"]:
                events.append({
                    "event_type": "click",
                    "query": query,
                    "doc_id": click.get("doc_id"),
                    "position": click.get("position"),
                    "timestamp": timestamp,
                    "value": 1.0
                })

        # 2. 停留时间事件
        if "dwell_times" in interactions:
            for doc_id_str, dwell_time in interactions["dwell_times"].items():
                # 停留时间 > 30秒视为高相关性
                relevance_score = min(dwell_time / 30.0, 1.0)

                events.append({
                    "event_type": "dwell",
                    "query": query,
                    "doc_id": int(doc_id_str),
                    "dwell_time": dwell_time,
                    "timestamp": timestamp,
                    "value": relevance_score
                })

        # 3. 评分事件
        if "rating" in interactions:
            events.append({
                "event_type": "rating",
                "query": query,
                "doc_id": None,
                "rating": interactions["rating"],
                "timestamp": timestamp,
                "value": interactions["rating"]
            })

        # 4. 跳过事件
        if interactions.get("skipped"):
            events.append({
                "event_type": "skip",
                "query": query,
                "doc_id": None,
                "timestamp": timestamp,
                "value": -1.0
            })

        return events

    async def _get_feedback_context(
        self,
        query: str,
        user_id: Optional[str],
        session_id: Optional[str]
    ) -> Dict:
        """获取反馈上下文"""
        context = {
            "has_history": False,
            "user_preferences": None,
            "session_history": None,
            "query_stats": None
        }

        # 1. 会话历史
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            context["has_history"] = True
            context["session_history"] = {
                "event_count": len(session.events),
                "avg_click_position": session.get_avg_click_position()
            }

        # 2. 用户偏好
        if user_id and user_id in self.user_preferences:
            user_pref = self.user_preferences[user_id]
            context["user_preferences"] = {
                "preferred_terms": user_pref.preferred_terms,
                "avg_rating_given": user_pref.avg_rating_given,
                "interaction_patterns": user_pref.interaction_patterns
            }

        # 3. 查询模式
        if query in self.query_patterns:
            pattern = self.query_patterns[query]
            context["query_stats"] = {
                "count": pattern.count,
                "avg_rating": pattern.avg_rating,
                "avg_click_position": pattern.avg_click_position
            }

        return context

    def _log_query_event(
        self,
        original_query: str,
        optimized_query: str,
        user_id: Optional[str],
        session_id: Optional[str],
        params: Dict,
        context: Dict
    ):
        """记录查询事件"""
        if not session_id:
            return

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionHistory(session_id)

        self.sessions[session_id].add_event({
            "event_type": "query",
            "original_query": original_query,
            "optimized_query": optimized_query,
            "params": params,
            "context": context
        })

    def _update_session_history(
        self,
        session_id: str,
        events: List[Dict]
    ):
        """更新会话历史"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionHistory(session_id)

        for event in events:
            self.sessions[session_id].add_event(event)

    def _update_user_preferences(
        self,
        user_id: str,
        events: List[Dict]
    ):
        """更新用户偏好"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreference()

        user_pref = self.user_preferences[user_id]

        # 更新评分
        ratings = [e["value"] for e in events if e["event_type"] == "rating"]
        if ratings:
            user_pref.avg_rating_given = sum(ratings) / len(ratings)

        # 更新交互模式
        clicks = [e for e in events if e["event_type"] == "click"]
        if clicks:
            positions = [e["position"] for e in clicks]
            user_pref.interaction_patterns["avg_click_position"] = (
                sum(positions) / len(positions)
            )

    def _update_query_patterns(
        self,
        query: str,
        results: List[Dict],
        events: List[Dict]
    ):
        """更新查询模式统计"""
        if query not in self.query_patterns:
            self.query_patterns[query] = QueryPattern()

        pattern = self.query_patterns[query]
        pattern.count += 1

        # 更新评分
        ratings = [e["value"] for e in events if e["event_type"] == "rating"]
        if ratings:
            pattern.ratings.extend(ratings)
            pattern.avg_rating = sum(pattern.ratings) / len(pattern.ratings)

        # 更新点击位置
        clicks = [e for e in events if e["event_type"] == "click"]
        if clicks:
            positions = [e["position"] for e in clicks]
            pattern.avg_click_position = (
                (pattern.avg_click_position * (len(clicks) - 1) + sum(positions) / len(clicks))
                if pattern.count > 1
                else sum(positions) / len(clicks)
            )

    async def _persist_feedback(
        self,
        query: str,
        results: List[Dict],
        events: List[Dict],
        user_id: Optional[str],
        session_id: Optional[str]
    ):
        """
        持久化反馈到数据库

        TODO: 实现数据库持久化
        """
        # 这里可以连接到 FeedbackRecord 模型
        # 异步写入，不阻塞主流程
        pass

    def get_insights(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取反馈洞察"""
        insights = {
            "total_sessions": len(self.sessions),
            "total_users": len(self.user_preferences),
            "total_queries": len(self.query_patterns),
            "user_stats": None,
            "session_stats": None
        }

        if user_id and user_id in self.user_preferences:
            user_pref = self.user_preferences[user_id]
            insights["user_stats"] = {
                "avg_rating_given": user_pref.avg_rating_given,
                "interaction_patterns": user_pref.interaction_patterns,
                "query_count": len(user_pref.query_history)
            }

        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            insights["session_stats"] = {
                "event_count": len(session.events),
                "avg_click_position": session.get_avg_click_position(),
                "duration_minutes": (datetime.now() - session.created_at).total_seconds() / 60
            }

        return insights

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """清理旧会话"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        to_delete = [
            session_id for session_id, session in self.sessions.items()
            if session.created_at < cutoff_time
        ]

        for session_id in to_delete:
            del self.sessions[session_id]

        if to_delete:
            logger.info(f"清理了 {len(to_delete)} 个旧会话")


# 全局实例
_realtime_feedback_processor = None


def get_realtime_feedback_processor() -> RealTimeFeedbackProcessor:
    """获取实时反馈处理器实例"""
    global _realtime_feedback_processor
    if _realtime_feedback_processor is None:
        _realtime_feedback_processor = RealTimeFeedbackProcessor()
    return _realtime_feedback_processor


def reset_realtime_feedback_processor():
    """重置全局实例"""
    global _realtime_feedback_processor
    _realtime_feedback_processor = None
