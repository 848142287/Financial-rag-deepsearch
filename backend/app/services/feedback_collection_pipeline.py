"""
反馈数据收集管道
自动收集用户反馈和交互数据，为LTR模型训练做准备
"""

from app.core.structured_logging import get_structured_logger
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import select
from app.core.database import get_db

logger = get_structured_logger(__name__)


class FeedbackType(Enum):
    """反馈类型"""
    EXPLICIT = "explicit"  # 显式反馈（评分、评论）
    IMPLICIT = "implicit"  # 隐式反馈（行为数据）
    INFERRED = "inferred"  # 推断反馈（从行为推断）


@dataclass
class FeedbackEvent:
    """反馈事件"""
    event_type: str  # click, dwell, scroll, rating, etc.
    user_id: Optional[str]
    session_id: str
    query: str
    document_id: Optional[int]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    value: Optional[float] = None


class FeedbackCollectionPipeline:
    """反馈收集管道"""

    def __init__(self):
        self.buffer_size = 100  # 缓冲区大小
        self.flush_interval = 60  # 刷新间隔（秒）
        self.feedback_buffer: List[FeedbackEvent] = []

    async def collect_feedback(
        self,
        event: FeedbackEvent,
        immediate_flush: bool = False
    ):
        """
        收集反馈事件

        Args:
            event: 反馈事件
            immediate_flush: 是否立即刷新到数据库
        """
        try:
            # 添加到缓冲区
            self.feedback_buffer.append(event)

            # 缓冲区满或要求立即刷新
            if len(self.feedback_buffer) >= self.buffer_size or immediate_flush:
                await self._flush_buffer()

        except Exception as e:
            logger.error(f"收集反馈失败: {e}")

    async def collect_implicit_feedback(
        self,
        user_id: Optional[str],
        session_id: str,
        query: str,
        documents: List[Dict[str, Any]],
        interaction_data: Dict[str, Any]
    ):
        """
        收集隐式反馈

        Args:
            user_id: 用户ID
            session_id: 会话ID
            query: 查询
            documents: 检索到的文档
            interaction_data: 交互数据（点击、停留时间等）
        """
        try:
            # 解析交互数据
            events = self._parse_interaction_data(
                user_id, session_id, query, documents, interaction_data
            )

            # 批量收集
            for event in events:
                await self.collect_feedback(event)

            logger.info(f"收集了{len(events)}个隐式反馈事件")

        except Exception as e:
            logger.error(f"收集隐式反馈失败: {e}")

    def _parse_interaction_data(
        self,
        user_id: Optional[str],
        session_id: str,
        query: str,
        documents: List[Dict[str, Any]],
        interaction_data: Dict[str, Any]
    ) -> List[FeedbackEvent]:
        """解析交互数据为反馈事件"""
        events = []
        timestamp = datetime.now()

        # 1. 点击事件
        clicks = interaction_data.get('clicks', [])
        for click in clicks:
            doc_idx = click.get('doc_index')
            if doc_idx is not None and doc_idx < len(documents):
                events.append(FeedbackEvent(
                    event_type='click',
                    user_id=user_id,
                    session_id=session_id,
                    query=query,
                    document_id=documents[doc_idx].get('id'),
                    timestamp=timestamp,
                    metadata={'position': doc_idx, 'click_time': click.get('time')},
                    value=1.0  # 点击表示正相关
                ))

        # 2. 停留时间事件
        dwells = interaction_data.get('dwell_times', [])
        for dwell in dwells:
            doc_idx = dwell.get('doc_index')
            if doc_idx is not None and doc_idx < len(documents):
                dwell_time = dwell.get('duration', 0)
                # 将停留时间转换为相关性分数（0-1）
                dwell_score = self._normalize_dwell_time(dwell_time)

                events.append(FeedbackEvent(
                    event_type='dwell',
                    user_id=user_id,
                    session_id=session_id,
                    query=query,
                    document_id=documents[doc_idx].get('id'),
                    timestamp=timestamp,
                    metadata={'duration_seconds': dwell_time},
                    value=dwell_score
                ))

        # 3. 滚动深度事件
        scroll_depth = interaction_data.get('scroll_depth', 0)
        if scroll_depth > 0:
            scroll_score = min(1.0, scroll_depth / 100.0)
            events.append(FeedbackEvent(
                event_type='scroll',
                user_id=user_id,
                session_id=session_id,
                query=query,
                document_id=None,  # 全局事件
                timestamp=timestamp,
                metadata={'scroll_depth_percent': scroll_depth},
                value=scroll_score
            ))

        # 4. 阅读时间事件
        reading_time = interaction_data.get('reading_time', 0)
        if reading_time > 0:
            reading_score = self._normalize_dwell_time(reading_time)
            events.append(FeedbackEvent(
                event_type='reading_time',
                user_id=user_id,
                session_id=session_id,
                query=query,
                document_id=None,
                timestamp=timestamp,
                metadata={'reading_time_seconds': reading_time},
                value=reading_score
            ))

        return events

    def _normalize_dwell_time(self, dwell_time: float) -> float:
        """归一化停留时间为相关性分数"""
        # 基于行业标准的停留时间-相关性映射
        if dwell_time < 3:
            return 0.2  # 太短，可能不相关
        elif dwell_time <= 10:
            # 3-10秒，线性增长
            return 0.2 + (dwell_time - 3) / 7 * 0.5
        elif dwell_time <= 60:
            # 10-60秒，高分区间
            return 0.7 + (dwell_time - 10) / 50 * 0.3
        else:
            return 0.8  # 太长可能是误操作，但仍然是正面的

    async def _flush_buffer(self):
        """刷新缓冲区到数据库"""
        if not self.feedback_buffer:
            return

        try:
            # 批量插入数据库
            async with get_db() as db:
                from app.models.user_feedback import UserFeedback

                # 按查询-文档对聚合
                aggregated = self._aggregate_feedback_events(self.feedback_buffer)

                for key, agg_data in aggregated.items():
                    # 创建反馈记录
                    feedback = UserFeedback(
                        user_id=agg_data['user_id'],
                        session_id=agg_data['session_id'],
                        query=agg_data['query'],
                        document_id=agg_data['document_id'],
                        feedback_type=agg_data['feedback_type'],
                        signals=json.dumps(agg_data['signals']),
                        satisfaction_score=agg_data['satisfaction_score'],
                        confidence=agg_data['confidence'],
                        created_at=datetime.now()
                    )

                    db.add(feedback)

                await db.commit()
                logger.info(f"已保存{len(aggregated)}条反馈记录到数据库")

            # 清空缓冲区
            self.feedback_buffer.clear()

        except Exception as e:
            logger.error(f"刷新缓冲区失败: {e}")

    def _aggregate_feedback_events(
        self,
        events: List[FeedbackEvent]
    ) -> Dict[str, Dict[str, Any]]:
        """聚合反馈事件"""
        aggregated = {}

        for event in events:
            # 生成聚合键
            key = f"{event.session_id}:{event.query}:{event.document_id or 'global'}"

            if key not in aggregated:
                aggregated[key] = {
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'query': event.query,
                    'document_id': event.document_id,
                    'feedback_type': FeedbackType.IMPLICIT.value,
                    'signals': {},
                    'event_count': 0,
                    'total_value': 0.0
                }

            # 聚合信号
            agg = aggregated[key]
            agg['event_count'] += 1
            if event.value is not None:
                agg['total_value'] += event.value

            # 合并元数据
            if event.event_type not in agg['signals']:
                agg['signals'][event.event_type] = []

            agg['signals'][event.event_type].append({
                'value': event.value,
                'timestamp': event.timestamp.isoformat(),
                'metadata': event.metadata
            })

        # 计算聚合指标
        for agg in aggregated.values():
            if agg['event_count'] > 0:
                agg['satisfaction_score'] = agg['total_value'] / agg['event_count']
                agg['confidence'] = min(1.0, agg['event_count'] / 10.0)  # 更多事件=更高置信度

        return aggregated

    async def collect_explicit_feedback(
        self,
        user_id: Optional[str],
        session_id: str,
        query: str,
        document_id: Optional[int],
        feedback_type: str,
        rating: Optional[int] = None,
        comments: Optional[str] = None,
        highlighted_items: Optional[List[Dict]] = None
    ):
        """
        收集显式反馈

        Args:
            user_id: 用户ID
            session_id: 会话ID
            query: 查询
            document_id: 文档ID
            feedback_type: 反馈类型
            rating: 评分（1-5）
            comments: 评论
            highlighted_items: 高亮项
        """
        try:
            async with get_db() as db:
                from app.models.user_feedback import UserFeedback

                feedback = UserFeedback(
                    user_id=user_id,
                    session_id=session_id,
                    query=query,
                    document_id=document_id,
                    feedback_type=feedback_type,
                    rating=rating,
                    comments=comments,
                    highlighted_items=json.dumps(highlighted_items) if highlighted_items else None,
                    satisfaction_score=rating / 5.0 if rating else None,
                    confidence=1.0,  # 显式反馈置信度高
                    created_at=datetime.now()
                )

                db.add(feedback)
                await db.commit()

                logger.info(f"已保存显式反馈: {feedback_type}, rating={rating}")

        except Exception as e:
            logger.error(f"保存显式反馈失败: {e}")


class AutomaticFeedbackCollector:
    """自动反馈收集器 - 从现有数据源收集反馈"""

    async def collect_from_retrieval_logs(
        self,
        days_back: int = 7,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        从检索日志中提取反馈数据

        Args:
            days_back: 回溯天数
            batch_size: 批次大小

        Returns:
            收集统计信息
        """
        try:
            async with get_db() as db:
                from app.models.retrieval_log import RetrievalLog

                cutoff_date = datetime.now() - timedelta(days=days_back)

                # 查询检索日志
                stmt = select(RetrievalLog).where(
                    RetrievalLog.created_at >= cutoff_date
                ).limit(batch_size)

                result = await db.execute(stmt)
                logs = result.scalars().all()

                # 从日志中提取反馈信号
                feedback_count = 0
                for log in logs:
                    # 解析检索结果中的隐式信号
                    if log.response_details:
                        details = json.loads(log.response_details) if isinstance(
                            log.response_details, str) else log.response_details

                        # 提取用户交互数据
                        interactions = details.get('user_interactions', {})
                        if interactions:
                            # 转换为反馈事件
                            feedback_count += await self._convert_interactions_to_feedback(
                                log, interactions
                            )

                return {
                    'logs_processed': len(logs),
                    'feedback_extracted': feedback_count,
                    'days_analyzed': days_back
                }

        except Exception as e:
            logger.error(f"从检索日志收集反馈失败: {e}")
            return {'error': str(e)}

    async def _convert_interactions_to_feedback(
        self,
        log,
        interactions: Dict[str, Any]
    ) -> int:
        """将交互数据转换为反馈记录"""
        # TODO: 实现转换逻辑
        return 0


# 全局实例
_feedback_pipeline = None
_auto_collector = None


def get_feedback_pipeline() -> FeedbackCollectionPipeline:
    """获取反馈管道实例"""
    global _feedback_pipeline
    if _feedback_pipeline is None:
        _feedback_pipeline = FeedbackCollectionPipeline()
    return _feedback_pipeline


def get_auto_feedback_collector() -> AutomaticFeedbackCollector:
    """获取自动反馈收集器实例"""
    global _auto_collector
    if _auto_collector is None:
        _auto_collector = AutomaticFeedbackCollector()
    return _auto_collector
