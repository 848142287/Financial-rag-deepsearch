"""
反馈驱动检索优化器
核心协调器，整合会话管理、反馈分析和优化策略
"""

from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import logging
from datetime import datetime
import json

from .session_state import (
    session_manager, SearchSession, SessionState, FeedbackData, FeedbackType, OptimizationRecord
)
from .feedback_analyzer import FeedbackAnalyzer, OptimizationNeed
from .optimization_strategies import OptimizationStrategySelector, OptimizationParams

logger = logging.getLogger(__name__)


class FeedbackDrivenOptimizer:
    """反馈驱动检索优化器"""

    def __init__(self):
        self.session_manager = session_manager
        self.feedback_analyzer = FeedbackAnalyzer()
        self.strategy_selector = OptimizationStrategySelector()

        # RAG服务集成（将在实际使用时注入）
        self.rag_service = None

    def set_rag_service(self, rag_service):
        """设置RAG服务"""
        self.rag_service = rag_service

    async def start_search_session(self, user_id: Optional[int], initial_query: str) -> Dict[str, Any]:
        """开始新的搜索会话"""
        # 创建新会话
        session = self.session_manager.create_session(user_id, initial_query)

        # 更新状态为搜索中
        self.session_manager.update_state(session.session_id, SessionState.SEARCHING)

        try:
            # 执行初始检索
            results = await self._execute_search(initial_query, OptimizationParams(query=initial_query))

            # 更新会话状态
            self.session_manager.update_state(session.session_id, SessionState.SHOWING)
            self.session_manager.update_current_results(session.session_id, results)

            # 构建响应
            response = {
                "session_id": session.session_id,
                "round": 1,
                "max_rounds": session.max_rounds,
                "results": results,
                "can_continue": self.session_manager.can_continue(session.session_id),
                "feedback_options": self._get_feedback_options(),
                "state": "search_completed"
            }

            logger.info(f"会话 {session.session_id} 初始检索完成，返回 {len(results)} 个结果")
            return response

        except Exception as e:
            logger.error(f"初始检索失败：{str(e)}")
            self.session_manager.update_state(session.session_id, SessionState.COMPLETED, outcome="error")
            raise

    async def process_feedback(self, session_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户反馈"""
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"会话 {session_id} 不存在")

        if session.state != SessionState.SHOWING:
            raise ValueError(f"会话 {session_id} 当前状态不允许反馈：{session.state.value}")

        # 更新状态为处理中
        self.session_manager.update_state(session_id, SessionState.PROCESSING)

        try:
            # 解析反馈数据
            parsed_feedback = self._parse_feedback_data(feedback_data)

            # 添加反馈到会话
            self.session_manager.add_feedback(session_id, parsed_feedback)

            # 分析反馈需求
            optimization_need = self.feedback_analyzer.analyze_feedback(parsed_feedback)

            # 检查是否满意
            if parsed_feedback.rating and parsed_feedback.rating >= 4:
                # 用户满意，结束会话
                self.session_manager.update_state(
                    session_id, SessionState.COMPLETED,
                    outcome="success",
                    satisfaction_score=parsed_feedback.rating
                )
                return {
                    "session_id": session_id,
                    "state": "completed",
                    "message": "感谢您的反馈！会话已成功完成。",
                    "satisfaction_score": parsed_feedback.rating
                }

            # 检查是否可以继续优化
            if not self.session_manager.can_continue(session_id):
                # 达到最大轮次
                self.session_manager.update_state(session_id, SessionState.MAX_ROUNDS)
                return {
                    "session_id": session_id,
                    "state": "max_rounds",
                    "message": "已达到最大优化轮次，会话结束。",
                    "final_results": session.current_results
                }

            # 判断是否需要优化
            if not self.feedback_analyzer.should_optimize(optimization_need):
                # 不需要优化，提供通用建议
                return {
                    "session_id": session_id,
                    "state": "no_optimization",
                    "message": "基于您的反馈，建议您尝试重新表述查询或提供更具体的要求。",
                    "suggestions": optimization_need.suggested_actions
                }

            # 执行优化检索
            return await self._execute_optimization(session, optimization_need)

        except Exception as e:
            logger.error(f"处理反馈失败：{str(e)}")
            self.session_manager.update_state(session_id, SessionState.SHOWING)
            raise

    async def _execute_optimization(self, session: SearchSession, optimization_need: OptimizationNeed) -> Dict[str, Any]:
        """执行优化检索"""
        session_id = session.session_id

        # 更新状态为优化中
        self.session_manager.update_state(session_id, SessionState.OPTIMIZING)

        try:
            # 选择优化策略
            strategy = self.strategy_selector.select_strategy(optimization_need)

            # 构建原始参数
            original_params = OptimizationParams(
                query=session.current_query,
                top_k=10,
                similarity_threshold=0.7
            )

            # 应用优化策略
            start_time = datetime.now()
            optimized_params = await self.strategy_selector.apply_strategy(strategy, original_params, optimization_need)
            optimization_time = (datetime.now() - start_time).total_seconds()

            # 执行优化检索
            results = await self._execute_search(optimized_params.query, optimized_params)

            # 记录优化信息
            optimization_record = OptimizationRecord(
                round_number=session.current_round + 1,
                feedback_data=session.feedback_history[-1],
                optimization_strategy=strategy.name,
                adjusted_params=optimized_params.__dict__,
                original_query=session.current_query,
                optimized_query=optimized_params.query,
                results_count=len(results),
                optimization_time=optimization_time
            )
            self.session_manager.add_optimization_record(session_id, optimization_record)

            # 更新会话状态
            self.session_manager.update_state(session_id, SessionState.SHOWING)
            self.session_manager.update_current_results(session_id, results)

            # 构建响应
            response = {
                "session_id": session_id,
                "round": session.current_round,
                "max_rounds": session.max_rounds,
                "results": results,
                "can_continue": self.session_manager.can_continue(session_id),
                "optimization_info": {
                    "strategy_applied": strategy.name,
                    "strategy_description": strategy.description,
                    "optimization_time": round(optimization_time, 2),
                    "query_changes": {
                        "original": session.current_query,
                        "optimized": optimized_params.query
                    },
                    "highlights": self._get_optimization_highlights(original_params, optimized_params, strategy)
                },
                "feedback_options": self._get_feedback_options(),
                "state": "optimization_completed"
            }

            logger.info(f"会话 {session_id} 优化完成，策略：{strategy.name}，结果数：{len(results)}")
            return response

        except Exception as e:
            logger.error(f"优化检索失败：{str(e)}")
            self.session_manager.update_state(session_id, SessionState.SHOWING)
            raise

    async def _execute_search(self, query: str, params: OptimizationParams) -> List[Dict[str, Any]]:
        """执行检索（集成现有RAG服务）"""
        if not self.rag_service:
            # 模拟检索结果
            return await self._simulate_search_results(query, params)

        # 实际检索逻辑
        try:
            # 这里需要根据实际的RAG服务接口进行调整
            search_params = {
                "query": query,
                "top_k": params.top_k,
                "similarity_threshold": params.similarity_threshold,
                "filters": params.filters or {},
                "weight_config": params.weight_config or {}
            }

            # 调用RAG服务
            results = await self.rag_service.search(**search_params)

            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "source": result.get("source", ""),
                    "score": result.get("score", 0.0),
                    "metadata": result.get("metadata", {}),
                    "highlight": result.get("highlight", ""),
                    "date": result.get("date", ""),
                    "authority_level": result.get("authority_level", "medium")
                })

            return formatted_results

        except Exception as e:
            logger.error(f"RAG检索失败：{str(e)}")
            # 降级到模拟结果
            return await self._simulate_search_results(query, params)

    async def _simulate_search_results(self, query: str, params: OptimizationParams) -> List[Dict[str, Any]]:
        """模拟检索结果"""
        # 生成模拟结果
        mock_results = []
        for i in range(min(params.top_k, 5)):
            mock_results.append({
                "id": f"mock_{i}_{datetime.now().timestamp()}",
                "content": f"这是关于'{query}'的模拟检索结果 {i+1}。内容包含相关的财务信息和分析数据。",
                "title": f"{query} - 相关文档 {i+1}",
                "source": ["官方报告", "研究分析", "新闻资讯", "监管文件", "市场数据"][i % 5],
                "score": round(0.9 - i * 0.1, 2),
                "metadata": {
                    "page": i + 1,
                    "section": "财务分析",
                    "type": "document"
                },
                "highlight": f"<mark>{query}</mark>",
                "date": "2024-01-01",
                "authority_level": ["high", "medium", "high", "low", "medium"][i % 5]
            })

        return mock_results

    def _parse_feedback_data(self, feedback_data: Dict[str, Any]) -> FeedbackData:
        """解析反馈数据"""
        feedback_type = FeedbackType(feedback_data.get("feedback_type", "unclear"))
        rating = feedback_data.get("rating")
        comments = feedback_data.get("comments", "")
        rewritten_query = feedback_data.get("rewritten_query", "")
        highlighted_items = feedback_data.get("highlighted_items", [])
        specific_requirements = feedback_data.get("specific_requirements", [])

        return FeedbackData(
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            rewritten_query=rewritten_query,
            highlighted_items=highlighted_items,
            specific_requirements=specific_requirements
        )

    def _get_feedback_options(self) -> List[Dict[str, Any]]:
        """获取反馈选项"""
        return [
            {
                "type": "relevance_low",
                "label": "结果不相关",
                "description": "检索结果与我的问题不匹配"
            },
            {
                "type": "incomplete",
                "label": "信息不完整",
                "description": "缺少我需要的信息"
            },
            {
                "type": "accuracy_issue",
                "label": "准确性有问题",
                "description": "结果中的信息可能不准确或过时"
            },
            {
                "type": "sorting_issue",
                "label": "排序问题",
                "description": "重要结果的排序不合理"
            },
            {
                "type": "general",
                "label": "其他问题",
                "description": "其他类型的问题或建议"
            }
        ]

    def _get_optimization_highlights(self, original: OptimizationParams, optimized: OptimizationParams,
                                   strategy) -> List[str]:
        """获取优化亮点"""
        highlights = []

        if original.query != optimized.query:
            highlights.append(f"查询已优化：'{original.query}' → '{optimized.query}'")

        if optimized.top_k > original.top_k:
            highlights.append(f"结果数量增加：{original.top_k} → {optimized.top_k}")

        if optimized.similarity_threshold < original.similarity_threshold:
            highlights.append(f"相似度要求降低：{original.similarity_threshold} → {optimized.similarity_threshold}")

        if optimized.date_range_days and (not original.date_range_days or optimized.date_range_days > original.date_range_days):
            highlights.append(f"时间范围扩大：{original.date_range_days or '默认'} → {optimized.date_range_days}天")

        if optimized.authority_boost > original.authority_boost:
            highlights.append("权威性权重提升")

        if optimized.sort_by != original.sort_by:
            highlights.append(f"排序策略调整：{original.sort_by} → {optimized.sort_by}")

        if not highlights:
            highlights.append(f"应用了{strategy.description}策略")

        return highlights

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        return self.session_manager.get_session_summary(session_id)

    async def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话统计"""
        return self.session_manager.get_session_stats(session_id)

    async def abandon_session(self, session_id: str) -> bool:
        """放弃会话"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return False

        self.session_manager.update_state(session_id, SessionState.COMPLETED, outcome="abandoned")
        logger.info(f"会话 {session_id} 已放弃")
        return True

    async def get_optimization_suggestions(self, session_id: str) -> List[str]:
        """获取优化建议"""
        session = self.session_manager.get_session(session_id)
        if not session or not session.feedback_history:
            return []

        latest_feedback = session.feedback_history[-1]
        optimization_need = self.feedback_analyzer.analyze_feedback(latest_feedback)
        return optimization_need.suggested_actions


# 全局优化器实例
feedback_optimizer = FeedbackDrivenOptimizer()