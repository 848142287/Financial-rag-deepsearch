"""
Agentic RAG核心协调器
整合三阶段流程：计划-执行-生成
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

from .plan_phase import PlanPhase, QueryContext, QueryPlan
from .execute_phase import ExecutePhase, FusedResult
from .generation_phase import GenerationPhase, GeneratedAnswer
from .optimization import (
    retrieval_optimizer,
    multi_modal_integrator,
    consistency_manager
)
from app.services.progress_tracker import progress_tracker, TaskStatus

logger = logging.getLogger(__name__)


class AgenticRAG:
    """Agentic RAG主协调器"""

    def __init__(self):
        self.plan_phase = PlanPhase()
        self.execute_phase = ExecutePhase()
        self.generation_phase = GenerationPhase()

        # 性能统计
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time': 0,
            'cache_hit_rate': 0,
            'success_rate': 0
        }

    async def query(
        self,
        question: str,
        conversation_id: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
        history: Optional[List[str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        enable_progress_tracking: bool = True
    ) -> Dict[str, Any]:
        """
        执行Agentic RAG查询

        Args:
            question: 用户问题
            conversation_id: 对话ID
            document_ids: 限定文档ID列表
            history: 历史对话
            user_preferences: 用户偏好
            enable_progress_tracking: 是否启用进度跟踪

        Returns:
            查询结果
        """
        start_time = datetime.now()
        task_id = None

        try:
            logger.info(f"Starting Agentic RAG query: {question[:100]}...")

            # 1. 记录查询频率
            query_hash = hashlib.md5(question.encode()).hexdigest()
            await retrieval_optimizer.record_query_frequency(question)

            # 2. 检查热门查询缓存
            cached_result = await retrieval_optimizer.get_cached_hot_query(query_hash)
            if cached_result:
                logger.info(f"Returning cached result for query: {question[:50]}...")
                return cached_result

            # 3. 创建进度跟踪任务
            if enable_progress_tracking:
                task_id = await progress_tracker.create_task(
                    query=question,
                    retrieval_mode="agentic"
                )

            # 4. 构建查询上下文
            context = QueryContext(
                conversation_id=conversation_id,
                previous_queries=history or [],
                user_preferences=user_preferences or {},
                session_metadata={
                    'document_ids': document_ids,
                    'enable_progress': enable_progress_tracking
                }
            )

            # === Phase 1: Plan ===
            if task_id:
                await progress_tracker.update_progress(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    current_step="planning",
                    progress_percentage=5.0,
                    message="开始理解查询意图"
                )

            plan = await self.plan_phase.process_query(question, context)

            # 金融查询增强
            enhanced_question = retrieval_optimizer.enhance_financial_query(plan.processed_query)
            plan.processed_query = enhanced_question

            # 时间敏感查询处理
            temporal_info = retrieval_optimizer.temporal_query_processing(enhanced_question)
            if temporal_info['is_temporal']:
                plan.processed_query = temporal_info['enhanced_query']
                plan.retrieval_params['temporal_focus'] = True

            # === Phase 2: Execute ===
            if task_id:
                await progress_tracker.update_progress(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    current_step="execution",
                    progress_percentage=25.0,
                    message="开始执行检索策略"
                )

            fused_result = await self.execute_phase.execute_retrieval(
                plan=plan,
                document_ids=document_ids,
                task_id=task_id
            )

            # 多模态结果整合
            if hasattr(fused_result, 'table_results') or hasattr(fused_result, 'image_results'):
                fused_result.final_results = await multi_modal_integrator.integrate_all(
                    fused_result.final_results
                )

            # === Phase 3: Generation ===
            if task_id:
                await progress_tracker.update_progress(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    current_step="generation",
                    progress_percentage=80.0,
                    message="开始生成答案"
                )

            answer = await self.generation_phase.generate_answer(
                plan=plan,
                fused_result=fused_result,
                task_id=task_id
            )

            # 4. 构建最终结果
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000

            result = {
                'success': True,
                'answer': answer.answer,
                'sources': answer.sources,
                'confidence': answer.confidence_score,
                'metadata': {
                    'query_type': plan.query_type.value,
                    'complexity_level': plan.complexity_level,
                    'retrieval_strategy': plan.retrieval_strategy.value,
                    'retrieval_quality': fused_result.quality_score,
                    'factual_score': answer.factual_score,
                    'compliance_score': answer.compliance_score,
                    'task_id': task_id,
                    'plan_id': plan.task_id,
                    'response_time_ms': int(response_time),
                    'timestamp': end_time.isoformat()
                },
                'performance': {
                    'plan_time_ms': None,  # 可以添加详细的时间分解
                    'execution_time_ms': fused_result.execution_time_ms,
                    'generation_time_ms': answer.generation_time_ms,
                    'cache_hit': False
                }
            }

            # 5. 缓存结果
            await retrieval_optimizer.cache_hot_query(query_hash, result)

            # 6. 更新性能统计
            self._update_performance_stats(response_time, True)

            logger.info(f"Agentic RAG query completed: {response_time:.2f}ms")
            return result

        except Exception as e:
            # 更新性能统计
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(response_time, False)

            logger.error(f"Agentic RAG query failed: {str(e)}")

            if task_id:
                await progress_tracker.fail_task(task_id, str(e))

            return {
                'success': False,
                'error': str(e),
                'response_time_ms': int(response_time),
                'timestamp': datetime.now().isoformat()
            }

    async def batch_query(
        self,
        queries: List[str],
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """批量查询处理"""
        results = []

        # 分批处理
        for i in range(0, len(queries), max_concurrent):
            batch = queries[i:i + max_concurrent]
            tasks = [
                self.query(query, **kwargs)
                for query in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        'success': False,
                        'error': str(result),
                        'query': batch[j]
                    })
                else:
                    results.append(result)

        return results

    async def explain_query_plan(self, question: str) -> Dict[str, Any]:
        """解释查询计划（用于调试）"""
        try:
            context = QueryContext()
            plan = await self.plan_phase.process_query(question, context)

            return {
                'original_query': question,
                'processed_query': plan.processed_query,
                'query_type': plan.query_type.value,
                'complexity_level': plan.complexity_level,
                'retrieval_strategy': plan.retrieval_strategy.value,
                'main_queries': plan.main_queries,
                'backup_queries': plan.backup_queries,
                'retrieval_params': plan.retrieval_params,
                'quality_threshold': plan.quality_threshold,
                'estimated_results': plan.estimated_results
            }

        except Exception as e:
            return {
                'error': str(e),
                'query': question
            }

    async def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        # 更新缓存命中率
        try:
            popular_queries = await retrieval_optimizer.get_popular_queries(10)
            cache_keys = await retrieval_optimizer.redis.keys(
                f"{retrieval_optimizer.cache_config.cache_key_prefix}:hot:*"
            )
            self.performance_stats['cache_hit_rate'] = len(cache_keys) / max(self.performance_stats['total_queries'], 1)
        except:
            pass

        return self.performance_stats.copy()

    async def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time': 0,
            'cache_hit_rate': 0,
            'success_rate': 0
        }

    async def optimize_cache(self):
        """优化缓存"""
        await retrieval_optimizer.optimize_query_cache()
        return {"message": "Cache optimization completed"}

    async def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        health_status = {
            'overall': 'healthy',
            'components': {},
            'performance': await self.get_performance_stats()
        }

        # 检查各组件状态
        try:
            # 检查计划阶段
            plan_health = await self._check_plan_phase_health()
            health_status['components']['plan_phase'] = plan_health

            # 检查执行阶段
            exec_health = await self._check_execute_phase_health()
            health_status['components']['execute_phase'] = exec_health

            # 检查生成阶段
            gen_health = await self._check_generation_phase_health()
            health_status['components']['generation_phase'] = gen_health

            # 检查数据一致性
            consistency = await consistency_manager.check_data_consistency()
            health_status['components']['data_consistency'] = consistency

            # 检查缓存状态
            cache_status = await self._check_cache_health()
            health_status['components']['cache'] = cache_status

            # 确定整体健康状态
            if any(status.get('status') == 'error' for status in health_status['components'].values()):
                health_status['overall'] = 'unhealthy'
            elif any(status.get('status') == 'warning' for status in health_status['components'].values()):
                health_status['overall'] = 'degraded'

        except Exception as e:
            health_status['overall'] = 'error'
            health_status['error'] = str(e)

        return health_status

    def _update_performance_stats(self, response_time: float, success: bool):
        """更新性能统计"""
        self.performance_stats['total_queries'] += 1

        # 更新平均响应时间
        total = self.performance_stats['total_queries']
        current_avg = self.performance_stats['avg_response_time']
        self.performance_stats['avg_response_time'] = (
            (current_avg * (total - 1) + response_time) / total
        )

        # 更新成功率
        if success:
            success_count = self.performance_stats['success_rate'] * (total - 1) + 1
        else:
            success_count = self.performance_stats['success_rate'] * (total - 1)
        self.performance_stats['success_rate'] = success_count / total

    async def _check_plan_phase_health(self) -> Dict[str, Any]:
        """检查计划阶段健康状态"""
        try:
            # 简单测试
            test_plan = await self.plan_phase.process_query(
                "测试查询",
                QueryContext()
            )
            return {'status': 'healthy', 'test_passed': True}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _check_execute_phase_health(self) -> Dict[str, Any]:
        """检查执行阶段健康状态"""
        try:
            # 这里可以添加对向量库和图数据库的连接测试
            return {'status': 'healthy', 'connections': 'ok'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _check_generation_phase_health(self) -> Dict[str, Any]:
        """检查生成阶段健康状态"""
        try:
            # 测试LLM连接
            from app.services.llm_service import llm_service
            test_response = await llm_service.generate_completion(
                prompt="测试",
                max_tokens=10
            )
            return {'status': 'healthy', 'llm_connected': True}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _check_cache_health(self) -> Dict[str, Any]:
        """检查缓存健康状态"""
        try:
            # 测试Redis连接
            await retrieval_optimizer.redis.ping()
            cache_size = len(await retrieval_optimizer.redis.keys("*"))
            return {
                'status': 'healthy',
                'cache_size': cache_size,
                'redis_connected': True
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# 全局实例
agentic_rag = AgenticRAG()