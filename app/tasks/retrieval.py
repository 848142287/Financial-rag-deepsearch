"""
检索异步任务
包括向量检索、图谱检索、结果融合等功能
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from celery import current_task
from app.core.celery_config import celery_app, monitor_task, RetrievalTask
from app.core.websocket_manager import connection_manager, MessageType
from app.services.rag.vector_retriever import VectorRetriever
from app.services.rag.graph_retriever import GraphRetriever
from app.services.rag.keyword_retriever import KeywordRetriever
from app.services.rag.result_fusioner import ResultFusioner
from app.services.rag.context_builder import ContextBuilder
from app.services.llm.llm_client import llm_client
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=RetrievalTask,
    name='app.tasks.retrieval.hybrid_search',
    max_retries=2,
    soft_time_limit=300
)
@monitor_task('hybrid_search')
def hybrid_search(self, query_id: str, query: str, user_id: int,
                 search_options: Optional[Dict[str, Any]] = None):
    """
    混合检索任务
    """
    logger.info(f"开始混合检索: {query_id}")

    try:
        # 更新任务状态
        update_search_progress(query_id, 10, "开始检索")

        # 解析和改写查询
        processed_query = process_query(query, search_options)
        update_search_progress(query_id, 25, "查询处理完成")

        # 并行执行多种检索
        search_tasks = [
            perform_vector_search(processed_query, search_options),
            perform_graph_search(processed_query, search_options),
            perform_keyword_search(processed_query, search_options)
        ]

        # 等待所有检索完成
        vector_results, graph_results, keyword_results = asyncio.run(
            execute_parallel_search(search_tasks)
        )
        update_search_progress(query_id, 60, "并行检索完成")

        # 融合检索结果
        fused_results = fuse_search_results(
            vector_results, graph_results, keyword_results, search_options
        )
        update_search_progress(query_id, 75, "结果融合完成")

        # 构建上下文
        context = build_context(fused_results, processed_query)
        update_search_progress(query_id, 85, "上下文构建完成")

        # 生成答案
        answer = generate_answer(processed_query, context)
        update_search_progress(query_id, 95, "答案生成完成")

        # 保存搜索结果
        save_search_result(query_id, query, fused_results, answer, user_id)
        update_search_progress(query_id, 100, "检索完成")

        # 发送完成通知
        send_search_completion_notification(query_id, user_id, 'success', answer, fused_results)

        return {
            'status': 'success',
            'query_id': query_id,
            'answer': answer,
            'results': fused_results,
            'statistics': {
                'vector_results': len(vector_results),
                'graph_results': len(graph_results),
                'keyword_results': len(keyword_results),
                'fused_results': len(fused_results),
                'processing_time': datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"混合检索失败: {query_id}, 错误: {e}")

        # 发送失败通知
        send_search_completion_notification(query_id, user_id, 'failed', str(e))

        # 重新抛出异常以触发Celery重试
        raise


@celery_app.task(
    bind=True,
    base=RetrievalTask,
    name='app.tasks.retrieval.batch_search',
    max_retries=1,
    soft_time_limit=600
)
@monitor_task('batch_search')
def batch_search(self, queries: List[Dict[str, Any]], user_id: int,
                search_options: Optional[Dict[str, Any]] = None):
    """
    批量检索任务
    """
    logger.info(f"开始批量检索: {len(queries)} 个查询")

    results = []
    batch_id = f"batch_{datetime.now().timestamp()}"

    try:
        total_queries = len(queries)
        completed_queries = 0

        for i, query_data in enumerate(queries):
            query_id = query_data.get('id')
            query_text = query_data.get('query')

            try:
                # 更新批量进度
                batch_progress = int((i / total_queries) * 100)
                update_batch_search_progress(batch_id, user_id, batch_progress,
                                           f"处理查询 {i+1}/{total_queries}")

                # 启动单个检索任务
                task = hybrid_search.delay(query_id, query_text, user_id, search_options)
                results.append({
                    'query_id': query_id,
                    'query': query_text,
                    'task_id': task.id,
                    'status': 'started'
                })

                completed_queries += 1

            except Exception as e:
                logger.error(f"启动检索任务失败: {query_id}, 错误: {e}")
                results.append({
                    'query_id': query_id,
                    'query': query_text,
                    'task_id': None,
                    'status': 'failed',
                    'error': str(e)
                })

        # 发送批量检索完成通知
        send_batch_search_notification(batch_id, user_id, results)

        return {
            'status': 'success',
            'batch_id': batch_id,
            'total_queries': total_queries,
            'completed_queries': completed_queries,
            'results': results,
            'started_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"批量检索失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=RetrievalTask,
    name='app.tasks.retrieval.optimize_search',
    max_retries=1,
    soft_time_limit=180
)
@monitor_task('search_optimization')
def optimize_search(self, query_id: str, original_query: str, feedback: Dict[str, Any],
                   user_id: int):
    """
    基于反馈的搜索优化
    """
    logger.info(f"开始搜索优化: {query_id}")

    try:
        # 分析反馈
        optimization_strategy = analyze_feedback(feedback)

        # 优化查询
        optimized_query = optimize_query(original_query, optimization_strategy)

        # 执行优化后的检索
        search_result = hybrid_search.apply_async(
            args=[f"opt_{query_id}", optimized_query, user_id],
            kwargs={'search_options': optimization_strategy}
        ).get()

        # 保存优化结果
        save_optimization_result(query_id, original_query, optimized_query,
                               optimization_strategy, search_result)

        # 发送优化完成通知
        send_optimization_notification(query_id, user_id, optimization_strategy, search_result)

        return {
            'status': 'success',
            'query_id': query_id,
            'original_query': original_query,
            'optimized_query': optimized_query,
            'optimization_strategy': optimization_strategy,
            'search_result': search_result
        }

    except Exception as e:
        logger.error(f"搜索优化失败: {query_id}, 错误: {e}")
        raise


def update_search_progress(query_id: str, progress: int, message: str):
    """更新检索进度"""
    try:
        # 更新Celery任务状态
        current_task.update_state(
            state='PROGRESS',
            meta={
                'current': progress,
                'total': 100,
                'status': message
            }
        )

        # 通过WebSocket推送进度
        asyncio.run(connection_manager.send_user_message({
            'type': MessageType.QUERY_PROGRESS.value,
            'data': {
                'query_id': query_id,
                'progress': progress,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        }, get_user_id_from_query(query_id)))

        # 更新Redis中的进度状态
        progress_data = {
            'query_id': query_id,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        redis_client.setex(f"search_progress:{query_id}", 1800, json.dumps(progress_data))

    except Exception as e:
        logger.error(f"更新检索进度失败: {e}")


def update_batch_search_progress(batch_id: str, user_id: int, progress: int, message: str):
    """更新批量检索进度"""
    try:
        asyncio.run(connection_manager.send_user_message({
            'type': MessageType.QUERY_PROGRESS.value,
            'data': {
                'batch_id': batch_id,
                'progress': progress,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        }, user_id))

    except Exception as e:
        logger.error(f"更新批量检索进度失败: {e}")


def process_query(query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """处理和改写查询"""
    try:
        # 基础处理
        processed_query = {
            'original': query,
            'cleaned': clean_query(query),
            'expanded': expand_query(query),
            'intent': analyze_query_intent(query)
        }

        # 应用选项
        if options:
            processed_query.update({
                'max_results': options.get('max_results', 10),
                'similarity_threshold': options.get('similarity_threshold', 0.7),
                'filters': options.get('filters', {}),
                'boost_factors': options.get('boost_factors', {})
            })

        return processed_query

    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise


def clean_query(query: str) -> str:
    """清理查询文本"""
    import re
    # 移除特殊字符，保留中文、英文、数字
    cleaned = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
    # 合并多个空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def expand_query(query: str) -> List[str]:
    """扩展查询"""
    # 这里可以使用同义词词典或嵌入相似度来扩展查询
    # 简化实现，返回原查询
    return [query]


def analyze_query_intent(query: str) -> str:
    """分析查询意图"""
    # 简化的意图分析
    if any(word in query for word in ['分析', '评估', '对比']):
        return 'analysis'
    elif any(word in query for word in ['是什么', '定义', '解释']):
        return 'definition'
    elif any(word in query for word in ['如何', '怎么', '方法']):
        return 'howto'
    else:
        return 'search'


async def execute_parallel_search(search_tasks: List) -> Tuple:
    """执行并行检索"""
    try:
        results = await asyncio.gather(*search_tasks)
        return results
    except Exception as e:
        logger.error(f"并行检索失败: {e}")
        raise


def perform_vector_search(processed_query: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
    """执行向量检索"""
    try:
        retriever = VectorRetriever()
        search_params = {
            'query': processed_query['cleaned'],
            'top_k': processed_query.get('max_results', 10),
            'threshold': processed_query.get('similarity_threshold', 0.7),
            'filters': processed_query.get('filters', {})
        }

        results = retriever.search(**search_params)
        return results

    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []


def perform_graph_search(processed_query: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
    """执行图谱检索"""
    try:
        retriever = GraphRetriever()
        search_params = {
            'query': processed_query['cleaned'],
            'intent': processed_query.get('intent', 'search'),
            'max_depth': options.get('max_depth', 2) if options else 2,
            'limit': processed_query.get('max_results', 10)
        }

        results = retriever.search(**search_params)
        return results

    except Exception as e:
        logger.error(f"图谱检索失败: {e}")
        return []


def perform_keyword_search(processed_query: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
    """执行关键词检索"""
    try:
        retriever = KeywordRetriever()
        search_params = {
            'query': processed_query['cleaned'],
            'max_results': processed_query.get('max_results', 10),
            'filters': processed_query.get('filters', {})
        }

        results = retriever.search(**search_params)
        return results

    except Exception as e:
        logger.error(f"关键词检索失败: {e}")
        return []


def fuse_search_results(vector_results: List, graph_results: List, keyword_results: List,
                       options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """融合检索结果"""
    try:
        fusioner = ResultFusioner()

        fusion_params = {
            'vector_weight': options.get('vector_weight', 0.4) if options else 0.4,
            'graph_weight': options.get('graph_weight', 0.3) if options else 0.3,
            'keyword_weight': options.get('keyword_weight', 0.3) if options else 0.3,
            'diversity_threshold': options.get('diversity_threshold', 0.7) if options else 0.7,
            'max_results': options.get('max_results', 10) if options else 10
        }

        fused_results = fusioner.fuse_results(
            vector_results, graph_results, keyword_results, **fusion_params
        )

        return fused_results

    except Exception as e:
        logger.error(f"结果融合失败: {e}")
        return []


def build_context(results: List[Dict[str, Any]], processed_query: Dict[str, Any]) -> str:
    """构建上下文"""
    try:
        context_builder = ContextBuilder()

        context_params = {
            'max_length': processed_query.get('max_context_length', 4000),
            'intent': processed_query.get('intent', 'search')
        }

        context = context_builder.build_context(results, **context_params)
        return context

    except Exception as e:
        logger.error(f"上下文构建失败: {e}")
        return ""


def generate_answer(processed_query: Dict[str, Any], context: str) -> str:
    """生成答案"""
    try:
        prompt = f"""
        基于以下上下文回答问题：

        问题：{processed_query['original']}

        上下文：
        {context}

        请提供准确、详细的答案：
        """

        answer = llm_client.generate_response(prompt)
        return answer

    except Exception as e:
        logger.error(f"答案生成失败: {e}")
        return "抱歉，生成答案时出现错误。"


def save_search_result(query_id: str, query: str, results: List[Dict[str, Any]], answer: str, user_id: int):
    """保存搜索结果"""
    try:
        result_data = {
            'query_id': query_id,
            'query': query,
            'answer': answer,
            'results': results,
            'user_id': user_id,
            'created_at': datetime.now()
        }

        # 保存到Redis缓存
        redis_client.setex(f"search_result:{query_id}", 3600, json.dumps(result_data, ensure_ascii=False))

        # 保存到数据库（可选）
        # mysql_client.save_search_result(result_data)

    except Exception as e:
        logger.error(f"保存搜索结果失败: {e}")


def analyze_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """分析用户反馈"""
    try:
        feedback_type = feedback.get('type', 'general')
        rating = feedback.get('rating', 3)

        strategy = {
            'query_expansion': False,
            'weight_adjustment': False,
            'filter_modification': False,
            'similarity_threshold': 0.7
        }

        if feedback_type == 'relevance_low':
            strategy['query_expansion'] = True
            strategy['similarity_threshold'] = 0.6
        elif feedback_type == 'incomplete':
            strategy['weight_adjustment'] = True
            strategy['similarity_threshold'] = 0.5
        elif feedback_type == 'accuracy_issue':
            strategy['filter_modification'] = True
            strategy['similarity_threshold'] = 0.8

        return strategy

    except Exception as e:
        logger.error(f"反馈分析失败: {e}")
        return {}


def optimize_query(original_query: str, strategy: Dict[str, Any]) -> str:
    """优化查询"""
    try:
        optimized_query = original_query

        if strategy.get('query_expansion'):
            # 添加相关关键词
            optimized_query += " 详细分析 数据"

        return optimized_query

    except Exception as e:
        logger.error(f"查询优化失败: {e}")
        return original_query


def save_optimization_result(query_id: str, original_query: str, optimized_query: str,
                           strategy: Dict[str, Any], search_result: Dict[str, Any]):
    """保存优化结果"""
    try:
        optimization_data = {
            'query_id': query_id,
            'original_query': original_query,
            'optimized_query': optimized_query,
            'strategy': strategy,
            'result': search_result,
            'created_at': datetime.now()
        }

        redis_client.setex(f"optimization:{query_id}", 3600, json.dumps(optimization_data, ensure_ascii=False))

    except Exception as e:
        logger.error(f"保存优化结果失败: {e}")


def get_user_id_from_query(query_id: str) -> int:
    """从查询ID获取用户ID"""
    # 这里应该从数据库或缓存中获取
    # 简化实现
    return 1


def send_search_completion_notification(query_id: str, user_id: int, status: str, answer: str = None, results: List = None):
    """发送检索完成通知"""
    try:
        notification_data = {
            'query_id': query_id,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

        if status == 'success':
            message = f"检索完成: {query_id}"
            level = 'success'
            notification_data.update({
                'answer': answer,
                'results_count': len(results) if results else 0
            })
        else:
            message = f"检索失败: {query_id}"
            level = 'error'
            notification_data['error'] = answer

        asyncio.run(connection_manager.send_user_message({
            'type': MessageType.QUERY_COMPLETED.value,
            'data': {
                **notification_data,
                'message': message,
                'level': level
            }
        }, user_id))

    except Exception as e:
        logger.error(f"发送检索完成通知失败: {e}")


def send_batch_search_notification(batch_id: str, user_id: int, results: List[Dict[str, Any]]):
    """发送批量检索完成通知"""
    try:
        successful_count = sum(1 for r in results if r['status'] == 'started')
        failed_count = len(results) - successful_count

        message = f"批量检索完成: 成功 {successful_count} 个，失败 {failed_count} 个"

        asyncio.run(connection_manager.send_user_message({
            'type': MessageType.QUERY_COMPLETED.value,
            'data': {
                'batch_id': batch_id,
                'message': message,
                'total_queries': len(results),
                'successful_count': successful_count,
                'failed_count': failed_count,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        }, user_id))

    except Exception as e:
        logger.error(f"发送批量检索完成通知失败: {e}")


def send_optimization_notification(query_id: str, user_id: int, strategy: Dict[str, Any], search_result: Dict[str, Any]):
    """发送优化完成通知"""
    try:
        asyncio.run(connection_manager.send_user_message({
            'type': MessageType.QUERY_COMPLETED.value,
            'data': {
                'query_id': query_id,
                'optimization_strategy': strategy,
                'optimized_result': search_result,
                'message': '搜索优化完成',
                'timestamp': datetime.now().isoformat()
            }
        }, user_id))

    except Exception as e:
        logger.error(f"发送优化完成通知失败: {e}")