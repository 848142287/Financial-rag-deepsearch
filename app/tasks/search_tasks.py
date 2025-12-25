"""
搜索处理异步任务
"""

from celery import current_task
from app.core.async_tasks.celery_app import celery_app
import logging
import asyncio
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import time

from app.services.agentic_rag.plan_phase import PlanPhase
from app.services.agentic_rag.execute_phase import ExecutePhase
from app.services.agentic_rag.generation_phase import GenerationPhase
from app.services.fusion_agent.agents import WorkerCoordinator
from app.services.websocket_service import WebSocketManager
from app.core.redis_client import redis_client
from app.core.database import SessionLocal
from app.models.search import SearchLog, SearchResult
from app.models.conversation import Conversation

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.search_tasks.agentic_search")
def agentic_search(self, query: str, user_id: Optional[str] = None, conversation_id: Optional[str] = None,
                  search_mode: str = "enhanced", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Agentic RAG搜索任务
    """
    task_id = self.request.id
    start_time = time.time()
    logger.info(f"Starting agentic search task {task_id} for query: {query[:50]}...")

    websocket_manager = WebSocketManager()

    try:
        # 发送开始通知
        websocket_manager.send_message(
            "search_processing",
            {
                "task_id": task_id,
                "query": query,
                "status": "started",
                "progress": 0,
                "message": "开始智能检索..."
            }
        )

        # 1. 计划阶段
        self.update_state(
            state="PROGRESS",
            meta={"current": 10, "total": 100, "status": "分析查询意图..."}
        )

        plan_phase = PlanPhase()
        query_plan = await run_plan_phase_async(plan_phase, query, options or {})

        websocket_manager.send_message(
            "search_processing",
            {
                "task_id": task_id,
                "status": "planning",
                "progress": 20,
                "message": f"查询类型: {query_plan.query_type.value}",
                "query_plan": query_plan.to_dict()
            }
        )

        # 2. 执行阶段
        self.update_state(
            state="PROGRESS",
            meta={"current": 30, "total": 100, "status": "并行检索中..."}
        )

        execute_phase = ExecutePhase()
        search_results = await run_execute_phase_async(execute_phase, query_plan, search_mode)

        websocket_manager.send_message(
            "search_processing",
            {
                "task_id": task_id,
                "status": "executing",
                "progress": 60,
                "message": f"检索到 {len(search_results.fused_results)} 个结果",
                "results_count": len(search_results.fused_results)
            }
        )

        # 3. 生成阶段
        self.update_state(
            state="PROGRESS",
            meta={"current": 70, "total": 100, "status": "生成答案..."}
        )

        generation_phase = GenerationPhase()
        answer = await run_generation_phase_async(generation_phase, query, search_results, query_plan)

        websocket_manager.send_message(
            "search_processing",
            {
                "task_id": task_id,
                "status": "generating",
                "progress": 90,
                "message": "答案生成完成"
            }
        )

        # 4. 记录搜索日志
        response_time = (time.time() - start_time) * 1000
        await log_search_async(query, answer, search_results, response_time, user_id, conversation_id)

        # 5. 缓存结果
        cache_key = f"search:{hash(query)}:{search_mode}"
        cache_data = {
            "query": query,
            "answer": answer.to_dict(),
            "results": [r.to_dict() for r in search_results.fused_results],
            "plan": query_plan.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "response_time": response_time
        }
        redis_client.setex(cache_key, 1800, json.dumps(cache_data))  # 缓存30分钟

        # 完成搜索
        result = {
            "query": query,
            "answer": answer.to_dict(),
            "results": [r.to_dict() for r in search_results.fused_results],
            "sources": [r.get("metadata", {}).get("source", "未知") for r in search_results.fused_results],
            "response_time_ms": response_time,
            "query_plan": query_plan.to_dict(),
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        websocket_manager.send_message(
            "search_processing",
            {
                "task_id": task_id,
                "status": "completed",
                "progress": 100,
                "message": "搜索完成",
                "result": result
            }
        )

        logger.info(f"Agentic search completed in {response_time:.2f}ms")
        return result

    except Exception as e:
        logger.error(f"Agentic search failed: {str(e)}")

        websocket_manager.send_message(
            "search_processing",
            {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
        )

        raise


@celery_app.task(bind=True, name="app.tasks.search_tasks.fusion_agent_search")
def fusion_agent_search(self, query: str, user_id: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fusion Agent搜索任务
    """
    task_id = self.request.id
    start_time = time.time()
    logger.info(f"Starting fusion agent search task {task_id} for query: {query[:50]}...")

    websocket_manager = WebSocketManager()

    try:
        # 发送开始通知
        websocket_manager.send_message(
            "fusion_search",
            {
                "task_id": task_id,
                "query": query,
                "status": "started",
                "progress": 0,
                "message": "启动融合智能体..."
            }
        )

        # 初始化WorkerCoordinator
        coordinator = WorkerCoordinator()

        # 执行搜索流程
        self.update_state(
            state="PROGRESS",
            meta={"current": 20, "total": 100, "status": "任务分解..."}
        )

        websocket_manager.send_message(
            "fusion_search",
            {
                "task_id": task_id,
                "status": "task_decomposition",
                "progress": 20,
                "message": "分解搜索任务..."
            }
        )

        # 执行搜索
        result = await run_fusion_search_async(coordinator, query, context or {})

        # 记录响应时间
        response_time = (time.time() - start_time) * 1000

        # 保存结果
        final_result = {
            "query": query,
            "answer": result.get("final_answer", ""),
            "reasoning": result.get("reasoning_chain", []),
            "evidence": result.get("evidence_tracking", {}),
            "response_time_ms": response_time,
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        websocket_manager.send_message(
            "fusion_search",
            {
                "task_id": task_id,
                "status": "completed",
                "progress": 100,
                "message": "融合搜索完成",
                "result": final_result
            }
        )

        logger.info(f"Fusion agent search completed in {response_time:.2f}ms")
        return final_result

    except Exception as e:
        logger.error(f"Fusion agent search failed: {str(e)}")

        websocket_manager.send_message(
            "fusion_search",
            {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
        )

        raise


@celery_app.task(bind=True, name="app.tasks.search_tasks.batch_search")
def batch_search(self, queries: List[str], search_mode: str = "enhanced") -> Dict[str, Any]:
    """
    批量搜索任务
    """
    task_id = self.request.id
    logger.info(f"Starting batch search for {len(queries)} queries")

    results = []
    failed_queries = []

    for i, query in enumerate(queries):
        try:
            # 触发单个搜索任务
            task = agentic_search.delay(query, search_mode=search_mode)
            results.append({
                "query": query,
                "task_id": task.id,
                "status": "started"
            })

            # 更新进度
            progress = int((i + 1) / len(queries) * 100)
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": i + 1,
                    "total": len(queries),
                    "progress": progress,
                    "status": f"处理第 {i + 1}/{len(queries)} 个查询..."
                }
            )

        except Exception as e:
            logger.error(f"Failed to start search for query '{query}': {str(e)}")
            failed_queries.append({
                "query": query,
                "error": str(e)
            })

    return {
        "task_id": task_id,
        "total_queries": len(queries),
        "successful": len(results),
        "failed": len(failed_queries),
        "results": results,
        "failed_queries": failed_queries
    }


@celery_app.task(bind=True, name="app.tasks.search_tasks.search_recommendations")
def generate_search_recommendations(self, user_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    生成搜索推荐
    """
    logger.info(f"Generating search recommendations for user {user_id}")

    try:
        # 获取用户搜索历史
        recent_searches = get_user_search_history(user_id, limit=20)

        # 分析用户兴趣模式
        interests = analyze_user_interests(recent_searches)

        # 生成推荐查询
        recommendations = generate_recommendations(interests, limit)

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "interests": interests,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to generate recommendations for user {user_id}: {str(e)}")
        raise


@celery_app.task(name="app.tasks.search_tasks.cleanup_search_cache")
def cleanup_search_cache():
    """
    清理搜索缓存
    """
    logger.info("Starting search cache cleanup")

    try:
        # 获取所有搜索缓存键
        cache_keys = redis_client.keys("search:*")
        cleaned_count = 0

        for key in cache_keys:
            # 检查缓存是否过期
            ttl = redis_client.ttl(key)
            if ttl == -1:  # 没有过期时间的键
                # 检查内容的时间戳
                data = redis_client.get(key)
                if data:
                    try:
                        cache_data = json.loads(data)
                        timestamp = datetime.fromisoformat(cache_data.get("timestamp", ""))
                        # 如果缓存超过7天，删除
                        if datetime.utcnow() - timestamp > timedelta(days=7):
                            redis_client.delete(key)
                            cleaned_count += 1
                    except:
                        # 无法解析的缓存，直接删除
                        redis_client.delete(key)
                        cleaned_count += 1
            elif ttl == -2:  # 已过期的键
                redis_client.delete(key)
                cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        return {"cleaned_count": cleaned_count}

    except Exception as e:
        logger.error(f"Cache cleanup failed: {str(e)}")
        raise


# 异步辅助函数
async def run_plan_phase_async(plan_phase: PlanPhase, query: str, options: Dict[str, Any]):
    """异步运行计划阶段"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, plan_phase.process_query, query, options)


async def run_execute_phase_async(execute_phase: ExecutePhase, query_plan, search_mode: str):
    """异步运行执行阶段"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, execute_phase.execute, query_plan, search_mode)


async def run_generation_phase_async(generation_phase: GenerationPhase, query: str,
                                   search_results, query_plan):
    """异步运行生成阶段"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generation_phase.generate_answer,
                                     query, search_results, query_plan)


async def run_fusion_search_async(coordinator: WorkerCoordinator, query: str, context: Dict[str, Any]):
    """异步运行融合搜索"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, coordinator.coordinate_search, query, context)


async def log_search_async(query: str, answer, search_results, response_time: float,
                          user_id: Optional[str], conversation_id: Optional[str]):
    """异步记录搜索日志"""
    db = SessionLocal()
    try:
        # 创建搜索日志
        search_log = SearchLog(
            query=query,
            answer=answer.content,
            response_time_ms=response_time,
            results_count=len(search_results.fused_results),
            user_id=user_id,
            conversation_id=conversation_id,
            created_at=datetime.utcnow()
        )
        db.add(search_log)
        db.commit()

        # 保存搜索结果
        for i, result in enumerate(search_results.fused_results):
            search_result = SearchResult(
                search_log_id=search_log.id,
                rank=i + 1,
                content=result.get("content", ""),
                source=result.get("metadata", {}).get("source", "未知"),
                score=result.get("score", 0),
                metadata=result.get("metadata", {})
            )
            db.add(search_result)

        db.commit()

    finally:
        db.close()


def get_user_search_history(user_id: str, limit: int) -> List[Dict[str, Any]]:
    """获取用户搜索历史"""
    db = SessionLocal()
    try:
        logs = db.query(SearchLog).filter(
            SearchLog.user_id == user_id
        ).order_by(SearchLog.created_at.desc()).limit(limit).all()

        return [
            {
                "query": log.query,
                "timestamp": log.created_at.isoformat(),
                "results_count": log.results_count
            }
            for log in logs
        ]
    finally:
        db.close()


def analyze_user_interests(search_history: List[Dict[str, Any]]) -> List[str]:
    """分析用户兴趣"""
    # 简单的关键词提取
    # 实际应用中可以使用更复杂的NLP技术
    interests = []
    keywords = {}

    for search in search_history:
        query = search["query"]
        # 提取关键词（简化版）
        words = query.split()
        for word in words:
            if len(word) > 1:  # 过滤单字
                keywords[word] = keywords.get(word, 0) + 1

    # 获取高频关键词
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    interests = [word for word, count in sorted_keywords[:10]]

    return interests


def generate_recommendations(interests: List[str], limit: int) -> List[Dict[str, Any]]:
    """生成搜索推荐"""
    recommendations = []

    # 金融领域的推荐模板
    templates = [
        "最近的{interest}趋势如何？",
        "{interest}的投资机会分析",
        "关于{interest}的最新研报",
        "{interest}行业的发展前景",
        "如何评估{interest}的投资价值？"
    ]

    for i, interest in enumerate(interests[:limit]):
        if i < len(templates):
            template = templates[i % len(templates)]
            recommendations.append({
                "query": template.format(interest=interest),
                "reason": f"基于您对'{interest}'的兴趣",
                "type": "personalized"
            })

    return recommendations