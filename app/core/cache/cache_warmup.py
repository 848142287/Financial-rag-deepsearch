"""
缓存预热
系统启动时预加载热点数据，提高初始性能
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib

from .cache_manager import cache_manager
from .vector_cache import get_vector_cache
from .metadata_cache import get_metadata_cache
from .redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class WarmupPriority(Enum):
    """预热优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WarmupStatus(Enum):
    """预热状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class WarmupTask:
    """预热任务"""
    id: str
    name: str
    priority: WarmupPriority
    function: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: WarmupStatus = WarmupStatus.PENDING
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WarmupStrategy:
    """预热策略"""
    name: str
    description: str
    priority: WarmupPriority
    enabled: bool = True
    schedule: Optional[str] = None  # cron表达式
    conditions: Dict[str, Any] = field(default_factory=dict)


class CacheWarmer:
    """缓存预热器"""

    def __init__(self):
        self.tasks: Dict[str, WarmupTask] = {}
        self.strategies: Dict[str, WarmupStrategy] = {}
        self.running = False
        self._lock = asyncio.Lock()
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_items_loaded': 0,
            'cache_hits_after_warmup': 0,
            'warmup_time': 0.0
        }

    def register_strategy(self, strategy: WarmupStrategy) -> None:
        """注册预热策略"""
        self.strategies[strategy.name] = strategy
        logger.info(f"注册预热策略: {strategy.name}")

    def register_task(self, task: WarmupTask) -> None:
        """注册预热任务"""
        self.tasks[task.id] = task
        self._stats['total_tasks'] += 1
        logger.info(f"注册预热任务: {task.name}")

    def create_task(self, name: str, function: Callable, priority: WarmupPriority,
                   params: Dict[str, Any] = None, dependencies: List[str] = None,
                   max_retries: int = 3) -> str:
        """创建预热任务"""
        task_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()

        task = WarmupTask(
            id=task_id,
            name=name,
            priority=priority,
            function=function,
            params=params or {},
            dependencies=dependencies or [],
            max_retries=max_retries
        )

        self.register_task(task)
        return task_id

    async def warmup_all(self, force: bool = False) -> Dict[str, Any]:
        """执行所有预热任务"""
        if self.running and not force:
            logger.warning("预热正在运行中")
            return {'status': 'already_running'}

        start_time = datetime.now()
        self.running = True

        try:
            logger.info("开始缓存预热")

            # 按优先级排序任务
            sorted_tasks = self._get_sorted_tasks()

            # 执行任务
            results = []
            for task in sorted_tasks:
                if task.status == WarmupStatus.COMPLETED and not force:
                    continue

                result = await self._execute_task(task)
                results.append(result)

            # 更新统计
            end_time = datetime.now()
            self._stats['warmup_time'] = (end_time - start_time).total_seconds()
            self._stats['completed_tasks'] = len([t for t in self.tasks.values() if t.status == WarmupStatus.COMPLETED])
            self._stats['failed_tasks'] = len([t for t in self.tasks.values() if t.status == WarmupStatus.FAILED])

            logger.info(f"缓存预热完成，耗时: {self._stats['warmup_time']:.2f}秒")

            return {
                'status': 'completed',
                'total_tasks': len(self.tasks),
                'completed_tasks': self._stats['completed_tasks'],
                'failed_tasks': self._stats['failed_tasks'],
                'warmup_time': self._stats['warmup_time'],
                'results': results
            }

        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
        finally:
            self.running = False

    async def warmup_by_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """按策略执行预热"""
        if strategy_name not in self.strategies:
            raise ValueError(f"未知的预热策略: {strategy_name}")

        strategy = self.strategies[strategy_name]
        if not strategy.enabled:
            return {'status': 'disabled', 'strategy': strategy_name}

        logger.info(f"按策略执行预热: {strategy_name}")

        # 执行策略相关的任务
        strategy_tasks = [task for task in self.tasks.values()
                         if strategy.name in task.name or
                          any(dep in task.name for dep in strategy.dependencies)]

        results = []
        for task in strategy_tasks:
            result = await self._execute_task(task)
            results.append(result)

        return {
            'status': 'completed',
            'strategy': strategy_name,
            'tasks_executed': len(strategy_tasks),
            'results': results
        }

    async def _execute_task(self, task: WarmupTask) -> Dict[str, Any]:
        """执行预热任务"""
        async with self._lock:
            task.status = WarmupStatus.RUNNING
            task.start_time = datetime.now()
            task.error_message = None

        try:
            logger.info(f"执行预热任务: {task.name}")

            # 检查依赖
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"依赖任务不存在: {dep_id}")

                dep_task = self.tasks[dep_id]
                if dep_task.status != WarmupStatus.COMPLETED:
                    logger.warning(f"跳过任务 {task.name}，依赖 {dep_id} 未完成")
                    task.status = WarmupStatus.PENDING
                    return {
                        'task_id': task.id,
                        'status': 'skipped',
                        'reason': 'dependency_not_completed'
                    }

            # 执行任务函数
            if asyncio.iscoroutinefunction(task.function):
                result = await task.function(**task.params)
            else:
                result = task.function(**task.params)

            task.result = result
            task.status = WarmupStatus.COMPLETED
            task.progress = 100.0

            # 更新统计
            if isinstance(result, dict) and 'items_loaded' in result:
                self._stats['total_items_loaded'] += result['items_loaded']

            logger.info(f"预热任务完成: {task.name}")

            return {
                'task_id': task.id,
                'status': 'completed',
                'result': result
            }

        except Exception as e:
            task.retry_count += 1
            task.error_message = str(e)

            if task.retry_count <= task.max_retries:
                task.status = WarmupStatus.PENDING
                logger.warning(f"预热任务失败，将重试: {task.name}, 错误: {e}")
            else:
                task.status = WarmupStatus.FAILED
                logger.error(f"预热任务最终失败: {task.name}, 错误: {e}")

            return {
                'task_id': task.id,
                'status': 'failed',
                'error': str(e),
                'retry_count': task.retry_count
            }

        finally:
            task.end_time = datetime.now()

    def _get_sorted_tasks(self) -> List[WarmupTask]:
        """按优先级排序任务"""
        priority_order = {
            WarmupPriority.CRITICAL: 0,
            WarmupPriority.HIGH: 1,
            WarmupPriority.MEDIUM: 2,
            WarmupPriority.LOW: 3
        }

        return sorted(
            self.tasks.values(),
            key=lambda t: priority_order[t.priority]
        )

    async def preload_frequent_queries(self, query_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预加载频繁查询"""
        logger.info("预加载频繁查询")

        loaded_count = 0
        for query_info in query_data:
            try:
                query = query_info.get('query')
                params = query_info.get('params', {})
                frequency = query_info.get('frequency', 1)

                # 根据频率决定预加载策略
                if frequency >= 10:  # 高频查询，强制预加载
                    # 这里调用实际的搜索API并缓存结果
                    result = await self._execute_search(query, params)
                    if result:
                        loaded_count += 1

            except Exception as e:
                logger.warning(f"预加载查询失败 {query_info.get('query', 'unknown')}: {e}")

        return {
            'items_loaded': loaded_count,
            'total_queries': len(query_data)
        }

    async def preload_hot_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """预加载热点文档"""
        logger.info("预加载热点文档")

        loaded_count = 0
        for doc_id in doc_ids:
            try:
                # 预加载文档元数据
                metadata_cache = get_metadata_cache()
                metadata = await metadata_cache.get_document(doc_id)
                if metadata:
                    loaded_count += 1

                # 预加载文档内容
                # 这里应该调用实际的文档服务
                # content = await document_service.get_content(doc_id)

            except Exception as e:
                logger.warning(f"预加载文档失败 {doc_id}: {e}")

        return {
            'items_loaded': loaded_count,
            'total_documents': len(doc_ids)
        }

    async def preload_vector_index(self, index_name: str, sample_size: int = 1000) -> Dict[str, Any]:
        """预加载向量索引"""
        logger.info(f"预加载向量索引: {index_name}")

        try:
            # 生成示例查询向量
            sample_vectors = []
            for i in range(sample_size):
                # 这里应该根据实际向量维度生成
                vector = [random.random() for _ in range(768)]  # 假设768维
                sample_vectors.append(vector)

            # 执行示例搜索以预热缓存
            vector_cache = get_vector_cache()
            loaded_count = 0

            for vector in sample_vectors:
                search_params = {'top_k': 10, 'metric_type': 'cosine'}
                result = await vector_cache.get(vector, search_params)
                if result:
                    loaded_count += 1

            return {
                'items_loaded': loaded_count,
                'sample_size': sample_size,
                'index_name': index_name
            }

        except Exception as e:
            logger.error(f"预加载向量索引失败 {index_name}: {e}")
            return {
                'items_loaded': 0,
                'error': str(e)
            }

    async def preload_entity_relationships(self, entity_types: List[str]) -> Dict[str, Any]:
        """预加载实体关系"""
        logger.info("预加载实体关系")

        loaded_count = 0
        metadata_cache = get_metadata_cache()

        for entity_type in entity_types:
            try:
                # 预加载指定类型的实体
                entities = await metadata_cache.search_metadata(
                    data_type=metadata_cache.MetadataType.ENTITY,
                    filters={'type': entity_type},
                    limit=1000
                )

                loaded_count += len(entities)

                # 预加载这些实体的关系
                for entity_data in entities:
                    entity_id = entity_data['data'].id
                    relationships = await metadata_cache.search_metadata(
                        data_type=metadata_cache.MetadataType.RELATIONSHIP,
                        filters={'source_entity_id': entity_id},
                        limit=100
                    )
                    loaded_count += len(relationships)

            except Exception as e:
                logger.warning(f"预加载实体关系失败 {entity_type}: {e}")

        return {
            'items_loaded': loaded_count,
            'entity_types': entity_types
        }

    async def _execute_search(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """执行搜索（模拟）"""
        # 这里应该调用实际的搜索服务
        # 暂时返回模拟结果
        await asyncio.sleep(0.01)  # 模拟搜索延迟
        return {'results': [], 'query': query}

    def get_stats(self) -> Dict[str, Any]:
        """获取预热统计信息"""
        return {
            'total_tasks': self._stats['total_tasks'],
            'completed_tasks': self._stats['completed_tasks'],
            'failed_tasks': self._stats['failed_tasks'],
            'total_items_loaded': self._stats['total_items_loaded'],
            'cache_hits_after_warmup': self._stats['cache_hits_after_warmup'],
            'warmup_time': self._stats['warmup_time'],
            'task_statuses': {
                task.name: task.status.value for task in self.tasks.values()
            },
            'registered_strategies': list(self.strategies.keys())
        }

    async def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            'total_tasks': len(self.tasks),
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_items_loaded': 0,
            'cache_hits_after_warmup': 0,
            'warmup_time': 0.0
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            'id': task.id,
            'name': task.name,
            'status': task.status.value,
            'progress': task.progress,
            'start_time': task.start_time.isoformat() if task.start_time else None,
            'end_time': task.end_time.isoformat() if task.end_time else None,
            'error_message': task.error_message,
            'retry_count': task.retry_count,
            'max_retries': task.max_retries
        }


# 预定义的预热策略
class DefaultWarmupStrategies:
    """默认预热策略"""

    @staticmethod
    def get_critical_strategy() -> WarmupStrategy:
        """关键策略 - 系统核心功能"""
        return WarmupStrategy(
            name="critical",
            description="预热系统核心功能相关缓存",
            priority=WarmupPriority.CRITICAL,
            conditions={
                "system_load": "low",
                "time_window": "startup"
            }
        )

    @staticmethod
    def get_user_activity_strategy() -> WarmupStrategy:
        """用户活动策略 - 基于用户行为"""
        return WarmupStrategy(
            name="user_activity",
            description="基于历史用户行为预热热点数据",
            priority=WarmupPriority.HIGH,
            schedule="0 2 * * *",  # 每天凌晨2点
            conditions={
                "min_data_size": 1000,
                "recency_days": 7
            }
        )

    @staticmethod
    def get_business_hours_strategy() -> WarmupStrategy:
        """工作时间策略 - 业务高峰期预热"""
        return WarmupStrategy(
            name="business_hours",
            description="工作时间前预热热点数据",
            priority=WarmupPriority.MEDIUM,
            schedule="0 8 * * 1-5",  # 工作日上午8点
            conditions={
                "business_days_only": True
            }
        )


# 全局缓存预热器实例
cache_warmer: Optional[CacheWarmer] = None


def get_cache_warmer() -> CacheWarmer:
    """获取缓存预热器实例"""
    global cache_warmer

    if cache_warmer is None:
        cache_warmer = CacheWarmer()

        # 注册默认策略
        cache_warmer.register_strategy(DefaultWarmupStrategies.get_critical_strategy())
        cache_warmer.register_strategy(DefaultWarmupStrategies.get_user_activity_strategy())
        cache_warmer.register_strategy(DefaultWarmupStrategies.get_business_hours_strategy())

    return cache_warmer