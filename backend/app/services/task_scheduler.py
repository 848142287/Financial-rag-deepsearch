"""
异步任务调度系统
基于Celery的任务队列管理，支持优先级、重试、监控等功能
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

import redis.asyncio as redis
from celery import Celery, chain, chord, group
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure, task_success
import psutil

from app.core.config import settings
from app.services.upload_service import TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class TaskMetrics:
    """任务指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    active_tasks: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    throughput: float = 0.0  # tasks per minute


@dataclass
class WorkerInfo:
    """工作节点信息"""
    worker_id: str
    hostname: str
    pid: int
    status: WorkerStatus
    active_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)


class TaskScheduler:
    """任务调度器"""

    def __init__(self):
        self.celery_app = None
        self.redis_client = None
        self.workers: Dict[str, WorkerInfo] = {}
        self.metrics = TaskMetrics()
        self._running = False

    async def initialize(self):
        """初始化调度器"""
        # 初始化Celery
        self.celery_app = Celery(
            "document_processor",
            broker=settings.REDIS_URL,
            backend=settings.REDIS_URL,
            include=[
                "app.tasks.document_tasks",
                "app.tasks.analysis_tasks",
                "app.tasks.vector_tasks",
                "app.tasks.knowledge_tasks"
            ]
        )

        # 配置Celery
        self._configure_celery()

        # 初始化Redis客户端
        self.redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

        # 注册信号处理器
        self._register_signals()

        logger.info("Task scheduler initialized")

    def _configure_celery(self):
        """配置Celery"""
        # 任务路由配置
        self.celery_app.conf.task_routes = {
            "app.tasks.document_tasks.*": {"queue": "document_processing"},
            "app.tasks.analysis_tasks.*": {"queue": "analysis"},
            "app.tasks.vector_tasks.*": {"queue": "vectorization"},
            "app.tasks.knowledge_tasks.*": {"queue": "knowledge_graph"}
        }

        # 队列优先级配置
        self.celery_app.conf.task_queue_max_priority = 10
        self.celery_app.conf.worker_prefetch_multiplier = 1
        self.celery_app.conf.task_acks_late = True

        # 任务重试配置
        self.celery_app.conf.task_reject_on_worker_lost = True
        self.celery_app.conf.task_default_max_retries = 3
        self.celery_app.conf.task_default_retry_delay = 60
        self.celery_app.task_autoretry_for = (Exception,)

        # 任务结果过期
        self.celery_app.conf.result_expires = 3600  # 1小时

        # 工作节点配置
        self.celery_app.conf.worker_max_tasks_per_child = 50
        self.celery_app.conf.worker_disable_rate_limits = False

        # 监控配置
        self.celery_app.conf.worker_send_task_events = True
        self.celery_app.conf.task_send_sent_event = True

        # 时区设置
        self.celery_app.conf.timezone = "UTC"
        self.celery_app.conf.enable_utc = True

    def _register_signals(self):
        """注册Celery信号处理器"""
        @task_prerun.connect
        def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
            """任务开始前处理"""
            logger.info(f"Task {task_id} started: {task.name}")

        @task_postrun.connect
        def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
            """任务完成后处理"""
            logger.info(f"Task {task_id} completed with state: {state}")

        @task_failure.connect
        def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
            """任务失败处理"""
            logger.error(f"Task {task_id} failed: {exception}")

        @task_success.connect
        def task_success_handler(sender=None, result=None, task_id=None, args=None, kwargs=None, **kwds):
            """任务成功处理"""
            logger.info(f"Task {task_id} succeeded")

    async def start_monitoring(self):
        """启动监控"""
        self._running = True
        await asyncio.gather(
            self._monitor_workers(),
            self._collect_metrics(),
            self._process_delayed_tasks()
        )

    async def stop_monitoring(self):
        """停止监控"""
        self._running = False

    async def _monitor_workers(self):
        """监控工作节点"""
        while self._running:
            try:
                # 获取活跃的工作节点
                inspect = self.celery_app.control.inspect()
                stats = inspect.stats()

                if stats:
                    for worker_name, worker_stats in stats.items():
                        await self._update_worker_info(worker_name, worker_stats)

                # 检查离线工作节点
                await self._check_offline_workers()

                await asyncio.sleep(30)  # 30秒检查一次

            except Exception as e:
                logger.error(f"Worker monitoring error: {e}")
                await asyncio.sleep(60)

    async def _update_worker_info(self, worker_name: str, worker_stats: dict):
        """更新工作节点信息"""
        try:
            # 获取进程信息
            pid = worker_stats.get("pid")
            process = psutil.Process(pid) if pid else None

            worker_info = self.workers.get(worker_name, WorkerInfo(
                worker_id=worker_name,
                hostname=worker_stats.get("hostname", ""),
                pid=pid or 0,
                status=WorkerStatus.IDLE
            ))

            # 更新信息
            worker_info.memory_usage = process.memory_info().rss / 1024 / 1024 if process else 0
            worker_info.cpu_usage = process.cpu_percent() if process else 0
            worker_info.last_heartbeat = datetime.utcnow()

            # 获取活跃任务
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            if active_tasks and worker_name in active_tasks:
                worker_info.active_tasks = [task["id"] for task in active_tasks[worker_name]]
                worker_info.status = WorkerStatus.BUSY
            else:
                worker_info.active_tasks = []
                worker_info.status = WorkerStatus.IDLE

            self.workers[worker_name] = worker_info

        except Exception as e:
            logger.error(f"Error updating worker {worker_name}: {e}")

    async def _check_offline_workers(self):
        """检查离线工作节点"""
        now = datetime.utcnow()
        offline_threshold = timedelta(minutes=5)

        for worker_id, worker_info in self.workers.items():
            if now - worker_info.last_heartbeat > offline_threshold:
                if worker_info.status != WorkerStatus.OFFLINE:
                    worker_info.status = WorkerStatus.OFFLINE
                    logger.warning(f"Worker {worker_id} is offline")

    async def _collect_metrics(self):
        """收集指标"""
        while self._running:
            try:
                inspect = self.celery_app.control.inspect()

                # 获取任务统计
                stats = inspect.stats()
                if stats:
                    total_active = 0
                    total_completed = 0

                    for worker_stats in stats.values():
                        total_active += worker_stats.get("pool", {}).get("max-concurrency", 0)
                        total_completed += worker_stats.get("total", {})

                    self.metrics.active_tasks = total_active
                    # 这里可以添加更多指标计算逻辑

                # 从Redis获取队列状态
                await self._update_queue_metrics()

                await asyncio.sleep(60)  # 1分钟更新一次

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)

    async def _update_queue_metrics(self):
        """更新队列指标"""
        try:
            # 获取各队列长度
            queues = ["document_processing", "analysis", "vectorization", "knowledge_graph"]
            total_pending = 0

            for queue in queues:
                queue_length = await self.redis_client.llen(queue)
                total_pending += queue_length

            self.metrics.pending_tasks = total_pending

        except Exception as e:
            logger.error(f"Queue metrics error: {e}")

    async def _process_delayed_tasks(self):
        """处理延迟任务"""
        while self._running:
            try:
                # 获取到期的延迟任务
                current_time = datetime.utcnow().timestamp()
                expired_tasks = await self.redis_client.zrangebyscore(
                    "delayed_tasks",
                    0,
                    current_time
                )

                for task_id in expired_tasks:
                    # 获取任务详情
                    task_data = await self.redis_client.hgetall(f"task:{task_id}")
                    if task_data:
                        # 重新提交到队列
                        priority = TaskPriority(int(task_data.get("priority", TaskPriority.NORMAL)))
                        queue_name = f"document_parse:{priority.name.lower()}"

                        await self.redis_client.lpush(queue_name, task_id)

                        # 从延迟队列中移除
                        await self.redis_client.zrem("delayed_tasks", task_id)

                        logger.info(f"Delayed task {task_id} moved to queue {queue_name}")

                await asyncio.sleep(10)  # 10秒检查一次

            except Exception as e:
                logger.error(f"Delayed tasks processing error: {e}")
                await asyncio.sleep(30)

    async def submit_workflow(
        self,
        document_id: str,
        file_path: str,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """提交完整的工作流"""
        try:
            # 定义工作流链
            workflow = chain(
                # 1. 文档解析
                self.celery_app.signature(
                    "app.tasks.document_tasks.parse_document",
                    args=[document_id, file_path],
                    options={"queue": "document_processing", "priority": priority.value}
                ),
                # 2. 内容分析
                self.celery_app.signature(
                    "app.tasks.analysis_tasks.analyze_content",
                    options={"queue": "analysis", "priority": priority.value}
                ),
                # 3. 向量化
                self.celery_app.signature(
                    "app.tasks.vector_tasks.vectorize_content",
                    options={"queue": "vectorization", "priority": priority.value}
                ),
                # 4. 知识图谱构建
                self.celery_app.signature(
                    "app.tasks.knowledge_tasks.build_knowledge_graph",
                    options={"queue": "knowledge_graph", "priority": priority.value}
                )
            )

            # 执行工作流
            result = workflow.apply_async()
            workflow_id = str(uuid.uuid4())

            # 存储工作流信息
            await self.redis_client.hset(
                f"workflow:{workflow_id}",
                mapping={
                    "document_id": document_id,
                    "celery_task_id": result.id,
                    "status": "running",
                    "created_at": datetime.utcnow().isoformat()
                }
            )

            logger.info(f"Workflow {workflow_id} submitted for document {document_id}")
            return workflow_id

        except Exception as e:
            logger.error(f"Error submitting workflow: {e}")
            raise

    async def get_workflow_status(self, workflow_id: str) -> dict:
        """获取工作流状态"""
        try:
            # 获取工作流信息
            workflow_data = await self.redis_client.hgetall(f"workflow:{workflow_id}")
            if not workflow_data:
                return {"status": "not_found"}

            # 获取Celery任务状态
            celery_task_id = workflow_data.get("celery_task_id")
            if celery_task_id:
                result = AsyncResult(celery_task_id, app=self.celery_app)
                workflow_data["celery_status"] = result.state
                workflow_data["result"] = result.result if result.ready() else None

            return workflow_data

        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {"status": "error", "error": str(e)}

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        try:
            # 获取工作流信息
            workflow_data = await self.redis_client.hgetall(f"workflow:{workflow_id}")
            if not workflow_data:
                return False

            # 取消Celery任务
            celery_task_id = workflow_data.get("celery_task_id")
            if celery_task_id:
                self.celery_app.control.revoke(celery_task_id, terminate=True)

            # 更新状态
            await self.redis_client.hset(
                f"workflow:{workflow_id}",
                mapping={
                    "status": "cancelled",
                    "cancelled_at": datetime.utcnow().isoformat()
                }
            )

            logger.info(f"Workflow {workflow_id} cancelled")
            return True

        except Exception as e:
            logger.error(f"Error cancelling workflow: {e}")
            return False

    async def scale_workers(self, target_workers: int):
        """扩缩容工作节点"""
        try:
            current_workers = len(self.workers)

            if target_workers > current_workers:
                # 扩容
                scale_up = target_workers - current_workers
                logger.info(f"Scaling up workers by {scale_up}")

                # 这里应该调用容器编排API (如Kubernetes)
                # 简化实现，记录日志
                logger.info(f"Would scale up to {target_workers} workers")

            elif target_workers < current_workers:
                # 缩容
                scale_down = current_workers - target_workers
                logger.info(f"Scaling down workers by {scale_down}")

                # 优雅关闭空闲工作节点
                idle_workers = [
                    worker_id for worker_id, worker_info in self.workers.items()
                    if worker_info.status == WorkerStatus.IDLE
                ]

                for worker_id in idle_workers[:scale_down]:
                    self.celery_app.control.shutdown(worker_id)

        except Exception as e:
            logger.error(f"Error scaling workers: {e}")

    def get_workers_status(self) -> List[WorkerInfo]:
        """获取所有工作节点状态"""
        return list(self.workers.values())

    def get_metrics(self) -> TaskMetrics:
        """获取任务指标"""
        return self.metrics


# 全局调度器实例
task_scheduler = TaskScheduler()


async def get_task_scheduler() -> TaskScheduler:
    """获取任务调度器实例"""
    if not task_scheduler.celery_app:
        await task_scheduler.initialize()
    return task_scheduler