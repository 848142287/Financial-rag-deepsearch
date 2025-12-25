"""
批处理和异步多任务处理模块
提供高效的批处理、任务队列、并发控制和结果聚合功能
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Union, Tuple,
    AsyncGenerator, Awaitable, TypeVar, Generic
)
import uuid
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiofiles.os
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchTask:
    """批处理任务"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    max_retries: int = 3
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """批处理配置"""
    batch_size: int = 10
    max_concurrent_batches: int = 5
    timeout_per_task: float = 300.0
    retry_delay: float = 1.0
    max_retries: int = 3
    enable_progress_callback: bool = True
    result_aggregation_strategy: str = "collect_all"  # collect_all, first_success, best_result
    error_handling_strategy: str = "continue"  # continue, stop_on_error, retry_failed


class BaseBatchProcessor(ABC, Generic[T, R]):
    """批处理器基类"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_batches)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_batches)

    @abstractmethod
    async def process_item(self, item: T) -> R:
        """处理单个项目"""
        pass

    async def process_batch(self, items: List[T]) -> List[R]:
        """处理批次"""
        tasks = []
        for item in items:
            task = self.process_item(item)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing item {i}: {str(result)}")
                if self.config.error_handling_strategy == "stop_on_error":
                    raise result
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    async def process_all(self, items: List[T]) -> List[R]:
        """处理所有项目"""
        all_results = []

        # 分批处理
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)

            # 进度回调
            if self.config.enable_progress_callback:
                progress = (i + len(batch)) / len(items)
                await self._on_progress(progress, len(all_results), len(items))

        return all_results

    async def _on_progress(self, progress: float, completed: int, total: int):
        """进度回调"""
        logger.info(f"Progress: {progress:.2%} ({completed}/{total})")


class AsyncBatchProcessor(BaseBatchProcessor[T, R]):
    """异步批处理器"""

    def __init__(self, config: BatchConfig):
        super().__init__(config)
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.running = False

    async def add_task(
        self,
        func: Callable[..., Awaitable[R]],
        *args,
        priority: Priority = Priority.NORMAL,
        **kwargs
    ) -> str:
        """添加异步任务"""
        task = BatchTask(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )

        await self.task_queue.put(task)
        return task.task_id

    async def start_processing(self):
        """开始处理"""
        self.running = True
        worker_tasks = []

        for i in range(self.config.max_concurrent_batches):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            worker_tasks.append(worker)

        logger.info(f"Started {len(worker_tasks)} batch processing workers")

        return worker_tasks

    async def _worker(self, name: str):
        """工作线程"""
        logger.info(f"Batch processor worker {name} started")

        while self.running:
            try:
                # 获取任务
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                # 处理任务
                async with self.semaphore:
                    task_coro = self._process_task(task)
                    await task_coro

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {name} error: {str(e)}")

        logger.info(f"Batch processor worker {name} stopped")

    async def _process_task(self, task: BatchTask):
        """处理单个任务"""
        start_time = datetime.now()
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=start_time
        )

        try:
            # 执行任务
            if asyncio.iscoroutinefunction(task.func):
                task_result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=self.config.timeout_per_task
                )
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                task_result = await loop.run_in_executor(
                    self.executor,
                    lambda: task.func(*task.args, **task.kwargs)
                )

            # 成功结果
            result.status = TaskStatus.COMPLETED
            result.result = task_result

        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error = Exception(f"Task timeout after {self.config.timeout_per_task}s")
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = e

            # 重试逻辑
            if result.retry_count < task.max_retries:
                logger.info(f"Retrying task {task.task_id} (attempt {result.retry_count + 1})")
                result.retry_count += 1
                await asyncio.sleep(self.config.retry_delay ** result.retry_count)

                # 重新加入队列
                await self.task_queue.put(task)
                return

        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()

        # 结果回调
        if task.callback:
            try:
                if asyncio.iscoroutinefunction(task.callback):
                    await task.callback(result)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        lambda: task.callback(result)
                    )
            except Exception as e:
                logger.error(f"Task callback error: {str(e)}")

        await self.result_queue.put(result)

    async def get_results(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """获取结果"""
        results = []
        try:
            while True:
                result = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=timeout
                )
                results.append(result)
        except asyncio.TimeoutError:
            pass

        return results

    async def stop(self):
        """停止处理器"""
        self.running = False

        # 等待所有任务完成
        for task_id, task in self.active_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.executor.shutdown(wait=True)
        logger.info("Async batch processor stopped")


class FileBatchProcessor(BaseBatchProcessor[str, Dict[str, Any]]):
    """文件批处理器"""

    def __init__(self, config: BatchConfig, output_dir: Optional[str] = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else None
        self.processed_files = set()

    async def process_item(self, file_path: str) -> Dict[str, Any]:
        """处理单个文件"""
        try:
            file_path = Path(file_path)

            # 检查是否已处理
            if str(file_path) in self.processed_files:
                return {"file": str(file_path), "status": "skipped", "reason": "already processed"}

            # 获取文件信息
            stat = await aiofiles.os.stat(file_path)

            # 读取文件
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()

            # 处理文件内容
            result = await self._process_file_content(
                file_path,
                content,
                stat
            )

            # 保存结果
            if self.output_dir:
                await self._save_result(file_path, result)

            self.processed_files.add(str(file_path))

            return {
                "file": str(file_path),
                "status": "processed",
                "result": result,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime)
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "file": str(file_path),
                "status": "error",
                "error": str(e)
            }

    async def _process_file_content(
        self,
        file_path: Path,
        content: bytes,
        stat: Any
    ) -> Dict[str, Any]:
        """处理文件内容（子类实现）"""
        return {
            "path": str(file_path),
            "size": len(content),
            "type": file_path.suffix.lower()
        }

    async def _save_result(self, file_path: Path, result: Dict[str, Any]):
        """保存处理结果"""
        if not self.output_dir:
            return

        output_file = self.output_dir / f"{file_path.stem}_processed.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(output_file, 'w') as f:
            import json
            await f.write(json.dumps(result, ensure_ascii=False, indent=2))


class VectorBatchProcessor(BaseBatchProcessor[Dict[str, Any], List[float]]):
    """向量批处理器"""

    def __init__(
        self,
        config: BatchConfig,
        embedding_model: Callable[[str], List[float]],
        batch_embedding: Optional[Callable[[List[str]], List[List[float]]]] = None
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.batch_embedding = batch_embedding

    async def process_item(self, item: Dict[str, Any]) -> List[float]:
        """处理单个项目并生成向量"""
        text = item.get("text", "")
        if not text:
            return []

        if self.batch_embedding and len(text) > 100:
            # 使用批量嵌入
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.embedding_model(text)
            )
        else:
            # 单个嵌入
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.embedding_model(text)
            )

        return embedding

    async def process_batch_embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        if self.batch_embedding:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.batch_embedding(texts)
            )
            return embeddings
        else:
            # 逐个处理
            embeddings = []
            for text in texts:
                embedding = await self.process_item({"text": text})
                embeddings.append(embedding)
            return embeddings


class TaskScheduler:
    """任务调度器"""

    def __init__(self):
        self.tasks = {}
        self.task_groups = defaultdict(list)
        self.scheduled_tasks = asyncio.Queue()
        self.running = False

    async def schedule_task(
        self,
        coro: Awaitable,
        task_id: Optional[str] = None,
        group: Optional[str] = None,
        delay: Optional[float] = None,
        priority: Priority = Priority.NORMAL
    ) -> str:
        """调度任务"""
        if task_id is None:
            task_id = str(uuid.uuid4())

        task_info = {
            "task_id": task_id,
            "coro": coro,
            "group": group,
            "delay": delay,
            "priority": priority,
            "scheduled_at": datetime.now()
        }

        if delay:
            # 延迟执行
            asyncio.create_task(self._delayed_schedule(task_info))
        else:
            await self.scheduled_tasks.put(task_info)

        if group:
            self.task_groups[group].append(task_id)

        self.tasks[task_id] = task_info
        return task_id

    async def _delayed_schedule(self, task_info: Dict[str, Any]):
        """延迟调度"""
        await asyncio.sleep(task_info["delay"])
        await self.scheduled_tasks.put(task_info)

    async def start(self):
        """启动调度器"""
        self.running = True
        worker = asyncio.create_task(self._scheduler_worker())
        return worker

    async def _scheduler_worker(self):
        """调度器工作线程"""
        while self.running:
            try:
                task_info = await asyncio.wait_for(
                    self.scheduled_tasks.get(),
                    timeout=1.0
                )

                # 创建并执行任务
                task = asyncio.create_task(task_info["coro"])
                task.add_done_callback(lambda t: self._task_completed(task_info["task_id"]))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")

    def _task_completed(self, task_id: str):
        """任务完成回调"""
        if task_id in self.tasks:
            del self.tasks[task_id]

    async def get_group_status(self, group: str) -> Dict[str, Any]:
        """获取任务组状态"""
        group_tasks = self.task_groups.get(group, [])
        pending = len([t for t in group_tasks if t in self.tasks])

        return {
            "group": group,
            "total_tasks": len(group_tasks),
            "pending_tasks": pending,
            "completed_tasks": len(group_tasks) - pending
        }

    def stop(self):
        """停止调度器"""
        self.running = False


class ResultAggregator:
    """结果聚合器"""

    def __init__(self, strategy: str = "collect_all"):
        self.strategy = strategy
        self.results = []
        self.errors = []

    async def add_result(self, result: TaskResult):
        """添加结果"""
        if result.status == TaskStatus.COMPLETED:
            self.results.append(result)
        elif result.status == TaskStatus.FAILED:
            self.errors.append(result)

    def get_aggregated_result(self) -> Dict[str, Any]:
        """获取聚合结果"""
        if self.strategy == "collect_all":
            return {
                "results": [r.result for r in self.results],
                "errors": [str(r.error) for r in self.errors],
                "success_count": len(self.results),
                "error_count": len(self.errors),
                "total_count": len(self.results) + len(self.errors)
            }
        elif self.strategy == "first_success":
            for result in self.results:
                if result.result:
                    return {"result": result.result}
            return {"result": None}
        elif self.strategy == "best_result":
            if self.results:
                # 选择耗时最短的成功结果
                best = min(self.results, key=lambda r: r.duration)
                return {"result": best.result, "duration": best.duration}
            return {"result": None}
        else:
            return {"results": []}


class BatchProcessingManager:
    """批处理管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processors = {}
        self.scheduler = TaskScheduler()
        self.scheduler_task = None

    async def initialize(self):
        """初始化管理器"""
        self.scheduler_task = await self.scheduler.start()
        logger.info("Batch processing manager initialized")

    def create_processor(
        self,
        name: str,
        processor_class: type,
        config: BatchConfig,
        **kwargs
    ) -> BaseBatchProcessor:
        """创建处理器"""
        processor = processor_class(config, **kwargs)
        self.processors[name] = processor
        return processor

    def get_processor(self, name: str) -> BaseBatchProcessor:
        """获取处理器"""
        if name not in self.processors:
            raise ValueError(f"Processor {name} not found")
        return self.processors[name]

    async def process_files(
        self,
        file_paths: List[str],
        processor_name: str = "file_processor",
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """处理文件列表"""
        if processor_name not in self.processors:
            config = BatchConfig(
                batch_size=self.config.get("batch_size", 10),
                max_concurrent_batches=self.config.get("max_concurrent_batches", 5)
            )
            self.create_processor(
                processor_name,
                FileBatchProcessor,
                config,
                output_dir=output_dir
            )

        processor = self.get_processor(processor_name)
        return await processor.process_all(file_paths)

    async def process_with_retry(
        self,
        items: List[Any],
        process_func: Callable,
        max_retries: int = 3
    ) -> List[Any]:
        """带重试的处理"""
        results = []

        for item in items:
            retry_count = 0
            last_error = None

            while retry_count <= max_retries:
                try:
                    result = await process_func(item)
                    results.append(result)
                    break
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    if retry_count <= max_retries:
                        await asyncio.sleep(2 ** retry_count)
                    else:
                        logger.error(f"Failed to process item after {max_retries} retries: {str(e)}")
                        results.append(None)

        return results

    async def stop(self):
        """停止管理器"""
        self.scheduler.stop()

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        for processor in self.processors.values():
            if hasattr(processor, 'stop'):
                await processor.stop()

        logger.info("Batch processing manager stopped")


# 全局批处理管理器
_batch_manager = None


async def get_batch_manager(config: Optional[Dict[str, Any]] = None) -> BatchProcessingManager:
    """获取全局批处理管理器"""
    global _batch_manager

    if _batch_manager is None:
        if config is None:
            config = {
                "batch_size": 10,
                "max_concurrent_batches": 5,
                "timeout_per_task": 300.0,
                "max_retries": 3
            }

        _batch_manager = BatchProcessingManager(config)
        await _batch_manager.initialize()

    return _batch_manager


# 便利函数
async def batch_process(
    items: List[T],
    process_func: Callable[[T], Awaitable[R]],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[R]:
    """批处理便利函数"""
    manager = await get_batch_manager()

    config = BatchConfig(
        batch_size=batch_size,
        max_concurrent_batches=max_concurrent
    )

    processor = AsyncBatchProcessor(config)
    await processor.start_processing()

    try:
        # 添加所有任务
        task_ids = []
        for item in items:
            task_id = await processor.add_task(process_func, item)
            task_ids.append(task_id)

        # 等待所有任务完成
        results = []
        for _ in range(len(items)):
            result = await processor.get_results(timeout=60.0)
            results.extend(result)

        # 返回成功的结果
        successful_results = [
            r.result for r in results
            if r.status == TaskStatus.COMPLETED
        ]

        return successful_results
    finally:
        await processor.stop()