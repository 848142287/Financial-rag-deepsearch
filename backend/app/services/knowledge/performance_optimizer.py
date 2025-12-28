"""
性能优化器
实现批处理、并行处理、资源管理等功能
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging
import psutil
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_time: float
    items_processed: int
    throughput: float  # 每秒处理数
    memory_usage_mb: float
    cpu_usage_percent: float


class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self):
        # 线程池配置
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)

        # 批处理配置
        self.default_batch_size = 32
        self.max_batch_size = 128

        # 并发控制
        self.semaphore = asyncio.Semaphore(10)  # 最大并发任务数

        # 资源监控
        self.memory_threshold = 0.8  # 80%内存使用率
        self.cpu_threshold = 0.9     # 90%CPU使用率

    @asynccontextmanager
    async def resource_monitor(self):
        """资源监控上下文管理器"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024 / 1024

            metrics = PerformanceMetrics(
                total_time=end_time - start_time,
                items_processed=0,  # 需要在具体任务中设置
                throughput=0,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=psutil.cpu_percent()
            )

            logger.info(f"性能指标: {metrics}")

    async def batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: Optional[int] = None,
        use_threading: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        批量处理

        Args:
            items: 待处理项目列表
            process_func: 处理函数（可以是异步或同步）
            batch_size: 批次大小
            use_threading: 是否使用线程池
            progress_callback: 进度回调函数

        Returns:
            处理结果列表
        """
        if not batch_size:
            batch_size = min(self.default_batch_size, len(items))

        logger.info(f"开始批量处理: {len(items)} 项目, 批次大小: {batch_size}")

        # 分批
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

        results = []

        async def process_batch(batch):
            try:
                if use_threading and not asyncio.iscoroutinefunction(process_func):
                    # 使用线程池处理同步函数
                    loop = asyncio.get_event_loop()
                    batch_result = await loop.run_in_executor(
                        self.thread_pool,
                        process_func,
                        batch
                    )
                else:
                    # 异步处理
                    batch_result = await process_func(batch)

                if progress_callback:
                    progress = (len(results) + len(batch)) / len(items)
                    await progress_callback(progress)

                return batch_result

            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                return []

        # 并发处理批次
        semaphore = asyncio.Semaphore(4)  # 限制并发批次数

        async def process_with_semaphore(batch):
            async with semaphore:
                return await process_batch(batch)

        tasks = [process_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"批次处理异常: {batch_result}")
                continue

            if isinstance(batch_result, list):
                results.extend(batch_result)

        logger.info(f"批量处理完成: 处理了 {len(results)} 个结果")
        return results

    async def parallel_process(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """
        并行处理多个任务

        Args:
            tasks: 任务列表 [(func, args, kwargs), ...]
            max_workers: 最大工作线程数

        Returns:
            处理结果列表
        """
        if not max_workers:
            max_workers = min(8, len(tasks))

        logger.info(f"开始并行处理: {len(tasks)} 个任务, 最大工作数: {max_workers}")

        semaphore = asyncio.Semaphore(max_workers)

        async def execute_task(func, args, kwargs):
            async with semaphore:
                async with self.semaphore:
                    try:
                        # 检查资源使用
                        if await self._should_wait_for_resources():
                            await self._wait_for_resources()

                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            # 同步函数使用线程池
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(
                                self.thread_pool,
                                lambda: func(*args, **kwargs)
                            )
                    except Exception as e:
                        logger.error(f"任务执行失败: {e}")
                        return None

        # 执行所有任务
        results = await asyncio.gather(
            *[execute_task(func, args, kwargs) for func, args, kwargs in tasks],
            return_exceptions=True
        )

        # 过滤异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"任务异常: {result}")
            elif result is not None:
                processed_results.append(result)

        logger.info(f"并行处理完成: {len(processed_results)} 个成功结果")
        return processed_results

    async def adaptive_batch_size(
        self,
        items: List[Any],
        process_func: Callable,
        test_size: int = 10
    ) -> int:
        """
        自适应批次大小

        Args:
            items: 待处理项目
            process_func: 处理函数
            test_size: 测试大小

        Returns:
            最优批次大小
        """
        if len(items) <= self.default_batch_size:
            return len(items)

        logger.info("开始自适应批次大小测试")

        test_batch_sizes = [8, 16, 32, 64, 128]
        batch_times = {}

        for batch_size in test_batch_sizes:
            if batch_size > len(items):
                continue

            try:
                # 测试处理时间
                test_items = items[:min(test_size, len(items))]

                start_time = time.time()
                if asyncio.iscoroutinefunction(process_func):
                    await process_func(test_items)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.thread_pool,
                        process_func,
                        test_items
                    )
                end_time = time.time()

                # 计算吞吐量
                processing_time = end_time - start_time
                throughput = len(test_items) / processing_time
                batch_times[batch_size] = throughput

                logger.info(f"批次大小 {batch_size}: 吞吐量 {throughput:.2f} items/s")

            except Exception as e:
                logger.error(f"测试批次大小 {batch_size} 失败: {e}")

        # 选择最优批次大小
        if batch_times:
            optimal_size = max(batch_times, key=batch_times.get)
            logger.info(f"最优批次大小: {optimal_size}")
            return optimal_size
        else:
            logger.warning("无法确定最优批次大小，使用默认值")
            return self.default_batch_size

    async def optimize_embedding_batch(
        self,
        texts: List[str],
        embedding_model
    ) -> List[List[float]]:
        """
        优化嵌入生成批次

        Args:
            texts: 文本列表
            embedding_model: 嵌入模型

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        # 自适应批次大小
        optimal_batch_size = await self.adaptive_batch_size(
            texts,
            lambda batch: embedding_model.encode(batch)
        )

        # 批量生成嵌入
        async def generate_embeddings_batch(batch):
            # 同步调用嵌入模型
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.thread_pool,
                embedding_model.encode,
                batch
            )
            return embeddings.tolist()

        embeddings = await self.batch_process(
            texts,
            generate_embeddings_batch,
            batch_size=optimal_batch_size,
            use_threading=True
        )

        # 合并嵌入结果
        all_embeddings = []
        for batch_embeds in embeddings:
            all_embeddings.extend(batch_embeds)

        return all_embeddings

    async def _should_wait_for_resources(self) -> bool:
        """检查是否需要等待资源"""
        # 检查内存使用
        memory = psutil.virtual_memory()
        if memory.percent / 100 > self.memory_threshold:
            return True

        # 检查CPU使用
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent / 100 > self.cpu_threshold:
            return True

        return False

    async def _wait_for_resources(self):
        """等待资源可用"""
        while await self._should_wait_for_resources():
            logger.info("等待资源释放...")
            await asyncio.sleep(1)

    def get_system_stats(self) -> Dict[str, float]:
        """获取系统统计信息"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            'memory_used_percent': memory.percent,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count()
        }

    async def cleanup(self):
        """清理资源"""
        logger.info("清理性能优化器资源")

        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        # 取消所有等待的任务
        pending_tasks = [task for task in asyncio.all_tasks()
                        if task is not asyncio.current_task()
                        and not task.done()]
        for task in pending_tasks:
            task.cancel()


class VectorOptimizer:
    """向量处理优化器"""

    @staticmethod
    def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
        """标准化嵌入向量"""
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / (norms + 1e-8)  # 避免除零
        return normalized.tolist()

    @staticmethod
    def reduce_dimensions(
        embeddings: List[List[float]],
        target_dim: int = 256
    ) -> List[List[float]]:
        """降维处理"""
        try:
            from sklearn.decomposition import PCA
            import numpy as np

            if not embeddings or len(embeddings) == 0:
                return embeddings

            # 转换为numpy数组
            embeddings_array = np.array(embeddings)

            # 如果当前维度已经小于或等于目标维度，直接返回
            if embeddings_array.shape[1] <= target_dim:
                return embeddings

            # 使用PCA进行降维
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings_array)

            # 转换回列表格式
            return reduced_embeddings.tolist()

        except ImportError:
            # 如果sklearn不可用，返回原始嵌入
            import warnings
            warnings.warn("sklearn not available for dimension reduction, returning original embeddings")
            return embeddings
        except Exception as e:
            # 如果降维失败，返回原始嵌入
            import warnings
            warnings.warn(f"Dimension reduction failed: {e}, returning original embeddings")
            return embeddings

    @staticmethod
    def compress_vectors(vectors: List[List[float]]) -> List[bytes]:
        """压缩向量"""
        compressed = []
        for vector in vectors:
            # 转换为numpy数组并压缩
            arr = np.array(vector, dtype=np.float32)
            compressed.append(arr.tobytes())
        return compressed


# 全局性能优化器实例
performance_optimizer = PerformanceOptimizer()
vector_optimizer = VectorOptimizer()