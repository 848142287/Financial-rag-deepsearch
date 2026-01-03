"""
异步处理优化服务
提供并发控制、批处理、流水线等异步优化功能
"""

import asyncio
from app.core.structured_logging import get_structured_logger
from typing import List, Dict, Any, Callable, Optional, TypeVar, Coroutine
from dataclasses import dataclass
import time

logger = get_structured_logger(__name__)

T = TypeVar('T')


@dataclass
class BatchResult:
    """批处理结果"""
    total_items: int
    successful_items: int
    failed_items: int
    results: List[Any]
    errors: List[Exception]
    execution_time: float


class AsyncOptimizer:
    """异步处理优化器"""

    def __init__(self):
        # 并发控制配置
        self.default_max_concurrency = 10
        self.semaphores = {}  # 存储不同任务的信号量

    def get_semaphore(self, key: str, max_concurrency: Optional[int] = None) -> asyncio.Semaphore:
        """
        获取或创建信号量

        Args:
            key: 信号量键
            max_concurrency: 最大并发数

        Returns:
            Semaphore对象
        """
        if key not in self.semaphores:
            concurrency = max_concurrency or self.default_max_concurrency
            self.semaphores[key] = asyncio.Semaphore(concurrency)
            logger.debug(f"创建信号量: {key}, 并发数: {concurrency}")

        return self.semaphores[key]

    async def execute_with_concurrency_control(
        self,
        tasks: List[Coroutine],
        max_concurrency: int = 10,
        task_type: str = "default"
    ) -> List[Any]:
        """
        使用并发控制执行任务

        Args:
            tasks: 任务列表
            max_concurrency: 最大并发数
            task_type: 任务类型（用于信号量键）

        Returns:
            执行结果列表
        """
        semaphore = self.get_semaphore(task_type, max_concurrency)

        async def bounded_task(task):
            async with semaphore:
                return await task

        start_time = time.time()
        results = await asyncio.gather(*[bounded_task(task) for task in tasks], return_exceptions=True)
        execution_time = time.time() - start_time

        # 统计
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        logger.info(
            f"并发执行完成: {task_type}, "
            f"总数: {len(results)}, "
            f"成功: {successful}, "
            f"失败: {failed}, "
            f"耗时: {execution_time:.2f}s"
        )

        return results

    async def execute_batches(
        self,
        items: List[T],
        process_func: Callable[[List[T]], Coroutine],
        batch_size: int = 10,
        delay_between_batches: float = 0.1
    ) -> BatchResult:
        """
        批处理执行

        Args:
            items: 待处理的项目列表
            process_func: 处理函数（异步，接收批次）
            batch_size: 批次大小
            delay_between_batches: 批次间延迟（秒）

        Returns:
            批处理结果
        """
        start_time = time.time()
        all_results = []
        all_errors = []

        # 分批处理
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

        logger.info(f"开始批处理: 总数{len(items)}个, 分{len(batches)}批")

        for i, batch in enumerate(batches):
            try:
                logger.debug(f"处理第{i+1}/{len(batches)}批, 大小: {len(batch)}")

                batch_result = await process_func(batch)
                all_results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

                # 批次间延迟
                if i < len(batches) - 1 and delay_between_batches > 0:
                    await asyncio.sleep(delay_between_batches)

            except Exception as e:
                logger.error(f"批次{i+1}处理失败: {e}")
                all_errors.append(e)

        execution_time = time.time() - start_time

        return BatchResult(
            total_items=len(items),
            successful_items=len(all_results),
            failed_items=len(all_errors),
            results=all_results,
            errors=all_errors,
            execution_time=execution_time
        )

    async def execute_pipeline(
        self,
        stages: List[Callable[[Any], Coroutine]],
        initial_data: Any,
        stage_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        流水线处理

        Args:
            stages: 处理阶段函数列表
            initial_data: 初始数据
            stage_names: 阶段名称列表

        Returns:
            流水线执行结果
        """
        if stage_names is None:
            stage_names = [f"stage_{i}" for i in range(len(stages))]

        current_data = initial_data
        stage_results = {}

        logger.info(f"开始流水线处理: {len(stages)}个阶段")

        for i, (stage, name) in enumerate(zip(stages, stage_names)):
            stage_start = time.time()

            try:
                logger.info(f"执行阶段 {i+1}/{len(stages)}: {name}")

                # 执行阶段
                current_data = await stage(current_data)
                stage_time = time.time() - stage_start

                stage_results[name] = {
                    "status": "success",
                    "execution_time": stage_time,
                    "output_size": len(current_data) if isinstance(current_data, list) else 1
                }

                logger.info(f"阶段 {name} 完成, 耗时: {stage_time:.2f}s")

            except Exception as e:
                stage_time = time.time() - stage_start
                logger.error(f"阶段 {name} 失败: {e}")

                stage_results[name] = {
                    "status": "failed",
                    "error": str(e),
                    "execution_time": stage_time
                }

                # 流水线中断
                break

        total_time = sum(r.get("execution_time", 0) for r in stage_results.values())

        return {
            "final_data": current_data,
            "stage_results": stage_results,
            "total_execution_time": total_time,
            "completed_stages": len(stage_results)
        }

    async def execute_with_timeout(
        self,
        coro: Coroutine,
        timeout: float,
        default_value: Any = None
    ) -> Any:
        """
        带超时的异步执行

        Args:
            coro: 协程对象
            timeout: 超时时间（秒）
            default_value: 超时后的默认返回值

        Returns:
            执行结果或默认值
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"任务执行超时: {timeout}s")
            return default_value
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            return default_value

    async def execute_with_retry(
        self,
        func: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ) -> Any:
        """
        带重试的异步执行

        Args:
            func: 异步函数
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            backoff_factor: 退避因子
            exceptions: 需要重试的异常类型

        Returns:
            执行结果
        """
        last_exception = None
        current_delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"重试第 {attempt} 次")

                result = await func()
                return result

            except exceptions as e:
                last_exception = e

                if attempt < max_retries:
                    logger.warning(f"执行失败: {e}, {current_delay:.1f}s后重试")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    logger.error(f"达到最大重试次数: {max_retries}")

        # 所有重试都失败
        raise last_exception

    async def parallel_map(
        self,
        func: Callable[[T], Coroutine],
        items: List[T],
        max_concurrency: int = 10
    ) -> List[Any]:
        """
        并发映射

        Args:
            func: 异步处理函数
            items: 待处理项列表
            max_concurrency: 最大并发数

        Returns:
            处理结果列表
        """
        tasks = [func(item) for item in items]
        results = await self.execute_with_concurrency_control(tasks, max_concurrency)

        # 过滤异常
        return [r for r in results if not isinstance(r, Exception)]

    async def gather_with_results(
        self,
        *coros: Coroutine,
        return_exceptions: bool = True
    ) -> Dict[str, Any]:
        """
        收集协程结果并返回统计

        Args:
            *coros: 协程对象列表
            return_exceptions: 是否返回异常

        Returns:
            结果字典
        """
        start_time = time.time()

        results = await asyncio.gather(*coros, return_exceptions=return_exceptions)
        execution_time = time.time() - start_time

        successful = [r for r in results if not isinstance(r, Exception)]
        exceptions = [r for r in results if isinstance(r, Exception)]

        return {
            "results": results,
            "successful": successful,
            "exceptions": exceptions,
            "total_count": len(results),
            "success_count": len(successful),
            "exception_count": len(exceptions),
            "execution_time": execution_time
        }


# 全局异步优化器实例
async_optimizer = AsyncOptimizer()
