"""
性能优化模块
提供缓存、批处理、异步处理等性能优化功能
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
import time
import functools
import pickle
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.process = psutil.Process()

    def record_metric(self, name: str, value: float):
        """记录性能指标"""
        timestamp = time.time()
        self.metrics[name].append({
            'timestamp': timestamp,
            'value': value
        })

        # 保持最近1000个记录
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        return self.process.cpu_percent()

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': memory_percent
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024,  # MB
            'memory_available': psutil.virtual_memory().available / 1024 / 1024,  # MB
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }

    def get_metrics_summary(self, metric_name: str) -> Dict[str, float]:
        """获取指标摘要"""
        if metric_name not in self.metrics:
            return {}

        values = [m['value'] for m in self.metrics[metric_name]]
        if not values:
            return {}

        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1]
        }


# 全局性能监控器
performance_monitor = PerformanceMonitor()


class CacheManager:
    """缓存管理器"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl  # 生存时间（秒）
        self.lock = threading.Lock()

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = f"{func_name}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                if time.time() - item['timestamp'] < self.ttl:
                    self.access_times[key] = time.time()
                    return item['value']
                else:
                    # 过期，删除
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
        return None

    def set(self, key: str, value: Any):
        """设置缓存值"""
        with self.lock:
            # 如果缓存满了，删除最久未访问的项
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'memory_usage': sum(len(pickle.dumps(v['value'])) for v in self.cache.values())
            }


# 全局缓存管理器
cache_manager = CacheManager()


def cached(ttl: int = 3600, max_size: int = None):
    """缓存装饰器"""
    def decorator(func):
        cache = CacheManager(max_size=max_size or 1000, ttl=ttl)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            key = cache._generate_key(func.__name__, args, kwargs)

            # 尝试从缓存获取
            result = cache.get(key)
            if result is not None:
                return result

            # 执行函数
            result = func(*args, **kwargs)

            # 存储到缓存
            cache.set(key, result)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            key = cache._generate_key(func.__name__, args, kwargs)

            # 尝试从缓存获取
            result = cache.get(key)
            if result is not None:
                return result

            # 执行函数
            result = await func(*args, **kwargs)

            # 存储到缓存
            cache.set(key, result)
            return result

        # 根据函数类型返回合适的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class BatchProcessor:
    """批处理器"""

    def __init__(self, batch_size: int = 100, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.processing = False
        self.processed_count = 0
        self.error_count = 0

    async def add_item(self, item: Any):
        """添加项目到批处理队列"""
        await self.queue.put(item)

    async def process_batch(self, processor: Callable[[List[Any]], List[Any]]):
        """处理批次"""
        if self.processing:
            return

        self.processing = True
        batch = []

        try:
            while True:
                try:
                    # 等待第一个项目
                    if not batch:
                        item = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
                        batch.append(item)
                    else:
                        # 尝试获取更多项目（非阻塞）
                        try:
                            item = self.queue.get_nowait()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            break

                    # 如果批次满了，立即处理
                    if len(batch) >= self.batch_size:
                        await self._process_batch_items(batch, processor)
                        batch = []

                except asyncio.TimeoutError:
                    # 超时，处理当前批次
                    if batch:
                        await self._process_batch_items(batch, processor)
                        batch = []

        finally:
            # 处理剩余项目
            if batch:
                await self._process_batch_items(batch, processor)
            self.processing = False

    async def _process_batch_items(self, batch: List[Any], processor: Callable):
        """处理批次项目"""
        start_time = time.time()
        try:
            results = await processor(batch)
            self.processed_count += len(batch)
            performance_monitor.record_metric('batch_process_time', time.time() - start_time)
            performance_monitor.record_metric('batch_size', len(batch))
        except Exception as e:
            self.error_count += 1
            logger.error(f"批处理失败: {str(e)}")
            performance_monitor.record_metric('batch_error_count', 1)


class ThreadPoolManager:
    """线程池管理器"""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks = set()

    async def submit(self, func: Callable, *args, **kwargs):
        """提交任务到线程池"""
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(self.executor, func, *args, **kwargs)
        self.active_tasks.add(task)
        task.add_done_callback(lambda t: self.active_tasks.discard(t))
        return await task

    async def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)
        # 等待所有活动任务完成
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)


class ProcessPoolManager:
    """进程池管理器"""

    def __init__(self, max_processes: int = None):
        self.max_processes = max_processes or psutil.cpu_count() or 1
        self.executor = ProcessPoolExecutor(max_workers=self.max_processes)
        self.active_tasks = set()

    async def submit(self, func: Callable, *args, **kwargs):
        """提交任务到进程池"""
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(self.executor, func, *args, **kwargs)
        self.active_tasks.add(task)
        task.add_done_callback(lambda t: self.active_tasks.discard(t))
        return await task

    async def shutdown(self):
        """关闭进程池"""
        self.executor.shutdown(wait=True)
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)


class ResourceManager:
    """资源管理器"""

    def __init__(self):
        self.thread_pool = ThreadPoolManager()
        self.process_pool = ProcessPoolManager()
        self.resources = {}

    @asynccontextmanager
    async def acquire(self, resource_name: str, timeout: float = 30):
        """获取资源"""
        if resource_name not in self.resources:
            self.resources[resource_name] = {
                'lock': asyncio.Lock(),
                'count': 0,
                'max_concurrent': 10  # 默认最大并发数
            }

        resource = self.resources[resource_name]
        try:
            await asyncio.wait_for(resource['lock'].acquire(), timeout=timeout)
            resource['count'] += 1
            yield resource
        except asyncio.TimeoutError:
            raise TimeoutError(f"获取资源 {resource_name} 超时")
        finally:
            resource['lock'].release()
            resource['count'] -= 1

    async def run_in_thread(self, func: Callable, *args, **kwargs):
        """在线程池中运行函数"""
        return await self.thread_pool.submit(func, *args, **kwargs)

    async def run_in_process(self, func: Callable, *args, **kwargs):
        """在进程池中运行函数"""
        return await self.process_pool.submit(func, *args, **kwargs)

    async def cleanup(self):
        """清理资源"""
        await self.thread_pool.shutdown()
        await self.process_pool.shutdown()
        gc.collect()  # 强制垃圾回收


# 全局资源管理器
resource_manager = ResourceManager()


def rate_limiter(calls: int, period: float):
    """速率限制装饰器"""
    calls_made = deque()
    lock = threading.Lock()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                # 清理过期的调用记录
                while calls_made and calls_made[0] <= now - period:
                    calls_made.popleft()

                # 检查是否超过限制
                if len(calls_made) >= calls:
                    sleep_time = period - (now - calls_made[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                calls_made.append(now)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def memory_limit(max_memory_mb: int):
    """内存限制装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            try:
                result = func(*args, **kwargs)

                # 检查内存使用
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                if current_memory > max_memory_mb:
                    logger.warning(
                        f"内存使用超限: {current_memory:.2f}MB > {max_memory_mb}MB, "
                        f"函数: {func.__name__}"
                    )
                    # 执行垃圾回收
                    gc.collect()

                return result

            except MemoryError:
                logger.error(f"函数 {func.__name__} 内存溢出")
                gc.collect()
                raise

        return wrapper

    return decorator


class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """调用函数，带熔断保护"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("熔断器开启，拒绝调用")

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # 成功调用，重置
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    logger.warning(
                        f"函数 {func.__name__} 调用失败，尝试 {attempt + 1}/{max_attempts}: {str(e)}"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    logger.warning(
                        f"函数 {func.__name__} 调用失败，尝试 {attempt + 1}/{max_attempts}: {str(e)}"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        # 根据函数类型返回合适的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# 性能优化的工具函数
async def parallel_map(func: Callable, items: List[Any], max_workers: int = None):
    """并行映射函数"""
    if not items:
        return []

    semaphore = asyncio.Semaphore(max_workers or 10)

    async def process_item(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                return await resource_manager.run_in_thread(func, item)

    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)


def chunked(iterable, size: int):
    """将可迭代对象分块"""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


# 定期清理任务
async def cleanup_task():
    """定期清理任务"""
    while True:
        try:
            # 清理缓存
            cache_manager.clear()

            # 垃圾回收
            gc.collect()

            # 记录系统状态
            stats = performance_monitor.get_system_stats()
            performance_monitor.record_metric('system_cpu', stats['cpu_percent'])
            performance_monitor.record_metric('system_memory', stats['memory_percent'])

            logger.debug(f"系统状态: CPU {stats['cpu_percent']}%, 内存 {stats['memory_percent']}%")

        except Exception as e:
            logger.error(f"清理任务失败: {str(e)}")

        # 每5分钟执行一次
        await asyncio.sleep(300)


# 启动清理任务
async def start_cleanup_task():
    """启动清理任务"""
    asyncio.create_task(cleanup_task())