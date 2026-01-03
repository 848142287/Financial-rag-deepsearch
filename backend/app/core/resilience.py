"""
弹性和容错工具
提供重试、熔断、限流等功能
"""

import asyncio
import time
from app.core.structured_logging import get_structured_logger
from functools import wraps
from typing import Any, Callable, Optional, Type, Tuple, List, Dict
from enum import Enum
import threading

logger = get_structured_logger(__name__)

class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"          # 正常运行
    OPEN = "open"              # 熔断打开
    HALF_OPEN = "half_open"    # 半开（尝试恢复）

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: Tuple[Type[Exception], ...] = (Exception,)

@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5
    success_threshold: int = 2,
    timeout: float = 60.0
    half_open_max_calls: int = 3
    fallback_value: Any = None

@dataclass
class RateLimitConfig:
    """限流配置"""
    max_calls: int = 100
    time_window: float = 60.0  # 秒
    burst_size: int = 10

class CircuitBreaker:
    """熔断器"""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def record_success(self):
        """记录成功"""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info(f"熔断器 {self.name} 恢复到关闭状态")

    def record_failure(self):
        """记录失败"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            self.success_count = 0

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"熔断器 {self.name} 打开（失败次数: {self.failure_count}）")

    def can_execute(self) -> bool:
        """检查是否可以执行"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # 检查是否超时
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed > self.config.timeout:
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info(f"熔断器 {self.name} 进入半开状态")
                        return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls

            return False

    def get_state(self) -> CircuitState:
        """获取当前状态"""
        return self.state

    def reset(self):
        """重置熔断器"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.half_open_calls = 0
            logger.info(f"熔断器 {self.name} 已重置")

class RateLimiter:
    """滑动窗口限流器"""

    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        self.calls: List[float] = []
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取许可

        Args:
            timeout: 超时时间（秒）

        Returns:
            bool: 是否成功获取许可
        """
        start_time = time.time()

        while True:
            with self._lock:
                now = time.time()
                # 清理过期的调用记录
                self.calls = [t for t in self.calls if now - t < self.config.time_window]

                if len(self.calls) < self.config.max_calls:
                    self.calls.append(now)
                    return True

            # 检查是否超时
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning(f"限流器 {self.name} 超时")
                return False

            # 等待一小段时间后重试
            await asyncio.sleep(0.1)

    def get_available_permits(self) -> int:
        """获取可用许可数"""
        with self._lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.config.time_window]
            return self.config.max_calls - len(self.calls)

# 全局熔断器注册表
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_rate_limiters: Dict[str, RateLimiter] = {}

def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """获取或创建熔断器"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name,
            config or CircuitBreakerConfig()
        )
    return _circuit_breakers[name]

def get_rate_limiter(name: str, config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """获取或创建限流器"""
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter(
            name,
            config or RateLimitConfig()
        )
    return _rate_limiters[name]

def retry(
    max_attempts: int = 3,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True
):
    """
    重试装饰器

    Args:
        max_attempts: 最大尝试次数
        retry_on: 需要重试的异常类型
        base_delay: 基础延迟
        exponential_base: 指数退避基数
        max_delay: 最大延迟
        jitter: 是否添加随机抖动

    Usage:
        @retry(max_attempts=3, retry_on=(ConnectionError, TimeoutError))
        async def fetch_data():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        # 计算延迟时间
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )

                        # 添加随机抖动
                        if jitter:
                            import random
                            delay *= (0.5 + random.random())

                        logger.warning(
                            f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_attempts}): {e}, "
                            f"等待 {delay:.2f}秒后重试"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 在 {max_attempts} 次尝试后仍然失败")
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        if jitter:
                            import random
                            delay *= (0.5 + random.random())
                        logger.warning(
                            f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_attempts}): {e}, "
                            f"等待 {delay:.2f}秒后重试"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 在 {max_attempts} 次尝试后仍然失败")
            raise last_exception

        # 根据函数类型返回对应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    fallback_value: Any = None
):
    """
    熔断装饰器

    Args:
        name: 熔断器名称
        failure_threshold: 失败阈值
        timeout: 超时时间（秒）
        fallback_value: 降级值

    Usage:
        @circuit_breaker("database", failure_threshold=5, timeout=60.0)
        async def query_database():
            ...
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        fallback_value=fallback_value
    )
    breaker = get_circuit_breaker(name, config)

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                logger.warning(f"熔断器 {name} 打开，使用降级值")
                if fallback_value is not None:
                    return fallback_value
                raise Exception(f"熔断器 {name} 打开，服务暂时不可用")

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                logger.error(f"{name} 执行失败: {e}")
                if fallback_value is not None:
                    return fallback_value
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                logger.warning(f"熔断器 {name} 打开，使用降级值")
                if fallback_value is not None:
                    return fallback_value
                raise Exception(f"熔断器 {name} 打开，服务暂时不可用")

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                logger.error(f"{name} 执行失败: {e}")
                if fallback_value is not None:
                    return fallback_value
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def rate_limit(
    name: str,
    max_calls: int = 100,
    time_window: float = 60.0,
    timeout: Optional[float] = None
):
    """
    限流装饰器

    Args:
        name: 限流器名称
        max_calls: 最大调用次数
        time_window: 时间窗口（秒）
        timeout: 获取许可的超时时间

    Usage:
        @rate_limit("api_calls", max_calls=10, time_window=1.0)
        async def call_external_api():
            ...
    """
    config = RateLimitConfig(
        max_calls=max_calls,
        time_window=time_window
    )
    limiter = get_rate_limiter(name, config)

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not limiter.acquire(timeout=timeout):
                logger.warning(f"限流器 {name} 拒绝请求")
                raise Exception(f"请求被限流: {name}")
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not limiter.acquire(timeout=timeout):
                logger.warning(f"限流器 {name} 拒绝请求")
                raise Exception(f"请求被限流: {name}")
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def combine_decorators(*decorators):
    """
    组合多个装饰器

    Usage:
        @combine_decorators(
            retry(max_attempts=3),
            circuit_breaker("service"),
            rate_limit("api", max_calls=10)
        )
        async def service_call():
            ...
    """
    def decorator(func: Callable):
        for dec in reversed(decorators):
            func = dec(func)
        return func
    return decorator

# 便捷的组合装饰器
def resilient_service(
    service_name: str,
    max_attempts: int = 3,
    failure_threshold: int = 5,
    max_calls: int = 100
):
    """
    为服务添加完整的弹性保护

    Args:
        service_name: 服务名称
        max_attempts: 最大重试次数
        failure_threshold: 熔断失败阈值
        max_calls: 限流最大调用次数

    Usage:
        @resilient_service("milvus", max_attempts=3, failure_threshold=5)
        async def insert_vectors():
            ...
    """
    return combine_decorators(
        retry(max_attempts=max_attempts),
        circuit_breaker(service_name, failure_threshold=failure_threshold),
        rate_limit(f"{service_name}_calls", max_calls=max_calls)
    )
