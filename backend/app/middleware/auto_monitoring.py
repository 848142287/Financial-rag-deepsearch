"""
自动监控中间件 - Auto Monitoring Middleware

功能:
1. 自动记录所有API请求的响应时间
2. 自动记录错误率
3. 自动性能分析
4. 零侵入式监控
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.structured_logging import get_structured_logger


logger = get_structured_logger(__name__)


class AutoMonitoringMiddleware(BaseHTTPMiddleware):
    """
    自动监控中间件

    自动收集:
    1. 请求响应时间
    2. 错误率
    3. 状态码分布
    4. 端点性能
    """

    def __init__(self, app, enable_profiling: bool = True):
        super().__init__(app)
        self.enable_profiling = enable_profiling

        # 端点性能统计
        self.endpoint_stats = {}

        logger.info("✅ 自动监控中间件已启用")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"

        try:
            # 调用下一个中间件/路由
            response = await call_next(request)

            # 计算响应时间
            duration_ms = (time.time() - start_time) * 1000

            # 记录指标
            self._record_metrics(endpoint, duration_ms, response.status_code)

            # 记录性能数据
            if self.enable_profiling:
                self._record_performance(endpoint, duration_ms)

            return response

        except Exception as e:
            # 记录错误
            duration_ms = (time.time() - start_time) * 1000
            self._record_error(endpoint, duration_ms, str(e))
            raise

    def _record_metrics(self, endpoint: str, duration_ms: float, status_code: int):
        """记录指标"""
        try:
            monitoring_system = get_enhanced_monitoring_system()
            if not monitoring_system:
                return

            # 记录响应时间
            response_metric = Metric(
                name=f"http.response_time",
                type=MetricType.HISTOGRAM,
                value=duration_ms,
                timestamp=datetime.now(),
                labels={
                    "endpoint": endpoint,
                    "status_code": str(status_code)
                }
            )

            monitoring_system.collect_metric(response_metric)

            # 记录请求计数
            request_metric = Metric(
                name=f"http.requests_total",
                type=MetricType.COUNTER,
                value=1.0,
                timestamp=datetime.now(),
                labels={
                    "endpoint": endpoint,
                    "status_code": str(status_code)
                }
            )

            monitoring_system.collect_metric(request_metric)

        except Exception as e:
            logger.error(f"记录指标失败: {e}")

    def _record_performance(self, endpoint: str, duration_ms: float):
        """记录性能数据"""
        try:
            # 更新端点统计
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }

            stats = self.endpoint_stats[endpoint]
            stats["count"] += 1
            stats["total_time"] += duration_ms
            stats["min_time"] = min(stats["min_time"], duration_ms)
            stats["max_time"] = max(stats["max_time"], duration_ms)

        except Exception as e:
            logger.error(f"记录性能失败: {e}")

    def _record_error(self, endpoint: str, duration_ms: float, error: str):
        """记录错误"""
        try:
            logger.error(f"API错误: {endpoint} | {duration_ms:.2f}ms | {error}")

            # 记录错误指标
            monitoring_system = get_enhanced_monitoring_system()
            if monitoring_system:
                error_metric = Metric(
                    name=f"http.errors_total",
                    type=MetricType.COUNTER,
                    value=1.0,
                    timestamp=datetime.now(),
                    labels={
                        "endpoint": endpoint,
                        "error_type": type(error).__name__
                    }
                )

                monitoring_system.collect_metric(error_metric)

        except Exception as e:
            logger.error(f"记录错误失败: {e}")

    def get_endpoint_stats(self) -> dict:
        """获取端点统计"""
        stats = {}
        for endpoint, data in self.endpoint_stats.items():
            if data["count"] > 0:
                stats[endpoint] = {
                    "count": data["count"],
                    "avg_time": data["total_time"] / data["count"],
                    "min_time": data["min_time"],
                    "max_time": data["max_time"]
                }
        return stats


from datetime import datetime  # 确保datetime导入在顶部
