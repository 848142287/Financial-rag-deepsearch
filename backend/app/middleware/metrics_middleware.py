"""
指标收集中间件
"""

import time
from typing import Dict, Any
from collections import defaultdict, deque
from app.middleware.base_middleware import BaseMiddleware
from app.core.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)
from app.core.config import get_settings

class MetricsMiddleware(BaseMiddleware):
    """
    指标收集中间件

    收集RAG系统的性能指标，包括执行时间、成功率、调用频率等
    """

    def __init__(
        self,
        name: str = "MetricsMiddleware",
        max_history: int = 1000,
        alert_thresholds: Dict[str, float] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.settings = get_settings()
        self.max_history = max_history
        self.alert_thresholds = alert_thresholds or {
            "execution_time": 10.0,  # 10秒
            "error_rate": 0.1,      # 10%错误率
            "cache_miss_rate": 0.8  # 80%缓存未命中率
        }

        # 指标存储
        self.tool_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "call_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_execution_time": 0.0,
            "min_execution_time": float("inf"),
            "max_execution_time": 0.0,
            "recent_execution_times": deque(maxlen=max_history),
            "recent_success_rates": deque(maxlen=100)
        })

        self.system_metrics = {
            "start_time": time.time(),
            "total_calls": 0,
            "total_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    async def before_tool_run(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行前记录开始时间"""
        start_time = time.time()

        # 记录调用开始
        self.tool_metrics[tool_name]["call_count"] += 1
        self.system_metrics["total_calls"] += 1

        # 添加开始时间到输入中
        inputs["_metrics_start_time"] = start_time
        return inputs

    async def after_tool_run(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """工具执行后收集指标"""
        end_time = time.time()
        start_time = inputs.pop("_metrics_start_time", end_time)
        actual_execution_time = end_time - start_time

        # 更新工具指标
        metrics = self.tool_metrics[tool_name]
        metrics["total_execution_time"] += actual_execution_time
        metrics["recent_execution_times"].append(actual_execution_time)

        # 更新最小和最大执行时间
        if actual_execution_time < metrics["min_execution_time"]:
            metrics["min_execution_time"] = actual_execution_time
        if actual_execution_time > metrics["max_execution_time"]:
            metrics["max_execution_time"] = actual_execution_time

        # 更新成功/错误计数
        if outputs.get("success", False):
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1
            self.system_metrics["total_errors"] += 1

        # 更新缓存指标
        if outputs.get("_cache_hit", False):
            self.system_metrics["cache_hits"] += 1
        else:
            self.system_metrics["cache_misses"] += 1

        # 检查告警阈值
        await self._check_alerts(tool_name, actual_execution_time, outputs.get("success", False))

        # 更新成功率
        current_success_rate = metrics["success_count"] / metrics["call_count"]
        metrics["recent_success_rates"].append(current_success_rate)

        logger.debug(
            f"[{self.name}] Metrics updated for tool '{tool_name}': "
            f"time={actual_execution_time:.3f}s, success={outputs.get('success', False)}",
            extra={
                "tool_name": tool_name,
                "execution_time": actual_execution_time,
                "success": outputs.get("success", False),
                "call_count": metrics["call_count"],
                "success_rate": current_success_rate
            }
        )

        return outputs

    async def _check_alerts(self, tool_name: str, execution_time: float, success: bool):
        """检查告警条件"""
        # 执行时间告警
        if execution_time > self.alert_thresholds.get("execution_time", 10.0):
            logger.warning(
                f"[{self.name}] Performance alert: Tool '{tool_name}' execution time "
                f"({execution_time:.3f}s) exceeds threshold ({self.alert_thresholds['execution_time']}s)",
                extra={
                    "alert_type": "slow_execution",
                    "tool_name": tool_name,
                    "execution_time": execution_time,
                    "threshold": self.alert_thresholds["execution_time"]
                }
            )

        # 错误率告警
        metrics = self.tool_metrics[tool_name]
        if metrics["call_count"] >= 10:  # 至少10次调用才检查错误率
            error_rate = metrics["error_count"] / metrics["call_count"]
            if error_rate > self.alert_thresholds.get("error_rate", 0.1):
                logger.warning(
                    f"[{self.name}] Error rate alert: Tool '{tool_name}' error rate "
                    f"({error_rate:.2%}) exceeds threshold ({self.alert_thresholds['error_rate']:.2%})",
                    extra={
                        "alert_type": "high_error_rate",
                        "tool_name": tool_name,
                        "error_rate": error_rate,
                        "error_count": metrics["error_count"],
                        "call_count": metrics["call_count"],
                        "threshold": self.alert_thresholds["error_rate"]
                    }
                )

        # 缓存未命中率告警
        total_cache_requests = self.system_metrics["cache_hits"] + self.system_metrics["cache_misses"]
        if total_cache_requests >= 50:  # 至少50次缓存请求才检查
            cache_miss_rate = self.system_metrics["cache_misses"] / total_cache_requests
            if cache_miss_rate > self.alert_thresholds.get("cache_miss_rate", 0.8):
                logger.warning(
                    f"[{self.name}] Cache miss rate alert: Cache miss rate "
                    f"({cache_miss_rate:.2%}) exceeds threshold ({self.alert_thresholds['cache_miss_rate']:.2%})",
                    extra={
                        "alert_type": "high_cache_miss_rate",
                        "cache_miss_rate": cache_miss_rate,
                        "cache_hits": self.system_metrics["cache_hits"],
                        "cache_misses": self.system_metrics["cache_misses"],
                        "threshold": self.alert_thresholds["cache_miss_rate"]
                    }
                )

    def get_tool_metrics(self, tool_name: str = None) -> Dict[str, Any]:
        """获取工具指标"""
        if tool_name:
            if tool_name in self.tool_metrics:
                return self._calculate_tool_stats(tool_name)
            else:
                return {}
        else:
            return {name: self._calculate_tool_stats(name) for name in self.tool_metrics}

    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        current_time = time.time()
        uptime = current_time - self.system_metrics["start_time"]

        total_cache_requests = self.system_metrics["cache_hits"] + self.system_metrics["cache_misses"]
        cache_hit_rate = (
            self.system_metrics["cache_hits"] / total_cache_requests
            if total_cache_requests > 0 else 0.0
        )

        overall_error_rate = (
            self.system_metrics["total_errors"] / self.system_metrics["total_calls"]
            if self.system_metrics["total_calls"] > 0 else 0.0
        )

        return {
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "total_calls": self.system_metrics["total_calls"],
            "total_errors": self.system_metrics["total_errors"],
            "overall_error_rate": overall_error_rate,
            "cache_hits": self.system_metrics["cache_hits"],
            "cache_misses": self.system_metrics["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "tools_count": len(self.tool_metrics)
        }

    def _calculate_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """计算工具统计数据"""
        metrics = self.tool_metrics[tool_name]
        call_count = metrics["call_count"]

        if call_count == 0:
            return {}

        avg_execution_time = metrics["total_execution_time"] / call_count
        success_rate = metrics["success_count"] / call_count
        error_rate = metrics["error_count"] / call_count

        # 计算最近的平均执行时间
        recent_times = list(metrics["recent_execution_times"])
        recent_avg_time = sum(recent_times) / len(recent_times) if recent_times else 0

        return {
            "call_count": call_count,
            "success_count": metrics["success_count"],
            "error_count": metrics["error_count"],
            "success_rate": success_rate,
            "error_rate": error_rate,
            "avg_execution_time": avg_execution_time,
            "recent_avg_execution_time": recent_avg_time,
            "min_execution_time": metrics["min_execution_time"],
            "max_execution_time": metrics["max_execution_time"],
            "recent_calls": len(recent_times)
        }

    def _format_uptime(self, seconds: float) -> str:
        """格式化运行时间"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    def reset_metrics(self, tool_name: str = None):
        """重置指标"""
        if tool_name:
            if tool_name in self.tool_metrics:
                del self.tool_metrics[tool_name]
        else:
            self.tool_metrics.clear()
            self.system_metrics = {
                "start_time": time.time(),
                "total_calls": 0,
                "total_errors": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }