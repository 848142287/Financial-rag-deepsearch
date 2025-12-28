"""
SLA强制执行模块
用于确保P95响应时间符合SLA要求
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from app.core.redis_client import get_redis_client
from app.models.evaluation import RetrievalLog

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """检索模式枚举"""
    SIMPLE = "simple"
    ENHANCED = "enhanced"
    DEEP_SEARCH = "deep_search"
    AGENTIC = "agentic"


@dataclass
class SLAThreshold:
    """SLA阈值配置"""
    tier: str
    p95_threshold_ms: float
    p99_threshold_ms: float
    fallback_mode: RetrievalMode
    auto_switch: bool = True


class SLAEnforcement:
    """SLA强制执行器"""

    def __init__(self):
        self.redis = get_redis_client()

        # 定义各层级的SLA阈值
        self.sla_thresholds = {
            RetrievalMode.SIMPLE: SLAThreshold(
                tier="tier1",
                p95_threshold_ms=3000.0,  # 3秒
                p99_threshold_ms=5000.0,  # 5秒
                fallback_mode=None,  # 最简单模式，无降级
                auto_switch=False
            ),
            RetrievalMode.ENHANCED: SLAThreshold(
                tier="tier2",
                p95_threshold_ms=8000.0,  # 8秒
                p99_threshold_ms=12000.0,  # 12秒
                fallback_mode=RetrievalMode.SIMPLE,
                auto_switch=True
            ),
            RetrievalMode.DEEP_SEARCH: SLAThreshold(
                tier="tier3",
                p95_threshold_ms=20000.0,  # 20秒
                p99_threshold_ms=30000.0,  # 30秒
                fallback_mode=RetrievalMode.ENHANCED,
                auto_switch=True
            ),
            RetrievalMode.AGENTIC: SLAThreshold(
                tier="agentic",
                p95_threshold_ms=25000.0,  # 25秒
                p99_threshold_ms=40000.0,  # 40秒
                fallback_mode=RetrievalMode.ENHANCED,
                auto_switch=True
            )
        }

        # 滑动窗口配置
        self.window_size = 100  # 最近100个请求
        self.min_sample_size = 10  # 最小样本数

        # 降级模式冷却时间
        self.fallback_cooldown = 300  # 5分钟

        logger.info("SLA Enforcement initialized")

    async def check_sla_compliance(self, mode: RetrievalMode) -> Tuple[bool, Dict]:
        """
        检查指定模式的SLA合规性

        Args:
            mode: 检索模式

        Returns:
            (is_compliant, metrics)
        """
        try:
            # 获取最近的响应时间数据
            recent_latencies = await self._get_recent_latencies(mode)

            if len(recent_latencies) < self.min_sample_size:
                # 样本不足，假设合规
                return True, {
                    "sample_size": len(recent_latencies),
                    "status": "insufficient_data"
                }

            # 计算P95和P99
            p95 = float(np.percentile(recent_latencies, 95))
            p99 = float(np.percentile(recent_latencies, 99))

            # 获取SLA阈值
            sla = self.sla_thresholds.get(mode)
            if not sla:
                logger.warning(f"No SLA defined for mode: {mode}")
                return True, {"status": "no_sla_defined"}

            # 检查是否合规
            is_compliant = (p95 <= sla.p95_threshold_ms and
                          p99 <= sla.p99_threshold_ms)

            metrics = {
                "sample_size": len(recent_latencies),
                "p95_ms": p95,
                "p99_ms": p99,
                "p95_threshold_ms": sla.p95_threshold_ms,
                "p99_threshold_ms": sla.p99_threshold_ms,
                "is_compliant": is_compliant,
                "avg_ms": np.mean(recent_latencies),
                "max_ms": np.max(recent_latencies),
                "min_ms": np.min(recent_latencies)
            }

            # 记录SLA检查结果
            await self._log_sla_check(mode, metrics)

            return is_compliant, metrics

        except Exception as e:
            logger.error(f"Error checking SLA compliance for {mode}: {str(e)}")
            return False, {"error": str(e)}

    async def get_recommended_mode(self, initial_mode: RetrievalMode) -> RetrievalMode:
        """
        基于SLA表现推荐最佳检索模式

        Args:
            initial_mode: 初始请求的模式

        Returns:
            推荐的检索模式
        """
        try:
            # 检查当前模式的SLA合规性
            is_compliant, metrics = await self.check_sla_compliance(initial_mode)

            if is_compliant or not self.sla_thresholds[initial_mode].auto_switch:
                # 合规或不允许自动切换，保持原模式
                return initial_mode

            # 检查是否在冷却期内
            if await self._is_in_cooldown(initial_mode):
                logger.info(f"Mode {initial_mode} is in cooldown period")
                return initial_mode

            # 需要降级到更简单的模式
            fallback_mode = self.sla_thresholds[initial_mode].fallback_mode
            if fallback_mode:
                logger.warning(
                    f"SLA violation detected for {initial_mode}. "
                    f"Falling back to {fallback_mode}"
                )

                # 触发降级模式
                await self._trigger_fallback(initial_mode, fallback_mode)

                return fallback_mode

            return initial_mode

        except Exception as e:
            logger.error(f"Error getting recommended mode: {str(e)}")
            return initial_mode

    async def record_latency(self, mode: RetrievalMode, latency_ms: float):
        """记录响应时间"""
        try:
            key = f"sla:latency:{mode.value}"
            timestamp = int(time.time() * 1000)

            # 使用有序集合存储带时间戳的延迟数据
            await self.redis.zadd(key, {str(timestamp): latency_ms})

            # 保持滑动窗口大小
            await self.redis.zremrangebyrank(key, 0, -self.window_size - 1)

            # 设置过期时间（24小时）
            await self.redis.expire(key, 86400)

        except Exception as e:
            logger.error(f"Error recording latency: {str(e)}")

    async def _get_recent_latencies(self, mode: RetrievalMode) -> List[float]:
        """获取最近的响应时间数据"""
        try:
            key = f"sla:latency:{mode.value}"
            data = await self.redis.zrange(key, 0, -1, withscores=True)

            # 只返回分数（延迟值），不包含时间戳
            return [float(score) for _, score in data]

        except Exception as e:
            logger.error(f"Error getting recent latencies: {str(e)}")
            return []

    async def _is_in_cooldown(self, mode: RetrievalMode) -> bool:
        """检查模式是否在冷却期内"""
        try:
            key = f"sla:cooldown:{mode.value}"
            return await self.redis.exists(key)

        except Exception as e:
            logger.error(f"Error checking cooldown: {str(e)}")
            return False

    async def _trigger_fallback(self, from_mode: RetrievalMode, to_mode: RetrievalMode):
        """触发降级模式"""
        try:
            key = f"sla:cooldown:{from_mode.value}"
            await self.redis.setex(key, self.fallback_cooldown, 1)

            # 记录降级事件
            event_data = {
                "from_mode": from_mode.value,
                "to_mode": to_mode.value,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "sla_violation"
            }

            event_key = f"sla:fallback_events"
            await self.redis.lpush(event_key, str(event_data))
            await self.redis.expire(event_key, 86400)

            logger.info(f"Fallback triggered: {from_mode.value} -> {to_mode.value}")

        except Exception as e:
            logger.error(f"Error triggering fallback: {str(e)}")

    async def _log_sla_check(self, mode: RetrievalMode, metrics: Dict):
        """记录SLA检查结果"""
        try:
            log_data = {
                "mode": mode.value,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics
            }

            log_key = f"sla:checks:{mode.value}"
            await self.redis.lpush(log_key, str(log_data))
            await self.redis.expire(log_key, 86400)

        except Exception as e:
            logger.error(f"Error logging SLA check: {str(e)}")

    async def get_sla_report(self, mode: Optional[RetrievalMode] = None) -> Dict:
        """获取SLA报告"""
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "modes": {}
            }

            modes_to_check = [mode] if mode else list(RetrievalMode)

            for m in modes_to_check:
                is_compliant, metrics = await self.check_sla_compliance(m)
                report["modes"][m.value] = metrics

                # 添加降级状态
                if await self._is_in_cooldown(m):
                    report["modes"][m.value]["in_cooldown"] = True
                else:
                    report["modes"][m.value]["in_cooldown"] = False

            return report

        except Exception as e:
            logger.error(f"Error generating SLA report: {str(e)}")
            return {"error": str(e)}

    async def reset_sla_data(self, mode: Optional[RetrievalMode] = None):
        """重置SLA数据"""
        try:
            if mode:
                # 删除特定模式的数据
                await self.redis.delete(f"sla:latency:{mode.value}")
                await self.redis.delete(f"sla:checks:{mode.value}")
            else:
                # 删除所有SLA数据
                keys = await self.redis.keys("sla:*")
                if keys:
                    await self.redis.delete(*keys)

            logger.info(f"SLA data reset for mode: {mode.value if mode else 'all'}")

        except Exception as e:
            logger.error(f"Error resetting SLA data: {str(e)}")


# 全局实例
sla_enforcement = SLAEnforcement()

# SLAEnforcementService类别名，用于兼容性
SLAEnforcementService = SLAEnforcement


# 装饰器：自动应用SLA检查
def enforce_sla(mode: RetrievalMode):
    """SLA强制执行装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 获取推荐的模式
            recommended_mode = await sla_enforcement.get_recommended_mode(mode)

            # 如果模式被降级，修改请求参数
            if recommended_mode != mode:
                logger.info(f"Mode switched from {mode} to {recommended_mode} due to SLA")
                # 这里需要根据实际函数参数调整模式
                if 'retrieval_mode' in kwargs:
                    kwargs['retrieval_mode'] = recommended_mode
                elif len(args) > 1:
                    args = list(args)
                    args[1] = recommended_mode
                    args = tuple(args)

            # 记录开始时间
            start_time = time.time()

            try:
                # 执行原函数
                result = await func(*args, **kwargs)
                return result

            finally:
                # 记录响应时间
                latency_ms = (time.time() - start_time) * 1000
                await sla_enforcement.record_latency(recommended_mode, latency_ms)

        return wrapper
    return decorator