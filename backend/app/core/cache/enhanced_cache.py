"""
增强缓存服务
提供多层级缓存和智能缓存管理
"""

import json
import hashlib
import logging
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio

from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


class EnhancedCacheService:
    """增强的缓存服务"""

    def __init__(self):
        self.default_ttl = 3600  # 默认TTL 1小时
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total": 0
        }

        # 分层缓存配置
        self.cache_layers = {
            "L1": {"ttl": 300, "description": "热点数据缓存(5分钟)"},  # 5分钟
            "L2": {"ttl": 1800, "description": "频繁访问缓存(30分钟)"},  # 30分钟
            "L3": {"ttl": 7200, "description": "长期缓存(2小时)"},  # 2小时
        }

    def _generate_cache_key(
        self,
        prefix: str,
        *args,
        **kwargs
    ) -> str:
        """生成缓存键"""
        # 将参数序列化为字符串
        key_parts = [prefix]

        if args:
            key_parts.extend(str(arg) for arg in args)

        if kwargs:
            # 对kwargs排序以确保一致性
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}={v}" for k, v in sorted_kwargs)

        key_string = ":".join(key_parts)

        # 如果key太长，使用hash
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"

        return key_string

    async def get(
        self,
        key: str,
        layer: str = "L2"
    ) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键
            layer: 缓存层级 (L1, L2, L3)

        Returns:
            缓存的值，如果不存在返回None
        """
        try:
            cache_key = f"{layer}:{key}"
            cached_data = await redis_client.get(cache_key)

            if cached_data:
                self.cache_stats["hits"] += 1
                self.cache_stats["total"] += 1
                logger.debug(f"缓存命中: {cache_key}")
                return json.loads(cached_data)
            else:
                self.cache_stats["misses"] += 1
                self.cache_stats["total"] += 1
                logger.debug(f"缓存未命中: {cache_key}")
                return None

        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            self.cache_stats["misses"] += 1
            self.cache_stats["total"] += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        layer: str = "L2"
    ) -> bool:
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存的值
            ttl: 过期时间（秒），None表示使用层级默认TTL
            layer: 缓存层级

        Returns:
            是否设置成功
        """
        try:
            cache_key = f"{layer}:{key}"

            if ttl is None:
                ttl = self.cache_layers.get(layer, {}).get("ttl", self.default_ttl)

            serialized_value = json.dumps(value, ensure_ascii=False)
            await redis_client.setex(cache_key, ttl, serialized_value)

            logger.debug(f"缓存设置成功: {cache_key}, TTL: {ttl}s")
            return True

        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False

    async def delete(self, key: str, layer: str = "L2") -> bool:
        """
        删除缓存

        Args:
            key: 缓存键
            layer: 缓存层级

        Returns:
            是否删除成功
        """
        try:
            cache_key = f"{layer}:{key}"
            await redis_client.delete(cache_key)
            logger.debug(f"缓存删除成功: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        批量删除匹配模式的缓存

        Args:
            pattern: 匹配模式

        Returns:
            删除的缓存数量
        """
        try:
            # 使用SCAN遍历匹配的键
            count = 0
            async for key in redis_client.scan_iter(match=pattern):
                await redis_client.delete(key)
                count += 1

            logger.info(f"批量删除缓存: {pattern}, 删除数量: {count}")
            return count

        except Exception as e:
            logger.error(f"批量删除缓存失败: {e}")
            return 0

    async def get_or_set(
        self,
        key: str,
        value_func: Callable,
        ttl: Optional[int] = None,
        layer: str = "L2"
    ) -> Any:
        """
        获取缓存，如果不存在则设置

        Args:
            key: 缓存键
            value_func: 值生成函数
            ttl: 过期时间
            layer: 缓存层级

        Returns:
            缓存的值
        """
        # 尝试从缓存获取
        cached_value = await self.get(key, layer=layer)

        if cached_value is not None:
            return cached_value

        # 缓存未命中，调用函数生成值
        try:
            if asyncio.iscoroutinefunction(value_func):
                value = await value_func()
            else:
                value = value_func()

            # 存入缓存
            await self.set(key, value, ttl=ttl, layer=layer)

            return value

        except Exception as e:
            logger.error(f"获取或设置缓存失败: {e}")
            return None

    async def get_multi_layer(
        self,
        key: str
    ) -> Optional[Any]:
        """
        从多层缓存中获取值

        Args:
            key: 缓存键（不含层级前缀）

        Returns:
            缓存的值，并自动提升缓存层级
        """
        # 按层级顺序查找 L1 -> L2 -> L3
        for layer in ["L1", "L2", "L3"]:
            value = await self.get(key, layer=layer)
            if value is not None:
                # 找到后，提升到更高层级的缓存
                if layer != "L1":
                    ttl = self.cache_layers["L1"]["ttl"]
                    await self.set(key, value, ttl=ttl, layer="L1")
                    logger.debug(f"缓存提升: {layer} -> L1")
                return value

        return None

    async def warm_up(
        self,
        keys_and_values: Dict[str, Any],
        layer: str = "L3"
    ) -> int:
        """
        缓存预热

        Args:
            keys_and_values: 键值对字典
            layer: 缓存层级

        Returns:
            成功预热的数量
        """
        success_count = 0

        for key, value in keys_and_values.items():
            try:
                if await self.set(key, value, layer=layer):
                    success_count += 1
            except Exception as e:
                logger.error(f"预热缓存失败 {key}: {e}")

        logger.info(f"缓存预热完成: {success_count}/{len(keys_and_values)}")
        return success_count

    def cache_decorator(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        layer: str = "L2",
        exclude_args: Optional[List[str]] = None
    ):
        """
        缓存装饰器

        Args:
            prefix: 缓存键前缀
            ttl: 过期时间
            layer: 缓存层级
            exclude_args: 排除的参数名列表
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 排除指定参数
                filtered_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in (exclude_args or [])
                }

                # 生成缓存键
                cache_key = self._generate_cache_key(
                    prefix,
                    *args,
                    **filtered_kwargs
                )

                # 尝试获取缓存
                cached_result = await self.get(cache_key, layer=layer)
                if cached_result is not None:
                    return cached_result

                # 执行函数
                result = await func(*args, **kwargs)

                # 存入缓存
                await self.set(cache_key, result, ttl=ttl, layer=layer)

                return result

            return wrapper

        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self.cache_stats["total"]
        hits = self.cache_stats["hits"]

        return {
            "hits": hits,
            "misses": self.cache_stats["misses"],
            "total": total,
            "hit_rate": f"{(hits / total * 100):.2f}%" if total > 0 else "0%",
            "cache_layers": self.cache_layers
        }

    async def reset_stats(self):
        """重置统计信息"""
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total": 0
        }


# 全局缓存服务实例
enhanced_cache = EnhancedCacheService()
