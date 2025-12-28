"""
缓存中间件
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional
from app.middleware.base_middleware import BaseMiddleware
from app.services.redis_service import RedisService
from app.core.config import get_settings
from app.core.logging import logger

class CacheMiddleware(BaseMiddleware):
    """
    缓存中间件

    为RAG系统工具提供缓存功能，提高性能并减少重复计算
    """

    def __init__(
        self,
        name: str = "CacheMiddleware",
        cache_ttl: int = 3600,
        cache_prefix: str = "rag_cache:",
        enable_cache: bool = True,
        cache_size_limit: int = 1000,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.settings = get_settings()
        self.cache_ttl = cache_ttl
        self.cache_prefix = cache_prefix
        self.enable_cache = enable_cache
        self.cache_size_limit = cache_size_limit
        self.redis_service = None
        self._init_redis()

    def _init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_service = RedisService()
        except Exception as e:
            logger.error(f"Failed to initialize Redis for caching: {e}")
            self.enable_cache = False

    def _generate_cache_key(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 创建一个不包含敏感信息的inputs副本
        cache_inputs = {
            k: v for k, v in inputs.items()
            if not k.startswith("_") and k not in ["token", "password", "secret"]
        }

        # 序列化输入参数
        inputs_str = json.dumps(cache_inputs, sort_keys=True, default=str)

        # 生成哈希
        hash_obj = hashlib.md5(inputs_str.encode("utf-8"))
        input_hash = hash_obj.hexdigest()

        return f"{self.cache_prefix}{tool_name}:{input_hash}"

    async def before_tool_run(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行前检查缓存"""
        if not self.enable_cache or not self.redis_service:
            return inputs

        try:
            cache_key = self._generate_cache_key(tool_name, inputs)
            cached_result = await self.redis_service.get(cache_key)

            if cached_result:
                # 缓存命中
                cached_data = json.loads(cached_result)
                logger.info(
                    f"[{self.name}] Cache hit for tool '{tool_name}'",
                    extra={
                        "tool_name": tool_name,
                        "cache_key": cache_key,
                        "cached_at": cached_data.get("cached_at"),
                        "ttl": self.cache_ttl
                    }
                )

                # 设置缓存标志
                inputs["_cache_hit"] = True
                inputs["_cached_result"] = cached_data["result"]
            else:
                inputs["_cache_hit"] = False
                inputs["_cache_key"] = cache_key

        except Exception as e:
            logger.error(f"[{self.name}] Cache check failed: {e}")
            inputs["_cache_hit"] = False

        return inputs

    async def after_tool_run(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """工具执行后保存缓存"""
        if not self.enable_cache or not self.redis_service:
            return outputs

        try:
            # 检查是否缓存命中
            if inputs.get("_cache_hit", False):
                # 使用缓存结果，不需要保存
                outputs = inputs.get("_cached_result", outputs)
                outputs["_from_cache"] = True
                outputs["_cache_hit"] = True
                return outputs

            # 保存新的缓存结果
            cache_key = inputs.get("_cache_key")
            if cache_key:
                cache_data = {
                    "result": outputs,
                    "tool_name": tool_name,
                    "execution_time": execution_time,
                    "cached_at": time.time(),
                    "ttl": self.cache_ttl
                }

                await self.redis_service.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(cache_data, default=str)
                )

                logger.info(
                    f"[{self.name}] Cached result for tool '{tool_name}'",
                    extra={
                        "tool_name": tool_name,
                        "cache_key": cache_key,
                        "execution_time": execution_time,
                        "ttl": self.cache_ttl
                    }
                )

            outputs["_from_cache"] = False
            outputs["_cache_hit"] = False

        except Exception as e:
            logger.error(f"[{self.name}] Cache save failed: {e}")
            outputs["_from_cache"] = False
            outputs["_cache_hit"] = False

        return outputs

    async def invalidate_cache(self, tool_name: Optional[str] = None) -> int:
        """使缓存失效"""
        if not self.enable_cache or not self.redis_service:
            return 0

        try:
            if tool_name:
                # 删除特定工具的缓存
                pattern = f"{self.cache_prefix}{tool_name}:*"
                keys = await self.redis_service.keys(pattern)
                if keys:
                    await self.redis_service.delete(*keys)
                    return len(keys)
            else:
                # 删除所有RAG缓存
                pattern = f"{self.cache_prefix}*"
                keys = await self.redis_service.keys(pattern)
                if keys:
                    await self.redis_service.delete(*keys)
                    return len(keys)

        except Exception as e:
            logger.error(f"[{self.name}] Cache invalidation failed: {e}")

        return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.enable_cache or not self.redis_service:
            return {"enabled": False}

        try:
            pattern = f"{self.cache_prefix}*"
            keys = await self.redis_service.keys(pattern)

            stats = {
                "enabled": True,
                "total_keys": len(keys),
                "cache_prefix": self.cache_prefix,
                "ttl": self.cache_ttl
            }

            # 按工具名称分组统计
            tool_stats = {}
            for key in keys:
                # 提取工具名称
                key_parts = key.replace(self.cache_prefix, "").split(":")
                if len(key_parts) >= 2:
                    tool_name = key_parts[0]
                    tool_stats[tool_name] = tool_stats.get(tool_name, 0) + 1

            stats["tool_stats"] = tool_stats
            return stats

        except Exception as e:
            logger.error(f"[{self.name}] Cache stats failed: {e}")
            return {"enabled": True, "error": str(e)}

    async def clear_expired_cache(self) -> int:
        """清理过期缓存（Redis会自动清理，这里是手动清理）"""
        # Redis的TTL机制会自动清理过期键
        # 这里可以添加额外的清理逻辑
        return 0