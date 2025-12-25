"""
Redis客户端封装
提供Redis连接和基本操作
"""

import redis.asyncio as redis
from typing import Any, Optional, List, Union
import json
import logging
from .config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis异步客户端"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False

    async def connect(self):
        """连接Redis"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,  # 自动解码为字符串
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # 测试连接
            result = self.redis_client.ping()
            if hasattr(result, '__await__'):
                await result
            self.is_connected = True
            logger.info("Redis连接成功")

        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """断开Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Redis连接已断开")

    async def get(self, key: str) -> Optional[str]:
        """获取值"""
        if not self.is_connected:
            await self.connect()

        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis GET失败: {key}, 错误: {e}")
            return None

    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """设置值"""
        if not self.is_connected:
            await self.connect()

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)

            await self.redis_client.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.error(f"Redis SET失败: {key}, 错误: {e}")
            return False

    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        """设置带过期时间的值"""
        return await self.set(key, value, ex=seconds)

    async def delete(self, key: str) -> bool:
        """删除键"""
        if not self.is_connected:
            await self.connect()

        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE失败: {key}, 错误: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.is_connected:
            await self.connect()

        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis EXISTS失败: {key}, 错误: {e}")
            return False

    async def keys(self, pattern: str) -> List[str]:
        """获取匹配模式的所有键"""
        if not self.is_connected:
            await self.connect()

        try:
            keys = await self.redis_client.keys(pattern)
            return keys
        except Exception as e:
            logger.error(f"Redis KEYS失败: {pattern}, 错误: {e}")
            return []

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """递增"""
        if not self.is_connected:
            await self.connect()

        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCR失败: {key}, 错误: {e}")
            return None

    async def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        if not self.is_connected:
            await self.connect()

        try:
            result = await self.redis_client.expire(key, seconds)
            return result
        except Exception as e:
            logger.error(f"Redis EXPIRE失败: {key}, 错误: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """获取剩余生存时间"""
        if not self.is_connected:
            await self.connect()

        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL失败: {key}, 错误: {e}")
            return -1

    async def info(self, section: Optional[str] = None) -> dict:
        """获取Redis信息"""
        if not self.is_connected:
            await self.connect()

        try:
            info = await self.redis_client.info(section)
            return info
        except Exception as e:
            logger.error(f"Redis INFO失败: {section}, 错误: {e}")
            return {}

    async def ping(self) -> bool:
        """测试连接"""
        try:
            if not self.redis_client:
                await self.connect()
            result = self.redis_client.ping()
            if hasattr(result, '__await__'):
                await result
            return True
        except Exception as e:
            logger.error(f"Redis PING失败: {e}")
            self.is_connected = False
            return False

    async def flushdb(self) -> bool:
        """清空当前数据库"""
        if not self.is_connected:
            await self.connect()

        try:
            await self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB失败: {e}")
            return False

    # 哈希操作
    async def hget(self, name: str, key: str) -> Optional[str]:
        """获取哈希字段"""
        if not self.is_connected:
            await self.connect()

        try:
            return await self.redis_client.hget(name, key)
        except Exception as e:
            logger.error(f"Redis HGET失败: {name}:{key}, 错误: {e}")
            return None

    async def hset(self, name: str, key: str, value: Any) -> bool:
        """设置哈希字段"""
        if not self.is_connected:
            await self.connect()

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)

            result = await self.redis_client.hset(name, key, value)
            return True
        except Exception as e:
            logger.error(f"Redis HSET失败: {name}:{key}, 错误: {e}")
            return False

    async def hgetall(self, name: str) -> dict:
        """获取所有哈希字段"""
        if not self.is_connected:
            await self.connect()

        try:
            result = await self.redis_client.hgetall(name)
            return result
        except Exception as e:
            logger.error(f"Redis HGETALL失败: {name}, 错误: {e}")
            return {}

    # 列表操作
    async def lpush(self, name: str, *values: Any) -> int:
        """列表左侧推入"""
        if not self.is_connected:
            await self.connect()

        try:
            formatted_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    formatted_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    formatted_values.append(str(value))

            result = await self.redis_client.lpush(name, *formatted_values)
            return result
        except Exception as e:
            logger.error(f"Redis LPUSH失败: {name}, 错误: {e}")
            return 0

    async def rpop(self, name: str) -> Optional[str]:
        """列表右侧弹出"""
        if not self.is_connected:
            await self.connect()

        try:
            result = await self.redis_client.rpop(name)
            return result
        except Exception as e:
            logger.error(f"Redis RPOP失败: {name}, 错误: {e}")
            return None

    async def llen(self, name: str) -> int:
        """获取列表长度"""
        if not self.is_connected:
            await self.connect()

        try:
            result = await self.redis_client.llen(name)
            return result
        except Exception as e:
            logger.error(f"Redis LLEN失败: {name}, 错误: {e}")
            return 0

    # 集合操作
    async def sadd(self, name: str, *values: Any) -> int:
        """添加到集合"""
        if not self.is_connected:
            await self.connect()

        try:
            formatted_values = [str(v) for v in values]
            result = await self.redis_client.sadd(name, *formatted_values)
            return result
        except Exception as e:
            logger.error(f"Redis SADD失败: {name}, 错误: {e}")
            return 0

    async def smembers(self, name: str) -> set:
        """获取集合成员"""
        if not self.is_connected:
            await self.connect()

        try:
            result = await self.redis_client.smembers(name)
            return result
        except Exception as e:
            logger.error(f"Redis SMEMBERS失败: {name}, 错误: {e}")
            return set()


# 全局Redis客户端实例
redis_client = RedisClient()


# 兼容性函数
async def get_redis_client():
    """获取Redis客户端实例"""
    if not redis_client.is_connected:
        await redis_client.connect()
    return redis_client


# 同步版本的简单包装器（用于非异步上下文）
import redis

def get_sync_redis_client():
    """获取同步Redis客户端"""
    try:
        return redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
    except Exception as e:
        logger.error(f"同步Redis客户端创建失败: {e}")
        return None