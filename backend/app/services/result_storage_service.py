"""
解析结果缓存与存储服务
提供专门针对文档解析结果的存储和缓存功能
"""

import json
from app.core.structured_logging import get_structured_logger
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
import aiofiles.os

import redis.asyncio as redis

from app.core.config import settings
from app.db.mysql import get_db

logger = get_structured_logger(__name__)


class StorageType(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"
    FILESYSTEM = "filesystem"
    MYSQL = "mysql"


@dataclass
class StorageResult:
    """存储结果"""
    success: bool
    location: str
    size_bytes: int
    stored_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultStorageService:
    """解析结果存储服务"""

    def __init__(self):
        self.redis_client = None
        self.file_storage_dir = Path(settings.FILE_STORAGE_PATH or "/tmp/document_results")
        self.memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_memory_entries = 1000
        self.default_ttl = 3600  # 1小时

    async def initialize(self):
        """初始化服务"""
        # 初始化Redis
        self.redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

        # 创建文件存储目录
        await aiofiles.os.makedirs(self.file_storage_dir, exist_ok=True)

        logger.info("Result storage service initialized")

    async def store_parse_result(
        self,
        document_id: str,
        parse_result: Dict[str, Any],
        storage_type: StorageType = StorageType.REDIS,
        ttl: Optional[int] = None
    ) -> StorageResult:
        """存储文档解析结果"""
        try:
            storage_key = f"parse_result:{document_id}"
            ttl = ttl or self.default_ttl

            if storage_type == StorageType.MEMORY:
                result = await self._store_to_memory(storage_key, parse_result, ttl)
            elif storage_type == StorageType.REDIS:
                result = await self._store_to_redis(storage_key, parse_result, ttl)
            elif storage_type == StorageType.FILESYSTEM:
                result = await self._store_to_filesystem(storage_key, parse_result, ttl)
            else:
                result = await self._store_to_memory(storage_key, parse_result, ttl)

            logger.debug(f"Stored parse result for document {document_id}")
            return result

        except Exception as e:
            logger.error(f"Error storing embedding cache: {e}")
            return StorageResult(
                success=False,
                location="",
                size_bytes=0,
                metadata={"error": str(e)}
            )

    async def get_embedding_cache(self, text_hash: str) -> Optional[List[float]]:
        """获取嵌入缓存"""
        storage_key = f"embedding:{text_hash}"

        try:
            # 优先从Redis获取
            if self.redis_client:
                result = await self._get_from_storage(storage_key, StorageType.REDIS)
                if result is not None:
                    return result

            # 其次从内存获取
            result = await self._get_from_storage(storage_key, StorageType.MEMORY)
            if result is not None:
                return result

            return None

        except Exception as e:
            logger.error(f"Error getting embedding cache: {e}")
            return None

    async def store_document_summary(
        self,
        document_id: str,
        summary: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> StorageResult:
        """存储文档摘要"""
        try:
            storage_key = f"document_summary:{document_id}"
            ttl = ttl or 7200  # 2小时

            # 摘要存储到多个地方以确保可用性
            results = []

            # 存储到Redis（快速访问）
            if self.redis_client:
                result = await self._store_to_redis(storage_key, summary, ttl)
                results.append(result)

                results.append(result)

            # 存储到文件系统（备份）
            file_result = await self._store_to_filesystem(storage_key, summary, ttl)
            results.append(file_result)

            # 返回主要结果（Redis）
            main_result = results[0] if results else StorageResult(success=False, location="", size_bytes=0)

            logger.info(f"Stored document summary for {document_id}")
            return main_result

        except Exception as e:
            logger.error(f"Error storing document summary for {document_id}: {e}")
            return StorageResult(
                success=False,
                location="",
                size_bytes=0,
                metadata={"error": str(e)}
            )

    async def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档摘要"""
        storage_key = f"document_summary:{document_id}"

        try:
            # 按优先级获取
            result = await self._get_from_storage(storage_key, storage_type)
            if result is not None:
                return result

            return None

        except Exception as e:
            logger.error(f"Error getting document summary for {document_id}: {e}")
            return None

    async def batch_store_chunk_analyses(
        self,
        analyses: Dict[str, Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> Dict[str, StorageResult]:
        """批量存储块分析结果"""
        results = {}

        try:
            # 逐个存储
            for chunk_id, analysis in analyses.items():
                result = await self.store_chunk_analysis(chunk_id, analysis, ttl)
                results[chunk_id] = result

            logger.info(f"Batch stored {len(analyses)} chunk analyses")
            return results

        except Exception as e:
            logger.error(f"Error in batch store chunk analyses: {e}")
            # 为所有分析返回失败结果
            for chunk_id in analyses:
                results[chunk_id] = StorageResult(
                    success=False,
                    location="",
                    size_bytes=0,
                    metadata={"error": str(e)}
                )
            return results

    async def delete_document_results(self, document_id: str) -> bool:
        """删除文档的所有相关结果"""
        try:
            success_count = 0
            total_count = 0

            # 删除解析结果
            storage_key = f"parse_result:{document_id}"
            for storage_type in StorageType:
                if await self._delete_from_storage(storage_key, storage_type):
                    success_count += 1
                total_count += 1

            # 删除文档摘要
            summary_key = f"document_summary:{document_id}"
            for storage_type in StorageType:
                if await self._delete_from_storage(summary_key, storage_type):
                    success_count += 1
                total_count += 1

            # 删除块分析结果（需要查询）
            chunk_keys = await self._get_document_chunk_keys(document_id)
            for chunk_key in chunk_keys:
                for storage_type in StorageType:
                    if await self._delete_from_storage(chunk_key, storage_type):
                        success_count += 1
                    total_count += 1

            # 删除MySQL索引
            await self._remove_storage_index(document_id)

            logger.info(f"Deleted document results for {document_id}: {success_count}/{total_count}")
            return success_count > 0

        except Exception as e:
            logger.error(f"Error deleting document results for {document_id}: {e}")
            return False

    async def cleanup_expired_results(self) -> Dict[str, int]:
        """清理过期结果"""
        cleanup_stats = {
            "memory": 0,
            "redis": 0,
            "filesystem": 0
        }

        try:
            # 清理内存缓存
            current_time = datetime.utcnow()
            expired_keys = [
                key for key, (_, timestamp) in self.memory_cache.items()
                if current_time - timestamp > timedelta(seconds=self.default_ttl)
            ]
            for key in expired_keys:
                del self.memory_cache[key]
                cleanup_stats["memory"] += 1

            # 清理Redis过期键（Redis会自动清理，这里只是统计）
            if self.redis_client:
                # 获取所有相关键并检查TTL
                pattern = "*"
                keys = await self.redis_client.keys(pattern)
                for key in keys:
                    ttl = await self.redis_client.ttl(key)
                    if ttl == -1:  # 没有过期时间的键
                        pass
                    elif ttl == -2:  # 已过期的键
                        await self.redis_client.delete(key)
                        cleanup_stats["redis"] += 1

                result = await collection.delete_many({
                    "expires_at": {"$lt": current_time}
                })

            # 清理文件系统过期文件
            for file_path in self.file_storage_dir.glob("*.cache"):
                if await aiofiles.os.path.exists(file_path):
                    stat = await aiofiles.os.stat(file_path)
                    file_age = current_time - datetime.fromtimestamp(stat.st_mtime)
                    if file_age > timedelta(hours=24):  # 24小时后删除
                        await aiofiles.os.remove(file_path)
                        cleanup_stats["filesystem"] += 1

            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return cleanup_stats

    async def get_storage_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            "memory_cache": {
                "entries": len(self.memory_cache),
                "max_entries": self.max_memory_entries
            },
            "redis": {},
            "filesystem": {}
        }

        try:
            # Redis统计
            if self.redis_client:
                info = await self.redis_client.info("memory")
                count = await collection.count_documents({})
                stats["redis"] = {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "used_memory_peak": info.get("used_memory_peak_human", "N/A"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "cached_documents": count
                }

            # 文件系统统计
            file_count = 0
            total_size = 0
            for file_path in self.file_storage_dir.glob("*.cache"):
                file_count += 1
                stat = await aiofiles.os.stat(file_path)
                total_size += stat.st_size

            stats["filesystem"] = {
                "file_count": file_count,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            }

        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")

        return stats

    # 私有方法
    async def _store_to_memory(self, key: str, value: Any, ttl: int) -> StorageResult:
        """存储到内存"""
        # 检查内存限制
        if len(self.memory_cache) >= self.max_memory_entries:
            # 删除最旧的条目
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k][1]
            )
            del self.memory_cache[oldest_key]

        self.memory_cache[key] = (value, datetime.utcnow())
        return StorageResult(
            success=True,
            location="memory",
            size_bytes=len(str(value)),
            metadata={"ttl": ttl}
        )

    async def _store_to_redis(self, key: str, value: Any, ttl: int) -> StorageResult:
        """存储到Redis"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")

        serialized = json.dumps(value, default=str)
        success = await self.redis_client.setex(key, ttl, serialized)

        return StorageResult(
            success=success,
            location="redis",
            size_bytes=len(serialized.encode()),
            metadata={"ttl": ttl}
        )


        document = {
            "key": key,
            "value": value,
            "type": self._get_cache_type_from_key(key),
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl)
        }

        result = await collection.replace_one(
            {"key": key},
            document,
            upsert=True
        )

        return StorageResult(
            success=result.acknowledged,
            size_bytes=len(str(value)),
            metadata={"ttl": ttl, "upserted_id": str(result.upserted_id) if result.upserted_id else None}
        )

    async def _store_to_filesystem(self, key: str, value: Any, ttl: int) -> StorageResult:
        """存储到文件系统"""
        file_path = self.file_storage_dir / f"{key}.cache"

        cache_data = {
            "key": key,
            "value": value,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()
        }

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(pickle.dumps(cache_data))

        return StorageResult(
            success=True,
            location=str(file_path),
            size_bytes=len(pickle.dumps(cache_data)),
            metadata={"ttl": ttl}
        )

    async def _get_from_storage(self, key: str, storage_type: StorageType) -> Optional[Any]:
        """从指定存储获取数据"""
        try:
            if storage_type == StorageType.MEMORY:
                if key in self.memory_cache:
                    value, _ = self.memory_cache[key]
                    return value

            elif storage_type == StorageType.REDIS and self.redis_client:
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data)
