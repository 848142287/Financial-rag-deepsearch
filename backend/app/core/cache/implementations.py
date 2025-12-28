"""
Cache Implementation Classes

Concrete implementations of different cache backends for each cache level.
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as aioredis
from diskcache import Cache as DiskCacheLib

from .base import (
    BaseCache,
    CacheEntry,
    CacheConfig,
    CacheLevel,
    CachePolicy,
    CacheException,
    CacheConnectionException,
    CacheCapacityException
)

logger = logging.getLogger(__name__)


class MemoryCache(BaseCache):
    """L1 Memory cache implementation"""

    def __init__(self, config: CacheConfig):
        super().__init__(config, CacheLevel.L1_MEMORY)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_usage_bytes = 0
        self._is_async = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            with self._lock:
                if key not in self._cache:
                    self.metrics.misses += 1
                    return None

                entry = self._cache[key]

                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self.metrics.misses += 1
                    return None

                # Update access info
                entry.touch()

                # Move to end for LRU
                if self.config.policy == CachePolicy.LRU:
                    self._cache.move_to_end(key)

                self.metrics.hits += 1
                return entry.value

        except Exception as e:
            logger.error(f"Memory cache get error for key {key}: {e}")
            self.metrics.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        try:
            with self._lock:
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl or self.config.ttl
                )

                # Check capacity limits
                if not await self._check_capacity(entry):
                    logger.warning(f"Memory cache capacity exceeded for key {key}")
                    return False

                # Remove old entry if exists
                old_entry = self._cache.get(key)
                if old_entry:
                    self._memory_usage_bytes -= old_entry.size_bytes

                # Add new entry
                self._cache[key] = entry
                self._memory_usage_bytes += entry.size_bytes

                # Move to end for LRU
                if self.config.policy == CachePolicy.LRU:
                    self._cache.move_to_end(key)

                self.metrics.sets += 1
                return True

        except Exception as e:
            logger.error(f"Memory cache set error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry by key"""
        try:
            with self._lock:
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._memory_usage_bytes -= entry.size_bytes
                    self.metrics.deletes += 1
                    return True
                return False

        except Exception as e:
            logger.error(f"Memory cache delete error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def clear(self) -> bool:
        """Clear all entries"""
        try:
            with self._lock:
                self._cache.clear()
                self._memory_usage_bytes = 0
                self.metrics.deletes += len(self._cache)
                return True

        except Exception as e:
            logger.error(f"Memory cache clear error: {e}")
            self.metrics.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            with self._lock:
                if key not in self._cache:
                    return False
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    return False
                return True

        except Exception as e:
            logger.error(f"Memory cache exists error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get keys matching pattern"""
        try:
            with self._lock:
                if not pattern:
                    return list(self._cache.keys())

                # Simple pattern matching
                import fnmatch
                keys = []
                for key in self._cache.keys():
                    if fnmatch.fnmatch(key, pattern):
                        keys.append(key)
                return keys

        except Exception as e:
            logger.error(f"Memory cache keys error: {e}")
            self.metrics.errors += 1
            return []

    async def size(self) -> int:
        """Get number of entries"""
        with self._lock:
            return len(self._cache)

    async def _check_capacity(self, new_entry: CacheEntry) -> bool:
        """Check if cache can accept new entry"""
        # Check max entries
        if self.config.max_size and len(self._cache) >= self.config.max_size:
            await self._evict_entries()

        # Check memory limit
        if self.config.max_memory_mb:
            current_mb = self._memory_usage_bytes / (1024 * 1024)
            if current_mb >= self.config.max_memory_mb:
                await self._evict_entries()

        # Final check
        if self.config.max_size and len(self._cache) >= self.config.max_size:
            return False

        if self.config.max_memory_mb:
            current_mb = (self._memory_usage_bytes + new_entry.size_bytes) / (1024 * 1024)
            if current_mb >= self.config.max_memory_mb:
                return False

        return True

    async def _evict_entries(self):
        """Evict entries based on policy"""
        if not self._cache:
            return

        entries_to_remove = []

        if self.config.policy == CachePolicy.LRU:
            # Remove oldest accessed entries
            num_to_remove = max(1, len(self._cache) // 10)  # Remove 10%
            for _ in range(num_to_remove):
                if self._cache:
                    key, _ = self._cache.popitem(last=False)
                    entries_to_remove.append(key)

        elif self.config.policy == CachePolicy.TTL:
            # Remove expired entries
            current_time = datetime.utcnow()
            for key, entry in list(self._cache.items()):
                if entry.is_expired():
                    del self._cache[key]
                    entries_to_remove.append(key)

        elif self.config.policy == CachePolicy.LFU:
            # Remove least frequently used entries
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )
            num_to_remove = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:num_to_remove]:
                del self._cache[key]
                entries_to_remove.append(key)

        # Update memory usage
        for key in entries_to_remove:
            if key in self._cache:
                self._memory_usage_bytes -= self._cache[key].size_bytes
                self.metrics.evictions += 1

    async def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        removed_count = 0
        try:
            with self._lock:
                expired_keys = []
                for key, entry in self._cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)

                for key in expired_keys:
                    entry = self._cache.pop(key)
                    self._memory_usage_bytes -= entry.size_bytes
                    removed_count += 1

                self.metrics.evictions += removed_count

        except Exception as e:
            logger.error(f"Memory cache cleanup error: {e}")
            self.metrics.errors += 1

        return removed_count


class RedisCache(BaseCache):
    """L2 Redis distributed cache implementation"""

    def __init__(self, config: CacheConfig, host: str = "localhost",
                 port: int = 6379, db: int = 0, password: Optional[str] = None):
        super().__init__(config, CacheLevel.L2_REDIS)
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._redis: Optional[aioredis.Redis] = None
        self._is_async = True

    async def _get_connection(self) -> aioredis.Redis:
        """Get Redis connection"""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False
            )

            # Test connection
            await self._redis.ping()

        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            redis = await self._get_connection()
            data = await redis.get(self._make_key(key))

            if data is None:
                self.metrics.misses += 1
                return None

            # Deserialize
            entry = self._deserialize_entry(data)
            if entry.is_expired():
                await self.delete(key)
                self.metrics.misses += 1
                return None

            self.metrics.hits += 1
            return entry.value

        except Exception as e:
            logger.error(f"Redis cache get error for key {key}: {e}")
            self.metrics.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        try:
            redis = await self._get_connection()
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.config.ttl
            )

            data = self._serialize_entry(entry)

            if entry.ttl:
                success = await redis.setex(
                    self._make_key(key),
                    int(entry.ttl),
                    data
                )
            else:
                success = await redis.set(self._make_key(key), data)

            if success:
                self.metrics.sets += 1

            return success

        except Exception as e:
            logger.error(f"Redis cache set error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry by key"""
        try:
            redis = await self._get_connection()
            result = await redis.delete(self._make_key(key))

            if result > 0:
                self.metrics.deletes += 1
                return True

            return False

        except Exception as e:
            logger.error(f"Redis cache delete error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def clear(self) -> bool:
        """Clear all entries"""
        try:
            redis = await self._get_connection()
            pattern = self._make_key("*")
            keys = await redis.keys(pattern)

            if keys:
                result = await redis.delete(*keys)
                self.metrics.deletes += result

            return True

        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            self.metrics.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            redis = await self._get_connection()
            return await redis.exists(self._make_key(key)) > 0

        except Exception as e:
            logger.error(f"Redis cache exists error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get keys matching pattern"""
        try:
            redis = await self._get_connection()
            search_pattern = self._make_key(pattern or "*")
            keys = await redis.keys(search_pattern)

            # Remove prefix
            prefix = self._make_key("")
            return [key.decode().replace(prefix, "", 1) for key in keys]

        except Exception as e:
            logger.error(f"Redis cache keys error: {e}")
            self.metrics.errors += 1
            return []

    async def size(self) -> int:
        """Get number of entries"""
        try:
            redis = await self._get_connection()
            pattern = self._make_key("*")
            keys = await redis.keys(pattern)
            return len(keys)

        except Exception as e:
            logger.error(f"Redis cache size error: {e}")
            self.metrics.errors += 1
            return 0

    async def increment(self, key: str, delta: int = 1) -> Optional[int]:
        """Increment numeric value"""
        try:
            redis = await self._get_connection()
            return await redis.incrby(self._make_key(key), delta)

        except Exception as e:
            logger.error(f"Redis cache increment error for key {key}: {e}")
            self.metrics.errors += 1
            return None

    async def get_ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL for key"""
        try:
            redis = await self._get_connection()
            ttl = await redis.ttl(self._make_key(key))
            return ttl if ttl > 0 else None

        except Exception as e:
            logger.error(f"Redis cache TTL error for key {key}: {e}")
            self.metrics.errors += 1
            return None

    async def touch(self, key: str) -> bool:
        """Update access time for key"""
        # Redis doesn't have a direct touch, but we can get and reset TTL
        try:
            redis = await self._get_connection()
            ttl = await redis.ttl(self._make_key(key))
            if ttl > 0:
                await redis.expire(self._make_key(key), ttl)
                return True
            return False

        except Exception as e:
            logger.error(f"Redis cache touch error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    def _make_key(self, key: str) -> str:
        """Make Redis key with prefix"""
        return f"cache:{self.level}:{key}"

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry"""
        data = entry.to_dict()
        return json.dumps(data, default=str).encode()

    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry"""
        dict_data = json.loads(data.decode())
        return CacheEntry.from_dict(dict_data)


class DiskCache(BaseCache):
    """L3 Disk cache implementation using diskcache library"""

    def __init__(self, config: CacheConfig, directory: Optional[str] = None):
        super().__init__(config, CacheLevel.L3_DISK)
        self.directory = directory or tempfile.mkdtemp(prefix="cache_")
        self._disk_cache: Optional[DiskCacheLib] = None
        self._is_async = False

    async def _get_cache(self) -> DiskCacheLib:
        """Get disk cache instance"""
        if self._disk_cache is None:
            self._disk_cache = DiskCacheLib(
                self.directory,
                size_limit=self.config.max_memory_mb * 1024 * 1024 if self.config.max_memory_mb else None,
                eviction_policy=self.config.policy.value
            )
        return self._disk_cache

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            cache = await self._get_cache()
            data = cache.get(key)

            if data is None:
                self.metrics.misses += 1
                return None

            # Deserialize
            entry = self._deserialize_entry(data)
            if entry.is_expired():
                cache.delete(key)
                self.metrics.misses += 1
                return None

            self.metrics.hits += 1
            return entry.value

        except Exception as e:
            logger.error(f"Disk cache get error for key {key}: {e}")
            self.metrics.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        try:
            cache = await self._get_cache()
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.config.ttl
            )

            data = self._serialize_entry(entry)

            if entry.ttl:
                expire = timedelta(seconds=entry.ttl)
            else:
                expire = None

            cache.set(key, data, expire=expire)
            self.metrics.sets += 1
            return True

        except Exception as e:
            logger.error(f"Disk cache set error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry by key"""
        try:
            cache = await self._get_cache()
            result = cache.delete(key)

            if result:
                self.metrics.deletes += 1
                return True

            return False

        except Exception as e:
            logger.error(f"Disk cache delete error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def clear(self) -> bool:
        """Clear all entries"""
        try:
            cache = await self._get_cache()
            cache.clear()
            self.metrics.deletes += await self.size()
            return True

        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")
            self.metrics.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            cache = await self._get_cache()
            return cache.get(key) is not None

        except Exception as e:
            logger.error(f"Disk cache exists error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get keys matching pattern"""
        try:
            cache = await self._get_cache()
            import fnmatch

            if not pattern:
                return list(cache.iterkeys())

            keys = []
            for key in cache.iterkeys():
                if fnmatch.fnmatch(key, pattern):
                    keys.append(key)
            return keys

        except Exception as e:
            logger.error(f"Disk cache keys error: {e}")
            self.metrics.errors += 1
            return []

    async def size(self) -> int:
        """Get number of entries"""
        try:
            cache = await self._get_cache()
            return len(cache)

        except Exception as e:
            logger.error(f"Disk cache size error: {e}")
            self.metrics.errors += 1
            return 0

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry"""
        data = entry.to_dict()
        return pickle.dumps(data)

    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry"""
        dict_data = pickle.loads(data)
        return CacheEntry.from_dict(dict_data)


class DatabaseCache(BaseCache):
    """L4 Database cache implementation (fallback cache)"""

    def __init__(self, config: CacheConfig, connection_string: str):
        super().__init__(config, CacheLevel.L4_DATABASE)
        self.connection_string = connection_string
        self._connection = None
        self._is_async = False

    async def _get_connection(self):
        """Get database connection"""
        if self._connection is None:
            # Parse connection string and create connection
            # This would depend on the database type
            # For now, using SQLite as example
            if self.connection_string.startswith("sqlite"):
                self._connection = sqlite3.connect(
                    self.connection_string.replace("sqlite:///", ""),
                    check_same_thread=False
                )
                self._init_table()

        return self._connection

    def _init_table(self):
        """Initialize cache table"""
        cursor = self._connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value BLOB,
                ttl REAL,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER,
                tags TEXT,
                metadata TEXT
            )
        """)
        self._connection.commit()

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT value, ttl, created_at
                FROM cache_entries
                WHERE key = ?
            """, (key,))

            row = cursor.fetchone()

            if row is None:
                self.metrics.misses += 1
                return None

            value_data, ttl, created_at_str = row

            # Check expiration
            if ttl is not None:
                created_at = datetime.fromisoformat(created_at_str)
                if (datetime.utcnow() - created_at).total_seconds() > ttl:
                    await self.delete(key)
                    self.metrics.misses += 1
                    return None

            # Update access info
            cursor.execute("""
                UPDATE cache_entries
                SET last_accessed = ?, access_count = access_count + 1
                WHERE key = ?
            """, (datetime.utcnow().isoformat(), key))
            conn.commit()

            # Deserialize
            value = pickle.loads(value_data)
            self.metrics.hits += 1
            return value

        except Exception as e:
            logger.error(f"Database cache get error for key {key}: {e}")
            self.metrics.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            value_data = pickle.dumps(value)
            ttl = ttl or self.config.ttl
            created_at = datetime.utcnow().isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO cache_entries
                (key, value, ttl, created_at, last_accessed, access_count, tags, metadata)
                VALUES (?, ?, ?, ?, ?, 0, '', '{}')
            """, (key, value_data, ttl, created_at, created_at))

            conn.commit()
            self.metrics.sets += 1
            return True

        except Exception as e:
            logger.error(f"Database cache set error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry by key"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()

            if cursor.rowcount > 0:
                self.metrics.deletes += 1
                return True

            return False

        except Exception as e:
            logger.error(f"Database cache delete error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def clear(self) -> bool:
        """Clear all entries"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM cache_entries")
            conn.commit()

            self.metrics.deletes += cursor.rowcount
            return True

        except Exception as e:
            logger.error(f"Database cache clear error: {e}")
            self.metrics.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM cache_entries WHERE key = ?", (key,))
            return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Database cache exists error for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get keys matching pattern"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            if pattern:
                cursor.execute("SELECT key FROM cache_entries WHERE key LIKE ?", (pattern,))
            else:
                cursor.execute("SELECT key FROM cache_entries")

            return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Database cache keys error: {e}")
            self.metrics.errors += 1
            return []

    async def size(self) -> int:
        """Get number of entries"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM cache_entries")
            return cursor.fetchone()[0]

        except Exception as e:
            logger.error(f"Database cache size error: {e}")
            self.metrics.errors += 1
            return 0

    async def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM cache_entries
                WHERE ttl IS NOT NULL AND
                      (julianday('now') - julianday(created_at)) * 86400 > ttl
            """)

            conn.commit()
            removed_count = cursor.rowcount
            self.metrics.evictions += removed_count

            return removed_count

        except Exception as e:
            logger.error(f"Database cache cleanup error: {e}")
            self.metrics.errors += 1
            return 0