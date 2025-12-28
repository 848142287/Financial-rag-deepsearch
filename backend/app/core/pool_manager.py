"""
连接池和线程池管理模块
提供数据库连接池、Redis连接池、线程池等资源管理
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator, Tuple
from queue import Queue, Empty
import redis
import psycopg2
import psycopg2.pool
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import motor.motor_asyncio
import aiomysql
import redis.asyncio as aioredis
from httpx import AsyncClient

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """连接池配置"""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_lifetime: float = 3600.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ThreadPoolConfig:
    """线程池配置"""
    max_workers: int = 10
    thread_name_prefix: str = "worker"
    queue_size: int = 1000


class DatabaseConnectionPool:
    """数据库连接池管理器"""

    def __init__(self, database_url: str, config: PoolConfig):
        self.database_url = database_url
        self.config = config
        self.engine = None
        self.session_factory = None
        self._pool = None
        self._lock = threading.Lock()

    def initialize(self):
        """初始化连接池"""
        with self._lock:
            if self._pool is not None:
                return

            logger.info(f"Initializing database connection pool: {self.config}")

            # 创建SQLAlchemy引擎
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.config.min_connections,
                max_overflow=self.config.max_connections - self.config.min_connections,
                pool_timeout=self.config.connection_timeout,
                pool_recycle=self.config.max_lifetime,
                pool_pre_ping=True,
                echo=False
            )

            # 创建会话工厂
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )

            # 注册连接事件监听器
            self._register_listeners()

            self._pool = True
            logger.info("Database connection pool initialized successfully")

    def _register_listeners(self):
        """注册连接事件监听器"""
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            logger.debug("New database connection established")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            logger.debug("Connection checked in to pool")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[Session, None]:
        """获取数据库会话"""
        if not self._pool:
            self.initialize()

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    def get_sync_session(self) -> Session:
        """获取同步数据库会话（用于非异步代码）"""
        if not self._pool:
            self.initialize()

        return self.session_factory()

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch_one: bool = False
    ) -> Union[List[Dict], Dict, None]:
        """执行查询"""
        async with self.get_session() as session:
            try:
                result = session.execute(query, params or {})
                if fetch_one:
                    row = result.fetchone()
                    return dict(row) if row else None
                else:
                    rows = result.fetchall()
                    return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Query execution failed: {str(e)}")
                raise

    async def execute_batch(
        self,
        queries: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Any]:
        """批量执行查询"""
        async with self.get_session() as session:
            results = []
            try:
                for query, params in queries:
                    result = session.execute(query, params)
                    results.append(result)
                session.commit()
                return results
            except Exception as e:
                session.rollback()
                logger.error(f"Batch execution failed: {str(e)}")
                raise

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            pool_status = self.engine.pool.status()

            return {
                "status": "healthy",
                "pool_size": self.engine.pool.size(),
                "checked_in": pool_status.get("pool_checked_in", 0),
                "checked_out": pool_status.get("pool_checked_out", 0),
                "overflow": pool_status.get("pool_overflow", 0),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    def close(self):
        """关闭连接池"""
        if self.engine:
            self.engine.dispose()
            self._pool = None
            logger.info("Database connection pool closed")


class RedisConnectionPool:
    """Redis连接池管理器"""

    def __init__(self, redis_url: str, config: PoolConfig):
        self.redis_url = redis_url
        self.config = config
        self._pool = None
        self._async_pool = None

    def initialize(self):
        """初始化连接池"""
        if self._pool:
            return

        logger.info(f"Initializing Redis connection pool: {self.config}")

        # 创建同步连接池
        self._pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.connection_timeout,
            socket_connect_timeout=self.config.connection_timeout,
            retry_on_timeout=True,
            health_check_interval=30
        )

        logger.info("Redis connection pool initialized successfully")

    async def initialize_async(self):
        """初始化异步连接池"""
        if self._async_pool:
            return

        logger.info(f"Initializing async Redis connection pool")

        # 创建异步连接池
        self._async_pool = await aioredis.create_redis_pool(
            self.redis_url,
            minsize=self.config.min_connections,
            maxsize=self.config.max_connections,
            timeout=self.config.connection_timeout
        )

        logger.info("Async Redis connection pool initialized successfully")

    def get_sync_connection(self) -> redis.Redis:
        """获取同步Redis连接"""
        if not self._pool:
            self.initialize()

        return redis.Redis(connection_pool=self._pool)

    async def get_async_connection(self):
        """获取异步Redis连接"""
        if not self._async_pool:
            await self.initialize_async()

        return self._async_pool

    async def execute_command(self, command: str, *args) -> Any:
        """执行Redis命令"""
        conn = await self.get_async_connection()
        try:
            return await conn.execute(command, *args)
        except Exception as e:
            logger.error(f"Redis command failed: {command}, error: {str(e)}")
            raise

    async def cache_get(self, key: str) -> Optional[str]:
        """获取缓存"""
        try:
            return await self.execute_command("GET", key)
        except Exception:
            return None

    async def cache_set(
        self,
        key: str,
        value: str,
        expire: Optional[int] = None
    ) -> bool:
        """设置缓存"""
        try:
            if expire:
                await self.execute_command("SETEX", key, expire, value)
            else:
                await self.execute_command("SET", key, value)
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")
            return False

    async def cache_delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            await self.execute_command("DEL", key)
            return True
        except Exception:
            return False

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            conn = self.get_sync_connection()
            conn.ping()

            return {
                "status": "healthy",
                "pool_size": self._pool.connection_kwargs.get("max_connections", 0),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    async def close(self):
        """关闭连接池"""
        if self._async_pool:
            self._async_pool.close()
            await self._async_pool.wait_closed()

        if self._pool:
            self._pool.disconnect()

        logger.info("Redis connection pools closed")


class ThreadPoolManager:
    """线程池管理器"""

    def __init__(self, config: ThreadPoolConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix=config.thread_name_prefix
        )
        self._task_queue = Queue(maxsize=config.queue_size)
        self._shutdown = False

    async def submit_task(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """提交任务到线程池"""
        if self._shutdown:
            raise RuntimeError("ThreadPoolManager is shutdown")

        loop = asyncio.get_event_loop()
        try:
            # 使用put_nowait避免阻塞，如果队列满了则抛出异常
            self._task_queue.put_nowait(func.__name__)

            # 提交任务到线程池
            future = self.executor.submit(func, *args, **kwargs)

            # 在事件循环中等待结果
            result = await loop.run_in_executor(None, lambda: future.result())

            # 从队列中移除任务
            try:
                self._task_queue.get_nowait()
            except Empty:
                pass

            return result

        except queue.Full:
            raise RuntimeError("Task queue is full")

    async def submit_batch(
        self,
        tasks: List[Tuple[Callable, tuple, dict]]
    ) -> List[Any]:
        """批量提交任务"""
        if self._shutdown:
            raise RuntimeError("ThreadPoolManager is shutdown")

        # 创建任务future列表
        futures = []
        loop = asyncio.get_event_loop()

        for func, args, kwargs in tasks:
            try:
                self._task_queue.put_nowait(func.__name__)
                future = self.executor.submit(func, *args, **kwargs)
                futures.append((future, func.__name__))
            except queue.Full:
                # 清理已提交的任务
                for f, _ in futures:
                    f.cancel()
                raise RuntimeError("Task queue is full")

        # 等待所有任务完成
        results = []
        try:
            for future, task_name in futures:
                try:
                    result = await loop.run_in_executor(
                        None, lambda f=future: f.result()
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {str(e)}")
                    results.append(None)
        finally:
            # 清理队列
            for _ in futures:
                try:
                    self._task_queue.get_nowait()
                except Empty:
                    pass

        return results

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self._task_queue.qsize()

    def get_active_count(self) -> int:
        """获取活跃线程数"""
        return self.executor._threads.__len__()

    def get_stats(self) -> Dict[str, Any]:
        """获取线程池统计信息"""
        return {
            "max_workers": self.config.max_workers,
            "queue_size": self.get_queue_size(),
            "active_threads": self.get_active_count(),
            "max_queue_size": self.config.queue_size,
            "shutdown": self._shutdown
        }

    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self._shutdown = True
        self.executor.shutdown(wait=wait)
        logger.info("ThreadPoolManager shutdown")


class AsyncClientPool:
    """HTTP异步客户端池"""

    def __init__(self, pool_size: int = 10, timeout: float = 30.0):
        self.pool_size = pool_size
        self.timeout = timeout
        self._clients = Queue(maxsize=pool_size)
        self._initialized = False

    async def initialize(self):
        """初始化客户端池"""
        if self._initialized:
            return

        logger.info(f"Initializing HTTP client pool with size {self.pool_size}")

        # 预创建客户端
        for _ in range(self.pool_size):
            client = AsyncClient(timeout=self.timeout)
            self._clients.put(client)

        self._initialized = True
        logger.info("HTTP client pool initialized successfully")

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[AsyncClient, None]:
        """获取HTTP客户端"""
        if not self._initialized:
            await self.initialize()

        client = None
        try:
            client = self._clients.get(timeout=1.0)
            yield client
        except Empty:
            # 如果池为空，创建临时客户端
            client = AsyncClient(timeout=self.timeout)
            yield client
        finally:
            if client and not self._clients.full():
                self._clients.put(client)

    async def close(self):
        """关闭客户端池"""
        while not self._clients.empty():
            try:
                client = self._clients.get_nowait()
                await client.aclose()
            except Empty:
                break

        self._initialized = False
        logger.info("HTTP client pool closed")


class PoolManager:
    """统一的资源池管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_pools = {}
        self.redis_pools = {}
        self.thread_pools = {}
        self.http_pools = {}
        self._initialized = False

    async def initialize(self):
        """初始化所有资源池"""
        if self._initialized:
            return

        logger.info("Initializing PoolManager")

        # 初始化数据库连接池
        if "databases" in self.config:
            for name, db_config in self.config["databases"].items():
                pool_config = PoolConfig(**db_config.get("pool", {}))
                pool = DatabaseConnectionPool(
                    db_config["url"],
                    pool_config
                )
                pool.initialize()
                self.db_pools[name] = pool

        # 初始化Redis连接池
        if "redis" in self.config:
            for name, redis_config in self.config["redis"].items():
                pool_config = PoolConfig(**redis_config.get("pool", {}))
                pool = RedisConnectionPool(
                    redis_config["url"],
                    pool_config
                )
                pool.initialize()
                await pool.initialize_async()
                self.redis_pools[name] = pool

        # 初始化线程池
        if "thread_pools" in self.config:
            for name, thread_config in self.config["thread_pools"].items():
                pool_config = ThreadPoolConfig(**thread_config)
                pool = ThreadPoolManager(pool_config)
                self.thread_pools[name] = pool

        # 初始化HTTP客户端池
        if "http_clients" in self.config:
            for name, http_config in self.config["http_clients"].items():
                pool = AsyncClientPool(
                    pool_size=http_config.get("pool_size", 10),
                    timeout=http_config.get("timeout", 30.0)
                )
                await pool.initialize()
                self.http_pools[name] = pool

        self._initialized = True
        logger.info("PoolManager initialized successfully")

    def get_db_pool(self, name: str = "default") -> DatabaseConnectionPool:
        """获取数据库连接池"""
        if name not in self.db_pools:
            raise ValueError(f"Database pool {name} not found")
        return self.db_pools[name]

    def get_redis_pool(self, name: str = "default") -> RedisConnectionPool:
        """获取Redis连接池"""
        if name not in self.redis_pools:
            raise ValueError(f"Redis pool {name} not found")
        return self.redis_pools[name]

    def get_thread_pool(self, name: str = "default") -> ThreadPoolManager:
        """获取线程池"""
        if name not in self.thread_pools:
            raise ValueError(f"Thread pool {name} not found")
        return self.thread_pools[name]

    def get_http_pool(self, name: str = "default") -> AsyncClientPool:
        """获取HTTP客户端池"""
        if name not in self.http_pools:
            raise ValueError(f"HTTP client pool {name} not found")
        return self.http_pools[name]

    async def health_check(self) -> Dict[str, Any]:
        """健康检查所有资源池"""
        health_status = {}

        # 检查数据库连接池
        for name, pool in self.db_pools.items():
            health_status[f"db_{name}"] = pool.health_check()

        # 检查Redis连接池
        for name, pool in self.redis_pools.items():
            health_status[f"redis_{name}"] = pool.health_check()

        # 检查线程池
        for name, pool in self.thread_pools.items():
            health_status[f"thread_{name}"] = pool.get_stats()

        return {
            "overall_status": "healthy",
            "pools": health_status,
            "timestamp": time.time()
        }

    async def close(self):
        """关闭所有资源池"""
        logger.info("Closing all resource pools")

        # 关闭数据库连接池
        for pool in self.db_pools.values():
            pool.close()

        # 关闭Redis连接池
        for pool in self.redis_pools.values():
            await pool.close()

        # 关闭线程池
        for pool in self.thread_pools.values():
            pool.shutdown()

        # 关闭HTTP客户端池
        for pool in self.http_pools.values():
            await pool.close()

        self._initialized = False
        logger.info("All resource pools closed")


# 全局池管理器实例
_pool_manager = None


async def get_pool_manager(config: Optional[Dict[str, Any]] = None) -> PoolManager:
    """获取全局池管理器实例"""
    global _pool_manager

    if _pool_manager is None:
        if config is None:
            # 默认配置
            config = {
                "databases": {
                    "default": {
                        "url": "sqlite:///./app.db",
                        "pool": {
                            "min_connections": 5,
                            "max_connections": 20
                        }
                    }
                },
                "redis": {
                    "default": {
                        "url": "redis://localhost:6379",
                        "pool": {
                            "min_connections": 5,
                            "max_connections": 20
                        }
                    }
                },
                "thread_pools": {
                    "default": {
                        "max_workers": 10,
                        "queue_size": 1000
                    }
                }
            }

        _pool_manager = PoolManager(config)
        await _pool_manager.initialize()

    return _pool_manager


# 便利函数
async def get_db_session(pool_name: str = "default"):
    """获取数据库会话"""
    manager = await get_pool_manager()
    pool = manager.get_db_pool(pool_name)
    async with pool.get_session() as session:
        yield session


async def get_redis_connection(pool_name: str = "default"):
    """获取Redis连接"""
    manager = await get_pool_manager()
    pool = manager.get_redis_pool(pool_name)
    return await pool.get_async_connection()


async def submit_to_thread_pool(
    func: Callable,
    *args,
    pool_name: str = "default",
    **kwargs
) -> Any:
    """提交任务到线程池"""
    manager = await get_pool_manager()
    pool = manager.get_thread_pool(pool_name)
    return await pool.submit_task(func, *args, **kwargs)