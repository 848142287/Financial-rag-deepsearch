"""
统一资源管理器
管理连接池、缓存、线程池等系统资源
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.logging_utils import get_contextual_logger
from app.core.errors.unified_errors import ResourceError

logger = get_contextual_logger(__name__)

@dataclass
class PoolConfig:
    """连接池配置"""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    timeout: int = 30
    recycle: int = 3600
    ping: bool = True

class ResourceManager:
    """
    统一资源管理器
    负责管理所有系统资源：连接池、线程池、缓存等
    """

    _instance: Optional['ResourceManager'] = None

    def __init__(self):
        self.logger = get_contextual_logger(__name__)

        # 资源池
        self._database_pools: Dict[str, Any] = {}
        self._thread_pools: Dict[str, ThreadPoolExecutor] = {}
        self._async_pools: Dict[str, any] = {}

        # 资源状态
        self._initialized = False
        self._shutting_down = False

    @classmethod
    def get_instance(cls) -> 'ResourceManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self, config: Dict[str, Any] = None):
        """初始化所有资源"""
        if self._initialized:
            logger.warning("ResourceManager 已经初始化")
            return

        config = config or {}

        try:
            # 初始化数据库连接池
            await self._init_database_pools(config.get('database', {}))

            # 初始化线程池
            self._init_thread_pools(config.get('thread', {}))

            # 初始化资源清理任务
            asyncio.create_task(self._periodic_cleanup())

            self._initialized = True
            logger.info("ResourceManager 初始化完成")

        except Exception as e:
            logger.error(f"ResourceManager 初始化失败: {e}")
            raise ResourceError(
                message="资源管理器初始化失败",
                resource_type="ResourceManager",
                details={'error': str(e)}
            )

    async def _init_database_pools(self, config: Dict[str, Any]):
        """初始化数据库连接池"""
        try:
            from app.core.database.unified_db import DatabaseConfig, RepositoryFactory

            db_config = DatabaseConfig(
                backend=config.get('backend', 'mysql'),
                host=config.get('host', 'localhost'),
                port=config.get('port', 3306),
                database=config.get('database', 'financial_rag'),
                username=config.get('username', 'root'),
                password=config.get('password', ''),
                pool_size=config.get('pool_size', 20),
                max_overflow=config.get('max_overflow', 40),
                pool_timeout=config.get('pool_timeout', 30),
                pool_recycle=config.get('pool_recycle', 3600)
            )

            RepositoryFactory.initialize(db_config)

            self._database_pools['default'] = RepositoryFactory._engine

            logger.info("数据库连接池初始化完成")

        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {e}")
            raise

    def _init_thread_pools(self, config: Dict[str, Any]):
        """初始化线程池"""
        # CPU 密集型任务线程池
        cpu_count = asyncio.get_event_loop()._num_default_workers or 4
        self._thread_pools['cpu'] = ThreadPoolExecutor(
            max_workers=config.get('cpu_workers', cpu_count),
            thread_name_prefix='cpu_worker'
        )

        # IO 密集型任务线程池
        self._thread_pools['io'] = ThreadPoolExecutor(
            max_workers=config.get('io_workers', 20),
            thread_name_prefix='io_worker'
        )

        logger.info(f"线程池初始化完成: cpu={cpu_count}, io=20")

    @asynccontextmanager
    async def get_database_connection(self, name: str = 'default'):
        """获取数据库连接"""
        if not self._initialized:
            raise RuntimeError("ResourceManager 未初始化")

        try:
            from app.core.database.unified_db import RepositoryFactory
            async with RepositoryFactory.session() as session:
                yield session

        except Exception as e:
            logger.error(f"获取数据库连接失败: {e}")
            raise ResourceError(
                message=f"获取数据库连接失败: {str(e)}",
                resource_type="database_connection",
                resource_id=name
            )

    def get_thread_pool(self, pool_type: str = 'io') -> ThreadPoolExecutor:
        """获取线程池"""
        if not self._initialized:
            raise RuntimeError("ResourceManager 未初始化")

        pool = self._thread_pools.get(pool_type)
        if pool is None:
            raise ValueError(f"未知的线程池类型: {pool_type}")

        return pool

    async def run_in_thread_pool(
        self,
        func,
        *args,
        pool_type: str = 'io',
        **kwargs
    ):
        """在线程池中执行函数"""
        pool = self.get_thread_pool(pool_type)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(pool, func, *args, **kwargs)

    async def _periodic_cleanup(self):
        """定期清理资源"""
        while not self._shutting_down:
            try:
                # 每 5 分钟清理一次
                await asyncio.sleep(300)

                # 清理空闲连接
                # 清理过期缓存
                # 等等...

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"资源清理失败: {e}")

    async def shutdown(self):
        """关闭所有资源"""
        if self._shutting_down:
            return

        self._shutting_down = True
        logger.info("开始关闭 ResourceManager...")

        try:
            # 关闭线程池
            for name, pool in self._thread_pools.items():
                pool.shutdown(wait=True)
                logger.info(f"线程池 {name} 已关闭")

            # 关闭数据库连接池
            await self._close_database_pools()

            # 关闭其他资源...

            self._initialized = False
            logger.info("ResourceManager 已关闭")

        except Exception as e:
            logger.error(f"ResourceManager 关闭失败: {e}")

    async def _close_database_pools(self):
        """关闭数据库连接池"""
        try:
            from app.core.database.unified_db import RepositoryFactory
            await RepositoryFactory.close()
            logger.info("数据库连接池已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接池失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        return {
            'initialized': self._initialized,
            'database_pools': list(self._database_pools.keys()),
            'thread_pools': {
                name: pool._max_workers
                for name, pool in self._thread_pools.items()
            },
            'shutting_down': self._shutting_down
        }

# 全局实例
_resource_manager: Optional[ResourceManager] = None

def get_resource_manager() -> ResourceManager:
    """获取资源管理器实例"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager.get_instance()
    return _resource_manager

async def init_resources(config: Dict[str, Any] = None):
    """初始化所有资源（便捷函数）"""
    manager = get_resource_manager()
    await manager.initialize(config)

async def shutdown_resources():
    """关闭所有资源（便捷函数）"""
    manager = get_resource_manager()
    await manager.shutdown()

# FastAPI 生命周期事件
@asynccontextmanager
async def lifespan(app):
    """FastAPI 应用生命周期管理"""
    # 启动时初始化资源
    logger.info("应用启动：初始化资源...")
    await init_resources()
    yield
    # 关闭时清理资源
    logger.info("应用关闭：清理资源...")
    await shutdown_resources()
