"""
统一数据库访问层
提供 Repository 模式的统一数据库接口
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum

from app.core.logging_utils import get_contextual_logger
from app.core.errors.unified_errors import DatabaseError

logger = get_contextual_logger(__name__)

T = TypeVar('T')
ModelT = TypeVar('ModelT')

class DatabaseBackend(Enum):
    """数据库后端类型"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    NEBULA = "nebula"  # 图数据库
    MILVUS = "milvus"   # 向量数据库

@dataclass
class DatabaseConfig:
    """数据库配置"""
    backend: DatabaseBackend
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

class IRepository(ABC, Generic[T]):
    """
    Repository 接口
    定义所有数据访问的标准接口
    """

    @abstractmethod
    async def find_by_id(self, id: int) -> Optional[T]:
        """根据 ID 查找"""
        pass

    @abstractmethod
    async def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Dict[str, Any] = None
    ) -> List[T]:
        """查找所有记录"""
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """创建实体"""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """更新实体"""
        pass

    @abstractmethod
    async def delete(self, id: int) -> bool:
        """删除实体"""
        pass

    @abstractmethod
    async def count(self, filters: Dict[str, Any] = None) -> int:
        """统计记录数"""
        pass

    @abstractmethod
    async def exists(self, id: int) -> bool:
        """检查实体是否存在"""
        pass

class BaseRepository(IRepository[T]):
    """
    基础 Repository 实现
    提供通用的 CRUD 操作
    """

    def __init__(self, model: Type[ModelT], session_factory: sessionmaker):
        self.model = model
        self.session_factory = session_factory
        self.logger = get_contextual_logger(self.__class__.__name__)

    async def find_by_id(self, id: int) -> Optional[T]:
        """根据 ID 查找"""
        try:
            with self._get_session() as session:
                entity = session.query(self.model).filter(self.model.id == id).first()
                return entity
        except Exception as e:
            self.logger.error(f"查找失败 {self.model.__name__} id={id}: {e}")
            raise DatabaseError(
                message=f"查找失败: {str(e)}",
                table=self.model.__tablename__
            )

    async def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Dict[str, Any] = None
    ) -> List[T]:
        """查找所有记录"""
        try:
            with self._get_session() as session:
                query = session.query(self.model)

                # 应用过滤器
                if filters:
                    for key, value in filters.items():
                        if hasattr(self.model, key):
                            query = query.filter(getattr(self.model, key) == value)

                return query.offset(offset).limit(limit).all()

        except Exception as e:
            self.logger.error(f"查询失败 {self.model.__name__}: {e}")
            raise DatabaseError(
                message=f"查询失败: {str(e)}",
                table=self.model.__tablename__
            )

    async def create(self, entity: T) -> T:
        """创建实体"""
        try:
            with self._get_session() as session:
                session.add(entity)
                session.commit()
                session.refresh(entity)
                return entity

        except Exception as e:
            self.logger.error(f"创建失败 {self.model.__name__}: {e}")
            session.rollback()
            raise DatabaseError(
                message=f"创建失败: {str(e)}",
                table=self.model.__tablename__
            )

    async def update(self, entity: T) -> T:
        """更新实体"""
        try:
            with self._get_session() as session:
                session.merge(entity)
                session.commit()
                session.refresh(entity)
                return entity

        except Exception as e:
            self.logger.error(f"更新失败 {self.model.__name__}: {e}")
            session.rollback()
            raise DatabaseError(
                message=f"更新失败: {str(e)}",
                table=self.model.__tablename__
            )

    async def delete(self, id: int) -> bool:
        """删除实体"""
        try:
            with self._get_session() as session:
                entity = session.query(self.model).filter(self.model.id == id).first()
                if entity:
                    session.delete(entity)
                    session.commit()
                    return True
                return False

        except Exception as e:
            self.logger.error(f"删除失败 {self.model.__name__} id={id}: {e}")
            session.rollback()
            raise DatabaseError(
                message=f"删除失败: {str(e)}",
                table=self.model.__tablename__
            )

    async def count(self, filters: Dict[str, Any] = None) -> int:
        """统计记录数"""
        try:
            with self._get_session() as session:
                query = session.query(self.model)

                if filters:
                    for key, value in filters.items():
                        if hasattr(self.model, key):
                            query = query.filter(getattr(self.model, key) == value)

                return query.count()

        except Exception as e:
            self.logger.error(f"统计失败 {self.model.__name__}: {e}")
            raise DatabaseError(
                message=f"统计失败: {str(e)}",
                table=self.model.__tablename__
            )

    async def exists(self, id: int) -> bool:
        """检查实体是否存在"""
        try:
            entity = await self.find_by_id(id)
            return entity is not None
        except Exception:
            return False

    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

class DocumentRepository(BaseRepository):
    """文档 Repository"""

    async def find_by_status(self, status: str) -> List[T]:
        """根据状态查找文档"""
        return await self.find_all(filters={'status': status})

    async def find_by_user(self, user_id: int) -> List[T]:
        """根据用户 ID 查找文档"""
        return await self.find_all(filters={'user_id': user_id})

class RepositoryFactory:
    """
    Repository 工厂
    负责创建和管理 Repository 实例
    """

    _repositories: Dict[str, BaseRepository] = {}
    _session_factory: Optional[sessionmaker] = None
    _engine = None

    @classmethod
    def initialize(cls, config: DatabaseConfig):
        """初始化数据库连接"""
        try:
            # 构建数据库 URL
            if config.backend == DatabaseBackend.MYSQL:
                url = f"mysql+pymysql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            elif config.backend == DatabaseBackend.POSTGRESQL:
                url = f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            elif config.backend == DatabaseBackend.SQLITE:
                url = f"sqlite:///{config.database}"
            else:
                raise ValueError(f"不支持的数据库类型: {config.backend}")

            # 创建引擎
            cls._engine = create_engine(
                url,
                poolclass=QueuePool,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.pool_timeout,
                pool_recycle=config.pool_recycle,
                echo=config.echo,
                pool_pre_ping=True  # 检测连接是否有效
            )

            # 创建 Session 工厂
            cls._session_factory = sessionmaker(
                bind=cls._engine,
                autocommit=False,
                autoflush=False
            )

            logger.info(f"数据库连接已建立: {config.backend.value}://{config.host}:{config.port}")

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(
                message=f"数据库初始化失败: {str(e)}",
                details={'backend': config.backend.value}
            )

    @classmethod
    def get_repository(cls, model: Type[ModelT]) -> BaseRepository:
        """获取 Repository 实例"""
        model_name = model.__name__

        if model_name not in cls._repositories:
            if cls._session_factory is None:
                raise RuntimeError("数据库未初始化，请先调用 RepositoryFactory.initialize()")

            cls._repositories[model_name] = BaseRepository(model, cls._session_factory)

        return cls._repositories[model_name]

    @classmethod
    def get_document_repository(cls, model: Type[ModelT]) -> DocumentRepository:
        """获取文档 Repository"""
        if 'Document' not in cls._repositories:
            if cls._session_factory is None:
                raise RuntimeError("数据库未初始化")

            cls._repositories['Document'] = DocumentRepository(model, cls._session_factory)

        return cls._repositories['Document']

    @classmethod
    async def close(cls):
        """关闭数据库连接"""
        if cls._engine:
            cls._engine.dispose()
            logger.info("数据库连接已关闭")

    @classmethod
    @contextmanager
    def session(cls):
        """获取数据库会话上下文管理器"""
        if cls._session_factory is None:
            raise RuntimeError("数据库未初始化")

        session = cls._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

# 便捷函数
def init_database(config: DatabaseConfig):
    """初始化数据库"""
    RepositoryFactory.initialize(config)

def get_repository(model: Type[ModelT]) -> BaseRepository:
    """获取 Repository"""
    return RepositoryFactory.get_repository(model)

async def with_transaction(func):
    """
    事务装饰器
    确保函数在事务中执行
    """
    async def wrapper(*args, **kwargs):
        with RepositoryFactory.session() as session:
            # 将 session 注入到 kwargs
            kwargs['_session'] = session
            return await func(*args, **kwargs)

    return wrapper
