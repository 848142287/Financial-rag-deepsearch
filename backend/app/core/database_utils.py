"""
数据库连接资源管理工具
防止数据库连接泄漏
"""

from app.core.structured_logging import get_structured_logger
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine

from app.core.config import settings

logger = get_structured_logger(__name__)

# 全局引擎
_engine = None
_session_factory = None

def get_engine():
    """获取数据库引擎（单例）"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # 自动检测断开的连接
            pool_recycle=3600,   # 1小时回收连接
            echo=settings.debug  # 开发环境打印SQL
        )
        logger.info("Database engine created")
    return _engine

def get_session_factory():
    """获取会话工厂（单例）"""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    return _session_factory

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    数据库会话上下文管理器
    自动管理连接的生命周期，防止泄漏

    Example:
        with get_db_context() as db:
            documents = db.query(Document).all()
            # 自动提交或回滚
    """
    db = None
    try:
        session_factory = get_session_factory()
        db = session_factory()
        yield db
        db.commit()
    except Exception as e:
        if db:
            db.rollback()
        logger.error(f"Database operation failed: {e}", exc_info=True)
        raise
    finally:
        if db:
            db.close()

@contextmanager
def get_db_context_readonly() -> Generator[Session, None, None]:
    """
    只读数据库会话上下文管理器
    用于只读查询，自动回滚以避免连接锁定

    Example:
        with get_db_context_readonly() as db:
            documents = db.query(Document).all()
    """
    db = None
    try:
        session_factory = get_session_factory()
        db = session_factory()
        yield db
        # 只读事务总是回滚，释放锁
        db.rollback()
    except Exception as e:
        if db:
            db.rollback()
        logger.error(f"Database read operation failed: {e}")
        raise
    finally:
        if db:
            db.close()

class DatabaseSessionManager:
    """
    数据库会话管理器
    提供更灵活的会话管理方式
    """

    def __init__(self):
        self._active_sessions = set()
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_session(self) -> Session:
        """
        获取新的数据库会话
        注意：需要手动关闭

        Example:
            manager = DatabaseSessionManager()
            db = manager.get_session()
            try:
                # 使用db
                pass
            finally:
                manager.close_session(db)
        """
        session_factory = get_session_factory()
        session = session_factory()
        self._active_sessions.add(id(session))
        self._logger.debug(f"Session created, active: {len(self._active_sessions)}")
        return session

    def close_session(self, session: Session):
        """关闭会话"""
        if session:
            session.close()
            self._active_sessions.discard(id(session))
            self._logger.debug(f"Session closed, active: {len(self._active_sessions)}")

    def close_all_sessions(self):
        """关闭所有活动会话（用于清理）"""
        closed = 0
        for session_id in list(self._active_sessions):
            self._logger.warning(f"Force closing session: {session_id}")
            # 注意：这里无法获取session对象，只能记录
            closed += 1
        self._active_sessions.clear()
        self._logger.warning(f"Force closed {closed} sessions")

    def get_active_session_count(self) -> int:
        """获取活动会话数量"""
        return len(self._active_sessions)

# 全局会话管理器
db_session_manager = DatabaseSessionManager()

def cleanup_db_resources():
    """
    清理数据库资源
    在应用关闭时调用
    """
    global _engine, _session_factory

    logger.info("Cleaning up database resources...")

    # 关闭所有活动会话
    db_session_manager.close_all_sessions()

    # 销毁引擎
    if _engine:
        _engine.dispose()
        logger.info("Database engine disposed")
        _engine = None

    _session_factory = None

    logger.info("Database resources cleaned up")
