"""
数据库连接和会话管理
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy 2.0+ 基础模型类"""
    pass


# 创建数据库引擎
try:
    # 确保使用pymysql作为连接器
    database_url = settings.database_url
    if database_url.startswith("mysql://"):
        database_url = database_url.replace("mysql://", "mysql+pymysql://")

    engine = create_engine(
        database_url,
        # 优化连接池配置
        pool_size=20,  # 连接池大小
        max_overflow=30,  # 最大溢出连接数
        pool_pre_ping=True,  # 连接前ping测试
        pool_recycle=3600,  # 连接回收时间（1小时）
        echo=settings.debug,  # 调试模式下打印SQL
        # 异步引擎配置
        future=True,  # 使用SQLAlchemy 2.0 API
    )
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # 防止对象在提交后过期
)

# 创建异步引擎和会话
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

    # 转换数据库URL为异步格式
    async_database_url = settings.database_url.replace(
        "mysql+pymysql://",
        "mysql+aiomysql://",
        1
    )

    async_engine = create_async_engine(
        async_database_url,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.debug,
        future=True,
    )

    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    logger.info("Async database engine created successfully")
except ImportError:
    logger.warning("Async database driver (aiomysql) not installed, async features disabled")
    async_engine = None
    AsyncSessionLocal = None
except Exception as e:
    logger.error(f"Failed to create async database engine: {e}")
    async_engine = None
    AsyncSessionLocal = None


def get_db():
    """获取数据库会话（同步）"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_db():
    """获取异步数据库会话"""
    if AsyncSessionLocal is None:
        raise RuntimeError("Async database engine not available")

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise
        # 不需要显式关闭，async with 会自动处理


async def check_database_connection():
    """检查数据库连接"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False