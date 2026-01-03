"""
数据库事务管理工具
提供事务保护和数据完整性保证
"""

from contextlib import contextmanager
from functools import wraps
from app.core.structured_logging import get_structured_logger
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import Session

logger = get_structured_logger(__name__)

class TransactionError(Exception):
    """事务错误基类"""
    pass

class TransactionRetryError(TransactionError):
    """事务重试错误"""
    pass

def transactional(
    session: Session,
    max_retries: int = 3,
    retry_on: tuple = (IntegrityError,),
    rollback_on_error: bool = True
):
    """
    事务装饰器，提供自动重试和回滚功能

    Args:
        session: 数据库会话
        max_retries: 最大重试次数
        retry_on: 需要重试的异常类型
        rollback_on_error: 错误时是否回滚

    Example:
        @transactional(db_session, max_retries=3)
        def create_user(data):
            user = User(**data)
            db_session.add(user)
            db_session.commit()
            return user
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result

                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"事务失败，正在进行第 {attempt + 1} 次重试: {str(e)}"
                        )
                        try:
                            session.rollback()
                        except Exception as rollback_error:
                            logger.error(f"回滚失败: {rollback_error}")
                        continue
                    else:
                        logger.error(f"事务达到最大重试次数: {max_retries}")
                        try:
                            session.rollback()
                        except Exception as rollback_error:
                            logger.error(f"最终回滚失败: {rollback_error}")
                        raise TransactionRetryError(
                            f"事务失败，已重试 {max_retries} 次: {str(e)}"
                        ) from e

                except SQLAlchemyError as e:
                    last_exception = e
                    if rollback_on_error:
                        try:
                            session.rollback()
                            logger.error(f"事务失败已回滚: {str(e)}")
                        except Exception as rollback_error:
                            logger.error(f"回滚失败: {rollback_error}")
                    raise TransactionError(f"事务失败: {str(e)}") from e

                except Exception as e:
                    last_exception = e
                    if rollback_on_error:
                        try:
                            session.rollback()
                            logger.error(f"非数据库错误，事务已回滚: {str(e)}")
                        except Exception as rollback_error:
                            logger.error(f"回滚失败: {rollback_error}")
                    raise TransactionError(f"事务执行失败: {str(e)}") from e

            # 不应该到这里
            raise TransactionError("未知的事务错误")

        return wrapper
    return decorator

@contextmanager
def transaction_scope(
    session: Session,
    auto_commit: bool = True,
    rollback_on_error: bool = True
):
    """
    事务上下文管理器

    Args:
        session: 数据库会话
        auto_commit: 是否自动提交
        rollback_on_error: 错误时是否回滚

    Example:
        with transaction_scope(db_session) as session:
            user = User(name="test")
            session.add(user)
            # 自动提交或回滚
    """
    try:
        yield session

        if auto_commit:
            try:
                session.commit()
                logger.debug("事务提交成功")
            except Exception as e:
                logger.error(f"事务提交失败: {e}")
                if rollback_on_error:
                    session.rollback()
                    raise TransactionError(f"事务提交失败已回滚: {e}") from e
                raise

    except SQLAlchemyError as e:
        logger.error(f"数据库错误: {e}")
        if rollback_on_error:
            try:
                session.rollback()
                logger.debug("事务已回滚")
            except Exception as rollback_error:
                logger.error(f"回滚失败: {rollback_error}")
        raise TransactionError(f"事务失败: {e}") from e

    except Exception as e:
        logger.error(f"未知错误: {e}")
        if rollback_on_error:
            try:
                session.rollback()
                logger.debug("事务已回滚")
            except Exception as rollback_error:
                logger.error(f"回滚失败: {rollback_error}")
        raise TransactionError(f"事务执行失败: {e}") from e

def batch_operation(
    session: Session,
    batch_size: int = 100,
    auto_commit: bool = True
):
    """
    批量操作装饰器

    Args:
        session: 数据库会话
        batch_size: 批次大小
        auto_commit: 是否自动提交

    Example:
        @batch_operation(db_session, batch_size=500)
        def bulk_insert_users(users):
            for user in users:
                db_session.add(user)
                yield user  # 每batch_size个提交一次
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            generator = func(*args, **kwargs)

            if not hasattr(generator, '__iter__'):
                return generator

            batch_count = 0
            total_count = 0

            try:
                for item in generator:
                    batch_count += 1
                    total_count += 1
                    yield item

                    if batch_count >= batch_size:
                        if auto_commit:
                            session.commit()
                            logger.debug(f"批量提交: {batch_size} 条记录")
                        batch_count = 0

                # 提交剩余的记录
                if batch_count > 0 and auto_commit:
                    session.commit()
                    logger.debug(f"批量提交: {batch_count} 条记录 (最后一批)")

                logger.info(f"批量操作完成，总计: {total_count} 条记录")

            except Exception as e:
                logger.error(f"批量操作失败: {e}")
                try:
                    session.rollback()
                    logger.debug("批量操作已回滚")
                except Exception as rollback_error:
                    logger.error(f"回滚失败: {rollback_error}")
                raise TransactionError(f"批量操作失败: {e}") from e

        return wrapper
    return decorator

class TransactionManager:
    """事务管理器"""

    def __init__(self, session: Session):
        self.session = session
        self._transaction_count = 0
        self._savepoints = []

    def begin(self):
        """开始事务"""
        if self._transaction_count == 0:
            self.session.begin()
            logger.debug("开始新事务")
        self._transaction_count += 1
        return self

    def commit(self):
        """提交事务"""
        if self._transaction_count > 0:
            self._transaction_count -= 1
            if self._transaction_count == 0:
                self.session.commit()
                logger.debug("事务提交")
        else:
            logger.warning("没有活动的事务可提交")

    def rollback(self):
        """回滚事务"""
        if self._transaction_count > 0:
            self._transaction_count = 0
            self.session.rollback()
            logger.debug("事务回滚")
        else:
            logger.warning("没有活动的事务可回滚")

    def create_savepoint(self, name: str):
        """创建保存点"""
        savepoint_name = f"savepoint_{name}_{len(self._savepoints)}"
        self.session.execute(f"SAVEPOINT {savepoint_name}")
        self._savepoints.append(savepoint_name)
        logger.debug(f"创建保存点: {savepoint_name}")
        return savepoint_name

    def release_savepoint(self, name: str):
        """释放保存点"""
        savepoint_name = f"savepoint_{name}"
        if savepoint_name in self._savepoints:
            self.session.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            self._savepoints.remove(savepoint_name)
            logger.debug(f"释放保存点: {savepoint_name}")

    def rollback_to_savepoint(self, name: str):
        """回滚到保存点"""
        savepoint_name = f"savepoint_{name}"
        if savepoint_name in self._savepoints:
            self.session.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            logger.debug(f"回滚到保存点: {savepoint_name}")

    @contextmanager
    def transaction(self, auto_commit: bool = True):
        """事务上下文"""
        self.begin()
        try:
            yield self.session
            if auto_commit:
                self.commit()
        except Exception:
            self.rollback()
            raise
        finally:
            if self._transaction_count == 0:
                self.session.close()

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        return False

# 便捷函数
def with_transaction(
    session: Session,
    auto_commit: bool = True,
    rollback_on_error: bool = True
):
    """便捷的事务上下文管理器"""
    return transaction_scope(session, auto_commit, rollback_on_error)
