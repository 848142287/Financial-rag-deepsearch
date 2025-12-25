"""
错误处理和日志记录模块
"""

import logging
import traceback
import functools
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Type, Union, Callable
from sqlalchemy.orm import Session
from sqlalchemy import and_
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.database import get_db
from app.models.synchronization import SyncLog


# 配置日志
class SyncLogger:
    """同步系统专用日志器"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_logger()

    def setup_logger(self):
        """设置日志器"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def log_sync_event(
        self,
        db: Session,
        document_sync_id: Optional[int],
        level: str,
        component: str,
        message: str,
        details: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ):
        """记录同步事件到数据库"""
        try:
            # 记录到标准日志
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(f"[{component}] {message}")

            # 记录到数据库（如果有document_sync_id）
            if document_sync_id and db:
                stack_trace = traceback.format_exc() if exception else None

                sync_log = SyncLog(
                    document_sync_id=document_sync_id,
                    log_level=level.upper(),
                    component=component,
                    message=message,
                    details=details,
                    stack_trace=stack_trace
                )

                db.add(sync_log)
                db.commit()

        except Exception as e:
            # 避免日志记录本身出错导致的问题
            print(f"Failed to log sync event: {str(e)}")


# 全局日志器实例
sync_logger = SyncLogger("sync_system")


class SyncError(Exception):
    """同步系统基础异常类"""

    def __init__(self, message: str, details: Optional[Dict] = None, document_sync_id: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.document_sync_id = document_sync_id
        self.timestamp = datetime.utcnow()


class DocumentProcessingError(SyncError):
    """文档处理错误"""
    pass


class VectorSyncError(SyncError):
    """向量同步错误"""
    pass


class GraphSyncError(SyncError):
    """图谱同步错误"""
    pass


class EntityLinkError(SyncError):
    """实体关联错误"""
    pass


class ConfigurationError(SyncError):
    """配置错误"""
    pass


class StateTransitionError(SyncError):
    """状态转换错误"""
    pass


def handle_sync_error(
    error_type: Type[SyncError] = SyncError,
    document_sync_id: Optional[int] = None,
    component: str = "unknown"
):
    """同步错误处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # 获取数据库会话
                db = None
                for arg in args:
                    if isinstance(arg, Session):
                        db = arg
                        break

                # 如果是已知错误类型，直接处理
                if isinstance(e, SyncError):
                    await sync_logger.log_sync_event(
                        db, e.document_sync_id, "ERROR", component,
                        e.message, e.details, e
                    )
                    raise e

                # 转换为适当的错误类型
                error_msg = str(e)
                error_details = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs": list(kwargs.keys()),
                    "error_type": type(e).__name__
                }

                sync_error = error_type(
                    error_msg, error_details, document_sync_id
                )

                await sync_logger.log_sync_event(
                    db, document_sync_id, "ERROR", component,
                    error_msg, error_details, e
                )

                raise sync_error

        return wrapper
    return decorator


def handle_database_error(func: Callable) -> Callable:
    """数据库错误处理装饰器"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Database operation failed in {func.__name__}: {str(e)}"

            # 检查是否是数据库连接错误
            if "connection" in str(e).lower():
                error_msg = "Database connection error. Please try again later."

            # 记录错误
            sync_logger.logger.error(error_msg)

            # 抛出更友好的错误信息
            raise SyncError(error_msg, {"original_error": str(e)})

    return wrapper


class ErrorHandler:
    """全局错误处理器"""

    @staticmethod
    async def handle_sync_error(error: SyncError, db: Optional[Session] = None) -> JSONResponse:
        """处理同步错误"""
        await sync_logger.log_sync_event(
            db, error.document_sync_id, "ERROR", "global_handler",
            error.message, error.details, error
        )

        # 根据错误类型返回不同的HTTP状态码
        status_code = 500
        if isinstance(error, ConfigurationError):
            status_code = 400
        elif isinstance(error, StateTransitionError):
            status_code = 409
        elif isinstance(error, (DocumentProcessingError, VectorSyncError, GraphSyncError, EntityLinkError)):
            status_code = 422

        return JSONResponse(
            status_code=status_code,
            content={
                "error": True,
                "error_type": type(error).__name__,
                "message": error.message,
                "details": error.details,
                "timestamp": error.timestamp.isoformat(),
                "document_sync_id": error.document_sync_id
            }
        )

    @staticmethod
    async def handle_validation_error(error: Exception) -> JSONResponse:
        """处理验证错误"""
        return JSONResponse(
            status_code=422,
            content={
                "error": True,
                "error_type": "ValidationError",
                "message": "Validation failed",
                "details": {"validation_error": str(error)},
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @staticmethod
    async def handle_not_found_error(resource: str, identifier: Any) -> JSONResponse:
        """处理资源未找到错误"""
        message = f"{resource} not found: {identifier}"
        return JSONResponse(
            status_code=404,
            content={
                "error": True,
                "error_type": "NotFound",
                "message": message,
                "details": {"resource": resource, "identifier": str(identifier)},
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @staticmethod
    async def handle_permission_error(message: str = "Permission denied") -> JSONResponse:
        """处理权限错误"""
        return JSONResponse(
            status_code=403,
            content={
                "error": True,
                "error_type": "PermissionError",
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, db: Session, document_sync_id: Optional[int] = None):
        self.db = db
        self.document_sync_id = document_sync_id
        self.start_time = None
        self.checkpoints = []

    def start(self):
        """开始监控"""
        self.start_time = datetime.utcnow()
        self.checkpoints = []

        async def log_start():
            await sync_logger.log_sync_event(
                self.db, self.document_sync_id, "INFO", "performance_monitor",
                f"Performance monitoring started"
            )

        return log_start()

    def checkpoint(self, name: str, details: Optional[Dict] = None):
        """添加检查点"""
        if not self.start_time:
            return

        current_time = datetime.utcnow()
        elapsed = (current_time - self.start_time).total_seconds()

        checkpoint_data = {
            "name": name,
            "timestamp": current_time.isoformat(),
            "elapsed_seconds": elapsed,
            "details": details or {}
        }

        self.checkpoints.append(checkpoint_data)

        async def log_checkpoint():
            await sync_logger.log_sync_event(
                self.db, self.document_sync_id, "INFO", "performance_monitor",
                f"Checkpoint: {name} (elapsed: {elapsed:.2f}s)", checkpoint_data
            )

        return log_checkpoint()

    def finish(self, details: Optional[Dict] = None):
        """完成监控"""
        if not self.start_time:
            return

        current_time = datetime.utcnow()
        total_elapsed = (current_time - self.start_time).total_seconds()

        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": current_time.isoformat(),
            "total_elapsed_seconds": total_elapsed,
            "checkpoints": self.checkpoints,
            "details": details or {}
        }

        async def log_finish():
            await sync_logger.log_sync_event(
                self.db, self.document_sync_id, "INFO", "performance_monitor",
                f"Performance monitoring finished (total: {total_elapsed:.2f}s)", summary
            )

        return log_finish()


class HealthChecker:
    """系统健康检查器"""

    def __init__(self, db: Session):
        self.db = db

    async def check_database_health(self) -> Dict:
        """检查数据库健康状态"""
        try:
            # 执行简单查询测试连接
            from sqlalchemy import text
            result = self.db.execute(text("SELECT 1"))
            result.fetchone()

            # 检查同步日志表
            log_count = self.db.query(SyncLog).count()

            return {
                "status": "healthy",
                "connection": "ok",
                "log_entries": log_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def check_sync_system_health(self) -> Dict:
        """检查同步系统健康状态"""
        try:
            from app.models.synchronization import DocumentSync, SyncStatus

            # 统计各状态的同步任务
            status_counts = {}
            for status in SyncStatus:
                count = self.db.query(DocumentSync).filter(
                    DocumentSync.sync_status == status
                ).count()
                status_counts[status.value] = count

            # 检查最近失败的同步
            recent_failures = self.db.query(DocumentSync).filter(
                DocumentSync.sync_status == SyncStatus.FAILED
            ).count()

            health_score = 100
            if status_counts.get("failed", 0) > 10:
                health_score -= 20
            if status_counts.get("vector_ing", 0) > 50:
                health_score -= 10
            if status_counts.get("graph_ing", 0) > 50:
                health_score -= 10

            status = "healthy"
            if health_score < 70:
                status = "degraded"
            if health_score < 50:
                status = "unhealthy"

            return {
                "status": status,
                "health_score": health_score,
                "status_counts": status_counts,
                "recent_failures": recent_failures,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_system_health(self) -> Dict:
        """获取系统整体健康状态"""
        db_health = await self.check_database_health()
        sync_health = await self.check_sync_system_health()

        overall_status = "healthy"
        if db_health["status"] != "healthy" or sync_health["status"] == "unhealthy":
            overall_status = "unhealthy"
        elif sync_health["status"] == "degraded":
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "database": db_health,
            "sync_system": sync_health,
            "timestamp": datetime.utcnow().isoformat()
        }


# FastAPI错误处理器
async def sync_exception_handler(request: Request, exc: SyncError) -> JSONResponse:
    """FastAPI同步异常处理器"""
    return await ErrorHandler.handle_sync_error(exc)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """FastAPI通用异常处理器"""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)},
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# 日志清理器
class LogCleaner:
    """日志清理器"""

    def __init__(self, db: Session):
        self.db = db

    async def cleanup_old_logs(self, days: int = 30) -> int:
        """清理指定天数前的日志"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            deleted_count = self.db.query(SyncLog).filter(
                SyncLog.created_at < cutoff_date
            ).delete()

            self.db.commit()

            await sync_logger.log_sync_event(
                None, None, "INFO", "log_cleaner",
                f"Cleaned up {deleted_count} old log entries"
            )

            return deleted_count
        except Exception as e:
            await sync_logger.log_sync_event(
                None, None, "ERROR", "log_cleaner",
                f"Failed to cleanup old logs: {str(e)}"
            )
            return 0

    async def cleanup_error_logs(self, days: int = 7) -> int:
        """清理指定天数前的错误日志"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            deleted_count = self.db.query(SyncLog).filter(
                and_(
                    SyncLog.log_level == "ERROR",
                    SyncLog.created_at < cutoff_date
                )
            ).delete()

            self.db.commit()

            await sync_logger.log_sync_event(
                None, None, "INFO", "log_cleaner",
                f"Cleaned up {deleted_count} error log entries"
            )

            return deleted_count
        except Exception as e:
            await sync_logger.log_sync_event(
                None, None, "ERROR", "log_cleaner",
                f"Failed to cleanup error logs: {str(e)}"
            )
            return 0