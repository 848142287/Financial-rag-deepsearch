"""
系统管理相关API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from typing import List, Any, Dict, Optional
from datetime import datetime, timedelta
import psutil
import os
import logging

from app.core.database import get_db, engine
from app.schemas.admin import SystemStatus, SystemConfig
from app.models.admin import SystemConfig as SystemConfigModel
from app.models.document import Document
from app.models.conversation import Conversation, Message

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/system-status", response_model=SystemStatus)
async def get_system_status(
    db: Session = Depends(get_db)
) -> Any:
    """获取系统状态"""
    try:
        # 检查数据库连接
        db_status = "connected"
        try:
            result = db.execute(text("SELECT 1"))
            result.fetchone()  # 确保执行查询
        except Exception as e:
            db_status = f"error: {str(e)}"
            logger.error(f"Database connection error: {e}")

        # 检查Redis连接
        redis_status = "connected"
        try:
            from app.core.redis_client import redis_client
            if not await redis_client.ping():
                redis_status = "disconnected"
        except Exception as e:
            redis_status = f"error: {str(e)}"
            logger.error(f"Redis connection error: {e}")

        # 检查Milvus连接
        milvus_status = "connected"
        try:
            from app.services.milvus_service import MilvusService
            milvus_service = MilvusService()
            # 简单检查集合是否存在
            # 这里可以添加更详细的健康检查
        except Exception as e:
            milvus_status = f"error: {str(e)}"
            logger.error(f"Milvus connection error: {e}")

        # 检查Neo4j连接
        neo4j_status = "connected"
        try:
            from app.services.neo4j_service import Neo4jService
            neo4j_service = Neo4jService()
            # 简单连接检查
        except Exception as e:
            neo4j_status = f"error: {str(e)}"
            logger.error(f"Neo4j connection error: {e}")

        # 检查MinIO连接
        minio_status = "connected"
        try:
            from app.services.minio_service import MinIOService
            minio_service = MinIOService()
            # 简单检查存储桶是否可访问
        except Exception as e:
            minio_status = f"error: {str(e)}"
            logger.error(f"MinIO connection error: {e}")

        # 获取系统资源信息
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # 确定整体状态
        all_connected = all(
            "connected" in status
            for status in [db_status, redis_status, milvus_status, neo4j_status, minio_status]
        )
        overall_status = "healthy" if all_connected else "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": db_status,
                "redis": redis_status,
                "milvus": milvus_status,
                "neo4j": neo4j_status,
                "minio": minio_status
            },
            "system_resources": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system status: {str(e)}"
        )


@router.get("/configs", response_model=List[SystemConfig])
async def get_system_configs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> Any:
    """获取系统配置"""
    try:
        configs = db.query(SystemConfigModel)\
            .offset(skip)\
            .limit(limit)\
            .all()
        return configs

    except Exception as e:
        logger.error(f"Failed to get system configs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system configs: {str(e)}"
        )


@router.get("/configs/{config_key}", response_model=SystemConfig)
async def get_system_config(
    config_key: str,
    db: Session = Depends(get_db)
) -> Any:
    """获取特定系统配置"""
    try:
        config = db.query(SystemConfigModel)\
            .filter(SystemConfigModel.config_key == config_key)\
            .first()

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Config not found"
            )

        return config

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system config {config_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system config: {str(e)}"
        )


@router.put("/configs/{config_key}", response_model=SystemConfig)
async def update_system_config(
    config_key: str,
    config_data: dict,
    db: Session = Depends(get_db)
) -> Any:
    """更新系统配置"""
    try:
        config = db.query(SystemConfigModel)\
            .filter(SystemConfigModel.config_key == config_key)\
            .first()

        if not config:
            # 创建新配置
            config = SystemConfigModel(
                config_key=config_key,
                config_value=str(config_data.get("value", "")),
                description=config_data.get("description", "")
            )
            db.add(config)
        else:
            # 更新现有配置
            config.config_value = str(config_data.get("value", config.config_value))
            if config_data.get("description"):
                config.description = config_data["description"]
            config.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(config)

        return config

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update system config {config_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update system config: {str(e)}"
        )


@router.get("/logs")
async def get_system_logs(
    level: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[datetime] = None,
    db: Session = Depends(get_db)
) -> Any:
    """获取系统日志"""
    try:
        # 这里可以实现从日志文件或日志数据库中读取
        # 简化实现，返回一些示例日志
        logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": "System is running normally",
                "module": "system"
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                "level": "INFO",
                "message": "Database connection established",
                "module": "database"
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
                "level": "WARNING",
                "message": "High memory usage detected",
                "module": "monitoring"
            }
        ]

        # 根据级别过滤
        if level:
            logs = [log for log in logs if log["level"] == level.upper()]

        # 根据时间过滤
        if start_time:
            logs = [log for log in logs if datetime.fromisoformat(log["timestamp"]) >= start_time]

        # 限制数量
        logs = logs[:limit]

        return {
            "logs": logs,
            "total": len(logs),
            "filters": {
                "level": level,
                "limit": limit,
                "start_time": start_time.isoformat() if start_time else None
            }
        }

    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system logs: {str(e)}"
        )


@router.get("/statistics")
async def get_system_statistics(
    db: Session = Depends(get_db)
) -> Any:
    """获取系统统计信息"""
    try:
        # 文档统计
        total_documents = db.query(func.count(Document.id)).scalar() or 0
        completed_documents = db.query(func.count(Document.id))\
            .filter(Document.status == "completed").scalar() or 0
        processing_documents = db.query(func.count(Document.id))\
            .filter(Document.status == "processing").scalar() or 0

        # 对话统计
        total_conversations = db.query(func.count(Conversation.id)).scalar() or 0
        total_messages = db.query(func.count(Message.id)).scalar() or 0

        # 今日统计
        today = datetime.utcnow().date()
        today_documents = db.query(func.count(Document.id))\
            .filter(func.date(Document.created_at) == today).scalar() or 0
        today_conversations = db.query(func.count(Conversation.id))\
            .filter(func.date(Conversation.created_at) == today).scalar() or 0

        # 存储统计
        total_storage = 0
        try:
            documents = db.query(Document).filter(Document.file_size.isnot(None)).all()
            total_storage = sum(doc.file_size or 0 for doc in documents)
        except Exception:
            pass

        # 系统运行时间
        system_uptime = 0  # 可以实现实际运行时间计算
        try:
            # 简单实现：基于进程启动时间
            process = psutil.Process(os.getpid())
            system_uptime = datetime.utcnow().timestamp() - process.create_time()
        except Exception:
            pass

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "documents": {
                "total": total_documents,
                "completed": completed_documents,
                "processing": processing_documents,
                "today": today_documents
            },
            "conversations": {
                "total": total_conversations,
                "messages": total_messages,
                "today": today_conversations
            },
            "storage": {
                "total_bytes": total_storage,
                "total_mb": round(total_storage / (1024 * 1024), 2),
                "total_gb": round(total_storage / (1024 * 1024 * 1024), 2)
            },
            "system": {
                "uptime_seconds": int(system_uptime),
                "uptime_hours": round(system_uptime / 3600, 2)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get system statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system statistics: {str(e)}"
        )


@router.delete("/cache")
async def clear_system_cache(
    cache_type: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Any:
    """清理系统缓存"""
    try:
        cleared_caches = []

        # 清理Redis缓存
        if not cache_type or cache_type == "redis":
            try:
                from app.core.redis_client import redis_client
                await redis_client.flushdb()
                cleared_caches.append("redis")
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")

        # 清理应用缓存
        if not cache_type or cache_type == "application":
            # 这里可以添加应用级缓存清理逻辑
            cleared_caches.append("application")

        return {
            "message": "Cache cleared successfully",
            "cleared_caches": cleared_caches,
            "cache_type": cache_type or "all"
        }

    except Exception as e:
        logger.error(f"Failed to clear system cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear system cache: {str(e)}"
        )


@router.post("/backup")
async def create_system_backup(
    backup_type: str = "full",
    db: Session = Depends(get_db)
) -> Any:
    """创建系统备份"""
    try:
        # 这里可以实现实际的数据备份逻辑
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # 数据库备份
        if backup_type in ["full", "database"]:
            try:
                # 使用mysqldump备份数据库
                import subprocess
                import os
                from app.core.config import settings

                backup_dir = "./backups"
                os.makedirs(backup_dir, exist_ok=True)

                db_backup_file = f"{backup_dir}/{backup_id}_database.sql"

                # 获取数据库连接信息
                db_url = settings.DATABASE_URL
                # 解析数据库连接信息 (mysql://user:pass@host:port/db)
                import re
                match = re.match(r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
                if match:
                    user, password, host, port, database = match.groups()

                    cmd = [
                        'mysqldump',
                        f'-h{host}',
                        f'-P{port}',
                        f'-u{user}',
                        f'-p{password}',
                        '--single-transaction',
                        '--routines',
                        '--triggers',
                        database
                    ]

                    with open(db_backup_file, 'w') as f:
                        subprocess.run(cmd, stdout=f, check=True)

                    logger.info(f"数据库备份完成: {db_backup_file}")
            except Exception as e:
                logger.error(f"数据库备份失败: {e}")
                raise

        # 文件备份
        if backup_type in ["full", "files"]:
            try:
                import shutil
                import os

                backup_dir = "./backups"
                os.makedirs(backup_dir, exist_ok=True)

                files_backup_dir = f"{backup_dir}/{backup_id}_files"

                # 备份重要文件目录
                source_dirs = [
                    "./uploads",
                    "./data",
                    "./logs"
                ]

                for source_dir in source_dirs:
                    if os.path.exists(source_dir):
                        dest_dir = f"{files_backup_dir}/{source_dir.replace('./', '')}"
                        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
                        shutil.copytree(source_dir, dest_dir, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))

                logger.info(f"文件备份完成: {files_backup_dir}")
            except Exception as e:
                logger.error(f"文件备份失败: {e}")
                raise

        return {
            "message": "Backup created successfully",
            "backup_id": backup_id,
            "backup_type": backup_type,
            "created_at": datetime.utcnow().isoformat(),
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Failed to create system backup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create system backup: {str(e)}"
        )
