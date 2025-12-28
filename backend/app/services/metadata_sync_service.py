"""
元数据同步与状态管理服务
负责协调各个存储系统之间的数据一致性
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import redis.asyncio as redis
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.mysql import get_db
from app.models.document import Document, DocumentTask, VectorStorage, KnowledgeGraphNode
from app.services.milvus_service import MilvusService
from app.services.neo4j_service import Neo4jService
from app.services.minio_service import MinioService

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class StorageSystem(str, Enum):
    MYSQL = "mysql"
    MILVUS = "milvus"
    NEO4J = "neo4j"
    MINIO = "minio"
    REDIS = "redis"
    MONGODB = "mongodb"


@dataclass
class SyncTask:
    """同步任务"""
    task_id: str
    document_id: str
    source_system: StorageSystem
    target_system: StorageSystem
    operation: str  # create, update, delete
    data: Dict[str, Any]
    status: SyncStatus = SyncStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None


@dataclass
class SyncConflict:
    """同步冲突"""
    document_id: str
    conflict_type: str
    source_data: Dict[str, Any]
    target_data: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolution: Optional[str] = None


class MetadataSyncService:
    """元数据同步服务"""

    def __init__(self):
        self.redis_client = None
        self.active_syncs: Dict[str, SyncTask] = {}
        self.conflicts: List[SyncConflict] = []
        self._running = False

    async def initialize(self):
        """初始化服务"""
        self.redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Metadata sync service initialized")

    async def start_sync_monitoring(self):
        """启动同步监控"""
        self._running = True
        await asyncio.gather(
            self._process_sync_queue(),
            self._detect_conflicts(),
            self._cleanup_completed_syncs(),
            self._monitor_system_health()
        )

    async def stop_sync_monitoring(self):
        """停止同步监控"""
        self._running = False

    async def submit_sync_task(self, sync_task: SyncTask) -> bool:
        """提交同步任务"""
        try:
            # 存储任务到Redis
            task_key = f"sync_task:{sync_task.task_id}"
            task_data = {
                "document_id": sync_task.document_id,
                "source_system": sync_task.source_system.value,
                "target_system": sync_task.target_system.value,
                "operation": sync_task.operation,
                "data": json.dumps(sync_task.data),
                "status": sync_task.status.value,
                "created_at": sync_task.created_at.isoformat(),
                "retry_count": sync_task.retry_count,
                "max_retries": sync_task.max_retries
            }

            await self.redis_client.hset(task_key, mapping=task_data)
            await self.redis_client.lpush("sync_queue", sync_task.task_id)

            # 添加到活动同步
            self.active_syncs[sync_task.task_id] = sync_task

            logger.info(f"Sync task submitted: {sync_task.task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit sync task: {e}")
            return False

    async def _process_sync_queue(self):
        """处理同步队列"""
        while self._running:
            try:
                # 获取待处理任务
                task_id = await self.redis_client.brpop("sync_queue", timeout=10)
                if not task_id:
                    continue

                task_id = task_id[1]  # brpop returns (key, value)
                await self._execute_sync_task(task_id)

            except Exception as e:
                logger.error(f"Error processing sync queue: {e}")
                await asyncio.sleep(5)

    async def _execute_sync_task(self, task_id: str):
        """执行同步任务"""
        try:
            # 获取任务数据
            task_key = f"sync_task:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                logger.warning(f"Sync task not found: {task_id}")
                return

            # 重建SyncTask对象
            sync_task = SyncTask(
                task_id=task_id,
                document_id=task_data["document_id"],
                source_system=StorageSystem(task_data["source_system"]),
                target_system=StorageSystem(task_data["target_system"]),
                operation=task_data["operation"],
                data=json.loads(task_data["data"]),
                status=SyncStatus(task_data["status"]),
                created_at=datetime.fromisoformat(task_data["created_at"]),
                retry_count=int(task_data.get("retry_count", 0)),
                max_retries=int(task_data.get("max_retries", 3))
            )

            # 更新状态为进行中
            sync_task.status = SyncStatus.IN_PROGRESS
            await self._update_sync_task_status(sync_task)

            # 执行同步
            success = await self._perform_sync(sync_task)

            if success:
                sync_task.status = SyncStatus.COMPLETED
                await self._update_sync_task_status(sync_task)
                logger.info(f"Sync task completed: {task_id}")
            else:
                await self._handle_sync_failure(sync_task)

        except Exception as e:
            logger.error(f"Error executing sync task {task_id}: {e}")
            await self._handle_sync_error(task_id, str(e))

    async def _perform_sync(self, sync_task: SyncTask) -> bool:
        """执行具体的同步操作"""
        try:
            # 根据源系统和目标系统选择同步策略
            if sync_task.source_system == StorageSystem.MYSQL:
                return await self._sync_from_mysql(sync_task)
            elif sync_task.source_system == StorageSystem.MILVUS:
                return await self._sync_from_milvus(sync_task)
            elif sync_task.source_system == StorageSystem.NEO4J:
                return await self._sync_from_neo4j(sync_task)
            elif sync_task.source_system == StorageSystem.MINIO:
                return await self._sync_from_minio(sync_task)
            else:
                logger.error(f"Unsupported source system: {sync_task.source_system}")
                return False

        except Exception as e:
            logger.error(f"Sync operation failed: {e}")
            return False

    async def _sync_from_mysql(self, sync_task: SyncTask) -> bool:
        """从MySQL同步"""
        if sync_task.target_system == StorageSystem.MILVUS:
            return await self._sync_mysql_to_milvus(sync_task)
        elif sync_task.target_system == StorageSystem.NEO4J:
            return await self._sync_mysql_to_neo4j(sync_task)
        elif sync_task.target_system == StorageSystem.MINIO:
            return await self._sync_mysql_to_minio(sync_task)
        else:
            return False

    async def _sync_mysql_to_milvus(self, sync_task: SyncTask) -> bool:
        """MySQL到Milvus的同步"""
        milvus_service = MilvusService()

        async with get_db() as db:
            # 获取文档数据
            document = await db.get(Document, sync_task.document_id)
            if not document:
                return False

            # 根据操作类型执行相应动作
            if sync_task.operation == "delete":
                await milvus_service.delete_document(sync_task.document_id)
            elif sync_task.operation == "update":
                # 重新索引文档向量
                from app.services.qwen_embedding_service import QwenEmbeddingService as EmbeddingService
                embedding_service = EmbeddingService()

                # 获取文档块
                result = await db.execute(
                    select(DocumentChunk).where(
                        DocumentChunk.document_id == sync_task.document_id
                    )
                )
                chunks = result.scalars().all()

                # 生成向量并插入
                texts = [chunk.content for chunk in chunks]
                embeddings = await embedding_service.embed_batch(texts)

                await milvus_service.insert_vectors(
                    document_id=sync_task.document_id,
                    chunk_ids=[chunk.id for chunk in chunks],
                    embeddings=embeddings,
                    contents=texts
                )

        return True

    async def _sync_mysql_to_neo4j(self, sync_task: SyncTask) -> bool:
        """MySQL到Neo4j的同步"""
        neo4j_service = Neo4jService()

        async with get_db() as db:
            document = await db.get(Document, sync_task.document_id)
            if not document:
                return False

            if sync_task.operation == "delete":
                await neo4j_service.delete_document_graph(sync_task.document_id)
            elif sync_task.operation == "update":
                # 这里应该有重建图谱的逻辑
                # 简化实现
                pass

        return True

    async def _sync_mysql_to_minio(self, sync_task: SyncTask) -> bool:
        """MySQL到MinIO的同步"""
        minio_service = MinioService()

        async with get_db() as db:
            document = await db.get(Document, sync_task.document_id)
            if not document:
                return False

            if sync_task.operation == "delete":
                # 删除MinIO中的文件
                if document.file_path:
                    await minio_service.delete_file(document.file_path)

        return True

    async def _detect_conflicts(self):
        """检测数据冲突"""
        while self._running:
            try:
                # 检测文档状态冲突
                await self._detect_document_conflicts()

                # 检测向量数据冲突
                await self._detect_vector_conflicts()

                # 检测图谱数据冲突
                await self._detect_graph_conflicts()

                await asyncio.sleep(300)  # 5分钟检查一次

            except Exception as e:
                logger.error(f"Conflict detection error: {e}")
                await asyncio.sleep(60)

    async def _detect_document_conflicts(self):
        """检测文档状态冲突"""
        async with get_db() as db:
            # 检查处理状态不一致的文档
            result = await db.execute(
                select(Document).where(
                    Document.status == "completed",
                    Document.processed_at.is_(None)
                )
            )
            conflicted_docs = result.scalars().all()

            for doc in conflicted_docs:
                conflict = SyncConflict(
                    document_id=doc.id,
                    conflict_type="status_inconsistency",
                    source_data={"status": doc.status},
                    target_data={"processed_at": None}
                )
                self.conflicts.append(conflict)

                # 尝试自动修复
                await self._auto_resolve_conflict(conflict)

    async def _detect_vector_conflicts(self):
        """检测向量数据冲突"""
        milvus_service = MilvusService()

        async with get_db() as db:
            # 检查MySQL中有记录但Milvus中缺失的向量
            result = await db.execute(
                select(Document.id).join(VectorStorage)
            )
            mysql_doc_ids = [row[0] for row in result]

            # 获取Milvus中的文档ID
            milvus_doc_ids = await milvus_service.get_all_document_ids()

            # 找出差异
            missing_in_milvus = set(mysql_doc_ids) - set(milvus_doc_ids)
            for doc_id in missing_in_milvus:
                conflict = SyncConflict(
                    document_id=doc_id,
                    conflict_type="vector_missing",
                    source_data={"mysql": True},
                    target_data={"milvus": False}
                )
                self.conflicts.append(conflict)

    async def _detect_graph_conflicts(self):
        """检测图谱数据冲突"""
        neo4j_service = Neo4jService()

        async with get_db() as db:
            # 检查MySQL中有记录但Neo4j中缺失的图谱数据
            result = await db.execute(
                select(Document.id).join(KnowledgeGraphNode)
            )
            mysql_doc_ids = [row[0] for row in result]

            # 获取Neo4j中的文档ID
            neo4j_doc_ids = await neo4j_service.get_all_document_ids()

            # 找出差异
            missing_in_neo4j = set(mysql_doc_ids) - set(neo4j_doc_ids)
            for doc_id in missing_in_neo4j:
                conflict = SyncConflict(
                    document_id=doc_id,
                    conflict_type="graph_missing",
                    source_data={"mysql": True},
                    target_data={"neo4j": False}
                )
                self.conflicts.append(conflict)

    async def _auto_resolve_conflict(self, conflict: SyncConflict):
        """自动解决冲突"""
        try:
            if conflict.conflict_type == "status_inconsistency":
                # 修复状态不一致
                async with get_db() as db:
                    document = await db.get(Document, conflict.document_id)
                    if document:
                        document.processed_at = datetime.utcnow()
                        await db.commit()
                conflict.resolution = "auto_fixed"

            elif conflict.conflict_type == "vector_missing":
                # 重新同步向量
                sync_task = SyncTask(
                    task_id=f"auto_sync_{conflict.document_id}_{int(datetime.utcnow().timestamp())}",
                    document_id=conflict.document_id,
                    source_system=StorageSystem.MYSQL,
                    target_system=StorageSystem.MILVUS,
                    operation="create",
                    data={"reason": "conflict_resolution"}
                )
                await self.submit_sync_task(sync_task)
                conflict.resolution = "sync_triggered"

            elif conflict.conflict_type == "graph_missing":
                # 重新同步图谱
                sync_task = SyncTask(
                    task_id=f"auto_sync_{conflict.document_id}_{int(datetime.utcnow().timestamp())}",
                    document_id=conflict.document_id,
                    source_system=StorageSystem.MYSQL,
                    target_system=StorageSystem.NEO4J,
                    operation="create",
                    data={"reason": "conflict_resolution"}
                )
                await self.submit_sync_task(sync_task)
                conflict.resolution = "sync_triggered"

        except Exception as e:
            logger.error(f"Failed to auto-resolve conflict: {e}")
            conflict.resolution = "failed"

    async def _cleanup_completed_syncs(self):
        """清理已完成的同步任务"""
        while self._running:
            try:
                # 清理24小时前完成的任务
                cutoff_time = datetime.utcnow() - timedelta(hours=24)

                # 清理Redis中的任务记录
                pattern = "sync_task:*"
                keys = await self.redis_client.keys(pattern)

                for key in keys:
                    task_data = await self.redis_client.hgetall(key)
                    if task_data:
                        created_at = datetime.fromisoformat(task_data.get("created_at", ""))
                        status = task_data.get("status")

                        if status in ["completed", "failed"] and created_at < cutoff_time:
                            await self.redis_client.delete(key)

                # 清理内存中的活动同步
                self.active_syncs = {
                    task_id: task for task_id, task in self.active_syncs.items()
                    if task.created_at > cutoff_time
                }

                await asyncio.sleep(3600)  # 1小时清理一次

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)

    async def _monitor_system_health(self):
        """监控系统健康状态"""
        while self._running:
            try:
                # 检查各系统连接状态
                health_status = await self._check_system_health()

                # 存储健康状态到Redis
                await self.redis_client.setex(
                    "system_health",
                    300,  # 5分钟过期
                    json.dumps(health_status)
                )

                # 如果有系统不健康，记录告警
                unhealthy_systems = [
                    system for system, status in health_status.items()
                    if not status["healthy"]
                ]

                if unhealthy_systems:
                    logger.warning(f"Unhealthy systems detected: {unhealthy_systems}")

                await asyncio.sleep(60)  # 1分钟检查一次

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _check_system_health(self) -> Dict[str, Dict[str, Any]]:
        """检查各系统健康状态"""
        health_status = {}

        # 检查MySQL
        try:
            async with get_db() as db:
                await db.execute("SELECT 1")
            health_status["mysql"] = {"healthy": True, "last_check": datetime.utcnow().isoformat()}
        except Exception as e:
            health_status["mysql"] = {"healthy": False, "error": str(e), "last_check": datetime.utcnow().isoformat()}

        # 检查Redis
        try:
            await self.redis_client.ping()
            health_status["redis"] = {"healthy": True, "last_check": datetime.utcnow().isoformat()}
        except Exception as e:
            health_status["redis"] = {"healthy": False, "error": str(e), "last_check": datetime.utcnow().isoformat()}

        # 检查Milvus
        try:
            milvus_service = MilvusService()
            await milvus_service.health_check()
            health_status["milvus"] = {"healthy": True, "last_check": datetime.utcnow().isoformat()}
        except Exception as e:
            health_status["milvus"] = {"healthy": False, "error": str(e), "last_check": datetime.utcnow().isoformat()}

        # 检查Neo4j
        try:
            neo4j_service = Neo4jService()
            await neo4j_service.health_check()
            health_status["neo4j"] = {"healthy": True, "last_check": datetime.utcnow().isoformat()}
        except Exception as e:
            health_status["neo4j"] = {"healthy": False, "error": str(e), "last_check": datetime.utcnow().isoformat()}

        # 检查MinIO
        try:
            minio_service = MinioService()
            await minio_service.health_check()
            health_status["minio"] = {"healthy": True, "last_check": datetime.utcnow().isoformat()}
        except Exception as e:
            health_status["minio"] = {"healthy": False, "error": str(e), "last_check": datetime.utcnow().isoformat()}

        return health_status

    async def _update_sync_task_status(self, sync_task: SyncTask):
        """更新同步任务状态"""
        task_key = f"sync_task:{sync_task.task_id}"
        update_data = {
            "status": sync_task.status.value,
            "retry_count": sync_task.retry_count
        }

        if sync_task.error_message:
            update_data["error_message"] = sync_task.error_message

        await self.redis_client.hset(task_key, mapping=update_data)

    async def _handle_sync_failure(self, sync_task: SyncTask):
        """处理同步失败"""
        sync_task.retry_count += 1

        if sync_task.retry_count < sync_task.max_retries:
            # 重试
            sync_task.status = SyncStatus.PENDING
            await self._update_sync_task_status(sync_task)
            await self.redis_client.lpush("sync_queue", sync_task.task_id)

            # 延迟重试
            delay = min(60 * (2 ** sync_task.retry_count), 300)  # 最大5分钟
            await asyncio.sleep(delay)
        else:
            # 标记为失败
            sync_task.status = SyncStatus.FAILED
            await self._update_sync_task_status(sync_task)
            logger.error(f"Sync task failed after max retries: {sync_task.task_id}")

    async def _handle_sync_error(self, task_id: str, error_message: str):
        """处理同步错误"""
        try:
            task_key = f"sync_task:{task_id}"
            await self.redis_client.hset(
                task_key,
                mapping={
                    "status": SyncStatus.FAILED.value,
                    "error_message": error_message
                }
            )
        except Exception as e:
            logger.error(f"Failed to handle sync error: {e}")

    async def get_sync_status(self, document_id: str) -> Dict[str, Any]:
        """获取文档同步状态"""
        try:
            # 从Redis获取相关的同步任务
            pattern = f"sync_task:*"
            keys = await self.redis_client.keys(pattern)

            related_tasks = []
            for key in keys:
                task_data = await self.redis_client.hgetall(key)
                if task_data.get("document_id") == document_id:
                    related_tasks.append({
                        "task_id": key.split(":")[-1],
                        "source_system": task_data.get("source_system"),
                        "target_system": task_data.get("target_system"),
                        "operation": task_data.get("operation"),
                        "status": task_data.get("status"),
                        "created_at": task_data.get("created_at"),
                        "retry_count": task_data.get("retry_count"),
                        "error_message": task_data.get("error_message")
                    })

            # 获取冲突信息
            conflicts = [
                {
                    "conflict_type": conflict.conflict_type,
                    "detected_at": conflict.detected_at.isoformat(),
                    "resolution": conflict.resolution
                }
                for conflict in self.conflicts
                if conflict.document_id == document_id
            ]

            return {
                "document_id": document_id,
                "sync_tasks": related_tasks,
                "conflicts": conflicts,
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {
                "document_id": document_id,
                "error": str(e)
            }

    async def force_resync(self, document_id: str) -> Dict[str, Any]:
        """强制重新同步文档"""
        try:
            # 创建同步任务
            sync_tasks = [
                SyncTask(
                    task_id=f"force_sync_{document_id}_milvus_{int(datetime.utcnow().timestamp())}",
                    document_id=document_id,
                    source_system=StorageSystem.MYSQL,
                    target_system=StorageSystem.MILVUS,
                    operation="update",
                    data={"reason": "force_resync"}
                ),
                SyncTask(
                    task_id=f"force_sync_{document_id}_neo4j_{int(datetime.utcnow().timestamp())}",
                    document_id=document_id,
                    source_system=StorageSystem.MYSQL,
                    target_system=StorageSystem.NEO4J,
                    operation="update",
                    data={"reason": "force_resync"}
                )
            ]

            results = []
            for task in sync_tasks:
                success = await self.submit_sync_task(task)
                results.append({
                    "task_id": task.task_id,
                    "target_system": task.target_system.value,
                    "submitted": success
                })

            return {
                "document_id": document_id,
                "sync_tasks": results,
                "initiated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error forcing resync: {e}")
            return {
                "document_id": document_id,
                "error": str(e)
            }


# 全局服务实例
metadata_sync_service = MetadataSyncService()


async def get_metadata_sync_service() -> MetadataSyncService:
    """获取元数据同步服务实例"""
    if not metadata_sync_service.redis_client:
        await metadata_sync_service.initialize()
    return metadata_sync_service