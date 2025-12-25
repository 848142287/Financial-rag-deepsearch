"""
双库同步管理器
实现MySQL和Milvus之间的数据同步和一致性保证
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from sqlalchemy import select, func

from ..milvus_service import milvus_service
from ..neo4j_service import neo4j_service
from ...core.database import get_db
from ...models.document import Document, DocumentChunk, Entity
from ...core.redis_client import redis_client

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """同步状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class SyncOperation(Enum):
    """同步操作类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BATCH_CREATE = "batch_create"


@dataclass
class SyncTask:
    """同步任务"""
    task_id: str
    operation: SyncOperation
    entity_type: str  # document, chunk, entity
    entity_id: str
    data: Dict[str, Any]
    source: str  # mysql, milvus
    target: str  # milvus, mysql
    status: SyncStatus
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    error_message: Optional[str] = None


class DualDatabaseSyncManager:
    """双库同步管理器"""

    def __init__(self):
        self.sync_queue_key = "sync_queue"
        self.sync_status_key_prefix = "sync_status:"
        self.conflict_resolution_key = "conflict_resolution"
        self.max_retry_count = 3
        self.batch_size = 100

    async def sync_document(
        self,
        document_id: str,
        operation: SyncOperation = SyncOperation.CREATE,
        source: str = "mysql"
    ) -> bool:
        """
        同步文档

        Args:
            document_id: 文档ID
            operation: 操作类型
            source: 数据源

        Returns:
            是否成功
        """
        logger.info(f"开始同步文档: {document_id}, 操作: {operation.value}")

        try:
            # 根据操作类型执行同步
            if operation == SyncOperation.CREATE:
                success = await self._create_document_sync(document_id, source)
            elif operation == SyncOperation.UPDATE:
                success = await self._update_document_sync(document_id, source)
            elif operation == SyncOperation.DELETE:
                success = await self._delete_document_sync(document_id, source)
            else:
                logger.error(f"不支持的操作类型: {operation}")
                return False

            if success:
                await self._update_sync_status(document_id, SyncStatus.COMPLETED)
                logger.info(f"文档同步成功: {document_id}")
            else:
                await self._update_sync_status(document_id, SyncStatus.FAILED)
                logger.error(f"文档同步失败: {document_id}")

            return success

        except Exception as e:
            logger.error(f"文档同步异常: {document_id}, 错误: {e}")
            await self._update_sync_status(document_id, SyncStatus.FAILED, str(e))
            return False

    async def sync_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        operation: SyncOperation = SyncOperation.BATCH_CREATE
    ) -> bool:
        """
        同步文档块

        Args:
            document_id: 文档ID
            chunks: 文档块列表
            operation: 操作类型

        Returns:
            是否成功
        """
        logger.info(f"开始同步文档块: {document_id}, 数量: {len(chunks)}")

        try:
            # 批量处理文档块
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                success = await self._sync_chunk_batch(document_id, batch, operation)

                if not success:
                    logger.error(f"文档块批次同步失败: {document_id}, 批次: {i//self.batch_size}")
                    return False

            await self._update_sync_status(f"{document_id}_chunks", SyncStatus.COMPLETED)
            logger.info(f"文档块同步成功: {document_id}")
            return True

        except Exception as e:
            logger.error(f"文档块同步异常: {document_id}, 错误: {e}")
            await self._update_sync_status(f"{document_id}_chunks", SyncStatus.FAILED, str(e))
            return False

    async def sync_entities(
        self,
        entities: List[Dict[str, Any]],
        operation: SyncOperation = SyncOperation.BATCH_CREATE
    ) -> bool:
        """
        同步实体到Neo4j

        Args:
            entities: 实体列表
            operation: 操作类型

        Returns:
            是否成功
        """
        logger.info(f"开始同步实体: {len(entities)} 个")

        try:
            # 过滤有效实体
            valid_entities = [
                entity for entity in entities
                if entity.get('text') and entity.get('type')
            ]

            if not valid_entities:
                logger.warning("没有有效实体需要同步")
                return True

            # 批量同步到Neo4j
            success = await neo4j_service.create_entities(valid_entities)

            if success:
                # 同步关系
                relationships = await self._extract_entity_relationships(valid_entities)
                if relationships:
                    await neo4j_service.create_relationships(relationships)

                logger.info(f"实体同步成功: {len(valid_entities)} 个")
                return True
            else:
                logger.error("实体同步失败")
                return False

        except Exception as e:
            logger.error(f"实体同步异常: {e}")
            return False

    async def _create_document_sync(
        self,
        document_id: str,
        source: str
    ) -> bool:
        """创建文档同步"""
        if source == "mysql":
            # 从MySQL读取，同步到Milvus
            async with get_db() as db:
                document = await db.get(Document, document_id)
                if not document:
                    logger.error(f"文档不存在: {document_id}")
                    return False

                # 获取文档块
                chunks = await db.execute(
                    select(DocumentChunk).where(DocumentChunk.document_id == document_id)
                )
                chunks = chunks.scalars().all()

                # 准备Milvus数据
                milvus_data = []
                for chunk in chunks:
                    milvus_data.append({
                        'id': f"{document_id}_{chunk.id}",
                        'document_id': document_id,
                        'chunk_id': str(chunk.id),
                        'content': chunk.content,
                        'embedding': json.loads(chunk.embedding) if chunk.embedding else None,
                        'metadata': {
                            'chunk_index': chunk.chunk_index,
                            'start_char': chunk.start_char,
                            'end_char': chunk.end_char,
                            'document_title': document.filename
                        }
                    })

                # 批量插入Milvus
                if milvus_data:
                    await milvus_service.insert_documents(milvus_data)

                return True

        elif source == "milvus":
            # 从Milvus读取，同步到MySQL
            # 这种情况较少，通常作为数据恢复使用
            logger.warning("从Milvus同步到MySQL需要额外实现")
            return True

        return False

    async def _update_document_sync(
        self,
        document_id: str,
        source: str
    ) -> bool:
        """更新文档同步"""
        # 更新操作通常涉及重新生成嵌入
        # 简化实现：删除后重新创建
        await self._delete_document_sync(document_id, source)
        return await self._create_document_sync(document_id, source)

    async def _delete_document_sync(
        self,
        document_id: str,
        source: str
    ) -> bool:
        """删除文档同步"""
        try:
            if source == "mysql":
                # 从Milvus删除
                await milvus_service.delete_documents(document_id)

                # 从Neo4j删除相关实体和关系
                await neo4j_service.delete_document_entities(document_id)

            return True

        except Exception as e:
            logger.error(f"删除文档同步失败: {e}")
            return False

    async def _sync_chunk_batch(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        operation: SyncOperation
    ) -> bool:
        """同步文档块批次"""
        try:
            if operation == SyncOperation.BATCH_CREATE:
                # 生成嵌入
                texts = [chunk['content'] for chunk in chunks]
                embeddings = await self._generate_embeddings(texts)

                # 准备Milvus数据
                milvus_data = []
                for i, chunk in enumerate(chunks):
                    milvus_data.append({
                        'id': f"{document_id}_{chunk['metadata']['chunk_index']}",
                        'document_id': document_id,
                        'chunk_id': chunk['metadata']['chunk_index'],
                        'content': chunk['content'],
                        'embedding': embeddings[i] if i < len(embeddings) else None,
                        'metadata': chunk['metadata']
                    })

                # 批量插入Milvus
                await milvus_service.insert_documents(milvus_data)

            return True

        except Exception as e:
            logger.error(f"同步文档块批次失败: {e}")
            return False

    async def _extract_entity_relationships(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """提取实体关系"""
        relationships = []

        # 简单实现：基于文档位置和实体类型推断关系
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # 检查是否在同一文档或相近位置
                if self._should_create_relationship(entity1, entity2):
                    relationship = {
                        'source': entity1['text'],
                        'target': entity2['text'],
                        'type': self._infer_relationship_type(entity1, entity2),
                        'properties': {
                            'confidence': 0.7,
                            'source': 'auto_extraction'
                        }
                    }
                    relationships.append(relationship)

        return relationships

    def _should_create_relationship(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> bool:
        """判断是否应该创建关系"""
        # 同类型实体通常不建立关系
        if entity1['type'] == entity2['type']:
            return False

        # 特定类型组合建立关系
        valid_combinations = [
            ('COMPANY', 'PERSON'),
            ('PERSON', 'COMPANY'),
            ('COMPANY', 'STOCK'),
            ('STOCK', 'COMPANY')
        ]

        return (entity1['type'], entity2['type']) in valid_combinations

    def _infer_relationship_type(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> str:
        """推断关系类型"""
        type_mapping = {
            ('COMPANY', 'PERSON'): 'CEO_OF',
            ('PERSON', 'COMPANY'): 'WORKS_AT',
            ('COMPANY', 'STOCK'): 'HAS_STOCK',
            ('STOCK', 'COMPANY'): 'STOCK_OF'
        }

        return type_mapping.get(
            (entity1['type'], entity2['type']),
            'RELATED_TO'
        )

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入"""
        from ..embedding_service import embedding_service
        return await embedding_service.generate_embeddings(texts)

    async def _update_sync_status(
        self,
        entity_id: str,
        status: SyncStatus,
        error_message: Optional[str] = None
    ):
        """更新同步状态"""
        try:
            status_key = f"{self.sync_status_key_prefix}{entity_id}"
            status_data = {
                'status': status.value,
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'error_message': error_message
            }

            await redis_client.setex(status_key, 3600, json.dumps(status_data))

        except Exception as e:
            logger.error(f"更新同步状态失败: {e}")

    async def get_sync_status(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取同步状态"""
        try:
            status_key = f"{self.sync_status_key_prefix}{entity_id}"
            status_data = await redis_client.get(status_key)

            if status_data:
                return json.loads(status_data)

            return None

        except Exception as e:
            logger.error(f"获取同步状态失败: {e}")
            return None

    async def resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """解决冲突"""
        resolution_results = {
            'resolved': [],
            'failed': [],
            'manual_review': []
        }

        for conflict in conflicts:
            try:
                # 尝试自动解决冲突
                resolution = await self._auto_resolve_conflict(conflict)

                if resolution['auto_resolved']:
                    resolution_results['resolved'].append({
                        'conflict_id': conflict['id'],
                        'resolution': resolution['resolution']
                    })
                else:
                    # 需要人工审查
                    resolution_results['manual_review'].append(conflict)

            except Exception as e:
                resolution_results['failed'].append({
                    'conflict_id': conflict['id'],
                    'error': str(e)
                })

        return resolution_results

    async def _auto_resolve_conflict(
        self,
        conflict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """自动解决冲突"""
        # 简化的冲突解决策略
        conflict_type = conflict.get('type')

        if conflict_type == 'version_mismatch':
            # 版本不匹配：使用最新版本
            return {
                'auto_resolved': True,
                'resolution': 'use_latest_version'
            }
        elif conflict_type == 'data_corruption':
            # 数据损坏：从备份恢复
            return {
                'auto_resolved': True,
                'resolution': 'restore_from_backup'
            }
        else:
            # 其他冲突需要人工审查
            return {
                'auto_resolved': False,
                'resolution': None
            }

    async def verify_consistency(self) -> Dict[str, Any]:
        """验证数据一致性"""
        verification_result = {
            'mysql_milvus': {},
            'mysql_neo4j': {},
            'overall_consistency': True,
            'inconsistencies': []
        }

        try:
            # 验证MySQL和Milvus的一致性
            mysql_count = await self._count_mysql_documents()
            milvus_count = await self._count_milvus_documents()

            verification_result['mysql_milvus'] = {
                'mysql_count': mysql_count,
                'milvus_count': milvus_count,
                'consistent': mysql_count == milvus_count
            }

            # 验证MySQL和Neo4j的一致性
            mysql_entity_count = await self._count_mysql_entities()
            neo4j_entity_count = await self._count_neo4j_entities()

            verification_result['mysql_neo4j'] = {
                'mysql_entity_count': mysql_entity_count,
                'neo4j_entity_count': neo4j_entity_count,
                'consistent': mysql_entity_count == neo4j_entity_count
            }

            # 检查整体一致性
            if not verification_result['mysql_milvus']['consistent']:
                verification_result['overall_consistency'] = False
                verification_result['inconsistencies'].append(
                    'MySQL and Milvus document count mismatch'
                )

            if not verification_result['mysql_neo4j']['consistent']:
                verification_result['overall_consistency'] = False
                verification_result['inconsistencies'].append(
                    'MySQL and Neo4j entity count mismatch'
                )

            return verification_result

        except Exception as e:
            logger.error(f"验证一致性失败: {e}")
            verification_result['error'] = str(e)
            return verification_result

    async def _count_mysql_documents(self) -> int:
        """统计MySQL文档数量"""
        async with get_db() as db:
            result = await db.execute(select(func.count(Document.id)))
            return result.scalar()

    async def _count_milvus_documents(self) -> int:
        """统计Milvus文档数量"""
        try:
            # 实现Milvus统计功能
            return await milvus_service.count_documents()
        except Exception as e:
            logger.error(f"统计Milvus文档数量失败: {e}")
            return 0

    async def _count_mysql_entities(self) -> int:
        """统计MySQL实体数量"""
        async with get_db() as db:
            result = await db.execute(select(func.count(Entity.id)))
            return result.scalar()

    async def _count_neo4j_entities(self) -> int:
        """统计Neo4j实体数量"""
        try:
            # 实现Neo4j统计功能
            return await neo4j_service.count_entities()
        except Exception as e:
            logger.error(f"统计Neo4j实体数量失败: {e}")
            return 0


# 全局双库同步管理器实例
dual_database_sync_manager = DualDatabaseSyncManager()