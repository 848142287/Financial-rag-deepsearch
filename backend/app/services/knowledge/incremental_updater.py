"""
增量更新机制
基于文档ID+版本号+内容哈希识别差异，仅更新变化的部分
"""

import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from ...sync_state_machine import sync_state_machine, SyncState
from ...core.redis_client import redis_client
from ...core.database import get_db
from ...models.document import Document, DocumentChunk, Entity

logger = logging.getLogger(__name__)


@dataclass
class DocumentVersion:
    """文档版本信息"""
    document_id: str
    version: int
    content_hash: str
    chunk_hashes: Dict[str, str]  # chunk_id -> hash
    entity_hashes: Dict[str, str]  # entity_id -> hash
    metadata: Dict[str, Any]
    updated_at: datetime


@dataclass
class DiffResult:
    """差异检测结果"""
    document_id: str
    has_changes: bool
    new_chunks: List[str]
    modified_chunks: List[str]
    deleted_chunks: List[str]
    new_entities: List[str]
    modified_entities: List[str]
    deleted_entities: List[str]


class IncrementalUpdater:
    """增量更新器"""

    def __init__(self):
        self.version_key_prefix = "doc_version:"
        self.chunk_key_prefix = "chunk_hash:"
        self.entity_key_prefix = "entity_hash:"

    async def detect_changes(
        self,
        document_id: str,
        new_content_hash: str,
        new_chunks: List[Dict[str, Any]],
        new_entities: List[Dict[str, Any]]
    ) -> DiffResult:
        """
        检测文档变化

        Args:
            document_id: 文档ID
            new_content_hash: 新内容哈希
            new_chunks: 新文档块列表
            new_entities: 新实体列表

        Returns:
            差异检测结果
        """
        try:
            # 获取当前版本信息
            current_version = await self._get_document_version(document_id)

            if not current_version:
                # 新文档，全部为新增
                return self._create_new_document_diff(
                    document_id, new_chunks, new_entities
                )

            # 检查内容是否变化
            if current_version.content_hash == new_content_hash:
                # 内容无变化
                return DiffResult(
                    document_id=document_id,
                    has_changes=False,
                    new_chunks=[],
                    modified_chunks=[],
                    deleted_chunks=[],
                    new_entities=[],
                    modified_entities=[],
                    deleted_entities=[]
                )

            # 检测文档块变化
            chunk_diff = await self._detect_chunk_changes(
                current_version, new_chunks
            )

            # 检测实体变化
            entity_diff = await self._detect_entity_changes(
                current_version, new_entities
            )

            return DiffResult(
                document_id=document_id,
                has_changes=True,
                new_chunks=chunk_diff['new'],
                modified_chunks=chunk_diff['modified'],
                deleted_chunks=chunk_diff['deleted'],
                new_entities=entity_diff['new'],
                modified_entities=entity_diff['modified'],
                deleted_entities=entity_diff['deleted']
            )

        except Exception as e:
            logger.error(f"检测文档变化失败 {document_id}: {e}")
            raise

    async def apply_incremental_update(
        self,
        diff_result: DiffResult,
        new_chunks: List[Dict[str, Any]],
        new_entities: List[Dict[str, Any]]
    ) -> bool:
        """
        应用增量更新

        Args:
            diff_result: 差异检测结果
            new_chunks: 新文档块列表
            new_entities: 新实体列表

        Returns:
            是否更新成功
        """
        try:
            document_id = diff_result.document_id

            if not diff_result.has_changes:
                logger.info(f"文档无变化，跳过更新: {document_id}")
                return True

            # 初始化同步状态
            await sync_state_machine.initialize_sync(
                document_id,
                await self._get_next_version(document_id),
                await self._calculate_content_hash(new_chunks)
            )

            # 删除已删除的块和实体
            await self._delete_removed_items(diff_result)

            # 更新修改的块
            await self._update_modified_chunks(
                diff_result.modified_chunks, new_chunks
            )

            # 更新修改的实体
            await self._update_modified_entities(
                diff_result.modified_entities, new_entities
            )

            # 添加新的块
            await self._add_new_chunks(
                diff_result.new_chunks, new_chunks
            )

            # 添加新的实体
            await self._add_new_entities(
                diff_result.new_entities, new_entities
            )

            # 更新版本信息
            await self._update_document_version(
                document_id, new_chunks, new_entities
            )

            # 完成同步
            await sync_state_machine.transition_to_completed(document_id)

            logger.info(f"增量更新完成: {document_id}")
            return True

        except Exception as e:
            logger.error(f"增量更新失败 {diff_result.document_id}: {e}")
            await sync_state_machine.transition_to_failed(
                diff_result.document_id, str(e)
            )
            return False

    async def _get_document_version(self, document_id: str) -> Optional[DocumentVersion]:
        """获取文档版本信息"""
        try:
            key = f"{self.version_key_prefix}{document_id}"
            version_data = await redis_client.get(key)

            if version_data:
                data = json.loads(version_data)
                return DocumentVersion(
                    document_id=data['document_id'],
                    version=data['version'],
                    content_hash=data['content_hash'],
                    chunk_hashes=data['chunk_hashes'],
                    entity_hashes=data['entity_hashes'],
                    metadata=data.get('metadata', {}),
                    updated_at=datetime.fromisoformat(data['updated_at'])
                )

            return None

        except Exception as e:
            logger.error(f"获取文档版本失败 {document_id}: {e}")
            return None

    async def _calculate_chunk_hash(self, chunk: Dict[str, Any]) -> str:
        """计算文档块哈希"""
        content = chunk.get('content', '')
        metadata = json.dumps(chunk.get('metadata', {}), sort_keys=True)
        hash_source = f"{content}|{metadata}"
        return hashlib.md5(hash_source.encode()).hexdigest()

    async def _calculate_entity_hash(self, entity: Dict[str, Any]) -> str:
        """计算实体哈希"""
        text = entity.get('text', '')
        entity_type = entity.get('type', '')
        metadata = json.dumps(entity.get('metadata', {}), sort_keys=True)
        hash_source = f"{text}|{entity_type}|{metadata}"
        return hashlib.md5(hash_source.encode()).hexdigest()

    async def _calculate_content_hash(self, chunks: List[Dict[str, Any]]) -> str:
        """计算整个文档内容哈希"""
        all_content = ''.join(chunk.get('content', '') for chunk in chunks)
        return hashlib.md5(all_content.encode()).hexdigest()

    def _create_new_document_diff(
        self,
        document_id: str,
        new_chunks: List[Dict[str, Any]],
        new_entities: List[Dict[str, Any]]
    ) -> DiffResult:
        """创建新文档的差异结果"""
        new_chunk_ids = [str(chunk.get('id', i)) for i, chunk in enumerate(new_chunks)]
        new_entity_ids = [str(entity.get('id', i)) for i, entity in enumerate(new_entities)]

        return DiffResult(
            document_id=document_id,
            has_changes=True,
            new_chunks=new_chunk_ids,
            modified_chunks=[],
            deleted_chunks=[],
            new_entities=new_entity_ids,
            modified_entities=[],
            deleted_entities=[]
        )

    async def _detect_chunk_changes(
        self,
        current_version: DocumentVersion,
        new_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """检测文档块变化"""
        current_hashes = current_version.chunk_hashes
        new_hashes = {}

        # 计算新块的哈希
        for chunk in new_chunks:
            chunk_id = str(chunk.get('id', ''))
            if chunk_id:
                new_hashes[chunk_id] = await self._calculate_chunk_hash(chunk)

        # 分类变化
        new_chunk_ids = []
        modified_chunk_ids = []
        deleted_chunk_ids = []

        # 新增和修改的块
        for chunk_id, new_hash in new_hashes.items():
            if chunk_id not in current_hashes:
                new_chunk_ids.append(chunk_id)
            elif current_hashes[chunk_id] != new_hash:
                modified_chunk_ids.append(chunk_id)

        # 删除的块
        for chunk_id in current_hashes:
            if chunk_id not in new_hashes:
                deleted_chunk_ids.append(chunk_id)

        return {
            'new': new_chunk_ids,
            'modified': modified_chunk_ids,
            'deleted': deleted_chunk_ids
        }

    async def _detect_entity_changes(
        self,
        current_version: DocumentVersion,
        new_entities: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """检测实体变化"""
        current_hashes = current_version.entity_hashes
        new_hashes = {}

        # 计算新实体的哈希
        for entity in new_entities:
            entity_id = str(entity.get('id', ''))
            if entity_id:
                new_hashes[entity_id] = await self._calculate_entity_hash(entity)

        # 分类变化
        new_entity_ids = []
        modified_entity_ids = []
        deleted_entity_ids = []

        # 新增和修改的实体
        for entity_id, new_hash in new_hashes.items():
            if entity_id not in current_hashes:
                new_entity_ids.append(entity_id)
            elif current_hashes[entity_id] != new_hash:
                modified_entity_ids.append(entity_id)

        # 删除的实体
        for entity_id in current_hashes:
            if entity_id not in new_hashes:
                deleted_entity_ids.append(entity_id)

        return {
            'new': new_entity_ids,
            'modified': modified_entity_ids,
            'deleted': deleted_entity_ids
        }

    async def _delete_removed_items(self, diff_result: DiffResult):
        """删除已移除的项目"""
        document_id = diff_result.document_id

        # 删除已删除的文档块（从向量库）
        if diff_result.deleted_chunks:
            from ..milvus_service import milvus_service
            await milvus_service.delete_chunks(document_id, diff_result.deleted_chunks)

        # 删除已删除的实体（从图谱库）
        if diff_result.deleted_entities:
            from ..neo4j_service import neo4j_service
            await neo4j_service.delete_entities(diff_result.deleted_entities)

    async def _update_modified_chunks(
        self,
        modified_chunk_ids: List[str],
        new_chunks: List[Dict[str, Any]]
    ):
        """更新修改的文档块"""
        if not modified_chunk_ids:
            return

        # 找到修改的块
        modified_chunks = []
        for chunk in new_chunks:
            chunk_id = str(chunk.get('id', ''))
            if chunk_id in modified_chunk_ids:
                modified_chunks.append(chunk)

        if modified_chunks:
            from ..embedding_service import embedding_service
            from ..milvus_service import milvus_service

            # 重新生成嵌入
            texts = [chunk['content'] for chunk in modified_chunks]
            embeddings = await embedding_service.generate_embeddings(texts)

            # 更新向量库
            await milvus_service.update_chunks(modified_chunks, embeddings)

    async def _update_modified_entities(
        self,
        modified_entity_ids: List[str],
        new_entities: List[Dict[str, Any]]
    ):
        """更新修改的实体"""
        if not modified_entity_ids:
            return

        # 找到修改的实体
        modified_entities = []
        for entity in new_entities:
            entity_id = str(entity.get('id', ''))
            if entity_id in modified_entity_ids:
                modified_entities.append(entity)

        if modified_entities:
            from ..neo4j_service import neo4j_service
            await neo4j_service.update_entities(modified_entities)

    async def _add_new_chunks(
        self,
        new_chunk_ids: List[str],
        new_chunks: List[Dict[str, Any]]
    ):
        """添加新的文档块"""
        if not new_chunk_ids:
            return

        # 找到新的块
        new_chunk_objects = []
        for chunk in new_chunks:
            chunk_id = str(chunk.get('id', ''))
            if chunk_id in new_chunk_ids:
                new_chunk_objects.append(chunk)

        if new_chunk_objects:
            from ..embedding_service import embedding_service
            from ..milvus_service import milvus_service

            # 生成嵌入
            texts = [chunk['content'] for chunk in new_chunk_objects]
            embeddings = await embedding_service.generate_embeddings(texts)

            # 插入向量库
            await milvus_service.insert_chunks(new_chunk_objects, embeddings)

    async def _add_new_entities(
        self,
        new_entity_ids: List[str],
        new_entities: List[Dict[str, Any]]
    ):
        """添加新的实体"""
        if not new_entity_ids:
            return

        # 找到新的实体
        new_entity_objects = []
        for entity in new_entities:
            entity_id = str(entity.get('id', ''))
            if entity_id in new_entity_ids:
                new_entity_objects.append(entity)

        if new_entity_objects:
            from ..neo4j_service import neo4j_service
            await neo4j_service.create_entities(new_entity_objects)

    async def _update_document_version(
        self,
        document_id: str,
        new_chunks: List[Dict[str, Any]],
        new_entities: List[Dict[str, Any]]
    ):
        """更新文档版本信息"""
        try:
            # 计算新的哈希
            chunk_hashes = {}
            for chunk in new_chunks:
                chunk_id = str(chunk.get('id', ''))
                if chunk_id:
                    chunk_hashes[chunk_id] = await self._calculate_chunk_hash(chunk)

            entity_hashes = {}
            for entity in new_entities:
                entity_id = str(entity.get('id', ''))
                if entity_id:
                    entity_hashes[entity_id] = await self._calculate_entity_hash(entity)

            # 更新版本
            new_version = await self._get_next_version(document_id)
            content_hash = await self._calculate_content_hash(new_chunks)

            version_info = {
                'document_id': document_id,
                'version': new_version,
                'content_hash': content_hash,
                'chunk_hashes': chunk_hashes,
                'entity_hashes': entity_hashes,
                'metadata': {
                    'chunk_count': len(new_chunks),
                    'entity_count': len(new_entities),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                },
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            key = f"{self.version_key_prefix}{document_id}"
            await redis_client.setex(key, 86400 * 30, json.dumps(version_info))  # 30天过期

            # 更新数据库中的版本信息
            async with asynccontextmanager(get_db)() as db:
                document = await db.get(Document, document_id)
                if document:
                    document.version = new_version
                    document.content_hash = content_hash
                    document.updated_at = datetime.utcnow()
                    await db.commit()

        except Exception as e:
            logger.error(f"更新文档版本失败 {document_id}: {e}")

    async def _get_next_version(self, document_id: str) -> int:
        """获取下一个版本号"""
        current_version = await self._get_document_version(document_id)
        return (current_version.version + 1) if current_version else 1


# 全局增量更新器实例
incremental_updater = IncrementalUpdater()