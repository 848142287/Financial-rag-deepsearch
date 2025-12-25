"""
向量库同步服务
负责将文档分块同步到向量数据库
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models.document import DocumentChunk
from app.models.synchronization import DocumentSync, VectorSync, SyncStatus, SyncLog
from app.services.sync_state_machine import SyncStateMachine

logger = logging.getLogger(__name__)


class VectorDBClient:
    """向量数据库客户端接口"""

    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        self.collection_name = config.get("collection_name", "financial_documents")

    async def connect(self):
        """连接向量数据库"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            connections.connect(
                alias="default",
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 19530)
            )
            self.client = connections.get_connection("default")
            logger.info("Connected to Milvus vector database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {str(e)}")
            return False

    async def create_collection(self):
        """创建集合"""
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} already exists")
                return True

            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
            ]

            # 创建集合
            schema = CollectionSchema(fields, f"Financial documents collection")
            self.collection = Collection(self.collection_name, schema)
            logger.info(f"Created collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False

    async def insert_vectors(self, vectors: List[Dict]) -> List[str]:
        """插入向量"""
        try:
            if not self.collection:
                await self.create_collection()

            # 准备数据
            ids = [v["id"] for v in vectors]
            document_ids = [v["document_id"] for v in vectors]
            chunk_ids = [v["chunk_id"] for v in vectors]
            contents = [v["content"] for v in vectors]
            metadatas = [v.get("metadata", {}) for v in vectors]
            embeddings = [v["embedding"] for v in vectors]

            # 插入数据
            data = [
                ids,
                document_ids,
                chunk_ids,
                contents,
                metadatas,
                embeddings
            ]

            from pymilvus import Collection
            collection = Collection(self.collection_name)
            result = collection.insert(data)
            collection.flush()

            logger.info(f"Inserted {len(vectors)} vectors to {self.collection_name}")
            return ids
        except Exception as e:
            logger.error(f"Failed to insert vectors: {str(e)}")
            return []

    async def search_vectors(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """搜索向量"""
        try:
            if not self.collection:
                await self.create_collection()

            from pymilvus import Collection
            collection = Collection(self.collection_name)
            collection.load()

            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            results = collection.search(
                [query_vector],
                "embedding",
                search_params,
                limit=limit,
                output_fields=["document_id", "chunk_id", "content", "metadata"]
            )

            # 格式化结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.id,
                        "score": hit.score,
                        "document_id": hit.entity.get("document_id"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata", {})
                    })

            return search_results
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            return []

    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """删除向量"""
        try:
            if not self.collection:
                return False

            from pymilvus import Collection
            collection = Collection(self.collection_name)

            quoted_ids = [f'"{vid}"' for vid in vector_ids]
            expr = f"id in {quoted_ids}"
            collection.delete(expr)
            collection.flush()

            logger.info(f"Deleted {len(vector_ids)} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False


class VectorSyncService:
    """向量同步服务"""

    def __init__(self, db: Session, config: Dict = None):
        self.db = db
        self.config = config or {}
        self.state_machine = SyncStateMachine()
        self.vector_client = VectorDBClient(self.config)
        self.collection = None

    async def initialize(self):
        """初始化服务"""
        try:
            await self.vector_client.connect()
            await self.vector_client.create_collection()
            logger.info("Vector sync service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vector sync service: {str(e)}")
            return False

    async def sync_document(self, document_id: str, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """同步文档到向量数据库"""
        try:
            # 创建同步记录
            sync_record = DocumentSync(
                document_id=document_id,
                status=SyncStatus.PROCESSING,
                started_at=datetime.now()
            )
            self.db.add(sync_record)
            self.db.commit()

            # 准备向量数据
            vectors = []
            for chunk in chunks:
                if chunk.embedding:
                    vector_data = {
                        "id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "chunk_id": str(chunk.id),
                        "content": chunk.content[:8192],  # 限制长度
                        "metadata": {
                            "chunk_index": chunk.chunk_index,
                            "page_number": chunk.page_number,
                            "chunk_type": chunk.chunk_type
                        },
                        "embedding": chunk.embedding
                    }
                    vectors.append(vector_data)

            if not vectors:
                logger.warning(f"No vectors to sync for document {document_id}")
                return {"status": "no_vectors", "document_id": document_id}

            # 插入向量
            vector_ids = await self.vector_client.insert_vectors(vectors)

            # 更新同步记录
            sync_record.status = SyncStatus.COMPLETED
            sync_record.completed_at = datetime.now()
            sync_record.vector_count = len(vector_ids)
            self.db.commit()

            # 创建向量同步记录
            for i, vector_id in enumerate(vector_ids):
                vector_sync = VectorSync(
                    sync_id=sync_record.id,
                    chunk_id=chunks[i].id,
                    vector_id=vector_id,
                    status=SyncStatus.COMPLETED
                )
                self.db.add(vector_sync)

            self.db.commit()

            logger.info(f"Successfully synced {len(vector_ids)} vectors for document {document_id}")
            return {
                "status": "success",
                "document_id": document_id,
                "vector_count": len(vector_ids),
                "sync_id": sync_record.id
            }

        except Exception as e:
            logger.error(f"Failed to sync document {document_id}: {str(e)}")
            if sync_record:
                sync_record.status = SyncStatus.FAILED
                sync_record.error_message = str(e)
                sync_record.completed_at = datetime.now()
                self.db.commit()
            return {"status": "error", "document_id": document_id, "error": str(e)}

    async def search_similar_chunks(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """搜索相似文档块"""
        try:
            results = await self.vector_client.search_vectors(query_vector, limit)

            # 获取详细的chunk信息
            chunk_ids = [r["chunk_id"] for r in results]
            chunks = self.db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()

            chunk_map = {str(chunk.id): chunk for chunk in chunks}

            detailed_results = []
            for result in results:
                chunk = chunk_map.get(result["chunk_id"])
                if chunk:
                    detailed_results.append({
                        "chunk": chunk,
                        "score": result["score"],
                        "vector_id": result["id"]
                    })

            return detailed_results
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {str(e)}")
            return []

    async def delete_document_vectors(self, document_id: str) -> bool:
        """删除文档的所有向量"""
        try:
            # 查找向量同步记录
            vector_syncs = self.db.query(VectorSync).join(DocumentSync).filter(
                DocumentSync.document_id == document_id
            ).all()

            if not vector_syncs:
                return True

            # 删除向量
            vector_ids = [vs.vector_id for vs in vector_syncs if vs.vector_id]
            if vector_ids:
                await self.vector_client.delete_vectors(vector_ids)

            # 删除同步记录
            self.db.query(VectorSync).filter(VectorSync.id.in_([vs.id for vs in vector_syncs])).delete()
            self.db.commit()

            logger.info(f"Deleted {len(vector_ids)} vectors for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors for document {document_id}: {str(e)}")
            return False

    async def get_sync_status(self, document_id: str) -> Dict[str, Any]:
        """获取文档同步状态"""
        try:
            sync_record = self.db.query(DocumentSync).filter(
                DocumentSync.document_id == document_id
            ).order_by(DocumentSync.created_at.desc()).first()

            if not sync_record:
                return {"status": "not_found"}

            vector_syncs = self.db.query(VectorSync).filter(
                VectorSync.sync_id == sync_record.id
            ).all()

            return {
                "status": sync_record.status.value,
                "created_at": sync_record.created_at.isoformat(),
                "started_at": sync_record.started_at.isoformat() if sync_record.started_at else None,
                "completed_at": sync_record.completed_at.isoformat() if sync_record.completed_at else None,
                "vector_count": sync_record.vector_count,
                "error_message": sync_record.error_message,
                "synced_chunks": len([vs for vs in vector_syncs if vs.status == SyncStatus.COMPLETED]),
                "failed_chunks": len([vs for vs in vector_syncs if vs.status == SyncStatus.FAILED])
            }
        except Exception as e:
            logger.error(f"Failed to get sync status for document {document_id}: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def cleanup_failed_syncs(self, older_than_hours: int = 24) -> int:
        """清理失败的同步记录"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

            failed_syncs = self.db.query(DocumentSync).filter(
                DocumentSync.status == SyncStatus.FAILED,
                DocumentSync.created_at < cutoff_time
            ).all()

            count = 0
            for sync in failed_syncs:
                # 删除相关的向量同步记录
                self.db.query(VectorSync).filter(VectorSync.sync_id == sync.id).delete()

                # 删除同步记录
                self.db.delete(sync)
                count += 1

            self.db.commit()
            logger.info(f"Cleaned up {count} failed sync records")
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup failed syncs: {str(e)}")
            return 0


# 创建全局实例
vector_sync_service = None

def get_vector_sync_service(db: Session, config: Dict = None) -> VectorSyncService:
    """获取向量同步服务实例"""
    global vector_sync_service
    if vector_sync_service is None:
        vector_sync_service = VectorSyncService(db, config)
    return vector_sync_service