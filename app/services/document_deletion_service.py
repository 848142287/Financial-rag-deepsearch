"""
完整的文档删除服务
负责删除文档相关的所有数据：MySQL、Milvus向量、MongoDB、Redis、Neo4j、本地文件等
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text, delete

from app.core.config import settings
from app.models.document import Document
from app.services.upload_service import UploadService
from app.services.milvus_service import MilvusService
from app.services.neo4j_service import Neo4jService
from app.services.mongodb_client import MongoDBClient
from app.core.redis_client import redis_client
from app.tasks.vector_tasks import delete_document_vectors
from app.tasks.knowledge_tasks import delete_document_knowledge_graph

logger = logging.getLogger(__name__)


class DocumentDeletionService:
    """完整的文档删除服务"""

    def __init__(self):
        self.milvus_service = MilvusService()
        self.neo4j_service = Neo4jService()
        self.mongodb_client = MongoDBClient()
        self.upload_service = UploadService()

    async def delete_document_complete(self, document_id: int, db: Session) -> Dict[str, Any]:
        """
        完整删除文档及其所有相关数据
        """
        try:
            # 1. 获取文档信息
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"文档不存在: {document_id}")

            logger.info(f"开始完整删除文档: {document_id}, 标题: {document.title}")

            deletion_results = {
                "document_id": document_id,
                "document_title": document.title,
                "deleted_items": {},
                "errors": []
            }

            # 2. 删除向量数据库中的向量数据
            await self._delete_vector_data(document_id, deletion_results)

            # 3. 删除Neo4j图数据库中的图谱数据
            await self._delete_graph_data(document_id, deletion_results)

            # 4. 删除MongoDB中的文档数据
            await self._delete_mongodb_data(document_id, deletion_results)

            # 5. 删除Redis中的缓存数据
            await self._delete_redis_data(document_id, deletion_results)

            # 6. 删除MinIO/本地文件存储中的文件
            await self._delete_file_storage(document, deletion_results)

            # 7. 删除MySQL中的相关数据（文档分块、任务记录等）
            await self._delete_mysql_related_data(document_id, db, deletion_results)

            # 8. 最后删除MySQL中的主文档记录
            await self._delete_main_document(document_id, db, deletion_results)

            # 统计删除结果
            total_deleted = sum(
                len(result) if isinstance(result, list) else (1 if result else 0)
                for result in deletion_results["deleted_items"].values()
                if result not in [None, False, []]
            )

            logger.info(f"文档 {document_id} 完整删除完成，共删除 {total_deleted} 项数据")

            return {
                "success": True,
                "message": f"文档 '{document.title}' 及其所有相关数据删除成功",
                "document_id": document_id,
                "deletion_summary": deletion_results,
                "total_deleted_items": total_deleted
            }

        except Exception as e:
            logger.error(f"完整删除文档失败: {document_id}, 错误: {e}")
            return {
                "success": False,
                "message": f"文档删除失败: {str(e)}",
                "document_id": document_id
            }

    async def _delete_vector_data(self, document_id: int, results: Dict[str, Any]):
        """删除向量数据库数据"""
        try:
            logger.info(f"删除文档 {document_id} 的向量数据")

            # 使用Celery任务删除向量数据
            task = delete_document_vectors.delay(str(document_id))
            task_result = task.get(timeout=60)  # 等待任务完成

            results["deleted_items"]["vectors"] = task_result.get("deleted_vectors", 0)
            results["deleted_items"]["vector_deletion_task"] = task_result

            logger.info(f"向量数据删除完成: {task_result}")

        except Exception as e:
            logger.error(f"删除向量数据失败: {e}")
            results["errors"].append(f"向量数据删除失败: {str(e)}")
            results["deleted_items"]["vectors"] = False

    async def _delete_graph_data(self, document_id: int, results: Dict[str, Any]):
        """删除Neo4j图数据库数据"""
        try:
            logger.info(f"删除文档 {document_id} 的图谱数据")

            # 使用Celery任务删除图谱数据
            task = delete_document_knowledge_graph.delay(str(document_id))
            task_result = task.get(timeout=60)

            results["deleted_items"]["graph_nodes"] = task_result.get("deleted_nodes", 0)
            results["deleted_items"]["graph_relationships"] = task_result.get("deleted_relationships", 0)
            results["deleted_items"]["graph_deletion_task"] = task_result

            logger.info(f"图谱数据删除完成: {task_result}")

        except Exception as e:
            logger.error(f"删除图谱数据失败: {e}")
            results["errors"].append(f"图谱数据删除失败: {str(e)}")
            results["deleted_items"]["graph_nodes"] = False
            results["deleted_items"]["graph_relationships"] = False

    async def _delete_mongodb_data(self, document_id: int, results: Dict[str, Any]):
        """删除MongoDB数据"""
        try:
            logger.info(f"删除文档 {document_id} 的MongoDB数据")

            deleted_count = 0
            document_str_id = str(document_id)

            # 删除文档解析结果
            if self.mongodb_client.client:
                db = self.mongodb_client.client[settings.mongodb_db_name]

                # 删除文档chunks集合
                chunks_result = await db.document_chunks.delete_many({"document_id": document_str_id})
                deleted_count += chunks_result.deleted_count
                results["deleted_items"]["mongodb_chunks"] = chunks_result.deleted_count

                # 删除文档entities集合
                entities_result = await db.document_entities.delete_many({"document_id": document_str_id})
                deleted_count += entities_result.deleted_count
                results["deleted_items"]["mongodb_entities"] = entities_result.deleted_count

                # 删除文档relationships集合
                relationships_result = await db.document_relationships.delete_many({"document_id": document_str_id})
                deleted_count += relationships_result.deleted_count
                results["deleted_items"]["mongodb_relationships"] = relationships_result.deleted_count

                # 删除其他可能的集合
                for collection_name in ["document_metadata", "document_analysis", "processing_logs"]:
                    collection_result = await db[collection_name].delete_many({"document_id": document_str_id})
                    if collection_result.deleted_count > 0:
                        results["deleted_items"][f"mongodb_{collection_name}"] = collection_result.deleted_count
                        deleted_count += collection_result.deleted_count

            results["deleted_items"]["mongodb_total"] = deleted_count
            logger.info(f"MongoDB数据删除完成，共删除 {deleted_count} 条记录")

        except Exception as e:
            logger.error(f"删除MongoDB数据失败: {e}")
            results["errors"].append(f"MongoDB数据删除失败: {str(e)}")
            results["deleted_items"]["mongodb_total"] = False

    async def _delete_redis_data(self, document_id: int, results: Dict[str, Any]):
        """删除Redis缓存数据"""
        try:
            logger.info(f"删除文档 {document_id} 的Redis缓存数据")

            deleted_count = 0
            document_str_id = str(document_id)

            # 删除文档相关的缓存键
            patterns = [
                f"document:{document_str_id}:*",
                f"doc:{document_str_id}:*",
                f"vector:{document_str_id}:*",
                f"chunk:{document_str_id}:*",
                f"processing:{document_str_id}:*"
            ]

            for pattern in patterns:
                keys = redis_client.keys(pattern)
                if keys:
                    count = redis_client.delete(*keys)
                    deleted_count += count
                    results[f"redis_keys_{pattern.split(':')[1]}"] = count

            # 删除搜索结果缓存
            search_keys = redis_client.keys(f"search:*:{document_str_id}")
            if search_keys:
                count = redis_client.delete(*search_keys)
                deleted_count += count
                results["redis_search_cache"] = count

            results["deleted_items"]["redis_total"] = deleted_count
            logger.info(f"Redis缓存删除完成，共删除 {deleted_count} 个键")

        except Exception as e:
            logger.error(f"删除Redis数据失败: {e}")
            results["errors"].append(f"Redis数据删除失败: {str(e)}")
            results["deleted_items"]["redis_total"] = False

    async def _delete_file_storage(self, document: Document, results: Dict[str, Any]):
        """删除文件存储数据"""
        try:
            logger.info(f"删除文档 {document.id} 的文件存储")

            deleted_files = []

            # 删除MinIO中的文件
            if document.file_path:
                try:
                    await self.upload_service.initialize()

                    # 解析文件路径
                    if '/' in document.file_path:
                        bucket_name, object_name = document.file_path.split('/', 1)
                        self.upload_service.minio_client.remove_object(bucket_name, object_name)
                        deleted_files.append(f"minio:{document.file_path}")
                        logger.info(f"MinIO文件删除成功: {document.file_path}")
                except Exception as e:
                    logger.warning(f"MinIO文件删除失败: {e}")
                    results["errors"].append(f"MinIO文件删除失败: {str(e)}")

            # 删除本地处理文件（如果有的话）
            local_paths = [
                f"/tmp/uploads/{document.filename}",
                f"/tmp/processing/{document.id}_{document.filename}",
                f"/tmp/chunks/{document.id}",
                f"/tmp/cache/{document.id}"
            ]

            for local_path in local_paths:
                if os.path.exists(local_path):
                    try:
                        if os.path.isfile(local_path):
                            os.remove(local_path)
                        elif os.path.isdir(local_path):
                            import shutil
                            shutil.rmtree(local_path)
                        deleted_files.append(f"local:{local_path}")
                        logger.info(f"本地文件删除成功: {local_path}")
                    except Exception as e:
                        logger.warning(f"本地文件删除失败: {local_path}, 错误: {e}")

            results["deleted_items"]["files"] = deleted_files
            logger.info(f"文件存储删除完成，共删除 {len(deleted_files)} 个文件")

        except Exception as e:
            logger.error(f"删除文件存储失败: {e}")
            results["errors"].append(f"文件存储删除失败: {str(e)}")
            results["deleted_items"]["files"] = False

    async def _delete_mysql_related_data(self, document_id: int, db: Session, results: Dict[str, Any]):
        """删除MySQL中的相关数据"""
        try:
            logger.info(f"删除文档 {document_id} 的MySQL相关数据")

            deleted_counts = {}
            document_str_id = str(document_id)

            # 删除文档分块
            try:
                chunk_result = db.execute(
                    text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
                    {"doc_id": document_str_id}
                )
                deleted_counts["chunks"] = chunk_result.rowcount
            except Exception as e:
                logger.warning(f"删除文档分块失败: {e}")

            # 删除向量存储记录
            try:
                vector_result = db.execute(
                    text("DELETE FROM vector_storage WHERE document_id = :doc_id"),
                    {"doc_id": document_str_id}
                )
                deleted_counts["vector_storage"] = vector_result.rowcount
            except Exception as e:
                logger.warning(f"删除向量存储记录失败: {e}")

            # 删除实体提取记录
            try:
                entity_result = db.execute(
                    text("DELETE FROM document_entities WHERE document_id = :doc_id"),
                    {"doc_id": document_str_id}
                )
                deleted_counts["entities"] = entity_result.rowcount
            except Exception as e:
                logger.warning(f"删除实体记录失败: {e}")

            # 删除处理日志
            try:
                log_result = db.execute(
                    text("DELETE FROM processing_logs WHERE document_id = :doc_id"),
                    {"doc_id": document_str_id}
                )
                deleted_counts["processing_logs"] = log_result.rowcount
            except Exception as e:
                logger.warning(f"删除处理日志失败: {e}")

            # 删除任务记录
            try:
                task_result = db.execute(
                    text("DELETE FROM task_results WHERE document_id = :doc_id"),
                    {"doc_id": document_str_id}
                )
                deleted_counts["task_results"] = task_result.rowcount
            except Exception as e:
                logger.warning(f"删除任务记录失败: {e}")

            db.commit()
            results["deleted_items"]["mysql_related"] = deleted_counts
            logger.info(f"MySQL相关数据删除完成: {deleted_counts}")

        except Exception as e:
            logger.error(f"删除MySQL相关数据失败: {e}")
            results["errors"].append(f"MySQL相关数据删除失败: {str(e)}")
            results["deleted_items"]["mysql_related"] = False

    async def _delete_main_document(self, document_id: int, db: Session, results: Dict[str, Any]):
        """删除主文档记录"""
        try:
            logger.info(f"删除主文档记录: {document_id}")

            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                db.delete(document)
                db.commit()
                results["deleted_items"]["main_document"] = True
                logger.info(f"主文档记录删除成功: {document_id}")
            else:
                results["deleted_items"]["main_document"] = False

        except Exception as e:
            logger.error(f"删除主文档记录失败: {e}")
            results["errors"].append(f"主文档记录删除失败: {str(e)}")
            results["deleted_items"]["main_document"] = False
            db.rollback()


# 全局实例
document_deletion_service = DocumentDeletionService()