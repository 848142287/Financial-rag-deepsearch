"""
文档删除服务
提供文档的完整删除功能，包括从MySQL、Milvus、Neo4j、MinIO和本地文件等存储中删除
"""

from app.core.structured_logging import get_structured_logger
from typing import Dict, Any
from sqlalchemy.orm import Session

logger = get_structured_logger(__name__)

class DocumentDeletionService:
    """文档删除服务"""

    async def delete_document_complete(
        self,
        document_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """
        完整删除文档（从所有存储中删除）

        删除顺序：
        1. MySQL (数据库记录)
        2. Milvus (向量数据)
        3. Neo4j (知识图谱)
        4. MinIO (对象存储)
        5. 本地文件 (解析后的文件)

        Args:
            document_id: 文档ID
            db: 数据库会话

        Returns:
            删除结果
        """
        try:
            from app.services.milvus_service import MilvusService
            from app.services.neo4j_service import Neo4jService
            from app.services.minio_service import MinIOService
            from app.services.storage.enhanced_local_storage import EnhancedLocalStorage
            from sqlalchemy import text

            logger.info(f"开始完整删除文档: {document_id}")

            # 0. 获取文档信息（用于后续删除MinIO和本地文件）
            document_info = db.execute(
                text("SELECT id, file_path, storage_path, filename FROM documents WHERE id = :doc_id"),
                {"doc_id": document_id}
            ).fetchone()

            if not document_info:
                logger.warning(f"文档 {document_id} 不存在")
                return {
                    "success": False,
                    "error": "文档不存在",
                    "document_id": document_id
                }

            file_path = document_info[1]  # MinIO路径
            storage_path = document_info[2]  # 本地存储路径
            filename = document_info[3]  # 原始文件名

            # 1. 获取chunk数量
            chunk_count_result = db.execute(
                text("SELECT COUNT(*) FROM document_chunks WHERE document_id = :doc_id"),
                {"doc_id": document_id}
            ).scalar()
            chunk_count = chunk_count_result or 0

            deletion_details = {
                "mysql_records": 0,
                "milvus_vectors": 0,
                "neo4j_nodes": 0,
                "minio_files": 0,
                "local_files": 0
            }

            # 2. 删除MySQL数据库记录
            try:
                # 删除向量存储记录
                result = db.execute(
                    text("DELETE FROM vector_storage WHERE document_id = :doc_id"),
                    {"doc_id": document_id}
                )
                deletion_details["mysql_records"] += result.rowcount

                # 删除文档块记录
                result = db.execute(
                    text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
                    {"doc_id": document_id}
                )
                deletion_details["mysql_records"] += result.rowcount

                # 删除文档记录
                result = db.execute(
                    text("DELETE FROM documents WHERE id = :doc_id"),
                    {"doc_id": document_id}
                )
                deletion_details["mysql_records"] += result.rowcount

                db.commit()
                logger.info(f"✅ 已从MySQL删除文档 {document_id} 的记录")
            except Exception as mysql_error:
                logger.error(f"MySQL删除失败: {mysql_error}")
                db.rollback()
                raise

            # 3. 从Milvus删除向量
            try:
                milvus_service = MilvusService()
                await milvus_service.ensure_connected()
                deleted_count = await milvus_service.delete_document(document_id)
                deletion_details["milvus_vectors"] = deleted_count or chunk_count
                logger.info(f"✅ 已从Milvus删除文档 {document_id} 的向量")
            except Exception as milvus_error:
                logger.warning(f"⚠️  Milvus删除失败（继续执行）: {milvus_error}")
                deletion_details["milvus_vectors"] = 0

            # 4. 从Neo4j删除节点
            try:
                neo4j_service = Neo4jService()
                await neo4j_service.connect()

                # 统计要删除的节点数量
                count_cypher = f"""
                MATCH (n)
                WHERE n.document_id = {document_id}
                RETURN count(n) as count
                """
                result = await neo4j_service.execute_query(count_cypher)
                node_count = result[0]["count"] if result else 0

                # 删除文档相关的所有节点
                delete_cypher = f"""
                MATCH (n)
                WHERE n.document_id = {document_id}
                DETACH DELETE n
                """
                await neo4j_service.execute_query(delete_cypher)
                deletion_details["neo4j_nodes"] = node_count
                logger.info(f"✅ 已从Neo4j删除文档 {document_id} 的节点")
            except Exception as neo4j_error:
                logger.warning(f"⚠️  Neo4j删除失败（继续执行）: {neo4j_error}")
                deletion_details["neo4j_nodes"] = 0

            # 5. 从MinIO删除文件
            try:
                minio_service = MinIOService()
                # 删除原始上传文件
                if file_path:
                    success = await minio_service.delete_file(file_path)
                    if success:
                        deletion_details["minio_files"] += 1
                        logger.info(f"✅ 已从MinIO删除文件: {file_path}")

                # 删除处理后的文件（如果有）
                processed_bucket = "processed-files"
                processed_prefix = f"document_{document_id}"
                try:
                    # 列出并删除所有相关文件
                    files = await minio_service.list_files(prefix=processed_prefix, bucket=processed_bucket)
                    for file_info in files:
                        await minio_service.delete_file(file_info["name"], bucket=processed_bucket)
                        deletion_details["minio_files"] += 1
                    logger.info(f"✅ 已从MinIO删除处理文件: {len(files)} 个")
                except Exception as e:
                    logger.warning(f"⚠️  MinIO处理文件删除失败: {e}")

            except Exception as minio_error:
                logger.warning(f"⚠️  MinIO删除失败（继续执行）: {minio_error}")
                deletion_details["minio_files"] = 0

            # 6. 删除本地解析文件
            try:
                local_storage = EnhancedLocalStorage()
                success = await local_storage.delete_document(str(document_id))
                if success:
                    deletion_details["local_files"] = 1
                    logger.info(f"✅ 已删除本地解析文件: {document_id}")
                else:
                    deletion_details["local_files"] = 0
            except Exception as local_error:
                logger.warning(f"⚠️  本地文件删除失败（继续执行）: {local_error}")
                deletion_details["local_files"] = 0

            # 7. 清理缓存
            try:
                from app.core.redis_client import redis_client
                cache_keys = [
                    f"document:{document_id}",
                    f"document_chunks:{document_id}",
                    f"doc_metadata:{document_id}"
                ]
                for key in cache_keys:
                    await redis_client.delete(key)
                logger.info(f"✅ 已清理文档 {document_id} 的缓存")
            except Exception as cache_error:
                logger.warning(f"⚠️  缓存清理失败（继续执行）: {cache_error}")

            total_deleted = sum(deletion_details.values())
            logger.info(f"✅✅✅ 文档 {document_id} 完整删除成功！总计删除 {total_deleted} 项数据")
            logger.info(f"   详情: {deletion_details}")

            return {
                "success": True,
                "document_id": document_id,
                "filename": filename,
                "chunks_deleted": chunk_count,
                "deletion_details": deletion_details,
                "total_deleted_items": total_deleted
            }

        except Exception as e:
            logger.error(f"❌ 文档删除失败: {e}")
            db.rollback()
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }

# 创建全局实例
document_deletion_service = DocumentDeletionService()
