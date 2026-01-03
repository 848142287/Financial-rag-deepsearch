"""
文档双存储服务
实现Redis和MySQL双存储，确保数据持久化
"""

from typing import Dict, Any, Optional
from app.core.structured_logging import get_structured_logger
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_structured_logger(__name__)

class DocumentDualStorageService:
    """
    文档双存储服务

    功能：
    1. Redis热数据存储（快速访问）
    2. MySQL持久化存储（长期保存）
    3. 自动同步和一致性保证
    4. TTL管理
    """

    def __init__(self):
        """初始化服务"""
        self.redis_client = None
        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        if self._initialized:
            return

        # 初始化Redis连接
        from app.core.redis_client import get_redis_client
        self.redis_client = await get_redis_client()

        self._initialized = True
        logger.info("✅ 文档双存储服务初始化完成")

    async def store_document(
        self,
        document_id: int,
        parsed_content: Dict[str, Any],
        db: AsyncSession,
        ttl: int = 2592000  # 30天
    ) -> Dict[str, Any]:
        """
        双存储文档信息

        Args:
            document_id: 文档ID
            parsed_content: 解析后的内容
            db: 数据库会话
            ttl: Redis过期时间（秒）

        Returns:
            存储结果
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 1. 存储到Redis（热数据）
            redis_key = f"document:{document_id}"
            redis_result = await self._store_to_redis(
                redis_key,
                parsed_content,
                ttl
            )

            # 2. 存储到MySQL（持久化）
            mysql_result = await self._store_to_mysql(
                document_id,
                parsed_content,
                db
            )

            # 3. 更新同步状态
            await self._update_sync_status(document_id, redis_key, db)

            logger.info(f"✅ 文档 {document_id} 双存储完成")

            return {
                'document_id': document_id,
                'redis_key': redis_key,
                'redis_stored': redis_result,
                'mysql_stored': mysql_result,
                'sync_status': 'completed',
                'ttl': ttl
            }

        except Exception as e:
            logger.error(f"❌ 文档 {document_id} 双存储失败: {e}")
            raise

    async def _store_to_redis(
        self,
        key: str,
        content: Dict[str, Any],
        ttl: int
    ) -> bool:
        """存储到Redis"""
        try:
            import json

            # 序列化内容
            content_json = json.dumps(content, ensure_ascii=False)

            # 存储到Redis
            await self.redis_client.setex(
                key,
                ttl,
                content_json
            )

            logger.debug(f"✅ 存储到Redis成功: {key}, TTL={ttl}s")
            return True

        except Exception as e:
            logger.error(f"❌ Redis存储失败: {e}")
            return False

    async def _store_to_mysql(
        self,
        document_id: int,
        content: Dict[str, Any],
        db: AsyncSession
    ) -> bool:
        """存储到MySQL"""
        try:
            from app.models.document import Document
            from sqlalchemy import select
            import json

            # 查询文档
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                logger.error(f"❌ 文档 {document_id} 不存在")
                return False

            # 更新parsed_content字段
            # 将内容转换为JSON字符串存储
            document.parsed_content = json.dumps(content, ensure_ascii=False)

            # 更新存储路径
            document.mysql_storage_path = f"documents/{document_id}"

            # 更新时间戳
            document.updated_at = datetime.now()

            # 提交事务
            await db.commit()

            logger.debug(f"✅ 存储到MySQL成功: document_id={document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ MySQL存储失败: {e}")
            await db.rollback()
            return False

    async def _update_sync_status(
        self,
        document_id: int,
        redis_key: str,
        db: AsyncSession
    ):
        """更新同步状态"""
        try:
            from app.models.document import Document
            from sqlalchemy import select, update

            # 更新同步状态
            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    redis_key=redis_key,
                    storage_sync_status='completed'
                )
            )
            await db.commit()

        except Exception as e:
            logger.error(f"❌ 更新同步状态失败: {e}")

    async def load_document(
        self,
        document_id: int,
        db: AsyncSession,
        prefer_redis: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        加载文档信息（优先从Redis）

        Args:
            document_id: 文档ID
            db: 数据库会话
            prefer_redis: 是否优先从Redis加载

        Returns:
            文档内容
        """
        if not self._initialized:
            await self.initialize()

        # 1. 尝试从Redis加载
        if prefer_redis:
            redis_data = await self._load_from_redis(document_id)
            if redis_data:
                logger.debug(f"✅ 从Redis加载文档 {document_id}")
                return redis_data
            else:
                logger.debug(f"⚠️ Redis中未找到文档 {document_id}，尝试MySQL")

        # 2. 从MySQL加载
        mysql_data = await self._load_from_mysql(document_id, db)
        if mysql_data:
            logger.debug(f"✅ 从MySQL加载文档 {document_id}")

            # 回写到Redis
            await self._store_to_redis(
                f"document:{document_id}",
                mysql_data,
                2592000  # 30天
            )

            return mysql_data

        logger.error(f"❌ 未找到文档 {document_id}")
        return None

    async def _load_from_redis(self, document_id: int) -> Optional[Dict[str, Any]]:
        """从Redis加载"""
        try:
            import json

            redis_key = f"document:{document_id}"
            content_json = await self.redis_client.get(redis_key)

            if content_json:
                return json.loads(content_json)

            return None

        except Exception as e:
            logger.error(f"❌ 从Redis加载失败: {e}")
            return None

    async def _load_from_mysql(
        self,
        document_id: int,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """从MySQL加载"""
        try:
            from app.models.document import Document
            from sqlalchemy import select
            import json

            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if not document or not document.parsed_content:
                return None

            # 解析JSON
            return json.loads(document.parsed_content)

        except Exception as e:
            logger.error(f"❌ 从MySQL加载失败: {e}")
            return None

    async def delete_document(
        self,
        document_id: int,
        db: AsyncSession
    ) -> bool:
        """
        删除文档（双存储都删除）

        Args:
            document_id: 文档ID
            db: 数据库会话

        Returns:
            是否成功
        """
        try:
            # 1. 删除Redis数据
            redis_key = f"document:{document_id}"
            await self.redis_client.delete(redis_key)
            logger.info(f"✅ 已从Redis删除文档 {document_id}")

            # 2. 标记MySQL为已删除（软删除）
            from app.models.document import Document
            from sqlalchemy import update

            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    storage_sync_status='deleted',
                    redis_key=None,
                    updated_at=datetime.now()
                )
            )
            await db.commit()
            logger.info(f"✅ 已从MySQL标记删除文档 {document_id}")

            return True

        except Exception as e:
            logger.error(f"❌ 删除文档失败: {e}")
            await db.rollback()
            return False

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查Redis连接
            redis_ok = await self.redis_client.ping() if self.redis_client else False

            return {
                'status': 'healthy' if redis_ok else 'degraded',
                'redis_connected': redis_ok,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# 全局实例
_storage_service_instance: Optional[DocumentDualStorageService] = None

def get_document_storage_service() -> DocumentDualStorageService:
    """获取文档存储服务实例"""
    global _storage_service_instance

    if _storage_service_instance is None:
        _storage_service_instance = DocumentDualStorageService()
        logger.info("✅ 初始化文档双存储服务")

    return _storage_service_instance

__all__ = [
    'DocumentDualStorageService',
    'get_document_storage_service'
]
