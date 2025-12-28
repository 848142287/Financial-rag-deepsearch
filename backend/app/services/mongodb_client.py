"""
MongoDB客户端服务
用于存储和处理文档结构化数据
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB异步客户端"""

    def __init__(self):
        self.client = None
        self.database = None
        self.mongodb_url = settings.mongodb_url
        self.database_name = settings.mongodb_database

    async def connect(self):
        """连接到MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url)
            self.database = self.client[self.database_name]

            # 测试连接
            await self.client.admin.command('ping')
            logger.info(f"成功连接到MongoDB: {self.database_name}")

        except Exception as e:
            logger.error(f"MongoDB连接失败: {e}")
            raise

    async def disconnect(self):
        """断开MongoDB连接"""
        if self.client:
            self.client.close()
            logger.info("MongoDB连接已关闭")

    async def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """插入文档"""
        try:
            if not self.database:
                await self.connect()

            collection = self.database[collection_name]
            result = await collection.insert_one(document)

            document_id = str(result.inserted_id)
            logger.info(f"MongoDB插入成功: collection={collection_name}, id={document_id}")
            return document_id

        except Exception as e:
            logger.error(f"MongoDB插入失败: {e}")
            raise

    async def find_document(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """查找文档"""
        try:
            if not self.database:
                await self.connect()

            collection = self.database[collection_name]
            document = await collection.find_one(query)

            if document:
                document['_id'] = str(document['_id'])
            return document

        except Exception as e:
            logger.error(f"MongoDB查找失败: {e}")
            return None

    async def update_document(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """更新文档"""
        try:
            if not self.database:
                await self.connect()

            collection = self.database[collection_name]
            result = await collection.update_one(query, {"$set": update})

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"MongoDB更新失败: {e}")
            return False

    async def find_documents(self, collection_name: str, query: Dict[str, Any], limit: int = None) -> List[Dict[str, Any]]:
        """查找多个文档"""
        try:
            if not self.database:
                await self.connect()

            collection = self.database[collection_name]

            cursor = collection.find(query)
            if limit:
                cursor = cursor.limit(limit)

            documents = []
            async for document in cursor:
                document['_id'] = str(document['_id'])
                documents.append(document)

            return documents

        except Exception as e:
            logger.error(f"MongoDB查找失败: {e}")
            return []

    async def count_documents(self, collection_name: str, query: Dict[str, Any] = None) -> int:
        """统计文档数量"""
        try:
            if not self.database:
                await self.connect()

            collection = self.database[collection_name]
            if query:
                count = await collection.count_documents(query)
            else:
                count = await collection.count_documents({})

            return count

        except Exception as e:
            logger.error(f"MongoDB统计失败: {e}")
            return 0

    async def delete_document(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """删除文档"""
        try:
            if not self.database:
                await self.connect()

            collection = self.database[collection_name]
            result = await collection.delete_one(query)

            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"MongoDB删除失败: {e}")
            return False

    async def create_index(self, collection_name: str, index_spec: Dict[str, Any]) -> str:
        """创建索引"""
        try:
            if not self.database:
                await self.connect()

            collection = self.database[collection_name]
            result = await collection.create_index(index_spec)

            return result

        except Exception as e:
            logger.error(f"MongoDB索引创建失败: {e}")
            raise


# 全局MongoDB客户端实例
mongodb_client = MongoDBClient()


async def get_mongodb_client():
    """获取MongoDB客户端实例"""
    if not mongodb_client.client:
        await mongodb_client.connect()
    return mongodb_client