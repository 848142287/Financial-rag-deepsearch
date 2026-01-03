"""
增强Milvus服务
提供向量存储和检索功能
"""

from typing import List, Dict, Any, Optional
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class EnhancedMilvusService:
    """增强Milvus服务"""

    def __init__(self):
        """初始化Milvus服务"""
        self.host = "milvus"
        self.port = 19530
        self.connected = False

    def connect(self):
        """连接Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.connected = True
            logger.info(f"成功连接Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            self.connected = False

    def disconnect(self):
        """断开连接"""
        try:
            if self.connected:
                connections.disconnect("default")
                self.connected = False
                logger.info("已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开连接失败: {e}")

    def insert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        插入文档

        Args:
            collection_name: 集合名称
            documents: 文档列表

        Returns:
            插入结果
        """
        try:
            if not self.connected:
                self.connect()

            collection = Collection(collection_name)
            result = collection.insert(documents)
            collection.flush()

            return {
                "success": True,
                "inserted_count": len(result.insert_ids),
                "ids": result.insert_ids
            }
        except Exception as e:
            logger.error(f"插入文档失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        top_k: int = 10,
        expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        Args:
            collection_name: 集合名称
            vectors: 查询向量列表
            top_k: 返回结果数量
            expr: 过滤表达式

        Returns:
            检索结果
        """
        try:
            if not self.connected:
                self.connect()

            collection = Collection(collection_name)
            collection.load()

            results = collection.search(
                data=vectors,
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=top_k,
                expr=expr,
                output_fields=["*"]
            )

            return results
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        description: str = ""
    ) -> bool:
        """
        创建集合

        Args:
            collection_name: 集合名称
            dimension: 向量维度
            description: 描述

        Returns:
            是否成功
        """
        try:
            if utility.has_collection(collection_name):
                logger.warning(f"集合 {collection_name} 已存在")
                return True

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]

            schema = CollectionSchema(
                fields=fields,
                description=description
            )

            collection = Collection(
                name=collection_name,
                schema=schema
            )

            logger.info(f"成功创建集合: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        删除集合

        Args:
            collection_name: 集合名称

        Returns:
            是否成功
        """
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"成功删除集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息

        Args:
            collection_name: 集合名称

        Returns:
            统计信息
        """
        try:
            if not utility.has_collection(collection_name):
                return {"exists": False}

            collection = Collection(collection_name)
            collection.load()

            stats = collection.describe()
            num_entities = collection.num_entities

            return {
                "exists": True,
                "stats": stats,
                "num_entities": num_entities
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {"exists": False, "error": str(e)}


# 全局实例
_enhanced_milvus_instance: Optional[EnhancedMilvusService] = None


def get_enhanced_milvus_service() -> EnhancedMilvusService:
    """
    获取Milvus服务实例

    Returns:
        Milvus服务实例
    """
    global _enhanced_milvus_instance

    if _enhanced_milvus_instance is None:
        _enhanced_milvus_instance = EnhancedMilvusService()
        logger.info("初始化增强Milvus服务")

    return _enhanced_milvus_instance


__all__ = [
    "EnhancedMilvusService",
    "get_enhanced_milvus_service"
]
