"""
向量存储基类
定义向量数据库的通用接口
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorConfig:
    """向量存储配置"""
    host: str = "localhost"
    port: int = 19530
    db_name: str = "default"
    user: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 10
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"


@dataclass
class VectorData:
    """向量数据"""
    id: str
    vector: Union[List[float], np.ndarray]
    content: str
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    content: str
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


class BaseVectorStore(ABC):
    """向量存储基类"""

    def __init__(self, config: VectorConfig):
        self.config = config
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """连接到向量数据库"""
        pass

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str, dimension: int, collection_type: str = "text") -> bool:
        """创建集合"""
        pass

    @abstractmethod
    async def insert_vectors(self, collection_name: str, vectors: List[VectorData]) -> List[str]:
        """插入向量"""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        collection_name: str,
        query_vectors: Union[List[float], List[List[float]]],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[List[SearchResult]]:
        """搜索向量"""
        pass

    @abstractmethod
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """删除向量"""
        pass

    @abstractmethod
    async def update_vector(self, collection_name: str, vector_data: VectorData) -> bool:
        """更新向量"""
        pass

    @abstractmethod
    async def create_index(self, collection_name: str, index_type: str = "IVF_FLAT") -> bool:
        """创建索引"""
        pass

    @abstractmethod
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self.is_connected:
                await self.connect()

            # 测试基本操作
            test_result = await self._test_connection()

            return {
                "status": "healthy" if test_result else "unhealthy",
                "connected": self.is_connected,
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "db_name": self.config.db_name
                },
                "test_result": test_result
            }
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "error",
                "connected": False,
                "error": str(e)
            }

    async def _test_connection(self) -> bool:
        """测试连接"""
        # 默认实现，子类可以重写
        return self.is_connected