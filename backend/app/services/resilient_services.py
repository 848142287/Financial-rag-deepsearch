"""
使用弹性工具的服务示例
展示如何为现有服务添加重试、熔断和限流保护
"""

from app.core.structured_logging import get_structured_logger
from typing import List, Dict, Any, Optional

from app.core.resilience import (
    retry,
    circuit_breaker,
    rate_limit,
    resilient_service,
    combine_decorators
)

logger = get_structured_logger(__name__)


class ResilientEmbeddingService:
    """
    带弹性保护的Embedding服务包装器

    使用示例:
        service = ResilientEmbeddingService(embedding_service)
        embeddings = await service.encode_documents(["文档1", "文档2"])
    """

    def __init__(self, base_service):
        """
        Args:
            base_service: 基础embedding服务实例
        """
        self.base_service = base_service

    @resilient_service(
        service_name="embedding_encode",
        max_attempts=3,
        failure_threshold=5,
        max_calls=50
    )
    async def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        编码文档（带弹性保护）

        Args:
            documents: 文档列表
            batch_size: 批处理大小

        Returns:
            List[List[float]]: 向量列表
        """
        return await self.base_service.encode_documents(documents, batch_size)

    @resilient_service(
        service_name="embedding_encode_single",
        max_attempts=3,
        failure_threshold=3,
        max_calls=100
    )
    async def encode_single(self, text: str) -> List[float]:
        """
        编码单个文本（带弹性保护）

        Args:
            text: 文本内容

        Returns:
            List[float]: 向量
        """
        return await self.base_service.encode_single(text)


class ResilientMilvusService:
    """
    带弹性保护的Milvus服务包装器
    """

    def __init__(self, base_service):
        """
        Args:
            base_service: 基础milvus服务实例
        """
        self.base_service = base_service

    @combine_decorators(
        retry(max_attempts=3, retry_on=(ConnectionError, TimeoutError)),
        circuit_breaker("milvus_insert", failure_threshold=5, timeout=60.0),
        rate_limit("milvus_insert_calls", max_calls=20, time_window=1.0)
    )
    async def insert_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        插入向量（带弹性保护）

        Args:
            vectors: 向量列表
            ids: ID列表
            metadata: 元数据列表

        Returns:
            Dict[str, Any]: 插入结果
        """
        return await self.base_service.insert_vectors(vectors, ids, metadata)

    @combine_decorators(
        retry(max_attempts=3, retry_on=(ConnectionError, TimeoutError)),
        circuit_breaker("milvus_search", failure_threshold=5, timeout=60.0),
        rate_limit("milvus_search_calls", max_calls=100, time_window=1.0)
    )
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索向量（带弹性保护）

        Args:
            query_vector: 查询向量
            top_k: 返回结果数
            filters: 过滤条件

        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        return await self.base_service.search_vectors(query_vector, top_k, filters)

    @combine_decorators(
        retry(max_attempts=2, retry_on=(ConnectionError,)),
        circuit_breaker("milvus_delete", failure_threshold=3, timeout=30.0),
    )
    async def delete_vectors(self, ids: List[str]) -> Dict[str, Any]:
        """
        删除向量（带弹性保护）

        Args:
            ids: 向量ID列表

        Returns:
            Dict[str, Any]: 删除结果
        """
        return await self.base_service.delete_vectors(ids)


class ResilientNeo4jService:
    """
    带弹性保护的Neo4j服务包装器
    """

    def __init__(self, base_service):
        """
        Args:
            base_service: 基础neo4j服务实例
        """
        self.base_service = base_service

    @resilient_service(
        service_name="neo4j_query",
        max_attempts=3,
        failure_threshold=5,
        max_calls=50
    )
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行Cypher查询（带弹性保护）

        Args:
            query: Cypher查询语句
            params: 查询参数

        Returns:
            List[Dict[str, Any]]: 查询结果
        """
        return await self.base_service.execute_query(query, params)

    @resilient_service(
        service_name="neo4j_create_entity",
        max_attempts=3,
        failure_threshold=3,
        max_calls=30
    )
    async def create_entity(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建实体（带弹性保护）

        Args:
            label: 实体标签
            properties: 实体属性

        Returns:
            Dict[str, Any]: 创建的实体
        """
        return await self.base_service.create_entity(label, properties)

    @resilient_service(
        service_name="neo4j_create_relation",
        max_attempts=3,
        failure_threshold=3,
        max_calls=30
    )
    async def create_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建关系（带弹性保护）

        Args:
            from_entity: 起始实体ID
            to_entity: 目标实体ID
            relation_type: 关系类型
            properties: 关系属性

        Returns:
            Dict[str, Any]: 创建的关系
        """
        return await self.base_service.create_relation(
            from_entity, to_entity, relation_type, properties
        )


class ResilientLLMService:
    """
    带弹性保护的LLM服务包装器
    """

    def __init__(self, base_service):
        """
        Args:
            base_service: 基础LLM服务实例
        """
        self.base_service = base_service

    @combine_decorators(
        retry(max_attempts=3, retry_on=(ConnectionError, TimeoutError)),
        circuit_breaker("llm_chat", failure_threshold=5, timeout=60.0, fallback_value=None),
        rate_limit("llm_chat_calls", max_calls=20, time_window=60.0)
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        聊天补全（带弹性保护）

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式返回

        Returns:
            Dict[str, Any]: 响应结果
        """
        return await self.base_service.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )


# 便捷工厂函数
def create_resilient_service(service_type: str, base_service):
    """
    创建弹性服务

    Args:
        service_type: 服务类型 (embedding, milvus, neo4j, llm)
        base_service: 基础服务实例

    Returns:
        带弹性保护的服务包装器

    Raises:
        ValueError: 不支持的服务类型
    """
    service_map = {
        "embedding": ResilientEmbeddingService,
        "milvus": ResilientMilvusService,
        "neo4j": ResilientNeo4jService,
        "llm": ResilientLLMService,
    }

    service_class = service_map.get(service_type)
    if not service_class:
        raise ValueError(f"不支持的服务类型: {service_type}")

    return service_class(base_service)


# 使用示例
if __name__ == "__main__":
    # 示例：为现有的embedding服务添加弹性保护
    # from app.services.qwen_embedding_service import qwen_embedding_service
    #
    # resilient_embedding = create_resilient_service(
    #     "embedding",
    #     qwen_embedding_service
    # )
    #
    # # 正常使用，自动获得重试、熔断和限流保护
    # embeddings = await resilient_embedding.encode_documents(["文档1", "文档2"])
    pass
