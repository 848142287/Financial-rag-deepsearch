"""
模型服务工厂
根据配置自动选择本地模型或在线模型
"""

from app.core.structured_logging import get_structured_logger
import numpy as np

from app.core.config import settings

logger = get_structured_logger(__name__)

class ModelServiceFactory:
    """模型服务工厂类"""

    def __init__(self):
        self._embedding_service = None
        self._reranker_service = None
        self._use_local_models = getattr(settings, 'use_local_models', True)

    def get_embedding_service(self):
        """
        获取嵌入服务实例

        根据配置返回本地BGE服务或在线Qwen服务
        """
        if self._embedding_service is None:
            try:
                from app.services.embeddings.unified_embedding_service import get_embedding_service
                # 统一服务会自动选择最佳提供者（BGE本地或Qwen API）
                self._embedding_service = get_embedding_service()

                if self._use_local_models:
                    logger.info("Using unified embedding service (prefer BGE local)")
                else:
                    logger.info("Using unified embedding service (prefer Qwen API)")

            except Exception as e:
                logger.error(f"Failed to initialize embedding service: {e}")
                raise RuntimeError("Failed to initialize any embedding service")

        return self._embedding_service

    def get_reranker_service(self):
        """
        获取重排序服务实例

        根据配置返回本地BGE服务或在线Qwen服务
        """
        if self._reranker_service is None:
            try:
                from app.services.reranking.unified_reranker_service import get_reranker_service
                # 统一服务会自动选择最佳提供者（BGE本地或Qwen API）
                self._reranker_service = get_reranker_service()

                if self._use_local_models:
                    logger.info("Using unified reranker service (prefer BGE local)")
                else:
                    logger.info("Using unified reranker service (prefer Qwen API)")

            except Exception as e:
                logger.error(f"Failed to initialize reranker service: {e}")
                raise RuntimeError("Failed to initialize any reranker service")

        return self._reranker_service

# 全局单例
_factory = None

def get_model_service_factory() -> ModelServiceFactory:
    """获取模型服务工厂单例"""
    global _factory
    if _factory is None:
        _factory = ModelServiceFactory()
    return _factory

def get_embedding_service():
    """获取嵌入服务的便捷函数"""
    return get_model_service_factory().get_embedding_service()

def get_reranker_service():
    """获取重排序服务的便捷函数"""
    return get_model_service_factory().get_reranker_service()

# 兼容性别名 - 用于向后兼容
async def get_qwen_embedding_service():
    """向后兼容的函数 - 返回当前配置的embedding服务"""
    service = get_embedding_service()
    # 如果是同步服务，包装为异步接口
    if hasattr(service, 'encode'):
        # BGE本地服务是同步的，创建异步包装器
        class AsyncWrapper:
            def __init__(self, sync_service):
                self.sync_service = sync_service

            async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
                """异步包装器"""
                embeddings = self.sync_service.encode(texts)
                return [emb.tolist() for emb in embeddings]

            async def encode(self, texts: List[str]) -> List[np.ndarray]:
                """异步包装器"""
                return self.sync_service.encode(texts)

            async def encode_single(self, text: str) -> np.ndarray:
                """异步包装器"""
                return self.sync_service.encode_single(text)

            async def rerank(self, query: str, documents: List[str], top_k: int = None):
                """异步包装器 - 需要使用reranker服务"""
                reranker = get_reranker_service()
                if hasattr(reranker, 'rerank'):
                    return reranker.rerank(query, documents, top_k)
                else:
                    return [(i, 1.0) for i in range(len(documents))]

        return AsyncWrapper(service)
    else:
        return service

async def get_qwen_rerank_service():
    """向后兼容的函数 - 返回当前配置的reranker服务"""
    service = get_reranker_service()

    # 如果是同步服务，包装为异步接口
    if hasattr(service, 'rerank') and not hasattr(service, 'initialize'):
        # BGE本地服务是同步的，创建异步包装器
        class AsyncRerankWrapper:
            def __init__(self, sync_service):
                self.sync_service = sync_service

            async def rerank(self, query: str, documents: List[str], top_k: int = None):
                """异步包装器"""
                return self.sync_service.rerank(query, documents, top_k)

            async def initialize(self):
                """异步初始化 - 本地服务不需要"""
                pass

            async def close(self):
                """异步关闭 - 本地服务不需要"""
                pass

        return AsyncRerankWrapper(service)
    else:
        return service
