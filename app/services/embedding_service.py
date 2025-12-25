"""
嵌入服务 - 向后兼容别名
指向 qwen_embedding_service 以保持现有功能不受影响
"""

# 向后兼容的导入
from app.services.qwen_embedding_service import QwenEmbeddingService

# 向后兼容的别名
EmbeddingService = QwenEmbeddingService

# 添加占位符 RerankService 以避免导入错误
class RerankService:
    """占位符重排序服务类"""
    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, *args, **kwargs):
        """占位符重排序方法"""
        return []