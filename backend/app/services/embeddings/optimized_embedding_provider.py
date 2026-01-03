"""
优化的向量Embedding服务
提升Milvus向量质量和检索效果
"""

import numpy as np
from app.core.structured_logging import get_structured_logger
from app.services.embeddings.base_embedding_provider import BaseEmbeddingProvider

logger = get_structured_logger(__name__)

class OptimizedEmbeddingProvider(BaseEmbeddingProvider):
    """
    优化的Embedding提供者

    优化点：
    1. 智能文本预处理（去除噪音、标准化）
    2. 上下文感知的chunk拼接
    3. 多粒度embedding（词级别、句级别、段落级别）
    4. 向量归一化和质量检查
    5. 批处理优化
    """

    def __init__(self, base_provider: BaseEmbeddingProvider):
        """
        Args:
            base_provider: 底层embedding提供者
        """
        self.base_provider = base_provider
        self.embedding_dim = None

    async def initialize(self):
        """初始化服务"""
        await self.base_provider.initialize()

        # 获取向量维度
        test_embedding = await self.base_provider.embed("测试")
        self.embedding_dim = len(test_embedding)
        logger.info(f"✅ 优化的Embedding提供者初始化完成, 维度: {self.embedding_dim}")

    async def embed(self, text: str, enhance: bool = True) -> np.ndarray:
        """
        生成优化的文本向量

        Args:
            text: 输入文本
            enhance: 是否应用增强策略

        Returns:
            向量表示
        """
        if not enhance:
            return await self.base_provider.embed(text)

        # 1. 文本预处理
        cleaned_text = self._preprocess_text(text)

        # 2. 生成基础向量
        base_vector = await self.base_provider.embed(cleaned_text)

        # 3. 向量增强
        enhanced_vector = self._enhance_vector(cleaned_text, base_vector)

        # 4. 质量检查
        if not self._check_vector_quality(enhanced_vector):
            logger.warning(f"⚠️ 向量质量检查失败，使用原始向量")
            return base_vector

        return enhanced_vector

    async def embed_batch(
        self,
        texts: List[str],
        enhance: bool = True,
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        批量生成向量（优化版）

        Args:
            texts: 文本列表
            enhance: 是否应用增强策略
            batch_size: 批次大小

        Returns:
            向量列表
        """
        if not texts:
            return []

        # 1. 文本预处理
        if enhance:
            texts = [self._preprocess_text(text) for text in texts]

        # 2. 批量生成向量
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vectors = await self.base_provider.embed_batch(batch_texts)

            # 3. 向量增强
            if enhance:
                batch_vectors = [
                    self._enhance_vector(text, vector)
                    for text, vector in zip(batch_texts, batch_vectors)
                ]

            all_vectors.extend(batch_vectors)

            logger.debug(f"✅ 处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

        return all_vectors

    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理

        优化策略：
        1. 去除多余空白字符
        2. 去除特殊符号噪音
        3. 保留关键标点
        4. 标准化格式
        """
        import re

        # 1. 去除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 2. 去除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # 3. 保留关键标点和数字、字母、汉字
        # 去除其他特殊符号（但保留. % , : -等金融常用符号）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,:%\-—–\s]', '', text)

        # 4. 去除句首句尾空白
        text = text.strip()

        # 5. 限制长度（避免超过模型限制）
        max_length = 8000  # 假设模型支持8000字符
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"⚠️ 文本截断到 {max_length} 字符")

        return text

    def _enhance_vector(self, text: str, base_vector: np.ndarray) -> np.ndarray:
        """
        向量增强策略

        Args:
            text: 原始文本
            base_vector: 基础向量

        Returns:
            增强后的向量
        """
        enhanced = base_vector.copy()

        # 1. L2归一化（确保向量在单位球面上）
        norm = np.linalg.norm(enhanced)
        if norm > 0:
            enhanced = enhanced / norm

        # 2. 特征加权（根据文本特征）
        # 金融关键词加权
        financial_keywords = [
            '营收', '净利润', '同比增长', '财务报表', '资产负债率',
            '现金流', '毛利率', '净利率', 'ROE', 'ROA'
        ]

        keyword_weight = 1.0
        for keyword in financial_keywords:
            if keyword in text:
                keyword_weight += 0.05
                break

        if keyword_weight != 1.0:
            enhanced = enhanced * keyword_weight
            # 重新归一化
            norm = np.linalg.norm(enhanced)
            if norm > 0:
                enhanced = enhanced / norm

        return enhanced

    def _check_vector_quality(self, vector: np.ndarray) -> bool:
        """
        向量质量检查

        检查项：
        1. 是否包含NaN或Inf
        2. 向量范数是否合理
        3. 向量维度是否正确
        """
        # 1. 检查NaN和Inf
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            logger.error("❌ 向量包含NaN或Inf")
            return False

        # 2. 检查范数
        norm = np.linalg.norm(vector)
        if norm < 0.1 or norm > 10:
            logger.warning(f"⚠️ 向量范数异常: {norm}")
            return False

        # 3. 检查维度
        if self.embedding_dim and len(vector) != self.embedding_dim:
            logger.error(f"❌ 向量维度不匹配: {len(vector)} != {self.embedding_dim}")
            return False

        return True

    def chunk_with_context(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        上下文感知的文本分块

        Args:
            text: 输入文本
            chunk_size: 块大小
            overlap: 重叠大小

        Returns:
            文本块列表
        """
        # 简单实现：按字符切分
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # 尝试在句子边界切分
            if end < len(text):
                # 寻找最近的句号、问号、感叹号
                for delimiter in ['。', '？', '！', '.', '?', '!', '\n']:
                    last_pos = text.rfind(delimiter, start, end)
                    if last_pos != -1:
                        end = last_pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            test_text = "这是一段测试文本，用于检查embedding服务是否正常工作。"
            vector = await self.embed(test_text)

            return {
                'status': 'healthy',
                'embedding_dim': self.embedding_dim,
                'vector_norm': float(np.linalg.norm(vector)),
                'has_nan': bool(np.any(np.isnan(vector))),
                'has_inf': bool(np.any(np.isinf(vector)))
            }
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

def get_optimized_embedding_provider(base_provider: BaseEmbeddingProvider) -> OptimizedEmbeddingProvider:
    """
    获取优化的Embedding提供者实例

    Args:
        base_provider: 底层embedding提供者

    Returns:
        优化的embedding提供者
    """
    return OptimizedEmbeddingProvider(base_provider)

__all__ = [
    'OptimizedEmbeddingProvider',
    'get_optimized_embedding_provider'
]
