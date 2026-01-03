"""
L1层: 基于嵌入相似度的快速压缩器

使用句子编码器计算查询与文档的相似度，快速过滤不相关文档
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base_compressor import BaseCompressor, Document, CompressionResult
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class EmbeddingsCompressor(BaseCompressor):
    """
    基于嵌入相似度的压缩器

    特点:
    - 速度快，无额外LLM调用
    - 可以复用现有的BGE模型
    - 适用于第一层快速过滤
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        similarity_threshold: float = 0.6,
        default_top_k: int = 10,
        config: Dict[str, Any] = None
    ):
        """
        初始化嵌入压缩器

        Args:
            model_name: 句子编码器模型名称
            similarity_threshold: 相似度阈值，低于此值的文档将被过滤
            default_top_k: 默认保留的文档数量
            config: 其他配置
        """
        super().__init__(config)

        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.default_top_k = default_top_k

        # 延迟加载模型
        self.model: Optional[SentenceTransformer] = None
        self._model_loaded = False

        logger.info(
            f"EmbeddingsCompressor初始化: "
            f"model={model_name}, threshold={similarity_threshold}, top_k={default_top_k}"
        )

    def _load_model(self):
        """延迟加载模型"""
        if not self._model_loaded:
            try:
                logger.info(f"加载句子编码器: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self._model_loaded = True
                logger.info("✅ 句子编码器加载成功")
            except Exception as e:
                logger.error(f"❌ 句子编码器加载失败: {e}")
                raise

    async def compress(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None,
        similarity_threshold: float = None,
        **kwargs
    ) -> CompressionResult:
        """
        基于相似度压缩文档

        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_k: 保留最相关的top_k个文档（None则使用default_top_k）
            similarity_threshold: 相似度阈值（None则使用默认值）
            **kwargs: 其他参数

        Returns:
            CompressionResult: 压缩结果
        """
        start_time = time.time()

        # 加载模型
        if not self._model_loaded:
            self._load_model()

        # 参数设置
        top_k = top_k or self.default_top_k
        threshold = similarity_threshold or self.similarity_threshold

        try:
            # 1. 编码查询和文档
            logger.debug(f"编码查询和{len(documents)}个文档...")
            query_embedding = self.model.encode([query])
            doc_texts = [doc.page_content for doc in documents]
            doc_embeddings = self.model.encode(doc_texts, show_progress_bar=False)

            # 2. 计算相似度
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

            # 3. 过滤和排序
            doc_score_pairs = list(zip(documents, similarities, range(len(documents))))

            # 过滤低于阈值的文档
            filtered = [
                (doc, score, idx)
                for doc, score, idx in doc_score_pairs
                if score >= threshold
            ]

            # 按相似度排序
            filtered.sort(key=lambda x: x[1], reverse=True)

            # 取top_k
            compressed_docs = [doc for doc, _, _ in filtered[:top_k]]

            # 4. 计算统计信息
            compression_time = time.time() - start_time
            tokens_saved = self._estimate_tokens_saved(documents, compressed_docs)

            result = CompressionResult(
                compressed_docs=compressed_docs,
                original_count=len(documents),
                compressed_count=len(compressed_docs),
                compression_ratio=len(compressed_docs) / len(documents) if documents else 0,
                tokens_saved=tokens_saved,
                compression_time=compression_time,
                metadata={
                    "method": "embeddings_similarity",
                    "model": self.model_name,
                    "threshold": threshold,
                    "top_k": top_k,
                    "avg_similarity": np.mean([score for _, score, _ in filtered[:top_k]]) if filtered else 0,
                    "filtered_scores": [score for _, score, _ in filtered[:top_k]]
                }
            )

            # 更新统计
            self._update_stats(len(documents), tokens_saved, compression_time)

            logger.info(
                f"EmbeddingsCompressor: {len(documents)} → {len(compressed_docs)} "
                f"(ratio={result.compression_ratio:.2%}, time={compression_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"EmbeddingsCompressor压缩失败: {e}")
            # 返回原始文档
            return CompressionResult(
                compressed_docs=documents,
                original_count=len(documents),
                compressed_count=len(documents),
                compression_ratio=1.0,
                tokens_saved=0,
                compression_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def _update_stats(self, doc_count: int, tokens_saved: int, compression_time: float):
        """更新统计信息"""
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_documents_processed"] += doc_count
        self.compression_stats["total_tokens_saved"] += tokens_saved
        self.compression_stats["total_time"] += compression_time

    async def compress_batch(
        self,
        queries: List[str],
        documents_list: List[List[Document]],
        **kwargs
    ) -> List[CompressionResult]:
        """
        批量压缩（优化性能）

        Args:
            queries: 查询列表
            documents_list: 文档列表的列表
            **kwargs: 其他参数

        Returns:
            压缩结果列表
        """
        results = []
        for query, documents in zip(queries, documents_list):
            result = await self.compress(query, documents, **kwargs)
            results.append(result)
        return results


def get_embeddings_compressor(
    model_name: str = "BAAI/bge-large-zh-v1.5",
    similarity_threshold: float = 0.6,
    default_top_k: int = 10
) -> EmbeddingsCompressor:
    """
    获取嵌入压缩器实例

    Args:
        model_name: 模型名称
        similarity_threshold: 相似度阈值
        default_top_k: 默认top_k

    Returns:
        EmbeddingsCompressor实例
    """
    return EmbeddingsCompressor(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        default_top_k=default_top_k
    )
