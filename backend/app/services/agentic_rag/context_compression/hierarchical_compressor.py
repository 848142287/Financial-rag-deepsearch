"""
分层上下文压缩器

整合L1(Embeddings)和L3(LLM)压缩，根据检索级别自动选择压缩策略
"""

import time
from typing import List, Dict, Any, Optional
from enum import Enum

from .base_compressor import BaseCompressor, Document, CompressionResult

# 可选导入
try:
    from .embeddings_compressor import EmbeddingsCompressor
except ImportError:
    EmbeddingsCompressor = None

try:
    from .llm_extractor import FinancialContextCompressor
except ImportError:
    FinancialContextCompressor = None

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class CompressionLevel(Enum):
    """压缩级别"""
    NONE = "none"           # 不压缩
    LIGHT = "light"         # 轻度压缩（仅L1）
    MEDIUM = "medium"       # 中度压缩（L1 + L3部分）
    AGGRESSIVE = "aggressive"  # 激进压缩（L1 + L3全部）


class HierarchicalContextCompressor(BaseCompressor):
    """
    分层上下文压缩器

    三层压缩策略:
    - L1: EmbeddingsCompressor - 快速相似度过滤
    - L2: BGE Reranker - 重排序（在外部完成）
    - L3: FinancialContextCompressor - LLM精细提取

    根据检索级别自动选择压缩策略:
    - fast: 不压缩或轻度压缩
    - enhanced: 中度压缩
    - deep_search: 激进压缩
    """

    def __init__(
        self,
        llm=None,
        embeddings_compressor: EmbeddingsCompressor = None,
        llm_compressor: FinancialContextCompressor = None,
        config: Dict[str, Any] = None
    ):
        """
        初始化分层压缩器

        Args:
            llm: LangChain LLM实例（用于LLM压缩器）
            embeddings_compressor: L1嵌入压缩器
            llm_compressor: L3 LLM压缩器
            config: 配置字典
        """
        super().__init__(config)

        # L1压缩器
        if embeddings_compressor:
            self.l1_compressor = embeddings_compressor
        else:
            # 使用默认配置
            self.l1_compressor = EmbeddingsCompressor(
                model_name=config.get("embeddings_model", "BAAI/bge-large-zh-v1.5") if config else "BAAI/bge-large-zh-v1.5",
                similarity_threshold=0.6,
                default_top_k=10
            )

        # L3压缩器（如果提供了LLM）
        self.l3_compressor = llm_compressor
        if llm and not llm_compressor:
            self.l3_compressor = FinancialContextCompressor(
                llm=llm,
                compression_rate=0.5,
                max_length=2000
            )

        # 不同检索级别的压缩配置
        self.level_configs = self._setup_level_configs()

        logger.info(
            f"HierarchicalContextCompressor初始化: "
            f"L1={'✓' if self.l1_compressor else '✗'}, "
            f"L3={'✓' if self.l3_compressor else '✗'}"
        )

    def _setup_level_configs(self) -> Dict[str, Dict[str, Any]]:
        """设置不同级别的压缩配置"""
        return {
            "fast": {
                "compression_level": CompressionLevel.LIGHT,
                "l1_top_k": 5,
                "l1_threshold": 0.65,
                "use_llm": False,
                "llm_rate": 1.0,
                "llm_max_length": 2000
            },
            "enhanced": {
                "compression_level": CompressionLevel.MEDIUM,
                "l1_top_k": 10,
                "l1_threshold": 0.6,
                "use_llm": True,
                "llm_rate": 0.6,
                "llm_max_length": 1500
            },
            "deep_search": {
                "compression_level": CompressionLevel.AGGRESSIVE,
                "l1_top_k": 15,
                "l1_threshold": 0.55,
                "use_llm": True,
                "llm_rate": 0.4,
                "llm_max_length": 1200
            }
        }

    async def compress(
        self,
        query: str,
        documents: List[Document],
        retrieval_level: str = "enhanced",
        **kwargs
    ) -> CompressionResult:
        """
        分层压缩

        Args:
            query: 用户查询
            documents: 检索到的文档列表
            retrieval_level: 检索级别 (fast/enhanced/deep_search)
            **kwargs: 其他参数

        Returns:
            CompressionResult: 压缩结果
        """
        start_time = time.time()

        if not documents:
            return CompressionResult(
                compressed_docs=[],
                original_count=0,
                compressed_count=0,
                compression_ratio=0,
                tokens_saved=0,
                compression_time=0,
                metadata={"method": "hierarchical", "level": retrieval_level}
            )

        # 获取配置
        config = self.level_configs.get(retrieval_level, self.level_configs["enhanced"])
        compression_level = config["compression_level"]

        logger.info(
            f"分层压缩开始: {len(documents)}个文档, "
            f"级别={retrieval_level}, "
            f"压缩等级={compression_level.value}"
        )

        try:
            # 记录原始文档
            original_docs = documents
            current_docs = documents
            stage_results = {}

            # L1: 嵌入相似度过滤（始终执行）
            if compression_level != CompressionLevel.NONE:
                l1_result = await self.l1_compressor.compress(
                    query=query,
                    documents=current_docs,
                    top_k=config["l1_top_k"],
                    similarity_threshold=config["l1_threshold"]
                )
                current_docs = l1_result.compressed_docs
                stage_results["l1_embeddings"] = {
                    "count": len(current_docs),
                    "time": l1_result.compression_time,
                    "avg_similarity": l1_result.metadata.get("avg_similarity", 0)
                }

                logger.info(
                    f"L1压缩完成: {len(original_docs)} → {len(current_docs)} "
                    f"(相似度阈值={config['l1_threshold']})"
                )

            # L3: LLM精细提取（根据配置）
            if config["use_llm"] and self.l3_compressor:
                l3_result = await self.l3_compressor.compress(
                    query=query,
                    documents=current_docs,
                    compression_rate=config["llm_rate"],
                    max_length=config["llm_max_length"]
                )
                current_docs = l3_result.compressed_docs
                stage_results["l3_llm"] = {
                    "count": len(current_docs),
                    "time": l3_result.compression_time,
                    "avg_compression": l3_result.metadata.get("avg_compression_per_doc", 0)
                }

                logger.info(
                    f"L3压缩完成: {len(original_docs)} → {len(current_docs)} "
                    f"(LLM压缩率={config['llm_rate']:.2%})"
                )

            # 计算总体统计
            total_time = time.time() - start_time
            tokens_saved = self._estimate_tokens_saved(original_docs, current_docs)

            result = CompressionResult(
                compressed_docs=current_docs,
                original_count=len(original_docs),
                compressed_count=len(current_docs),
                compression_ratio=len(current_docs) / len(original_docs) if original_docs else 0,
                tokens_saved=tokens_saved,
                compression_time=total_time,
                metadata={
                    "method": "hierarchical",
                    "level": retrieval_level,
                    "compression_level": compression_level.value,
                    "stages": stage_results,
                    "config": config
                }
            )

            # 更新统计
            self._update_stats(len(original_docs), tokens_saved, total_time)

            logger.info(
                f"分层压缩完成: {len(original_docs)} → {len(current_docs)} "
                f"(ratio={result.compression_ratio:.2%}, "
                f"tokens_saved={tokens_saved}, "
                f"time={total_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"分层压缩失败: {e}")
            # 返回原始文档
            return CompressionResult(
                compressed_docs=documents,
                original_count=len(documents),
                compressed_count=len(documents),
                compression_ratio=1.0,
                tokens_saved=0,
                compression_time=time.time() - start_time,
                metadata={
                    "method": "hierarchical",
                    "level": retrieval_level,
                    "error": str(e)
                }
            )

    def _update_stats(self, doc_count: int, tokens_saved: int, compression_time: float):
        """更新统计信息"""
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_documents_processed"] += doc_count
        self.compression_stats["total_tokens_saved"] += tokens_saved
        self.compression_stats["total_time"] += compression_time

    def get_recommended_level(self, doc_count: int, avg_doc_length: int) -> str:
        """
        根据文档数量和长度推荐压缩级别

        Args:
            doc_count: 文档数量
            avg_doc_length: 平均文档长度

        Returns:
            推荐的检索级别
        """
        # 计算总token数估算
        total_tokens = doc_count * avg_doc_length * 0.75  # 粗略估算

        if total_tokens < 2000:
            return "fast"
        elif total_tokens < 5000:
            return "enhanced"
        else:
            return "deep_search"

    def update_config(self, level: str, config_updates: Dict[str, Any]):
        """
        更新特定级别的配置

        Args:
            level: 检索级别 (fast/enhanced/deep_search)
            config_updates: 配置更新
        """
        if level in self.level_configs:
            self.level_configs[level].update(config_updates)
            logger.info(f"已更新{level}级别的压缩配置")
        else:
            logger.warning(f"未知的级别: {level}")


# 全局实例
_hierarchical_compressor_instance: Optional[HierarchicalContextCompressor] = None


def get_compressor(
    llm=None,
    config: Dict[str, Any] = None
) -> HierarchicalContextCompressor:
    """
    获取分层压缩器单例

    Args:
        llm: LangChain LLM实例
        config: 配置字典

    Returns:
        HierarchicalContextCompressor实例
    """
    global _hierarchical_compressor_instance

    if _hierarchical_compressor_instance is None:
        _hierarchical_compressor_instance = HierarchicalContextCompressor(
            llm=llm,
            config=config
        )

    return _hierarchical_compressor_instance


def reset_compressor():
    """重置全局压缩器实例"""
    global _hierarchical_compressor_instance
    _hierarchical_compressor_instance = None
