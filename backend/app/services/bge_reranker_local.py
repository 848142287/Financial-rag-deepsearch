"""
本地 BGE Reranker 服务
使用本地下载的 bge-reranker-v2-m3 模型进行文档重排序
"""

from app.core.structured_logging import get_structured_logger
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = get_structured_logger(__name__)


@dataclass
class BGERerankerConfig:
    """BGE本地重排序模型配置"""
    model_path: str = "/app/models/bge-reranker-v2-m3"  # 从settings读取的默认路径
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512


class BGERerankerLocal:
    """本地 BGE 重排序服务类"""

    def __init__(self, config: Optional[BGERerankerConfig] = None, eager_load: bool = False):
        self.config = config or BGERerankerConfig()
        self.model = None

        # 如果启用预加载，立即加载模型
        if eager_load:
            self._load_model()

    def _load_model(self):
        """加载本地模型"""
        if self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            model_path = self.config.model_path

            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            logger.info(f"Loading BGE reranker model from {model_path}")

            # 从本地路径加载模型
            self.model = CrossEncoder(
                model_path,
                device=self.config.device,
                local_files_only=True  # 强制只使用本地文件
            )

            logger.info(f"BGE reranker model loaded successfully on {self.config.device}")

        except ImportError as e:
            logger.error(f"sentence_transformers not installed: {e}")
            raise RuntimeError("Please install sentence_transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load BGE reranker model: {e}")
            raise

    def warmup(self):
        """预热模型（预加载）"""
        logger.info("Warming up BGE reranker model...")
        self._load_model()

        # 使用测试查询进行预热
        try:
            test_query = "测试查询"
            test_docs = ["测试文档1", "测试文档2"]
            test_result = self.rerank(test_query, test_docs, top_k=2)
            logger.info(f"BGE reranker model warmed up successfully (test returned {len(test_result)} results)")
        except Exception as e:
            logger.warning(f"BGE reranker warmup test failed: {e}")

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前K个结果，None则返回全部

        Returns:
            List[(原始索引, 重排序分数)]
        """
        if not documents:
            return []

        if self.model is None:
            raise RuntimeError("Model not loaded")

        if top_k is None:
            top_k = len(documents)

        try:
            # 构造 query-doc pairs
            pairs = [[query, doc] for doc in documents]

            # 批量预测分数
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )

            # 确保 scores 是 numpy array
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores, dtype=np.float32)
            else:
                scores = scores.astype(np.float32)

            # 按分数排序并返回 (原始索引, 分数) 列表
            indexed_scores = [(i, float(scores[i])) for i in range(len(scores))]

            # 按分数降序排序
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # 返回 top_k 结果
            result = indexed_scores[:top_k]

            logger.info(f"Reranked {len(documents)} documents, returning top {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Error reranking documents with BGE: {e}")
            # 降级：返回原始顺序
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(documents)))]

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        重排序包含元数据的文档

        Args:
            query: 查询文本
            documents: 文档列表，每个文档包含content和metadata
            top_k: 返回前K个结果

        Returns:
            重排序后的文档列表
        """
        # 提取文档内容
        contents = [doc.get("content", "") for doc in documents]

        # 调用重排序
        reranked_indices = self.rerank(query, contents, top_k)

        # 重新组织文档
        result = []
        for original_idx, score in reranked_indices:
            if original_idx < len(documents):
                doc = documents[original_idx].copy()
                doc["rerank_score"] = score
                result.append(doc)

        return result

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: Optional[int] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        批量重排序

        Args:
            queries: 查询文本列表
            documents_list: 文档列表的列表
            top_k: 返回前K个结果

        Returns:
            重排序结果列表
        """
        results = []

        for query, documents in zip(queries, documents_list):
            try:
                rerank_result = self.rerank(query, documents, top_k)
                results.append(rerank_result)
            except Exception as e:
                logger.error(f"Error in batch reranking for query '{query[:50]}...': {e}")
                # 返回原始顺序
                results.append([(i, 1.0 - i * 0.01) for i in range(len(documents))])

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "service": "BGERerankerLocal",
            "model_path": self.config.model_path,
            "device": self.config.device,
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试重排序
            test_query = "测试查询"
            test_docs = ["文档1", "文档2", "文档3"]
            test_result = self.rerank(test_query, test_docs, top_k=2)

            is_healthy = len(test_result) > 0

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "model_path": self.config.model_path,
                "device": self.config.device,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 全局服务实例
_bge_reranker_service = None


def get_bge_reranker_service(config: Optional[BGERerankerConfig] = None) -> BGERerankerLocal:
    """获取 BGE 重排序服务单例"""
    global _bge_reranker_service
    if _bge_reranker_service is None:
        _bge_reranker_service = BGERerankerLocal(config)
    return _bge_reranker_service
