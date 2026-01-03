"""
BGE Reranker 服务
从DocMind项目借鉴，使用Sigmoid校准的精细化重排序
"""

import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class BaseReranker(ABC):
    """Reranker基类"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前K个结果

        Returns:
            [(doc_idx, score), ...] 按分数降序
        """
        pass


class BGERerankerService(BaseReranker):
    """
    BGE Reranker 服务

    特性：
    1. Sigmoid偏差校准：将BGE logits映射到0-10可解释范围
    2. 支持本地和远程两种模式
    3. 批处理优化
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        score_bias: float = 4.0,
        use_remote: bool = False,
        api_key: str = None
    ):
        """
        初始化BGE Reranker

        Args:
            model_name: 模型名称
            score_bias: Sigmoid偏差值（用于校准分数）
            use_remote: 是否使用远程API
            api_key: 远程API密钥
        """
        self.model_name = model_name
        self.score_bias = score_bias
        self.use_remote = use_remote
        self.api_key = api_key
        self._model = None

        logger.info(
            f"BGEReranker初始化: model={model_name}, "
            f"bias={score_bias}, remote={use_remote}"
        )

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None and not self.use_remote:
            try:
                from FlagEmbedding import BGELMFlagModel
                self._model = BGELMFlagModel(
                    model_name_or_path=self.model_name,
                    use_fp16=True  # FP16加速
                )
                logger.info("✅ BGE Reranker模型加载成功")
            except Exception as e:
                logger.error(f"❌ BGE Reranker模型加载失败: {e}")
                raise
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前K个结果

        Returns:
            [(doc_idx, score), ...] 按分数降序
        """
        if not documents:
            return []

        try:
            if self.use_remote:
                reranked = self._rerank_remote(query, documents)
            else:
                reranked = self._rerank_local(query, documents)

            # 应用Sigmoid校准
            calibrated = self._calibrate_scores([r[1] for r in reranked])

            # 重新组合
            result = [(reranked[i][0], calibrated[i]) for i in range(len(reranked))]

            # 截取top_k
            if top_k:
                result = result[:top_k]

            return result

        except Exception as e:
            logger.error(f"❌ Rerank失败: {e}")
            # 返回原始顺序
            return [(i, 0.0) for i in range(len(documents))]

    def _rerank_local(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[int, float]]:
        """本地Rerank"""
        model = self.model

        # 构建pairs
        pairs = [[query, doc] for doc in documents]

        # 批处理推理
        raw_scores = model.compute_score(pairs)

        # 返回 (idx, raw_score)
        return list(enumerate(raw_scores))

    def _rerank_remote(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[int, float]]:
        """远程API Rerank"""
        import requests

        if not self.api_key:
            raise ValueError("远程模式需要api_key")

        url = "https://api.siliconflow.cn/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "return_documents": False,
            "top_n": len(documents)
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        results = response.json()['results']
        return [(r['index'], r['relevance_score']) for r in results]

    def _calibrate_scores(self, raw_scores: List[float]) -> List[float]:
        """
        Sigmoid偏差校准

        BGE原始logits范围：[-10, 10]
        使用偏移后的Sigmoid映射到[0, 10]

        例如：raw=-2.2, bias=4.0
        -> shifted=1.8
        -> sigmoid(1.8)=0.85
        -> score=8.5
        """
        calibrated = []
        for score in raw_scores:
            shifted = score + self.score_bias
            sigmoid = 1 / (1 + np.exp(-shifted))
            calibrated.append(sigmoid * 10)

        return calibrated

    def predict(self, pairs: List[List[str]]) -> List[float]:
        """
        兼容性接口

        Args:
            pairs: [[query1, doc1], [query2, doc2], ...]

        Returns:
            [score1, score2, ...]
        """
        queries = [p[0] for p in pairs]
        docs = [p[1] for p in pairs]

        reranked = self.rerank(queries[0], docs)

        # 按原始顺序返回分数
        scores = [0.0] * len(docs)
        for idx, score in reranked:
            scores[idx] = score

        return scores


class ThreeLevelConfidenceFilter:
    """
    三级置信度过滤

    根据rerank分数将结果分为三个置信度级别：
    - high: score >= 6.0（可信）
    - medium: 4.0 <= score < 6.0（中等）
    - low: score < 4.0（低可信，可过滤）
    """

    def __init__(
        self,
        threshold_low: float = 4.0,
        threshold_high: float = 6.0,
        filter_low: bool = True
    ):
        """
        初始化置信度过滤器

        Args:
            threshold_low: 低置信度阈值
            threshold_high: 高置信度阈值
            filter_low: 是否过滤低置信度结果
        """
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.filter_low = filter_low

    def filter_and_classify(
        self,
        results: List[Tuple[int, float, dict]]
    ) -> List[Tuple[int, float, dict, str]]:
        """
        过滤并分类结果

        Args:
            results: [(doc_idx, score, metadata), ...]

        Returns:
            [(doc_idx, score, metadata, confidence), ...]
            confidence in ["high", "medium", "low"]
        """
        filtered = []

        for doc_idx, score, metadata in results:
            # 分类置信度
            if score >= self.threshold_high:
                confidence = "high"
            elif score >= self.threshold_low:
                confidence = "medium"
            else:
                confidence = "low"

            # 过滤低置信度
            if self.filter_low and confidence == "low":
                continue

            filtered.append((doc_idx, score, metadata, confidence))

        return filtered

    def get_confidence_stats(
        self,
        results: List[Tuple[int, float]]
    ) -> dict:
        """
        获取置信度统计

        Returns:
            {
                "high": int,
                "medium": int,
                "low": int,
                "avg_score": float
            }
        """
        high = sum(1 for _, s in results if s >= self.threshold_high)
        medium = sum(1 for _, s in results if self.threshold_low <= s < self.threshold_high)
        low = sum(1 for _, s in results if s < self.threshold_low)
        avg = sum(s for _, s in results) / len(results) if results else 0

        return {
            "high": high,
            "medium": medium,
            "low": low,
            "avg_score": avg
        }


# 全局实例
_reranker_service = None


def get_bge_reranker_service() -> BGERerankerService:
    """获取BGE Reranker服务单例"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = BGERerankerService()
    return _reranker_service
