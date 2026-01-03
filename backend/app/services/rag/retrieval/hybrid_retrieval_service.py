"""
混合检索服务
借鉴DocMind项目的三路混合检索架构：
- Path A: HyDE向量检索
- Path B: 语义向量检索
- Path C: BM25关键词检索
"""

from dataclasses import dataclass, field
from app.core.structured_logging import get_structured_logger
from app.services.rag.retrieval.enhanced_query_processor import (
    get_query_processor,
    ProcessedQuery
)
from app.services.rag.retrieval.bge_reranker_service import (
    get_bge_reranker_service,
    ThreeLevelConfidenceFilter
)

logger = get_structured_logger(__name__)

@dataclass
class RetrievalResult:
    """检索结果"""
    doc_id: str
    content: str
    score: float
    confidence: str  # high/medium/low
    source: str  # 来源路径标识
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HybridRetrievalConfig:
    """混合检索配置"""
    # 召回配置
    candidate_multiplier: int = 3  # 每条路召回 top_k * multiplier
    enable_hyde: bool = True  # 启用HyDE路径
    enable_semantic: bool = True  # 启用语义检索路径
    enable_bm25: bool = True  # 启用BM25路径

    # Reranker配置
    enable_rerank: bool = True
    score_bias: float = 4.0  # Sigmoid偏差

    # 置信度过滤
    enable_confidence_filter: bool = True
    threshold_low: float = 4.0
    threshold_high: float = 6.0

    # 关键词硬过滤
    enable_keyword_filter: bool = True
    keyword_high_score_threshold: float = 8.5  # 高分豁免阈值

class HybridRetrievalService:
    """
    混合检索服务

    三路召回 + Reranker + 置信度过滤
    """

    def __init__(
        self,
        vector_store=None,
        bm25_store=None,
        config: HybridRetrievalConfig = None
    ):
        """
        初始化混合检索服务

        Args:
            vector_store: 向量存储实例
            bm25_store: BM25存储实例
            config: 检索配置
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.config = config or HybridRetrievalConfig()

        self.query_processor = get_query_processor()
        self.reranker = get_bge_reranker_service()
        self.confidence_filter = ThreeLevelConfidenceFilter(
            threshold_low=self.config.threshold_low,
            threshold_high=self.config.threshold_high
        )

        logger.info(
            f"HybridRetrievalService初始化: "
            f"hyde={self.config.enable_hyde}, "
            f"semantic={self.config.enable_semantic}, "
            f"bm25={self.config.enable_bm25}, "
            f"rerank={self.config.enable_rerank}"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        history: List[Dict[str, str]] = None,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        执行混合检索

        Args:
            query: 用户查询
            top_k: 返回结果数量
            history: 对话历史
            filters: 元数据过滤条件

        Returns:
            {
                "results": List[RetrievalResult],
                "query_info": ProcessedQuery,
                "stats": Dict,
                "direct_answer": Optional[str]  # 元问题的直接回答
            }
        """
        logger.info(f"🔍 开始混合检索: query='{query}', top_k={top_k}")

        # 1. 查询处理
        processed_query = await self.query_processor.process(query, history)

        # 如果是元问题，直接返回
        if processed_query.is_meta_question:
            return {
                "results": [],
                "query_info": processed_query,
                "stats": {},
                "direct_answer": processed_query.direct_answer
            }

        # 2. 三路召回
        candidates = await self._three_way_recall(processed_query, top_k, filters)

        logger.info(f"📊 三路召回完成: candidates={len(candidates)}")

        # 3. Reranker重排序
        if self.config.enable_rerank and candidates:
            reranked = await self._rerank_candidates(query, candidates, top_k)
        else:
            reranked = candidates[:top_k]

        logger.info(f"🔄 Reranker完成: reranked={len(reranked)}")

        # 4. 关键词硬过滤
        if self.config.enable_keyword_filter and processed_query.keywords:
            filtered = self._keyword_hard_filter(
                reranked,
                processed_query.keywords,
                query
            )
        else:
            filtered = reranked

        logger.info(f"🔑 关键词过滤完成: filtered={len(filtered)}")

        # 5. 置信度过滤
        if self.config.enable_confidence_filter:
            final_results = self._apply_confidence_filter(filtered)
        else:
            final_results = [
                RetrievalResult(
                    doc_id=r["doc_id"],
                    content=r["content"],
                    score=r["score"],
                    confidence="unknown",
                    source=r.get("source", "unknown"),
                    metadata=r.get("metadata", {})
                )
                for r in filtered
            ]

        logger.info(
            f"✅ 混合检索完成: final_results={len(final_results)}, "
            f"avg_score={sum(r.score for r in final_results)/len(final_results) if final_results else 0:.2f}"
        )

        # 统计信息
        stats = {
            "candidates": len(candidates),
            "after_rerank": len(reranked),
            "after_keyword_filter": len(filtered),
            "final": len(final_results),
            "avg_score": sum(r.score for r in final_results) / len(final_results) if final_results else 0,
            "confidence_distribution": self._get_confidence_distribution(final_results)
        }

        return {
            "results": final_results[:top_k],
            "query_info": processed_query,
            "stats": stats,
            "direct_answer": None
        }

    async def _three_way_recall(
        self,
        processed_query: ProcessedQuery,
        top_k: int,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        三路召回

        Path A: HyDE向量检索
        Path B: 语义向量检索
        Path C: BM25关键词检索
        """
        candidate_k = top_k * self.config.candidate_multiplier
        candidates = {}  # {doc_id: best_score, metadata}

        # Path A: HyDE向量检索
        if self.config.enable_hyde and self.vector_store and processed_query.hyde_doc:
            hyde_results = await self._vector_search(
                processed_query.hyde_doc,
                candidate_k,
                filters,
                source="hyde"
            )
            for r in hyde_results:
                self._merge_candidate(candidates, r)

            logger.debug(f"HyDE路径: 召回{len(hyde_results)}个")

        # Path B: 语义向量检索
        if self.config.enable_semantic and self.vector_store:
            semantic_results = await self._vector_search(
                processed_query.vector_query,
                candidate_k,
                filters,
                source="semantic"
            )
            for r in semantic_results:
                self._merge_candidate(candidates, r)

            logger.debug(f"语义路径: 召回{len(semantic_results)}个")

        # Path C: BM25关键词检索
        if self.config.enable_bm25 and self.bm25_store:
            bm25_results = await self._bm25_search(
                [processed_query.standalone_query] + processed_query.keywords,
                candidate_k,
                filters
            )
            for r in bm25_results:
                self._merge_candidate(candidates, r)

            logger.debug(f"BM25路径: 召回{len(bm25_results)}个")

        # 转换为列表
        return [
            {
                "doc_id": doc_id,
                "content": data["content"],
                "score": data["score"],
                "source": data["source"],
                "metadata": data.get("metadata", {})
            }
            for doc_id, data in candidates.items()
        ]

    def _merge_candidate(
        self,
        candidates: Dict[str, Dict],
        new_result: Dict[str, Any]
    ):
        """
        合并候选结果，保留最高分数

        Args:
            candidates: {doc_id: {content, score, source, metadata}}
            new_result: {"doc_id", "content", "score", "source", "metadata"}
        """
        doc_id = new_result["doc_id"]

        if doc_id not in candidates:
            candidates[doc_id] = {
                "content": new_result["content"],
                "score": new_result["score"],
                "source": new_result["source"],
                "metadata": new_result.get("metadata", {})
            }
        else:
            # 保留更高分数的结果
            if new_result["score"] > candidates[doc_id]["score"]:
                candidates[doc_id]["score"] = new_result["score"]
                candidates[doc_id]["source"] = new_result["source"]

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any],
        source: str
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        注：这是接口定义，实际实现需要调用具体的vector store
        """
        # TODO: 调用实际的vector store
        # 这里返回模拟结果
        return []

    async def _bm25_search(
        self,
        queries: List[str],
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        BM25检索

        注：这是接口定义，实际实现需要调用具体的BM25 store
        """
        # TODO: 调用实际的BM25 store
        # 这里返回模拟结果
        return []

    async def _rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        使用BGE Reranker重排序
        """
        if not candidates:
            return []

        # 提取文档内容
        documents = [c["content"] for c in candidates]

        # Rerank
        rerank_scores = self.reranker.rerank(query, documents, top_k=None)

        # 更新分数
        for i, (idx, score) in enumerate(rerank_scores):
            candidates[idx]["score"] = score

        # 按分数排序
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates

    def _keyword_hard_filter(
        self,
        results: List[Dict[str, Any]],
        keywords: List[str],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        关键词硬过滤

        规则：
        1. 如果文档包含任一关键词，保留
        2. 如果rerank分数 > threshold，直接保留（高分豁免）
        """
        filtered = []

        for r in results:
            # 高分豁免
            if r["score"] >= self.config.keyword_high_score_threshold:
                filtered.append(r)
                continue

            # 关键词检查
            content_lower = r["content"].lower()
            if any(kw.lower() in content_lower for kw in keywords):
                filtered.append(r)

        return filtered

    def _apply_confidence_filter(
        self,
        results: List[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """
        应用置信度过滤
        """
        # 转换为元组格式
        tuples = [
            (i, r["score"], {"content": r["content"], "metadata": r.get("metadata", {})})
            for i, r in enumerate(results)
        ]

        # 过滤并分类
        filtered = self.confidence_filter.filter_and_classify(tuples)

        # 转换为RetrievalResult
        return [
            RetrievalResult(
                doc_id=results[idx]["doc_id"],
                content=metadata["content"],
                score=score,
                confidence=confidence,
                source=results[idx].get("source", "unknown"),
                metadata=metadata.get("metadata", {})
            )
            for idx, score, metadata, confidence in filtered
        ]

    def _get_confidence_distribution(
        self,
        results: List[RetrievalResult]
    ) -> Dict[str, int]:
        """获取置信度分布"""
        dist = {"high": 0, "medium": 0, "low": 0}
        for r in results:
            dist[r.confidence] = dist.get(r.confidence, 0) + 1
        return dist

# 全局实例
_hybrid_retrieval_service = None

def get_hybrid_retrieval_service(
    vector_store=None,
    bm25_store=None,
    config: HybridRetrievalConfig = None
) -> HybridRetrievalService:
    """获取混合检索服务单例"""
    global _hybrid_retrieval_service
    if _hybrid_retrieval_service is None:
        _hybrid_retrieval_service = HybridRetrievalService(
            vector_store=vector_store,
            bm25_store=bm25_store,
            config=config
        )
    return _hybrid_retrieval_service
