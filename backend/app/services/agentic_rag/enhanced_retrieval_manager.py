"""
增强的Agentic RAG检索管理器
扩展现有的RetrieverManager，集成DocMind的核心增强功能
"""

from app.core.structured_logging import get_structured_logger
from app.services.rag.retrieval.bge_reranker_service import get_bge_reranker_service
from app.services.rag.retrieval.enhanced_query_processor import get_query_processor

logger = get_structured_logger(__name__)

class EnhancedRetrieverManager(RetrieverManager):
    """
    增强的检索管理器

    在现有RetrieverManager基础上，增加：
    1. 混合检索（三路召回）
    2. BGE Reranker
    3. 增强查询处理
    """

    def __init__(
        self,
        enable_enhancements: bool = True,
        use_hybrid_retrieval: bool = False,
        enable_bge_reranker: bool = True,
        skip_parent_init: bool = False
    ):
        """
        初始化增强检索管理器

        Args:
            enable_enhancements: 是否启用增强功能
            use_hybrid_retrieval: 是否使用混合检索（三路召回）
            enable_bge_reranker: 是否启用BGE Reranker
            skip_parent_init: 是否跳过父类初始化（用于测试环境）
        """
        # 只在非测试环境下调用父类初始化
        if not skip_parent_init:
            try:
                super().__init__()
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    # 在async上下文中，手动初始化父类的基础属性
                    self.retrievers = {}
                    self.configs = {}
                    self.initialization_status = {}
                    logger.warning("在async上下文中运行，跳过父类自动初始化")
                else:
                    raise
        else:
            # 测试模式：手动初始化父类的基础属性
            self.retrievers = {}
            self.configs = {}
            self.initialization_status = {}

        self.enable_enhancements = enable_enhancements
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.enable_bge_reranker = enable_bge_reranker

        # 增强组件
        self.bge_reranker = None
        self.query_processor = None
        self.hybrid_retriever = None

        if enable_enhancements:
            self._initialize_enhancements()

        logger.info(
            f"EnhancedRetrieverManager初始化: "
            f"enhancements={enable_enhancements}, "
            f"hybrid={use_hybrid_retrieval}, "
            f"bge_reranker={enable_bge_reranker}"
        )

    def _initialize_enhancements(self):
        """初始化增强组件"""
        try:
            # 1. BGE Reranker
            if self.enable_bge_reranker:
                self.bge_reranker = get_bge_reranker_service()
                logger.info("✅ BGE Reranker已加载")

            # 2. 查询处理器
            self.query_processor = get_query_processor()
            logger.info("✅ 查询处理器已加载")

            # 3. 混合检索（可选）
            if self.use_hybrid_retrieval:
                # 需要提供vector_store和bm25_store
                logger.info("ℹ️ 混合检索需要提供vector_store和bm25_store")

        except Exception as e:
            logger.warning(f"增强组件初始化失败: {e}")
            self.enable_enhancements = False

    def setup_hybrid_retrieval(
        self,
        vector_store,
        bm25_store,
        config: HybridRetrievalConfig = None
    ):
        """
        设置混合检索

        Args:
            vector_store: 向量存储实例
            bm25_store: BM25存储实例
            config: 混合检索配置
        """
        try:
            self.hybrid_retriever = HybridRetrievalService(
                vector_store=vector_store,
                bm25_store=bm25_store,
                config=config or HybridRetrievalConfig()
            )

            # 添加到检索器列表
            self.retrievers[RetrieverType.HYBRID] = self.hybrid_retriever

            logger.info("✅ 混合检索已设置")

        except Exception as e:
            logger.error(f"混合检索设置失败: {e}")

    async def enhanced_retrieve(
        self,
        query: str,
        top_k: int = 5,
        history: List[Dict[str, str]] = None,
        use_hybrid: bool = None
    ) -> Dict[str, Any]:
        """
        增强检索

        Args:
            query: 查询文本
            top_k: 返回数量
            history: 对话历史
            use_hybrid: 是否使用混合检索（None则使用初始化配置）

        Returns:
            {
                "results": [...],
                "query_info": {...},
                "stats": {...},
                "enhancement_used": bool
            }
        """
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid_retrieval

        if not self.enable_enhancements or not use_hybrid:
            # 使用原有检索逻辑
            return await self._standard_retrieve(query, top_k, history)

        # 使用增强检索
        try:
            # 1. 查询处理
            if self.query_processor:
                processed_query = await self.query_processor.process(query, history)

                # 元问题直接返回
                if processed_query.is_meta_question:
                    return {
                        "results": [],
                        "query_info": processed_query,
                        "stats": {},
                        "direct_answer": processed_query.direct_answer,
                        "enhancement_used": True
                    }

                query_to_use = processed_query.vector_query
            else:
                processed_query = None
                query_to_use = query

            # 2. 混合检索
            if self.hybrid_retriever:
                result = await self.hybrid_retriever.retrieve(
                    query=query_to_use,
                    top_k=top_k,
                    history=history
                )

                result["enhancement_used"] = True
                return result

            # 3. 降级到原有检索
            else:
                standard_result = await self._standard_retrieve(query, top_k, history)
                standard_result["enhancement_used"] = False
                return standard_result

        except Exception as e:
            logger.error(f"增强检索失败: {e}，降级到标准检索")
            return await self._standard_retrieve(query, top_k, history)

    async def _standard_retrieve(
        self,
        query: str,
        top_k: int,
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        标准检索（原有逻辑）

        使用现有的VECTOR、GRAPH等检索器
        """
        results = []
        query_info = {}

        # 使用向量检索
        if RetrieverType.VECTOR in self.retrievers:
            vector_results = await self.retrievers[RetrieverType.VECTOR].retrieve(
                query=query,
                top_k=top_k
            )
            results.extend(vector_results)

        # 使用图谱检索
        if RetrieverType.GRAPH in self.retrievers:
            try:
                graph_results = await self.retrievers[RetrieverType.GRAPH].retrieve(
                    query=query,
                    top_k=top_k
                )
                results.extend(graph_results)
            except Exception as e:
                logger.warning(f"图谱检索失败: {e}")

        # BGE Reranker重排序（如果启用）
        if self.enable_bge_reranker and self.bge_reranker and results:
            documents = [r.get("content", "") for r in results]
            reranked = self.bge_reranker.rerank(query, documents, top_k=top_k)

            # 重新排序
            reranked_results = []
            for idx, score in reranked:
                if idx < len(results):
                    result = results[idx].copy()
                    result["rerank_score"] = score
                    reranked_results.append(result)

            results = reranked_results

        return {
            "results": results[:top_k],
            "query_info": query_info,
            "stats": {
                "total_results": len(results),
                "enhancement_used": False
            }
        }

# 便捷函数
def get_enhanced_retriever_manager(
    enable_enhancements: bool = True,
    use_hybrid_retrieval: bool = False,
    enable_bge_reranker: bool = True,
    skip_parent_init: bool = False
) -> EnhancedRetrieverManager:
    """
    获取增强检索管理器实例

    Args:
        enable_enhancements: 是否启用增强功能
        use_hybrid_retrieval: 是否使用混合检索
        enable_bge_reranker: 是否启用BGE Reranker
        skip_parent_init: 是否跳过父类初始化（用于测试环境）

    Returns:
        EnhancedRetrieverManager
    """
    return EnhancedRetrieverManager(
        enable_enhancements=enable_enhancements,
        use_hybrid_retrieval=use_hybrid_retrieval,
        enable_bge_reranker=enable_bge_reranker,
        skip_parent_init=skip_parent_init
    )
