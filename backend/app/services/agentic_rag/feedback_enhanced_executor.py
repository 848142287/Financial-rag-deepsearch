"""
集成反馈回路的Agentic RAG执行器

在原有执行器基础上集成L1实时反馈回路
"""

import time
import asyncio
from typing import Dict, List, Any, Optional

from app.core.structured_logging import get_structured_logger
from .feedback_loop import get_realtime_feedback_processor
from .context_compression import HierarchicalContextCompressor, get_compressor, Document
from .context_compression.base_compressor import CompressionResult
from .adaptive_retrieval import get_adaptive_feedback_processor

logger = get_structured_logger(__name__)


class ExecutionResult:
    """执行结果"""
    def __init__(
        self,
        plan_id: str,
        fused_results: List[Dict],
        execution_time: float,
        metadata: Dict = None
    ):
        self.plan_id = plan_id
        self.fused_results = fused_results
        self.execution_time = execution_time
        self.metadata = metadata or {}


class FeedbackEnhancedExecutor:
    """
    集成反馈回路的执行器

    功能:
    1. 实时反馈优化查询
    2. 动态调整检索参数
    3. 上下文压缩
    4. 收集用户反馈
    5. 自适应检索方法选择 (新)
    """

    def __init__(
        self,
        base_executor=None,
        enable_feedback: bool = True,
        enable_compression: bool = True,
        enable_adaptive: bool = False,
        llm=None
    ):
        """
        初始化反馈增强执行器

        Args:
            base_executor: 基础执行器
            enable_feedback: 是否启用反馈回路
            enable_compression: 是否启用上下文压缩
            enable_adaptive: 是否启用自适应检索
            llm: LLM实例（用于压缩）
        """
        self.base_executor = base_executor
        self.enable_feedback = enable_feedback
        self.enable_compression = enable_compression
        self.enable_adaptive = enable_adaptive

        # 初始化反馈处理器
        if enable_feedback:
            try:
                self.feedback_processor = get_realtime_feedback_processor()
                logger.info("✅ 反馈回路已启用")
            except Exception as e:
                logger.warning(f"⚠ 反馈回路初始化失败: {e}")
                self.enable_feedback = False

        # 初始化压缩器
        if enable_compression:
            try:
                self.compressor = get_compressor(llm=llm)
                logger.info("✅ 上下文压缩已启用")
            except Exception as e:
                logger.warning(f"⚠ 上下文压缩初始化失败: {e}")
                self.enable_compression = False

        # 初始化自适应反馈处理器
        if enable_adaptive:
            try:
                self.adaptive_processor = get_adaptive_feedback_processor(
                    enable_classification=True,
                    enable_optimization=True,
                    enable_bandit=True,
                    enable_feedback=True
                )
                logger.info("✅ 自适应检索已启用")
            except Exception as e:
                logger.warning(f"⚠ 自适应检索初始化失败: {e}")
                self.enable_adaptive = False

    async def execute_with_feedback(
        self,
        plan,
        query: str,
        retrieval_level: str = "enhanced",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        执行检索计划 - 集成反馈优化

        Args:
            plan: 检索计划
            query: 用户查询
            retrieval_level: 检索级别
            user_id: 用户ID
            session_id: 会话ID
            **kwargs: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        plan_id = getattr(plan, 'plan_id', 'unknown')

        logger.info(
            f"开始执行检索: {plan_id}, query={query[:50]}, "
            f"反馈={'启用' if self.enable_feedback else '禁用'}, "
            f"压缩={'启用' if self.enable_compression else '禁用'}, "
            f"自适应={'启用' if self.enable_adaptive else '禁用'}"
        )

        try:
            # ========== 阶段1: 反馈优化 ==========
            optimized_query = query
            dynamic_params = {}
            selected_method = None
            query_features = None

            # 优先使用自适应处理器
            if self.enable_adaptive:
                adaptive_result = await self.adaptive_processor.process_query_with_adaptive_feedback(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    retrieval_level=retrieval_level,
                    base_params=kwargs
                )

                optimized_query = adaptive_result["query"]
                dynamic_params = adaptive_result["params"]
                selected_method = adaptive_result["method"]
                query_features = adaptive_result["features"]

                logger.info(
                    f"自适应优化: method={selected_method}, "
                    f"query_type={query_features['query_type']}, "
                    f"complexity={query_features['complexity']}, "
                    f"query changed={query != optimized_query}"
                )

            elif self.enable_feedback:
                feedback_result = await self.feedback_processor.enhance_query(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    retrieval_level=retrieval_level
                )

                optimized_query = feedback_result["optimized_query"]
                dynamic_params = feedback_result["params"]

                logger.info(
                    f"反馈优化: query changed={query != optimized_query}, "
                    f"params={dynamic_params}"
                )

            # ========== 阶段2: 执行检索 ==========
            if self.base_executor:
                # 使用基础执行器
                execution_result = await self._execute_with_base_executor(
                    plan, optimized_query, dynamic_params
                )
            else:
                # 模拟执行
                execution_result = await self._mock_execute(
                    optimized_query, dynamic_params
                )

            fused_results = execution_result.fused_results

            # ========== 阶段3: 上下文压缩 ==========
            if self.enable_compression and self.compressor and fused_results:
                compression_result = await self._apply_compression(
                    query=optimized_query,
                    documents=fused_results,
                    retrieval_level=retrieval_level,
                    dynamic_params=dynamic_params
                )

                fused_results = compression_result["compressed_docs"]

                # 记录压缩信息
                if not hasattr(execution_result, 'metadata'):
                    execution_result.metadata = {}
                execution_result.metadata["context_compression"] = compression_result

            # ========== 阶段4: 更新执行结果 ==========
            execution_time = time.time() - start_time
            execution_result.execution_time = execution_time

            # 添加反馈元数据
            if not hasattr(execution_result, 'metadata'):
                execution_result.metadata = {}

            execution_result.metadata.update({
                "feedback_enabled": self.enable_feedback,
                "compression_enabled": self.enable_compression,
                "adaptive_enabled": self.enable_adaptive,
                "query_optimized": query != optimized_query,
                "original_query": query,
                "optimized_query": optimized_query,
                "dynamic_params": dynamic_params,
                "selected_method": selected_method,
                "query_features": query_features
            })

            logger.info(
                f"检索完成: {plan_id}, "
                f"results={len(fused_results)}, "
                f"time={execution_time:.2f}s"
            )

            return execution_result

        except Exception as e:
            logger.error(f"检索执行失败: {plan_id}, 错误={e}")

            return ExecutionResult(
                plan_id=plan_id,
                fused_results=[],
                execution_time=time.time() - start_time,
                metadata={
                    "error": str(e),
                    "feedback_enabled": self.enable_feedback,
                    "compression_enabled": self.enable_compression,
                    "adaptive_enabled": self.enable_adaptive
                }
            )

    async def collect_result_feedback(
        self,
        query: str,
        results: List[Dict],
        user_interactions: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        method: str = "vector"
    ):
        """
        收集结果反馈

        Args:
            query: 查询
            results: 检索结果
            user_interactions: 用户交互
            user_id: 用户ID
            session_id: 会话ID
            method: 使用的检索方法
        """
        # 优先使用自适应处理器
        if self.enable_adaptive:
            try:
                await self.adaptive_processor.collect_result_feedback(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    method=method,
                    results=results,
                    user_interactions=user_interactions
                )

                logger.info(
                    f"自适应反馈收集完成: query={query[:30]}, "
                    f"method={method}, "
                    f"interactions={list(user_interactions.keys())}"
                )
                return

            except Exception as e:
                logger.error(f"自适应反馈收集失败: {e}")

        # 降级到基础反馈处理器
        if self.enable_feedback:
            try:
                await self.feedback_processor.collect_feedback(
                    query=query,
                    results=results,
                    user_interactions=user_interactions,
                    user_id=user_id,
                    session_id=session_id
                )

                logger.info(
                    f"反馈收集完成: query={query[:30]}, "
                    f"interactions={list(user_interactions.keys())}"
                )

            except Exception as e:
                logger.error(f"收集反馈失败: {e}")
        else:
            logger.info("反馈回路未启用，跳过收集")

    async def _execute_with_base_executor(
        self,
        plan,
        query: str,
        params: Dict
    ) -> ExecutionResult:
        """使用基础执行器执行"""
        # 这里调用实际的执行器
        # 根据你的实际实现调整
        try:
            # 假设 base_executor 有 execute_plan 方法
            result = await self.base_executor.execute_plan(plan)

            # 应用动态参数
            if "top_k" in params and hasattr(result, 'fused_results'):
                max_results = params["top_k"]
                if len(result.fused_results) > max_results:
                    result.fused_results = result.fused_results[:max_results]

            return result

        except Exception as e:
            logger.error(f"基础执行器调用失败: {e}")
            # 降级到模拟执行
            return await self._mock_execute(query, params)

    async def _mock_execute(
        self,
        query: str,
        params: Dict
    ) -> ExecutionResult:
        """模拟执行（用于测试）"""
        await asyncio.sleep(0.1)

        # 模拟结果
        mock_results = []
        top_k = params.get("top_k", 10)

        for i in range(top_k):
            mock_results.append({
                "content": f"这是查询'{query}'的第{i+1}个检索结果...",
                "metadata": {
                    "source": f"doc_{i+1}",
                    "score": 0.9 - i * 0.05
                }
            })

        return ExecutionResult(
            plan_id="mock",
            fused_results=mock_results,
            execution_time=0.1,
            metadata={"mock": True}
        )

    async def _apply_compression(
        self,
        query: str,
        documents: List[Dict],
        retrieval_level: str,
        dynamic_params: Dict
    ) -> Dict:
        """应用上下文压缩"""
        # 转换为Document格式
        doc_objects = []
        for doc in documents:
            doc_objects.append(Document(
                page_content=doc.get("content", ""),
                metadata=doc.get("metadata", {})
            ))

        # 使用动态压缩率
        compression_rate = dynamic_params.get("compression_rate", 0.5)

        # 更新压缩器配置
        if retrieval_level in ["enhanced", "deep_search"]:
            # 应用动态压缩率
            # 注意: 这里需要根据实际compressor实现调整
            pass

        # 执行压缩
        compression_result: CompressionResult = await self.compressor.compress(
            query=query,
            documents=doc_objects,
            retrieval_level=retrieval_level
        )

        # 转换回结果格式
        compressed_results = []
        for doc in compression_result.compressed_docs:
            compressed_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return {
            "compressed_docs": compressed_results,
            "original_count": compression_result.original_count,
            "compressed_count": compression_result.compressed_count,
            "compression_ratio": compression_result.compression_ratio,
            "tokens_saved": compression_result.tokens_saved,
            "compression_time": compression_result.compression_time
        }

    def get_feedback_insights(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """获取反馈洞察"""
        # 优先返回自适应洞察
        if self.enable_adaptive:
            return self.adaptive_processor.get_adaptive_insights(user_id, session_id)

        # 降级到基础反馈洞察
        if self.enable_feedback:
            return self.feedback_processor.get_insights(user_id, session_id)

        return {
            "feedback_enabled": self.enable_feedback,
            "adaptive_enabled": self.enable_adaptive
        }


# 全局实例
_feedback_enhanced_executor = None


def get_feedback_enhanced_executor(
    base_executor=None,
    enable_feedback: bool = True,
    enable_compression: bool = True,
    enable_adaptive: bool = False,
    llm=None
) -> FeedbackEnhancedExecutor:
    """获取反馈增强执行器实例"""
    global _feedback_enhanced_executor
    if _feedback_enhanced_executor is None:
        _feedback_enhanced_executor = FeedbackEnhancedExecutor(
            base_executor=base_executor,
            enable_feedback=enable_feedback,
            enable_compression=enable_compression,
            enable_adaptive=enable_adaptive,
            llm=llm
        )
    return _feedback_enhanced_executor


def reset_feedback_enhanced_executor():
    """重置全局实例"""
    global _feedback_enhanced_executor
    _feedback_enhanced_executor = None
