"""
增强的Agentic RAG执行器
集成上下文压缩功能
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from app.core.structured_logging import get_structured_logger
from .context_compression import (
    HierarchicalContextCompressor,
    get_compressor,
    Document,
    CompressionResult
)

logger = get_structured_logger(__name__)


@dataclass
class ExecutionResult:
    """执行结果"""
    plan_id: str
    fused_results: List[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedRAGExecutor:
    """
    增强的RAG执行器

    在原有执行器基础上集成上下文压缩:
    - 自动根据检索级别选择压缩策略
    - 减少传递给LLM的上下文长度
    - 提升生成质量和效率
    """

    def __init__(
        self,
        base_executor=None,
        enable_compression: bool = True,
        llm=None
    ):
        """
        初始化增强执行器

        Args:
            base_executor: 基础执行器实例
            enable_compression: 是否启用上下文压缩
            llm: LLM实例（用于L3压缩）
        """
        self.base_executor = base_executor
        self.enable_compression = enable_compression

        # 初始化压缩器
        self.compressor: Optional[HierarchicalContextCompressor] = None
        if enable_compression:
            try:
                self.compressor = get_compressor(llm=llm)
                logger.info("✅ 上下文压缩器已加载")
            except Exception as e:
                logger.warning(f"⚠ 上下文压缩器初始化失败: {e}")
                self.enable_compression = False

    async def execute_plan(
        self,
        plan,
        query: str,
        retrieval_level: str = "enhanced"
    ) -> ExecutionResult:
        """
        执行检索计划（带上下文压缩）

        Args:
            plan: 检索计划
            query: 用户查询
            retrieval_level: 检索级别 (fast/enhanced/deep_search)

        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        plan_id = getattr(plan, 'plan_id', 'unknown')

        logger.info(f"开始执行检索计划: {plan_id}, 级别={retrieval_level}")

        try:
            # 1. 执行基础检索
            if self.base_executor:
                execution_result = await self.base_executor.execute_plan(plan)
            else:
                # 模拟执行结果
                execution_result = await self._mock_execute_plan(plan)

            fused_results = execution_result.fused_results

            # 2. 上下文压缩（如果启用）
            if self.enable_compression and self.compressor and fused_results:
                # 决定是否需要压缩
                should_compress = self._should_compress(
                    retrieval_level,
                    len(fused_results),
                    self._estimate_total_tokens(fused_results)
                )

                if should_compress:
                    logger.info(
                        f"执行上下文压缩: {len(fused_results)}个文档, "
                        f"级别={retrieval_level}"
                    )

                    # 转换为Document格式
                    documents = self._convert_to_documents(fused_results)

                    # 执行压缩
                    compression_result = await self.compressor.compress(
                        query=query,
                        documents=documents,
                        retrieval_level=retrieval_level
                    )

                    # 转换回原始格式
                    compressed_results = self._convert_from_documents(
                        compression_result.compressed_docs
                    )

                    # 更新执行结果
                    fused_results = compressed_results

                    # 记录压缩信息
                    execution_result.metadata = execution_result.metadata or {}
                    execution_result.metadata["context_compression"] = {
                        "enabled": True,
                        "original_count": compression_result.original_count,
                        "compressed_count": compression_result.compressed_count,
                        "compression_ratio": compression_result.compression_ratio,
                        "tokens_saved": compression_result.tokens_saved,
                        "compression_time": compression_result.compression_time,
                        "stages": compression_result.metadata.get("stages", {})
                    }

                    logger.info(
                        f"上下文压缩完成: {compression_result.original_count} → "
                        f"{compression_result.compressed_count} 文档, "
                        f"节省 {compression_result.tokens_saved} tokens"
                    )

            # 3. 更新执行时间
            execution_time = time.time() - start_time
            execution_result.execution_time = execution_time

            logger.info(f"检索计划执行完成: {plan_id}, 耗时={execution_time:.2f}s")

            return execution_result

        except Exception as e:
            logger.error(f"检索计划执行失败: {plan_id}, 错误={e}")

            return ExecutionResult(
                plan_id=plan_id,
                fused_results=[],
                execution_time=time.time() - start_time,
                metadata={
                    "error": str(e),
                    "context_compression": {"enabled": False}
                }
            )

    def _should_compress(
        self,
        retrieval_level: str,
        doc_count: int,
        total_tokens: int
    ) -> bool:
        """
        判断是否需要压缩

        Args:
            retrieval_level: 检索级别
            doc_count: 文档数量
            total_tokens: 总token数

        Returns:
            是否需要压缩
        """
        # Fast模式：文档少或token少时不压缩
        if retrieval_level == "fast":
            return doc_count > 5 or total_tokens > 3000

        # Enhanced模式：总是压缩
        if retrieval_level == "enhanced":
            return doc_count > 3 or total_tokens > 2000

        # DeepSearch模式：总是压缩
        if retrieval_level == "deep_search":
            return True

        return False

    def _estimate_total_tokens(self, results: List[Dict[str, Any]]) -> int:
        """
        估算结果的总token数

        Args:
            results: 检索结果列表

        Returns:
            总token数估算
        """
        total_length = 0
        for result in results:
            content = result.get("content", "")
            total_length += len(content)

        # 粗略估算：中文1.5字符=1token
        return int(total_length / 1.5)

    def _convert_to_documents(self, results: List[Dict[str, Any]]) -> List[Document]:
        """
        将检索结果转换为Document格式

        Args:
            results: 检索结果列表

        Returns:
            Document列表
        """
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("source", ""),
                    "score": result.get("score", 0),
                    **result.get("metadata", {})
                }
            )
            documents.append(doc)

        return documents

    def _convert_from_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        将Document格式转换回检索结果

        Args:
            documents: Document列表

        Returns:
            检索结果列表
        """
        results = []
        for doc in documents:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            results.append(result)

        return results

    async def _mock_execute_plan(self, plan) -> ExecutionResult:
        """
        模拟执行检索计划（用于测试）

        Args:
            plan: 检索计划

        Returns:
            模拟的执行结果
        """
        # 模拟延迟
        await asyncio.sleep(0.1)

        # 返回模拟结果
        mock_results = [
            {
                "content": "这是模拟的检索结果内容...",
                "metadata": {"source": "mock_doc_1", "score": 0.9}
            }
        ]

        return ExecutionResult(
            plan_id=getattr(plan, 'plan_id', 'mock'),
            fused_results=mock_results,
            execution_time=0.1,
            metadata={"mock": True}
        )

    def update_compression_config(self, level: str, config: Dict[str, Any]):
        """
        更新压缩配置

        Args:
            level: 检索级别
            config: 配置更新
        """
        if self.compressor:
            self.compressor.update_config(level, config)
            logger.info(f"已更新{level}级别的压缩配置")


# 全局实例
_enhanced_executor_instance: Optional[EnhancedRAGExecutor] = None


def get_enhanced_executor(
    base_executor=None,
    enable_compression: bool = True,
    llm=None
) -> EnhancedRAGExecutor:
    """
    获取增强执行器实例

    Args:
        base_executor: 基础执行器
        enable_compression: 是否启用压缩
        llm: LLM实例

    Returns:
        EnhancedRAGExecutor实例
    """
    global _enhanced_executor_instance

    if _enhanced_executor_instance is None:
        _enhanced_executor_instance = EnhancedRAGExecutor(
            base_executor=base_executor,
            enable_compression=enable_compression,
            llm=llm
        )

    return _enhanced_executor_instance


def reset_enhanced_executor():
    """重置全局执行器实例"""
    global _enhanced_executor_instance
    _enhanced_executor_instance = None
