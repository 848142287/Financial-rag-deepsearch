"""
兼容性适配器
"""

import asyncio
from typing import List, Dict, Any, Optional
from .rag_toolkit import RAGToolkit
# DEPRECATED: Use ConsolidatedRAGService instead - from app.services.consolidated_rag_service import ConsolidatedRAGService RAGService
from app.core.logging import logger
from app.core.config import get_settings

class RAGCompatibilityAdapter:
    """
    RAG兼容性适配器

    确保新的LangChain 1.0+工具系统与现有RAG服务完全兼容
    """

    def __init__(
        self,
        enable_new_features: bool = True,
        fallback_to_legacy: bool = True,
        **toolkit_kwargs
    ):
        """
        初始化适配器

        Args:
            enable_new_features: 是否启用新功能特性
            fallback_to_legacy: 是否在新功能失败时回退到传统实现
            toolkit_kwargs: 传递给RAGToolkit的参数
        """
        self.settings = get_settings()
        self.enable_new_features = enable_new_features
        self.fallback_to_legacy = fallback_to_legacy

        # 初始化新工具包
        self.rag_toolkit = None
        if enable_new_features:
            try:
                self.rag_toolkit = RAGToolkit(**toolkit_kwargs)
                logger.info("RAGToolkit initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAGToolkit: {e}")
                if not fallback_to_legacy:
                    raise
                self.enable_new_features = False

        # 初始化传统RAG服务作为回退
        self.legacy_rag_service = None
        if fallback_to_legacy or not enable_new_features:
            try:
                self.legacy_rag_service = RAGService()
                logger.info("Legacy RAGService initialized as fallback")
            except Exception as e:
                logger.error(f"Failed to initialize legacy RAGService: {e}")
                if not enable_new_features:
                    raise

    async def retrieve_documents(
        self,
        query: str,
        knowledge_base_ids: Optional[List[int]] = None,
        retrieval_mode: str = "enhanced",
        top_k: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        文档检索（兼容接口）

        Args:
            query: 检索查询语句
            knowledge_base_ids: 知识库ID列表
            retrieval_mode: 检索模式
            top_k: 返回文档数量
            threshold: 相似度阈值
            filters: 过滤条件

        Returns:
            检索结果字典
        """
        if self.enable_new_features and self.rag_toolkit:
            try:
                # 使用新的工具系统
                result = await self.rag_toolkit.execute_tool(
                    "document_retrieval",
                    {
                        "query": query,
                        "knowledge_base_ids": knowledge_base_ids,
                        "retrieval_mode": retrieval_mode,
                        "top_k": top_k,
                        "threshold": threshold,
                        "filters": filters
                    }
                )

                # 转换为传统格式
                if result.get("success"):
                    return {
                        "success": True,
                        "documents": result.get("documents", []),
                        "query": result.get("query"),
                        "retrieval_mode": result.get("retrieval_mode"),
                        "total_results": result.get("total_results", 0),
                        "search_metadata": result.get("search_metadata", {})
                    }
                else:
                    raise Exception(result.get("error", "Unknown error"))

            except Exception as e:
                logger.error(f"New tool retrieval failed: {e}")
                if not self.fallback_to_legacy or not self.legacy_rag_service:
                    raise

        # 回退到传统实现
        if self.legacy_rag_service:
            try:
                result = await self.legacy_rag_service.retrieve_documents(
                    query=query,
                    knowledge_base_ids=knowledge_base_ids,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    threshold=threshold,
                    filters=filters or {}
                )
                logger.info("Used legacy RAG service for document retrieval")
                return result

            except Exception as e:
                logger.error(f"Legacy RAG retrieval failed: {e}")
                raise

        raise RuntimeError("No available retrieval implementation")

    async def generate_response(
        self,
        query: str,
        context: Optional[List[str]] = None,
        context_type: str = "retrieved",
        response_format: str = "text",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        生成响应（兼容接口）

        Args:
            query: 用户查询问题
            context: 上下文文档列表
            context_type: 上下文类型
            response_format: 响应格式
            max_tokens: 最大生成token数
            temperature: 生成温度
            include_citations: 是否包含引用

        Returns:
            生成结果字典
        """
        if self.enable_new_features and self.rag_toolkit:
            try:
                # 转换context为retrieved_docs格式
                retrieved_docs = None
                if context:
                    retrieved_docs = [
                        {"content": doc, "source": "legacy_context", "score": 1.0}
                        for doc in context
                    ]

                # 使用新的工具系统
                result = await self.rag_toolkit.execute_tool(
                    "rag_generation",
                    {
                        "query": query,
                        "retrieved_docs": retrieved_docs,
                        "context_type": context_type,
                        "response_format": response_format,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "include_citations": include_citations
                    }
                )

                # 转换为传统格式
                if result.get("success"):
                    return {
                        "success": True,
                        "answer": result.get("answer", ""),
                        "citations": result.get("citations", []),
                        "query": result.get("query"),
                        "context_type": result.get("context_type"),
                        "generation_metadata": result.get("generation_metadata", {})
                    }
                else:
                    raise Exception(result.get("error", "Unknown error"))

            except Exception as e:
                logger.error(f"New tool generation failed: {e}")
                if not self.fallback_to_legacy or not self.legacy_rag_service:
                    raise

        # 回退到传统实现
        if self.legacy_rag_service:
            try:
                result = await self.legacy_rag_service.generate_response(
                    query=query,
                    context=context or [],
                    context_type=context_type,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    include_citations=include_citations
                )
                logger.info("Used legacy RAG service for response generation")
                return result

            except Exception as e:
                logger.error(f"Legacy RAG generation failed: {e}")
                raise

        raise RuntimeError("No available generation implementation")

    async def multimodal_analysis(
        self,
        image_path: Optional[str] = None,
        document_path: Optional[str] = None,
        query: str = "",
        analysis_type: str = "description"
    ) -> Dict[str, Any]:
        """
        多模态分析（兼容接口）

        Args:
            image_path: 图像文件路径
            document_path: 文档文件路径
            query: 查询问题
            analysis_type: 分析类型

        Returns:
            分析结果字典
        """
        if self.enable_new_features and self.rag_toolkit:
            try:
                result = await self.rag_toolkit.execute_tool(
                    "multimodal_processing",
                    {
                        "image_path": image_path,
                        "document_path": document_path,
                        "query": query,
                        "analysis_type": analysis_type
                    }
                )

                if result.get("success"):
                    return {
                        "success": True,
                        "analysis": result.get("analysis", {}),
                        "analysis_type": result.get("analysis_type"),
                        "metadata": result.get("metadata", {})
                    }
                else:
                    raise Exception(result.get("error", "Unknown error"))

            except Exception as e:
                logger.error(f"New tool multimodal analysis failed: {e}")
                if not self.fallback_to_legacy:
                    raise

        # 多模态功能传统实现可能不存在，返回错误
        return {
            "success": False,
            "error": "Multimodal analysis not available in legacy mode",
            "analysis": {}
        }

    def get_toolkit_stats(self) -> Dict[str, Any]:
        """获取工具包统计信息"""
        if self.rag_toolkit:
            return self.rag_toolkit.get_middleware_stats()
        else:
            return {"toolkit_enabled": False}

    def get_compatibility_info(self) -> Dict[str, Any]:
        """获取兼容性信息"""
        return {
            "new_features_enabled": self.enable_new_features,
            "toolkit_available": self.rag_toolkit is not None,
            "legacy_fallback_enabled": self.fallback_to_legacy,
            "legacy_service_available": self.legacy_rag_service is not None,
            "available_tools": self.rag_toolkit.get_tool_names() if self.rag_toolkit else []
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "adapter_status": "healthy",
            "toolkit_status": "unknown",
            "legacy_status": "unknown"
        }

        # 检查新工具包
        if self.rag_toolkit:
            try:
                # 执行一个简单的工具测试
                result = await self.rag_toolkit.execute_tool(
                    "document_retrieval",
                    {"query": "test", "top_k": 1}
                )
                health_status["toolkit_status"] = "healthy" if result.get("success") else "unhealthy"
            except Exception as e:
                health_status["toolkit_status"] = f"error: {str(e)}"
        else:
            health_status["toolkit_status"] = "disabled"

        # 检查传统服务
        if self.legacy_rag_service:
            try:
                # 执行一个简单的传统服务测试
                result = await self.legacy_rag_service.retrieve_documents(
                    query="test", top_k=1
                )
                health_status["legacy_status"] = "healthy" if result.get("success") else "unhealthy"
            except Exception as e:
                health_status["legacy_status"] = f"error: {str(e)}"
        else:
            health_status["legacy_status"] = "disabled"

        return health_status