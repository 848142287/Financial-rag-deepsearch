"""
RAG生成Tool - 使用LangChain 1.0+ Tool接口封装
"""

import asyncio
from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
# DEPRECATED: Use ConsolidatedRAGService instead - from app.services.consolidated_rag_service import ConsolidatedRAGService RAGService
from app.core.config import get_settings
from app.core.logging import logger

class RAGGenerationInput(BaseModel):
    """RAG生成输入参数"""
    query: str = Field(description="用户查询问题")
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="已检索的文档列表，如果不提供将自动检索"
    )
    context_type: str = Field(
        default="retrieved",
        description="上下文类型: retrieved, full_document, none"
    )
    response_format: str = Field(
        default="text",
        description="响应格式: text, markdown, json"
    )
    max_tokens: int = Field(default=2048, description="最大生成token数")
    temperature: float = Field(default=0.1, description="生成温度")
    include_citations: bool = Field(default=True, description="是否包含引用")

class RAGGenerationTool(BaseTool):
    """
    RAG生成Tool

    基于LangChain Tool接口封装的RAG生成功能
    """
    name: str = "rag_generation"
    description: str = "基于检索到的文档内容生成回答，支持多种上下文类型和输出格式"
    args_schema: Type[BaseModel] = RAGGenerationInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        self.rag_service = None
        self._init_services()

    def _init_services(self):
        """初始化服务组件"""
        try:
            self.rag_service = RAGService()
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")

    def _run(
        self,
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        context_type: str = "retrieved",
        response_format: str = "text",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        同步执行RAG生成

        Args:
            query: 用户查询问题
            retrieved_docs: 已检索的文档列表
            context_type: 上下文类型
            response_format: 响应格式
            max_tokens: 最大生成token数
            temperature: 生成温度
            include_citations: 是否包含引用

        Returns:
            生成结果字典
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._arun(
                    query=query,
                    retrieved_docs=retrieved_docs,
                    context_type=context_type,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    include_citations=include_citations
                )
            )
        finally:
            loop.close()

    async def _arun(
        self,
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        context_type: str = "retrieved",
        response_format: str = "text",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        异步执行RAG生成

        Args:
            query: 用户查询问题
            retrieved_docs: 已检索的文档列表
            context_type: 上下文类型
            response_format: 响应格式
            max_tokens: 最大生成token数
            temperature: 生成温度
            include_citations: 是否包含引用

        Returns:
            生成结果字典
        """
        if not self.rag_service:
            return {
                "success": False,
                "error": "RAG service not initialized",
                "answer": "",
                "citations": []
            }

        try:
            # 如果没有提供检索文档，自动检索
            if not retrieved_docs and context_type != "none":
                retrieval_result = await self.rag_service.retrieve_documents(
                    query=query,
                    retrieval_mode="enhanced",
                    top_k=10
                )
                retrieved_docs = retrieval_result.get("documents", [])

            # 构建生成参数
            generation_params = {
                "query": query,
                "context_type": context_type,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "include_citations": include_citations
            }

            if retrieved_docs:
                generation_params["context"] = [
                    doc.get("content", "") for doc in retrieved_docs
                ]

            # 调用现有RAG服务进行生成
            result = await self.rag_service.generate_response(
                **generation_params
            )

            # 处理引用信息
            citations = []
            if include_citations and retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    citations.append({
                        "index": i,
                        "source": doc.get("source", ""),
                        "score": doc.get("score", 0.0),
                        "content_preview": doc.get("content", "")[:100] + "...",
                        "metadata": doc.get("metadata", {})
                    })

            return {
                "success": True,
                "answer": result.get("answer", ""),
                "citations": citations,
                "query": query,
                "context_type": context_type,
                "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0,
                "generation_metadata": {
                    "response_format": response_format,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            }

        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "citations": [],
                "query": query
            }

    def get_tool_description(self) -> str:
        """获取工具详细描述"""
        return """
        RAG生成工具，支持多种上下文类型：

        1. retrieved - 使用检索到的相关文档
        2. full_document - 使用完整文档内容
        3. none - 不使用任何上下文，直接回答

        支持多种输出格式：纯文本、Markdown、JSON
        可以自动检索文档或使用已提供的文档。
        包含引用功能，可追溯答案来源。
        """