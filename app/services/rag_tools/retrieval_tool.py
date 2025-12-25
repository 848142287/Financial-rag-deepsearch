"""
文档检索Tool - 使用LangChain 1.0+ Tool接口封装
"""

import asyncio
from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
# DEPRECATED: Use ConsolidatedRAGService instead - from app.services.consolidated_rag_service import ConsolidatedRAGService RAGService
from app.services.consolidated_document_service import ConsolidatedDocumentService as DocumentProcessor
from app.core.config import get_settings
from app.core.logging import logger

class DocumentRetrievalInput(BaseModel):
    """文档检索输入参数"""
    query: str = Field(description="检索查询语句")
    knowledge_base_ids: Optional[List[int]] = Field(default=None, description="知识库ID列表")
    retrieval_mode: str = Field(default="enhanced", description="检索模式: simple, enhanced, deep_search")
    top_k: int = Field(default=10, description="返回文档数量")
    threshold: float = Field(default=0.7, description="相似度阈值")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="过滤条件")

class DocumentRetrievalTool(BaseTool):
    """
    文档检索Tool

    基于LangChain Tool接口封装的文档检索功能，支持多种检索模式
    """
    name: str = "document_retrieval"
    description: str = "从知识库中检索相关文档片段，支持简单检索、增强检索和深度搜索模式"
    args_schema: Type[BaseModel] = DocumentRetrievalInput

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
        knowledge_base_ids: Optional[List[int]] = None,
        retrieval_mode: str = "enhanced",
        top_k: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        同步执行文档检索

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
        # 异步转同步执行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._arun(
                    query=query,
                    knowledge_base_ids=knowledge_base_ids,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    threshold=threshold,
                    filters=filters
                )
            )
        finally:
            loop.close()

    async def _arun(
        self,
        query: str,
        knowledge_base_ids: Optional[List[int]] = None,
        retrieval_mode: str = "enhanced",
        top_k: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        异步执行文档检索

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
        if not self.rag_service:
            return {
                "success": False,
                "error": "RAG service not initialized",
                "documents": []
            }

        try:
            # 调用现有RAG服务进行检索
            result = await self.rag_service.retrieve_documents(
                query=query,
                knowledge_base_ids=knowledge_base_ids,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                threshold=threshold,
                filters=filters or {}
            )

            # 转换为Tool标准输出格式
            documents = []
            for doc in result.get("documents", []):
                documents.append({
                    "content": doc.get("content", ""),
                    "source": doc.get("source", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                })

            return {
                "success": True,
                "documents": documents,
                "query": query,
                "retrieval_mode": retrieval_mode,
                "total_results": len(documents),
                "search_metadata": {
                    "threshold": threshold,
                    "top_k": top_k,
                    "filters": filters
                }
            }

        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "query": query
            }

    def get_tool_description(self) -> str:
        """获取工具详细描述"""
        return """
        文档检索工具，支持多种检索模式：

        1. simple - 简单向量检索
        2. enhanced - 增强检索（向量+知识图谱）
        3. deep_search - 深度搜索（多轮推理）

        可以指定知识库、返回数量、相似度阈值等参数。
        适用于从大量文档中快速定位相关内容。
        """