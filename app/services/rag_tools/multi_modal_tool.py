"""
多模态处理Tool - 使用LangChain 1.0+ Tool接口封装
"""

import asyncio
from typing import List, Dict, Any, Optional, Type, Union
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from app.services.multimodal.multimodal_processor import MultiModalProcessor
from app.core.config import get_settings
from app.core.logging import logger

class MultiModalInput(BaseModel):
    """多模态处理输入参数"""
    image_path: Optional[str] = Field(default=None, description="图像文件路径")
    document_path: Optional[str] = Field(default=None, description="文档文件路径")
    query: str = Field(description="关于图像或文档的查询问题")
    analysis_type: str = Field(
        default="description",
        description="分析类型: description, ocr, chart_analysis, table_extraction"
    )
    include_context: bool = Field(default=True, description="是否包含文档上下文")

class MultiModalProcessingTool(BaseTool):
    """
    多模态处理Tool

    基于LangChain Tool接口封装的多模态内容分析功能
    """
    name: str = "multimodal_processing"
    description: str = "分析图像和文档内容，支持OCR、图表分析、表格提取等多模态任务"
    args_schema: Type[BaseModel] = MultiModalInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        self.multimodal_processor = None
        self._init_services()

    def _init_services(self):
        """初始化服务组件"""
        try:
            self.multimodal_processor = MultiModalProcessor()
        except Exception as e:
            logger.error(f"Failed to initialize MultiModalProcessor: {e}")

    def _run(
        self,
        image_path: Optional[str] = None,
        document_path: Optional[str] = None,
        query: str = "",
        analysis_type: str = "description",
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        同步执行多模态处理

        Args:
            image_path: 图像文件路径
            document_path: 文档文件路径
            query: 查询问题
            analysis_type: 分析类型
            include_context: 是否包含上下文

        Returns:
            分析结果字典
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._arun(
                    image_path=image_path,
                    document_path=document_path,
                    query=query,
                    analysis_type=analysis_type,
                    include_context=include_context
                )
            )
        finally:
            loop.close()

    async def _arun(
        self,
        image_path: Optional[str] = None,
        document_path: Optional[str] = None,
        query: str = "",
        analysis_type: str = "description",
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        异步执行多模态处理

        Args:
            image_path: 图像文件路径
            document_path: 文档文件路径
            query: 查询问题
            analysis_type: 分析类型
            include_context: 是否包含上下文

        Returns:
            分析结果字典
        """
        if not self.multimodal_processor:
            return {
                "success": False,
                "error": "MultiModal processor not initialized",
                "analysis": {}
            }

        try:
            # 根据分析类型调用相应功能
            if analysis_type == "description" and image_path:
                result = await self._analyze_image_description(
                    image_path, query, include_context
                )
            elif analysis_type == "ocr" and image_path:
                result = await self._extract_text_from_image(image_path)
            elif analysis_type == "chart_analysis" and image_path:
                result = await self._analyze_chart(image_path, query)
            elif analysis_type == "table_extraction" and document_path:
                result = await self._extract_tables(document_path)
            else:
                result = await self._general_multimodal_analysis(
                    image_path, document_path, query
                )

            return {
                "success": True,
                "analysis": result,
                "analysis_type": analysis_type,
                "query": query,
                "metadata": {
                    "image_path": image_path,
                    "document_path": document_path
                }
            }

        except Exception as e:
            logger.error(f"MultiModal processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {},
                "query": query
            }

    async def _analyze_image_description(
        self,
        image_path: str,
        query: str,
        include_context: bool
    ) -> Dict[str, Any]:
        """分析图像描述"""
        result = await self.multimodal_processor.describe_image(
            image_path=image_path,
            query=query,
            include_context=include_context
        )
        return {
            "description": result.get("description", ""),
            "objects": result.get("objects", []),
            "confidence": result.get("confidence", 0.0),
            "detailed_analysis": result.get("detailed_analysis", "")
        }

    async def _extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """从图像提取文字"""
        result = await self.multimodal_processor.ocr_image(image_path)
        return {
            "extracted_text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "text_blocks": result.get("text_blocks", []),
            "languages": result.get("languages", [])
        }

    async def _analyze_chart(self, image_path: str, query: str) -> Dict[str, Any]:
        """分析图表"""
        result = await self.multimodal_processor.analyze_chart(
            image_path=image_path,
            query=query
        )
        return {
            "chart_type": result.get("chart_type", ""),
            "data_points": result.get("data_points", []),
            "trends": result.get("trends", []),
            "insights": result.get("insights", ""),
            "summary": result.get("summary", "")
        }

    async def _extract_tables(self, document_path: str) -> Dict[str, Any]:
        """提取表格"""
        result = await self.multimodal_processor.extract_tables(
            document_path=document_path
        )
        return {
            "tables": result.get("tables", []),
            "table_count": len(result.get("tables", [])),
            "summary": result.get("summary", "")
        }

    async def _general_multimodal_analysis(
        self,
        image_path: Optional[str],
        document_path: Optional[str],
        query: str
    ) -> Dict[str, Any]:
        """通用多模态分析"""
        result = await self.multimodal_processor.analyze_content(
            image_path=image_path,
            document_path=document_path,
            query=query
        )
        return {
            "analysis": result.get("analysis", ""),
            "modality": result.get("modality", ""),
            "confidence": result.get("confidence", 0.0),
            "extracted_info": result.get("extracted_info", {})
        }

    def get_tool_description(self) -> str:
        """获取工具详细描述"""
        return """
        多模态处理工具，支持多种内容分析：

        1. description - 图像描述和分析
        2. ocr - 图像文字识别和提取
        3. chart_analysis - 图表分析和数据提取
        4. table_extraction - 表格提取和结构化

        可以处理图片、PDF、Word等多种格式文件。
        支持图像描述、OCR识别、图表分析等高级功能。
        适用于文档理解、数据提取、内容分析等场景。
        """