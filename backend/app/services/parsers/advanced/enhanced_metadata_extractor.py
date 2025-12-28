"""
增强的元数据提取器 - LLM + LangChain 结构化提取

功能特点：
1. 使用现有系统的 qwen-vl-plus 模型
2. LangChain 结构化输出（Pydantic 模型）
3. 提取表格、关键点、摘要、主题等结构化信息
4. 支持批量并发处理

集成来源:
- 03_DataAnalysis_main/backend/core/analysis/data_analyzer.py
- backend/app/services/parsers/advanced/enhanced_llm_multimodal_extractor.py
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.services.llm_service import llm_service
from app.core.config import settings


# ==================== Pydantic 数据模型 ====================

class ExtractedTable(BaseModel):
    """提取的表格"""
    title: str = Field(description="表格标题")
    headers: List[str] = Field(description="表头列表")
    rows: List[List[str]] = Field(description="表格数据行")
    summary: str = Field(description="表格内容摘要")


class KeyPoint(BaseModel):
    """关键点"""
    point: str = Field(description="关键点内容")
    importance: str = Field(description="重要性: high/medium/low", default="medium")
    context: str = Field(description="上下文信息", default="")


class ExtractedTopic(BaseModel):
    """提取的主题"""
    topic: str = Field(description="主题名称")
    relevance_score: float = Field(description="相关性评分 0-1", ge=0, le=1)
    keywords: List[str] = Field(description="关键词列表", default_factory=list)


class ChunkMetadataExtraction(BaseModel):
    """Chunk元数据提取结果"""
    chunk_id: str = Field(description="Chunk ID")
    summary: str = Field(description="内容摘要 (2-3句话)")
    key_points: List[KeyPoint] = Field(description="关键点列表", default_factory=list)
    tables: List[ExtractedTable] = Field(description="表格列表", default_factory=list)
    topics: List[ExtractedTopic] = Field(description="主题列表", default_factory=list)
    sentiment: str = Field(description="情感倾向: positive/neutral/negative", default="neutral")
    language: str = Field(description="语言: zh/en/mixed", default="zh")


class DocumentMetadataExtraction(BaseModel):
    """文档级元数据提取结果"""
    document_title: str = Field(description="文档标题")
    overall_summary: str = Field(description="整体摘要")
    main_topics: List[str] = Field(description="主要主题", default_factory=list)
    key_statistics: List[Dict[str, Any]] = Field(description="关键统计数据", default_factory=list)
    document_type: str = Field(description="文档类型: report/article/paper/etc", default="unknown")
    target_audience: str = Field(description="目标受众", default="")
    creation_date: Optional[str] = Field(description="创建日期", default=None)


# ==================== 提取器配置 ====================

@dataclass
class ExtractionConfig:
    """提取配置"""
    # 模型配置
    model: str = field(default_factory=lambda: settings.qwen_multimodal_model)
    temperature: float = 0.1
    max_tokens: int = 2000

    # 批处理配置
    batch_size: int = 5
    max_workers: int = 10
    timeout: int = 60

    # 提取选项
    extract_tables: bool = True
    extract_key_points: bool = True
    extract_topics: bool = True
    extract_sentiment: bool = True

    # 提示词模板
    summary_prompt_template: str = """
请分析以下文本内容，提取结构化信息。

文本内容：
{text}

请提取以下信息：
1. 内容摘要：用2-3句话概括主要内容
2. 关键点：列出3-5个关键点，并标注重要性
3. 表格：如果文本中包含表格数据，提取表格信息
4. 主题：识别3-5个主要主题，并评估相关性
5. 情感倾向：判断文本的情感倾向

{format_instructions}

请确保提取的信息准确、完整、相关。
"""

    document_prompt_template: str = """
请分析以下文档，提取文档级的元数据信息。

文档内容（前5000字）：
{text}

请提取以下信息：
1. 文档标题：识别文档的主要标题
2. 整体摘要：概括整个文档的主要内容
3. 主要主题：列出文档涉及的主要主题
4. 关键统计数据：提取文档中的重要数字和统计数据
5. 文档类型：判断文档的类型（研究报告、新闻文章、学术论文等）
6. 目标受众：判断文档的目标受众

{format_instructions}

请确保提取的信息准确、全面。
"""


# ==================== 增强元数据提取器 ====================

class EnhancedMetadataExtractor:
    """
    增强的元数据提取器

    结合 LLM 和 LangChain 进行结构化元数据提取
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        初始化提取器

        Args:
            config: 提取配置
        """
        self.config = config or ExtractionConfig()
        self._setup_parsers()

    def _setup_parsers(self):
        """设置输出解析器"""
        self.chunk_parser = PydanticOutputParser(pydantic_object=ChunkMetadataExtraction)
        self.document_parser = PydanticOutputParser(pydantic_object=DocumentMetadataExtraction)

    async def extract_chunk_metadata(
        self,
        chunk: Document,
        chunk_id: Optional[str] = None
    ) -> ChunkMetadataExtraction:
        """
        提取单个chunk的元数据

        Args:
            chunk: 输入的Document
            chunk_id: Chunk ID（可选）

        Returns:
            ChunkMetadataExtraction 对象
        """
        chunk_id = chunk_id or chunk.metadata.get('chunk_id', 'unknown')

        # 构建提示词
        prompt = ChatPromptTemplate.from_template(
            self.config.summary_prompt_template
        )

        # 限制文本长度
        text = chunk.page_content[:5000]

        # 获取格式指令
        format_instructions = self.chunk_parser.get_format_instructions()

        # 构建输入
        input_data = {
            "text": text,
            "format_instructions": format_instructions
        }

        try:
            # 调用LLM
            chain = prompt | llm_service.get_llm() | self.chunk_parser
            result = await chain.ainvoke(input_data)

            # 更新chunk_id
            result.chunk_id = chunk_id

            return result

        except Exception as e:
            # 返回默认结果
            return ChunkMetadataExtraction(
                chunk_id=chunk_id,
                summary=f"提取失败: {str(e)}",
                key_points=[],
                tables=[],
                topics=[],
                sentiment="neutral",
                language="unknown"
            )

    async def extract_document_metadata(
        self,
        document: Document
    ) -> DocumentMetadataExtraction:
        """
        提取文档级元数据

        Args:
            document: 输入的Document

        Returns:
            DocumentMetadataExtraction 对象
        """
        # 构建提示词
        prompt = ChatPromptTemplate.from_template(
            self.config.document_prompt_template
        )

        # 限制文本长度
        text = document.page_content[:5000]

        # 获取格式指令
        format_instructions = self.document_parser.get_format_instructions()

        # 构建输入
        input_data = {
            "text": text,
            "format_instructions": format_instructions
        }

        try:
            # 调用LLM
            chain = prompt | llm_service.get_llm() | self.document_parser
            result = await chain.ainvoke(input_data)

            return result

        except Exception as e:
            # 返回默认结果
            return DocumentMetadataExtraction(
                document_title=document.metadata.get('filename', 'Unknown'),
                overall_summary=f"提取失败: {str(e)}",
                main_topics=[],
                key_statistics=[],
                document_type="unknown"
            )

    async def extract_batch_chunks_metadata(
        self,
        chunks: List[Document],
        show_progress: bool = False
    ) -> List[ChunkMetadataExtraction]:
        """
        批量提取chunk元数据（并发）

        Args:
            chunks: Document列表
            show_progress: 是否显示进度

        Returns:
            ChunkMetadataExtraction 列表
        """
        results = []
        total = len(chunks)

        # 使用 semaphore 控制并发数
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def extract_with_semaphore(chunk: Document, index: int):
            async with semaphore:
                result = await self.extract_chunk_metadata(chunk, chunk.metadata.get('chunk_id', f'chunk_{index}'))
                if show_progress and index % 10 == 0:
                    print(f"进度: {index}/{total} chunks 已处理")
                return result

        tasks = [extract_with_semaphore(chunk, i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ChunkMetadataExtraction(
                    chunk_id=chunks[i].metadata.get('chunk_id', f'chunk_{i}'),
                    summary=f"处理失败: {str(result)}",
                    key_points=[],
                    tables=[],
                    topics=[]
                ))
            else:
                final_results.append(result)

        return final_results

    def extract_chunk_metadata_sync(
        self,
        chunk: Document,
        chunk_id: Optional[str] = None
    ) -> ChunkMetadataExtraction:
        """
        同步方式提取chunk元数据

        Args:
            chunk: 输入的Document
            chunk_id: Chunk ID

        Returns:
            ChunkMetadataExtraction 对象
        """
        # 在线程池中运行异步方法
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.extract_chunk_metadata(chunk, chunk_id)
            )
            return result
        finally:
            loop.close()

    def extract_batch_chunks_metadata_sync(
        self,
        chunks: List[Document],
        show_progress: bool = False
    ) -> List[ChunkMetadataExtraction]:
        """
        同步批量提取chunk元数据（使用线程池）

        Args:
            chunks: Document列表
            show_progress: 是否显示进度

        Returns:
            ChunkMetadataExtraction 列表
        """
        results = []
        total = len(chunks)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交所有任务
            future_to_chunk = {
                executor.submit(
                    self.extract_chunk_metadata_sync,
                    chunk,
                    chunk.metadata.get('chunk_id', f'chunk_{i}')
                ): (i, chunk)
                for i, chunk in enumerate(chunks)
            }

            # 收集结果
            for future in as_completed(future_to_chunk):
                index, chunk = future_to_chunk[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append((index, result))

                    if show_progress and len(results) % 10 == 0:
                        print(f"进度: {len(results)}/{total} chunks 已处理")

                except Exception as e:
                    # 创建错误结果
                    error_result = ChunkMetadataExtraction(
                        chunk_id=chunk.metadata.get('chunk_id', f'chunk_{index}'),
                        summary=f"处理失败: {str(e)}",
                        key_points=[],
                        tables=[],
                        topics=[]
                    )
                    results.append((index, error_result))

        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]


# 创建全局实例
enhanced_metadata_extractor = EnhancedMetadataExtractor()


# 便捷函数
async def extract_chunk_metadata_async(
    chunk: Document,
    chunk_id: Optional[str] = None
) -> ChunkMetadataExtraction:
    """
    异步提取chunk元数据

    Args:
        chunk: 输入的Document
        chunk_id: Chunk ID

    Returns:
        ChunkMetadataExtraction 对象
    """
    return await enhanced_metadata_extractor.extract_chunk_metadata(chunk, chunk_id)


async def extract_batch_chunks_metadata_async(
    chunks: List[Document],
    show_progress: bool = False
) -> List[ChunkMetadataExtraction]:
    """
    异步批量提取chunk元数据

    Args:
        chunks: Document列表
        show_progress: 是否显示进度

    Returns:
        ChunkMetadataExtraction 列表
    """
    return await enhanced_metadata_extractor.extract_batch_chunks_metadata(chunks, show_progress)


def extract_chunk_metadata(
    chunk: Document,
    chunk_id: Optional[str] = None
) -> ChunkMetadataExtraction:
    """
    同步提取chunk元数据

    Args:
        chunk: 输入的Document
        chunk_id: Chunk ID

    Returns:
        ChunkMetadataExtraction 对象
    """
    return enhanced_metadata_extractor.extract_chunk_metadata_sync(chunk, chunk_id)


def extract_batch_chunks_metadata(
    chunks: List[Document],
    show_progress: bool = False
) -> List[ChunkMetadataExtraction]:
    """
    同步批量提取chunk元数据

    Args:
        chunks: Document列表
        show_progress: 是否显示进度

    Returns:
        ChunkMetadataExtraction 列表
    """
    return enhanced_metadata_extractor.extract_batch_chunks_metadata_sync(chunks, show_progress)
