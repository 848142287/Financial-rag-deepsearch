"""
增强的并发文档分析服务 - 完整的文档处理 Pipeline

功能特点：
1. 统一的 PDF 提取 + 高级 Markdown 分割 → 精准文档解析
2. LLM 多模态提取 + LangChain 结构化提取 → 丰富元数据
3. 优化批量嵌入 + 并发文档分析 → 极速处理
4. 完整的端到端文档处理流程

集成来源:
- backend/app/services/parsers/advanced/unified_pdf_extractor.py
- backend/app/services/parsers/advanced/enhanced_markdown_splitter.py
- backend/app/services/parsers/advanced/enhanced_metadata_extractor.py
- backend/app/services/vector_stores/optimized_batch_embedding.py
- 03_DataAnalysis_main/backend/core/analysis/data_analyzer.py
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from app.services.parsers.advanced.unified_pdf_extractor import unified_pdf_extractor
from app.services.parsers.advanced.enhanced_markdown_splitter import (
    enhanced_markdown_splitter,
    SplitConfig
)
from app.services.parsers.advanced.enhanced_metadata_extractor import (
    enhanced_metadata_extractor,
    ChunkMetadataExtraction,
    DocumentMetadataExtraction
)
from app.services.vector_stores.optimized_batch_embedding import (
    embed_batch_documents,
    embed_batch_documents_async
)


# ==================== 分析结果模型 ====================

@dataclass
class ChunkAnalysisResult:
    """Chunk分析结果"""
    chunk: Document
    metadata: ChunkMetadataExtraction
    embedding: Optional[List[float]] = None
    processing_time: float = 0.0


@dataclass
class DocumentAnalysisResult:
    """文档分析结果"""
    # 基础信息
    document_id: str
    filename: str
    file_path: str
    status: str  # success, partial_success, failed

    # PDF 提取结果
    raw_text: str
    extraction_method: str  # fast, accurate, auto
    extraction_time: float

    # 分割结果
    chunks: List[Document]
    chunk_count: int
    splitting_time: float

    # 元数据提取结果
    document_metadata: Optional[DocumentMetadataExtraction] = None
    chunks_metadata: List[ChunkMetadataExtraction] = field(default_factory=list)
    metadata_extraction_time: float = 0.0

    # 向量嵌入结果
    embeddings: List[List[float]] = field(default_factory=list)
    embedding_time: float = 0.0

    # 详细分析结果
    chunk_analyses: List[ChunkAnalysisResult] = field(default_factory=list)

    # 统计信息
    total_processing_time: float = 0.0
    total_tokens_used: int = 0

    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ==================== 分析器配置 ====================

@dataclass
class AnalyzerConfig:
    """分析器配置"""
    # PDF 提取配置
    extraction_mode: str = "auto"  # fast, accurate, auto
    force_extraction_mode: Optional[str] = None

    # 分割配置
    split_config: Optional[SplitConfig] = None

    # 元数据提取配置
    extract_metadata: bool = True
    extract_embeddings: bool = True
    max_workers: int = 10

    # 批处理配置
    batch_size: int = 20

    # 性能优化
    enable_parallel_processing: bool = True
    enable_caching: bool = True


# ==================== 增强文档分析器 ====================

class EnhancedDocumentAnalyzer:
    """
    增强的文档分析器

    提供完整的端到端文档处理pipeline：
    1. PDF 提取
    2. Markdown 分割
    3. 元数据提取
    4. 向量嵌入
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        初始化分析器

        Args:
            config: 分析器配置
        """
        self.config = config or AnalyzerConfig()
        self.splitter = enhanced_markdown_splitter
        self.metadata_extractor = enhanced_metadata_extractor

        # 更新分割器配置
        if self.config.split_config:
            self.splitter = enhanced_markdown_splitter.__class__(self.config.split_config)

    async def analyze_document(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
        extract_embeddings: bool = True
    ) -> DocumentAnalysisResult:
        """
        分析单个文档（完整pipeline）

        Args:
            file_path: 文件路径
            original_filename: 原始文件名
            extract_embeddings: 是否提取向量嵌入

        Returns:
            DocumentAnalysisResult 对象
        """
        start_time = time.time()
        original_filename = original_filename or Path(file_path).name
        document_id = f"doc_{original_filename}_{int(start_time)}"

        try:
            # 步骤1: PDF 提取
            extraction_start = time.time()
            extraction_result = await self._extract_pdf(file_path, original_filename)
            extraction_time = time.time() - extraction_start

            # 步骤2: Markdown 分割
            splitting_start = time.time()
            chunks = self._split_markdown(extraction_result['markdown_content'], original_filename)
            splitting_time = time.time() - splitting_start

            # 步骤3: 元数据提取
            metadata_time = 0.0
            document_metadata = None
            chunks_metadata = []

            if self.config.extract_metadata:
                metadata_start = time.time()

                # 并发提取chunk元数据和文档元数据
                document_metadata, chunks_metadata = await asyncio.gather(
                    self.metadata_extractor.extract_document_metadata(
                        Document(page_content=extraction_result['markdown_content'])
                    ),
                    self.metadata_extractor.extract_batch_chunks_metadata(
                        chunks[:50],  # 限制前50个chunk以避免过长处理时间
                        show_progress=False
                    ) if len(chunks) > 0 else asyncio.sleep(0)
                )

                metadata_time = time.time() - metadata_start

            # 步骤4: 向量嵌入
            embeddings = []
            embedding_time = 0.0

            if extract_embeddings and self.config.extract_embeddings:
                embedding_start = time.time()
                chunk_texts = [chunk.page_content for chunk in chunks]
                embeddings = await embed_batch_documents_async(
                    texts=chunk_texts,
                    chunk_size=self.config.batch_size,
                    show_progress=False
                )
                embedding_time = time.time() - embedding_start

            # 创建chunk分析结果
            chunk_analyses = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunks_metadata[i] if i < len(chunks_metadata) else None
                chunk_embedding = embeddings[i] if i < len(embeddings) else None

                chunk_analyses.append(
                    ChunkAnalysisResult(
                        chunk=chunk,
                        metadata=chunk_metadata,
                        embedding=chunk_embedding
                    )
                )

            # 构建结果
            total_time = time.time() - start_time

            return DocumentAnalysisResult(
                document_id=document_id,
                filename=original_filename,
                file_path=file_path,
                status="success",
                raw_text=extraction_result['markdown_content'],
                extraction_method=extraction_result.get('method', 'auto'),
                extraction_time=extraction_time,
                chunks=chunks,
                chunk_count=len(chunks),
                splitting_time=splitting_time,
                document_metadata=document_metadata,
                chunks_metadata=chunks_metadata,
                metadata_extraction_time=metadata_time,
                embeddings=embeddings,
                embedding_time=embedding_time,
                chunk_analyses=chunk_analyses,
                total_processing_time=total_time
            )

        except Exception as e:
            # 返回失败结果
            total_time = time.time() - start_time
            return DocumentAnalysisResult(
                document_id=document_id,
                filename=original_filename,
                file_path=file_path,
                status="failed",
                raw_text="",
                extraction_method="none",
                extraction_time=0.0,
                chunks=[],
                chunk_count=0,
                splitting_time=0.0,
                total_processing_time=total_time
            )

    async def _extract_pdf(
        self,
        file_path: str,
        original_filename: str
    ) -> Dict[str, Any]:
        """
        提取PDF内容

        Args:
            file_path: 文件路径
            original_filename: 原始文件名

        Returns:
            提取结果字典
        """
        if self.config.extraction_mode == "fast":
            result = await unified_pdf_extractor.extract_fast(file_path, original_filename)
            return {
                'markdown_content': result.markdown_content,
                'method': 'fast'
            }
        elif self.config.extraction_mode == "accurate":
            result = await unified_pdf_extractor.extract_accurate(file_path, original_filename)
            return {
                'markdown_content': result.markdown_content,
                'method': 'accurate'
            }
        else:  # auto
            result = await unified_pdf_extractor.extract_with_auto_mode(
                file_path,
                original_filename,
                force_mode=self.config.force_extraction_mode
            )
            return {
                'markdown_content': result.markdown_content,
                'method': result.get('method', 'auto')
            }

    def _split_markdown(
        self,
        markdown_content: str,
        original_filename: str
    ) -> List[Document]:
        """
        分割Markdown内容

        Args:
            markdown_content: Markdown内容
            original_filename: 原始文件名

        Returns:
            Document列表
        """
        base_metadata = {
            'source': original_filename,
            'filename': original_filename
        }
        return self.splitter.split_text(markdown_content, base_metadata)

    async def analyze_batch_documents(
        self,
        file_paths: List[str],
        show_progress: bool = True
    ) -> List[DocumentAnalysisResult]:
        """
        批量分析文档（并发）

        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度

        Returns:
            DocumentAnalysisResult 列表
        """
        results = []
        total = len(file_paths)

        if not self.config.enable_parallel_processing:
            # 顺序处理
            for i, file_path in enumerate(file_paths):
                result = await self.analyze_document(file_path)
                results.append(result)

                if show_progress:
                    print(f"进度: {i+1}/{total} 文档已处理")

        else:
            # 并发处理（使用semaphore控制并发数）
            semaphore = asyncio.Semaphore(self.config.max_workers)

            async def analyze_with_semaphore(file_path: str, index: int):
                async with semaphore:
                    result = await self.analyze_document(file_path)
                    if show_progress:
                        print(f"进度: {index+1}/{total} 文档已处理")
                    return result

            tasks = [analyze_with_semaphore(fp, i) for i, fp in enumerate(file_paths)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常结果
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # 创建失败结果
                    final_results.append(DocumentAnalysisResult(
                        document_id=f"failed_{i}",
                        filename=Path(file_paths[i]).name,
                        file_path=file_paths[i],
                        status="failed",
                        raw_text="",
                        extraction_method="none",
                        extraction_time=0.0,
                        chunks=[],
                        chunk_count=0,
                        splitting_time=0.0
                    ))
                else:
                    final_results.append(result)

            results = final_results

        return results

    def analyze_document_sync(
        self,
        file_path: str,
        original_filename: Optional[str] = None
    ) -> DocumentAnalysisResult:
        """
        同步方式分析文档

        Args:
            file_path: 文件路径
            original_filename: 原始文件名

        Returns:
            DocumentAnalysisResult 对象
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.analyze_document(file_path, original_filename)
            )
            return result
        finally:
            loop.close()

    def analyze_batch_documents_sync(
        self,
        file_paths: List[str],
        show_progress: bool = True
    ) -> List[DocumentAnalysisResult]:
        """
        同步批量分析文档（使用线程池）

        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度

        Returns:
            DocumentAnalysisResult 列表
        """
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self.analyze_document_sync, file_path)
                for file_path in file_paths
            ]

            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    results.append(result)

                    if show_progress:
                        print(f"进度: {i+1}/{len(file_paths)} 文档已处理")

                except Exception as e:
                    # 创建失败结果
                    results.append(DocumentAnalysisResult(
                        document_id=f"failed_{i}",
                        filename=Path(file_paths[i]).name,
                        file_path=file_paths[i],
                        status="failed",
                        raw_text="",
                        extraction_method="none",
                        extraction_time=0.0,
                        chunks=[],
                        chunk_count=0,
                        splitting_time=0.0
                    ))

            return results


# 创建全局实例
enhanced_document_analyzer = EnhancedDocumentAnalyzer()


# 便捷函数
async def analyze_document_async(
    file_path: str,
    original_filename: Optional[str] = None,
    extract_embeddings: bool = True
) -> DocumentAnalysisResult:
    """
    异步分析文档

    Args:
        file_path: 文件路径
        original_filename: 原始文件名
        extract_embeddings: 是否提取向量嵌入

    Returns:
        DocumentAnalysisResult 对象
    """
    return await enhanced_document_analyzer.analyze_document(
        file_path,
        original_filename,
        extract_embeddings
    )


async def analyze_batch_documents_async(
    file_paths: List[str],
    show_progress: bool = True
) -> List[DocumentAnalysisResult]:
    """
    异步批量分析文档

    Args:
        file_paths: 文件路径列表
        show_progress: 是否显示进度

    Returns:
        DocumentAnalysisResult 列表
    """
    return await enhanced_document_analyzer.analyze_batch_documents(
        file_paths,
        show_progress
    )


def analyze_document(
    file_path: str,
    original_filename: Optional[str] = None
) -> DocumentAnalysisResult:
    """
    同步分析文档

    Args:
        file_path: 文件路径
        original_filename: 原始文件名

    Returns:
        DocumentAnalysisResult 对象
    """
    return enhanced_document_analyzer.analyze_document_sync(
        file_path,
        original_filename
    )


def analyze_batch_documents(
    file_paths: List[str],
    show_progress: bool = True
) -> List[DocumentAnalysisResult]:
    """
    同步批量分析文档

    Args:
        file_paths: 文件路径列表
        show_progress: 是否显示进度

    Returns:
        DocumentAnalysisResult 列表
    """
    return enhanced_document_analyzer.analyze_batch_documents_sync(
        file_paths,
        show_progress
    )
