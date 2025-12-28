"""
增强的文档处理 API 端点

提供完整的文档处理pipeline API：
1. PDF 提取 + 高级 Markdown 分割
2. LLM 多模态提取 + LangChain 结构化提取
3. 优化批量嵌入 + 并发文档分析
4. 可视化报告生成
"""

import os
import time
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.services.parsers.advanced.enhanced_document_analyzer import (
    enhanced_document_analyzer,
    analyze_document_async,
    DocumentAnalysisResult,
    AnalyzerConfig
)
from app.services.parsers.advanced.enhanced_markdown_splitter import (
    enhanced_markdown_splitter,
    split_markdown
)
from app.services.parsers.advanced.enhanced_metadata_extractor import (
    enhanced_metadata_extractor,
    extract_batch_chunks_metadata_async,
    ChunkMetadataExtraction
)
from app.services.visualization.enhanced_report_generator import (
    enhanced_report_generator,
    generate_analysis_report,
    HTMLReport
)


# ==================== 请求/响应模型 ====================

class AnalysisRequest(BaseModel):
    """分析请求"""
    extraction_mode: str = Field(default="auto", description="提取模式: fast/accurate/auto")
    extract_metadata: bool = Field(default=True, description="是否提取元数据")
    extract_embeddings: bool = Field(default=False, description="是否提取向量嵌入")
    max_workers: int = Field(default=10, description="最大并发数")


class BatchAnalysisRequest(BaseModel):
    """批量分析请求"""
    extraction_mode: str = Field(default="auto", description="提取模式")
    extract_metadata: bool = Field(default=True, description="是否提取元数据")
    extract_embeddings: bool = Field(default=False, description="是否提取向量嵌入")
    max_workers: int = Field(default=10, description="最大并发数")


class MarkdownSplitRequest(BaseModel):
    """Markdown分割请求"""
    max_chunk_size: int = Field(default=2000, description="最大chunk大小")
    chunk_overlap: int = Field(default=200, description="chunk重叠大小")
    min_chunk_size: int = Field(default=100, description="最小chunk大小")


class MetadataExtractionRequest(BaseModel):
    """元数据提取请求"""
    extract_tables: bool = Field(default=True, description="是否提取表格")
    extract_key_points: bool = Field(default=True, description="是否提取关键点")
    extract_topics: bool = Field(default=True, description="是否提取主题")
    extract_sentiment: bool = Field(default=True, description="是否分析情感")
    max_workers: int = Field(default=10, description="最大并发数")


class ReportGenerationRequest(BaseModel):
    """报告生成请求"""
    user_query: Optional[str] = Field(default=None, description="用户查询")


# ==================== API响应模型 ====================

class AnalysisResponse(BaseModel):
    """分析响应"""
    success: bool
    message: str
    document_id: str
    filename: str
    status: str
    chunk_count: int
    extraction_method: str
    processing_time: float
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchAnalysisResponse(BaseModel):
    """批量分析响应"""
    success: bool
    message: str
    total_documents: int
    success_count: int
    failed_count: int
    total_time: float
    results: List[Dict[str, Any]]


class MarkdownSplitResponse(BaseModel):
    """Markdown分割响应"""
    success: bool
    chunk_count: int
    chunks: List[Dict[str, Any]]
    processing_time: float


class MetadataExtractionResponse(BaseModel):
    """元数据提取响应"""
    success: bool
    chunk_count: int
    extractions: List[Dict[str, Any]]
    processing_time: float


class ReportResponse(BaseModel):
    """报告响应"""
    success: bool
    title: str
    summary: str
    html_url: Optional[str] = None
    html: Optional[str] = None


# ==================== 路由器 ====================

router = APIRouter()


# ==================== 健康检查 ====================

@router.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "enhanced_document_analysis",
        "version": "1.0.0",
        "features": [
            "PDF提取 + 高级Markdown分割",
            "LLM结构化元数据提取",
            "并发文档分析",
            "ECharts可视化报告"
        ],
        "timestamp": time.time()
    }


# ==================== 1. 完整文档分析 ====================

@router.post("/analyze", response_model=AnalysisResponse, tags=["文档分析"])
async def analyze_document_endpoint(
    file: UploadFile = File(..., description="PDF文件"),
    extraction_mode: str = Form(default="auto", description="提取模式: fast/accurate/auto"),
    extract_metadata: bool = Form(default=True, description="是否提取元数据"),
    extract_embeddings: bool = Form(default=False, description="是否提取向量嵌入"),
    max_workers: int = Form(default=10, description="最大并发数")
):
    """
    完整文档分析（单文档）

    执行完整的文档处理pipeline：
    1. PDF提取
    2. Markdown分割
    3. 元数据提取
    4. 向量嵌入
    """
    start_time = time.time()

    try:
        # 保存上传的文件到临时位置
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # 创建分析器配置
            config = AnalyzerConfig(
                extraction_mode=extraction_mode,
                extract_metadata=extract_metadata,
                extract_embeddings=extract_embeddings,
                max_workers=max_workers
            )
            analyzer = enhanced_document_analyzer.__class__(config)

            # 执行分析
            result: DocumentAnalysisResult = await analyzer.analyze_document(
                tmp_file_path,
                file.filename,
                extract_embeddings=extract_embeddings
            )

            # 构建响应
            metadata = None
            if result.document_metadata:
                metadata = {
                    "document_title": result.document_metadata.document_title,
                    "document_type": result.document_metadata.document_type,
                    "overall_summary": result.document_metadata.overall_summary,
                    "main_topics": result.document_metadata.main_topics,
                }

            return AnalysisResponse(
                success=True,
                message="文档分析完成",
                document_id=result.document_id,
                filename=result.filename,
                status=result.status,
                chunk_count=result.chunk_count,
                extraction_method=result.extraction_method,
                processing_time=result.total_processing_time,
                summary=metadata.get("overall_summary") if metadata else None,
                metadata=metadata
            )

        finally:
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


# ==================== 2. Markdown分割 ====================

@router.post("/split-markdown", response_model=MarkdownSplitResponse, tags=["文档分割"])
async def split_markdown_endpoint(
    file: UploadFile = File(..., description="Markdown/文本文件"),
    max_chunk_size: int = Form(default=2000),
    chunk_overlap: int = Form(default=200),
    min_chunk_size: int = Form(default=100)
):
    """
    高级Markdown分割

    基于标题层级智能分割文档，保留完整的结构上下文
    """
    start_time = time.time()

    try:
        # 读取文件内容
        content = await file.read()
        text = content.decode('utf-8')

        # 执行分割
        from app.services.parsers.advanced.enhanced_markdown_splitter import SplitConfig
        config = SplitConfig(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        splitter = enhanced_markdown_splitter.__class__(config)
        chunks = splitter.split_text(text, {'filename': file.filename})

        # 构建响应
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "chunk_id": chunk.metadata.get('chunk_id'),
                "title_path": chunk.metadata.get('title_path'),
                "content": chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                "char_count": len(chunk.page_content),
                "metadata": chunk.metadata
            })

        processing_time = time.time() - start_time

        return MarkdownSplitResponse(
            success=True,
            chunk_count=len(chunks),
            chunks=chunks_data,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分割失败: {str(e)}")


# ==================== 3. 元数据提取 ====================

@router.post("/extract-metadata", response_model=MetadataExtractionResponse, tags=["元数据提取"])
async def extract_metadata_endpoint(
    chunks: List[Dict[str, Any]] = Form(..., description="Chunk列表"),
    extract_tables: bool = Form(default=True),
    extract_key_points: bool = Form(default=True),
    extract_topics: bool = Form(default=True),
    extract_sentiment: bool = Form(default=True),
    max_workers: int = Form(default=10)
):
    """
    批量提取chunk元数据

    使用LLM提取结构化元数据：摘要、关键点、表格、主题、情感等
    """
    start_time = time.time()

    try:
        # 将字典转换为Document对象
        from langchain_core.documents import Document
        documents = [
            Document(page_content=chunk.get('content', ''), metadata=chunk.get('metadata', {}))
            for chunk in chunks
        ]

        # 执行提取
        extractions: List[ChunkMetadataExtraction] = await extract_batch_chunks_metadata_async(
            documents,
            show_progress=False
        )

        # 构建响应
        extractions_data = []
        for extraction in extractions:
            extractions_data.append({
                "chunk_id": extraction.chunk_id,
                "summary": extraction.summary,
                "key_points": [
                    {"point": kp.point, "importance": kp.importance}
                    for kp in extraction.key_points
                ],
                "topics": [
                    {"topic": t.topic, "relevance_score": t.relevance_score}
                    for t in extraction.topics
                ],
                "tables_count": len(extraction.tables),
                "sentiment": extraction.sentiment,
                "language": extraction.language
            })

        processing_time = time.time() - start_time

        return MetadataExtractionResponse(
            success=True,
            chunk_count=len(extractions),
            extractions=extractions_data,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}")


# ==================== 4. 生成可视化报告 ====================

@router.post("/generate-report", response_model=ReportResponse, tags=["报告生成"])
async def generate_report_endpoint(
    analysis_data: Dict[str, Any] = Form(..., description="文档分析数据"),
    user_query: Optional[str] = Form(default=None, description="用户查询"),
    return_html: bool = Form(default=False, description="是否直接返回HTML")
):
    """
    生成ECharts可视化报告

    基于文档分析结果生成交互式HTML报告
    """
    try:
        # 这里简化处理，实际应用中需要完整的DocumentAnalysisResult对象
        # 为了演示，我们创建一个mock结果
        # 在实际使用中，应该从前端或数据库获取完整的分析结果

        # 生成报告
        report: HTMLReport = generate_analysis_report(
            analysis_data,
            user_query
        )

        if return_html:
            return ReportResponse(
                success=True,
                title=report.title,
                summary=report.summary,
                html=report.html
            )
        else:
            # 在实际应用中，应该保存HTML到文件并返回URL
            return ReportResponse(
                success=True,
                title=report.title,
                summary=report.summary
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")


# ==================== 5. 一体化处理 ====================

@router.post("/process-full-pipeline", tags=["完整Pipeline"])
async def process_full_pipeline(
    file: UploadFile = File(..., description="PDF文件"),
    user_query: Optional[str] = Form(default=None, description="用户查询"),
    generate_report: bool = Form(default=True, description="是否生成可视化报告")
):
    """
    一体化文档处理Pipeline

    执行完整流程：
    1. PDF提取
    2. Markdown分割
    3. 元数据提取
    4. 向量嵌入（可选）
    5. 生成可视化报告
    """
    start_time = time.time()

    try:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # 步骤1-3: 执行完整分析
            result: DocumentAnalysisResult = await analyze_document_async(
                tmp_file_path,
                file.filename,
                extract_embeddings=False
            )

            if result.status == "failed":
                raise HTTPException(status_code=500, detail="文档分析失败")

            # 步骤4: 生成报告
            report = None
            if generate_report:
                report = generate_analysis_report(result, user_query)

            # 构建响应
            response_data = {
                "success": True,
                "message": "文档处理完成",
                "document_id": result.document_id,
                "filename": result.filename,
                "status": result.status,
                "processing_time": time.time() - start_time,
                "analysis_summary": {
                    "chunk_count": result.chunk_count,
                    "extraction_method": result.extraction_method,
                    "extraction_time": result.extraction_time,
                    "splitting_time": result.splitting_time,
                    "metadata_extraction_time": result.metadata_extraction_time
                }
            }

            # 添加元数据信息
            if result.document_metadata:
                response_data["document_metadata"] = {
                    "title": result.document_metadata.document_title,
                    "type": result.document_metadata.document_type,
                    "summary": result.document_metadata.overall_summary,
                    "topics": result.document_metadata.main_topics
                }

            # 添加报告HTML（如果生成）
            if report and generate_report:
                response_data["report"] = {
                    "title": report.title,
                    "summary": report.summary,
                    "html": report.html
                }

            return JSONResponse(content=response_data)

        finally:
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


# ==================== 6. 批量文档分析 ====================

@router.post("/analyze-batch", tags=["批量分析"])
async def analyze_batch_documents_endpoint(
    files: List[UploadFile] = File(..., description="PDF文件列表"),
    extraction_mode: str = Form(default="auto"),
    extract_metadata: bool = Form(default=True),
    max_workers: int = Form(default=5)
):
    """
    批量文档分析（并发处理）
    """
    start_time = time.time()

    try:
        # 保存所有上传的文件
        file_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                file_paths.append(tmp_file.name)

        try:
            # 创建分析器配置
            config = AnalyzerConfig(
                extraction_mode=extraction_mode,
                extract_metadata=extract_metadata,
                extract_embeddings=False,
                max_workers=max_workers
            )
            analyzer = enhanced_document_analyzer.__class__(config)

            # 执行批量分析
            results = await analyzer.analyze_batch_documents(
                file_paths,
                show_progress=False
            )

            # 统计结果
            success_count = sum(1 for r in results if r.status == "success")
            failed_count = len(results) - success_count

            # 构建响应
            results_summary = []
            for result in results:
                results_summary.append({
                    "document_id": result.document_id,
                    "filename": result.filename,
                    "status": result.status,
                    "chunk_count": result.chunk_count,
                    "processing_time": result.total_processing_time
                })

            return BatchAnalysisResponse(
                success=True,
                message=f"批量分析完成：{success_count}成功, {failed_count}失败",
                total_documents=len(files),
                success_count=success_count,
                failed_count=failed_count,
                total_time=time.time() - start_time,
                results=results_summary
            )

        finally:
            # 清理所有临时文件
            for fp in file_paths:
                if os.path.exists(fp):
                    os.unlink(fp)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量分析失败: {str(e)}")
