"""
Qwen3-VL-Plus内容分析任务
处理文档内容的智能分析，包括摘要、图表分析、表格理解等
"""

import asyncio
import logging
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from celery import current_task
from celery.exceptions import Retry

from app.celery_app import celery_app
from app.services.qwen_service import QwenService
from app.services.consolidated_document_service import ConsolidatedDocumentService as DocumentParser
from app.core.config import settings
from app.db.mysql import get_db
from app.models.document import Document, DocumentChunk, ImageAnalysis, TableAnalysis

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,)
)
def analyze_content(self, document_id: str) -> Dict[str, Any]:
    """
    分析文档内容任务
    """
    try:
        logger.info(f"Starting content analysis for document_id: {document_id}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Initializing analysis"}
        )

        # 异步执行内容分析
        result = asyncio.run(_analyze_content_async(document_id, self))

        logger.info(f"Content analysis completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Content analysis failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=60)


async def _analyze_content_async(document_id: str, task) -> Dict[str, Any]:
    """异步执行内容分析"""
    qwen_service = QwenService()

    # 获取文档块
    async with get_db() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = result.scalars().all()

    if not chunks:
        raise ValueError(f"No chunks found for document {document_id}")

    analysis_results = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        task.update_state(
            state="PROGRESS",
            meta={
                "current": (i / total_chunks) * 100,
                "total": 100,
                "status": f"Analyzing chunk {i+1}/{total_chunks}"
            }
        )

        # 分析文本块
        chunk_analysis = await _analyze_chunk_async(qwen_service, chunk)
        analysis_results.append(chunk_analysis)

    # 生成文档整体摘要
    task.update_state(
        state="PROGRESS",
        meta={"current": 90, "total": 100, "status": "Generating document summary"}
    )

    document_summary = await _generate_document_summary_async(qwen_service, analysis_results)

    # 保存分析结果
    await _save_analysis_results(document_id, analysis_results, document_summary)

    return {
        "status": "completed",
        "document_id": document_id,
        "analyzed_chunks": len(analysis_results),
        "summary": document_summary,
        "analysis_time": datetime.utcnow().isoformat()
    }


async def _analyze_chunk_async(qwen_service: QwenService, chunk: DocumentChunk) -> Dict[str, Any]:
    """分析单个文本块"""
    # 根据块类型选择分析策略
    if chunk.chunk_type == "text":
        return await _analyze_text_chunk_async(qwen_service, chunk)
    elif chunk.chunk_type == "image":
        return await _analyze_image_chunk_async(qwen_service, chunk)
    elif chunk.chunk_type == "table":
        return await _analyze_table_chunk_async(qwen_service, chunk)
    elif chunk.chunk_type == "formula":
        return await _analyze_formula_chunk_async(qwen_service, chunk)
    else:
        return await _analyze_mixed_content_async(qwen_service, chunk)


async def _analyze_text_chunk_async(qwen_service: QwenService, chunk: DocumentChunk) -> Dict[str, Any]:
    """分析纯文本块"""
    try:
        # 生成摘要
        summary = await qwen_service.generate_summary(chunk.content)

        # 提取关键信息
        key_points = await qwen_service.extract_key_points(chunk.content)

        # 情感分析
        sentiment = await qwen_service.analyze_sentiment(chunk.content)

        # 金融实体识别
        entities = await qwen_service.extract_financial_entities(chunk.content)

        return {
            "chunk_id": chunk.id,
            "chunk_type": "text",
            "summary": summary,
            "key_points": key_points,
            "sentiment": sentiment,
            "entities": entities,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error analyzing text chunk {chunk.id}: {e}")
        return {
            "chunk_id": chunk.id,
            "chunk_type": "text",
            "error": str(e),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }


async def _analyze_image_chunk_async(qwen_service: QwenService, chunk: DocumentChunk) -> Dict[str, Any]:
    """分析图片块"""
    try:
        # 获取图片路径
        image_path = chunk.metadata.get("image_path")
        if not image_path:
            raise ValueError("No image path found in chunk metadata")

        # 图片内容描述
        description = await qwen_service.analyze_image(image_path)

        # 如果是图表，进行趋势分析
        is_chart = chunk.metadata.get("is_chart", False)
        chart_analysis = None
        if is_chart:
            chart_analysis = await qwen_service.analyze_chart(image_path)

        # OCR文本提取
        ocr_text = await qwen_service.extract_text_from_image(image_path)

        return {
            "chunk_id": chunk.id,
            "chunk_type": "image",
            "description": description,
            "is_chart": is_chart,
            "chart_analysis": chart_analysis,
            "ocr_text": ocr_text,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error analyzing image chunk {chunk.id}: {e}")
        return {
            "chunk_id": chunk.id,
            "chunk_type": "image",
            "error": str(e),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }


async def _analyze_table_chunk_async(qwen_service: QwenService, chunk: DocumentChunk) -> Dict[str, Any]:
    """分析表格块"""
    try:
        # 获取表格数据
        table_data = chunk.metadata.get("table_data")
        if not table_data:
            raise ValueError("No table data found in chunk metadata")

        # 表格内容理解
        table_summary = await qwen_service.analyze_table(table_data)

        # 数据趋势分析
        trend_analysis = await qwen_service.analyze_table_trends(table_data)

        # 关键指标提取
        key_metrics = await qwen_service.extract_table_metrics(table_data)

        return {
            "chunk_id": chunk.id,
            "chunk_type": "table",
            "table_summary": table_summary,
            "trend_analysis": trend_analysis,
            "key_metrics": key_metrics,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error analyzing table chunk {chunk.id}: {e}")
        return {
            "chunk_id": chunk.id,
            "chunk_type": "table",
            "error": str(e),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }


async def _analyze_formula_chunk_async(qwen_service: QwenService, chunk: DocumentChunk) -> Dict[str, Any]:
    """分析公式块"""
    try:
        # 获取公式内容
        formula = chunk.content
        if not formula:
            raise ValueError("No formula content found")

        # 公式解释
        explanation = await qwen_service.explain_formula(formula)

        # 变量识别
        variables = await qwen_service.extract_formula_variables(formula)

        # 计算步骤分解
        steps = await qwen_service.breakdown_formula_steps(formula)

        return {
            "chunk_id": chunk.id,
            "chunk_type": "formula",
            "formula": formula,
            "explanation": explanation,
            "variables": variables,
            "steps": steps,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error analyzing formula chunk {chunk.id}: {e}")
        return {
            "chunk_id": chunk.id,
            "chunk_type": "formula",
            "error": str(e),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }


async def _analyze_mixed_content_async(qwen_service: QwenService, chunk: DocumentChunk) -> Dict[str, Any]:
    """分析混合内容块"""
    try:
        # 综合分析
        comprehensive_analysis = await qwen_service.analyze_mixed_content(chunk.content, chunk.metadata)

        return {
            "chunk_id": chunk.id,
            "chunk_type": "mixed",
            "comprehensive_analysis": comprehensive_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error analyzing mixed content chunk {chunk.id}: {e}")
        return {
            "chunk_id": chunk.id,
            "chunk_type": "mixed",
            "error": str(e),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def batch_analyze_images(self, document_id: str, image_paths: List[str]) -> Dict[str, Any]:
    """
    批量分析图片
    """
    try:
        logger.info(f"Starting batch image analysis for document_id: {document_id}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": len(image_paths), "status": "Starting image analysis"}
        )

        # 异步执行批量分析
        result = asyncio.run(_batch_analyze_images_async(document_id, image_paths, self))

        logger.info(f"Batch image analysis completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Batch image analysis failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _batch_analyze_images_async(document_id: str, image_paths: List[str], task) -> Dict[str, Any]:
    """异步批量分析图片"""
    qwen_service = QwenService()
    analysis_results = []

    for i, image_path in enumerate(image_paths):
        task.update_state(
            state="PROGRESS",
            meta={
                "current": i,
                "total": len(image_paths),
                "status": f"Analyzing image {i+1}/{len(image_paths)}"
            }
        )

        try:
            # 分析单张图片
            analysis = await qwen_service.analyze_image(image_path)
            analysis_results.append({
                "image_path": image_path,
                "analysis": analysis,
                "status": "success"
            })

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            analysis_results.append({
                "image_path": image_path,
                "error": str(e),
                "status": "failed"
            })

    # 保存图片分析结果
    await _save_image_analysis_results(document_id, analysis_results)

    return {
        "status": "completed",
        "document_id": document_id,
        "total_images": len(image_paths),
        "successful_analyses": sum(1 for r in analysis_results if r["status"] == "success"),
        "failed_analyses": sum(1 for r in analysis_results if r["status"] == "failed"),
        "results": analysis_results
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def generate_document_insights(self, document_id: str) -> Dict[str, Any]:
    """
    生成文档洞察
    """
    try:
        logger.info(f"Generating insights for document_id: {document_id}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Analyzing document patterns"}
        )

        # 异步执行洞察生成
        result = asyncio.run(_generate_insights_async(document_id, self))

        logger.info(f"Document insights generated for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to generate insights for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _generate_insights_async(document_id: str, task) -> Dict[str, Any]:
    """异步生成文档洞察"""
    qwen_service = QwenService()

    # 获取分析结果
    async with get_db() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = result.scalars().all()

    # 收集所有分析结果
    all_analyses = []
    for chunk in chunks:
        analysis = chunk.metadata.get("analysis")
        if analysis:
            all_analyses.append(analysis)

    # 生成综合洞察
    task.update_state(
        state="PROGRESS",
        meta={"current": 50, "total": 100, "status": "Generating comprehensive insights"}
    )

    insights = await qwen_service.generate_document_insights(all_analyses)

    # 提取关键趋势
    trends = await qwen_service.extract_document_trends(all_analyses)

    # 识别风险和机会
    risk_opportunity = await qwen_service.identify_risks_and_opportunities(all_analyses)

    # 保存洞察结果
    await _save_insights_results(document_id, insights, trends, risk_opportunity)

    return {
        "status": "completed",
        "document_id": document_id,
        "insights": insights,
        "trends": trends,
        "risks_and_opportunities": risk_opportunity,
        "generation_time": datetime.utcnow().isoformat()
    }


async def _generate_document_summary_async(qwen_service: QwenService, analysis_results: List[Dict]) -> str:
    """生成文档整体摘要"""
    # 合并所有文本内容
    all_text = " ".join([
        result.get("summary", "") for result in analysis_results
        if result.get("summary")
    ])

    if len(all_text) > 8000:  # 限制输入长度
        all_text = all_text[:8000] + "..."

    return await qwen_service.generate_summary(all_text)


async def _save_analysis_results(document_id: str, analysis_results: List[Dict], summary: str):
    """保存分析结果"""
    async with get_db() as db:
        # 更新文档的摘要
        document = await db.get(Document, document_id)
        if document:
            document.summary = summary
            document.analyzed_at = datetime.utcnow()
            await db.commit()

        # 更新每个块的分析结果
        for analysis in analysis_results:
            chunk = await db.get(DocumentChunk, analysis["chunk_id"])
            if chunk:
                chunk.metadata = {
                    **chunk.metadata,
                    "analysis": analysis
                }
                await db.commit()


async def _save_image_analysis_results(document_id: str, results: List[Dict]):
    """保存图片分析结果"""
    async with get_db() as db:
        for result in results:
            if result["status"] == "success":
                image_analysis = ImageAnalysis(
                    document_id=document_id,
                    image_path=result["image_path"],
                    description=result["analysis"].get("description", ""),
                    chart_analysis=result["analysis"].get("chart_analysis"),
                    ocr_text=result["analysis"].get("ocr_text"),
                    analysis_timestamp=datetime.utcnow()
                )
                db.add(image_analysis)

        await db.commit()


async def _save_insights_results(document_id: str, insights: str, trends: Dict, risk_opportunity: Dict):
    """保存洞察结果"""
    async with get_db() as db:
        document = await db.get(Document, document_id)
        if document:
            document.doc_metadata = {
                **document.doc_metadata,
                "insights": {
                    "summary": insights,
                    "trends": trends,
                    "risks_and_opportunities": risk_opportunity,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            await db.commit()