"""
文档处理流水线 API - 完整版
整合所有步骤：解析 -> 多模态分析 -> 深度汇总 -> 增强Markdown -> 知识图谱 -> 向量存储 -> 本地存储
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from pydantic import BaseModel
import uuid

from app.core.structured_logging import get_structured_logger
from app.services.pipeline.document_pipeline_service import get_document_pipeline_service

logger = get_structured_logger(__name__)

router = APIRouter()


class PipelineProcessRequest(BaseModel):
    """流水线处理请求"""
    document_id: Optional[str] = None
    enable_multimodal: bool = True
    enable_deepseek_summary: bool = True
    enable_knowledge_graph: bool = True
    enable_vector_storage: bool = True
    enable_file_storage: bool = True


class PipelineStatusResponse(BaseModel):
    """流水线状态响应"""
    document_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0-100
    current_step: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# 存储处理状态（生产环境应使用 Redis）
pipeline_status_store: Dict[str, PipelineStatusResponse] = {}


@router.post("/process", summary="文档处理流水线")
async def process_document_pipeline(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    enable_multimodal: bool = True,
    enable_deepseek_summary: bool = True,
    enable_knowledge_graph: bool = True,
    enable_vector_storage: bool = True,
    enable_file_storage: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    完整的文档处理流水线

    流程：
    1. 文档解析 (openpyxl/python-pptx)
    2. 多模态分析 (qwen-vl-plus)
    3. 深度汇总 (基于规则 + deepseek检查)
    4. 增强Markdown生成
    5. 知识图谱抽取 (neo4j)
    6. 向量存储 (milvus)
    7. 本地文件存储

    支持的文件类型：.xlsx, .xls, .pptx, .ppt, .pdf, .docx, .md
    """
    try:
        # 生成文档ID
        if not document_id:
            document_id = str(uuid.uuid4())

        # 读取文件内容
        file_content = await file.read()

        # 初始化状态
        pipeline_status_store[document_id] = PipelineStatusResponse(
            document_id=document_id,
            status="processing",
            progress=0,
            current_step="初始化..."
        )

        # 获取流水线服务
        pipeline_service = get_document_pipeline_service()

        # 如果是后台任务
        if background_tasks:
            background_tasks.add_task(
                _process_document_background,
                pipeline_service,
                file_content,
                file.filename,
                document_id,
                {
                    "enable_multimodal": enable_multimodal,
                    "enable_deepseek_summary": enable_deepseek_summary,
                    "enable_knowledge_graph": enable_knowledge_graph,
                    "enable_vector_storage": enable_vector_storage,
                    "enable_file_storage": enable_file_storage
                }
            )

            return JSONResponse(content={
                "document_id": document_id,
                "filename": file.filename,
                "status": "processing",
                "message": "文档正在后台处理中，请使用 /status/{document_id} 查询进度",
                "status_url": f"/api/v1/pipeline/status/{document_id}"
            })

        # 同步处理
        else:
            result = await pipeline_service.process_document(
                file_content=file_content,
                filename=file.filename,
                document_id=document_id,
                options={
                    "enable_multimodal": enable_multimodal,
                    "enable_deepseek_summary": enable_deepseek_summary,
                    "enable_knowledge_graph": enable_knowledge_graph,
                    "enable_vector_storage": enable_vector_storage,
                    "enable_file_storage": enable_file_storage
                }
            )

            # 更新状态
            pipeline_status_store[document_id] = PipelineStatusResponse(
                document_id=document_id,
                status="completed" if result.success else "failed",
                progress=100,
                current_step="完成" if result.success else "失败",
                result={
                    "processing_time": result.processing_time,
                    "parsing_method": result.parsing_result.get("method", "unknown"),
                    "multimodal_analyzed": result.multimodal_analysis.get("analyzed_count", 0),
                    "kg_entities": result.knowledge_graph.get("entity_count", 0),
                    "kg_relations": result.knowledge_graph.get("relation_count", 0),
                    "markdown_length": len(result.enhanced_markdown),
                    "file_storage": result.file_storage.get("status", "unknown")
                },
                error=result.error_message if not result.success else None
            )

            return JSONResponse(content={
                "document_id": document_id,
                "filename": file.filename,
                "success": result.success,
                "processing_time": result.processing_time,
                "result": {
                    "parsing": {
                        "method": result.parsing_result.get("method"),
                        "success": result.parsing_result.get("success", True)
                    },
                    "multimodal": {
                        "status": result.multimodal_analysis.get("status"),
                        "analyzed_count": result.multimodal_analysis.get("analyzed_count", 0)
                    },
                    "deepseek_summary": {
                        "status": result.deepseek_summary.get("status")
                    },
                    "knowledge_graph": {
                        "status": result.knowledge_graph.get("status"),
                        "entity_count": result.knowledge_graph.get("entity_count", 0),
                        "relation_count": result.knowledge_graph.get("relation_count", 0)
                    },
                    "vector_storage": {
                        "status": result.vector_storage.get("status")
                    },
                    "file_storage": {
                        "status": result.file_storage.get("status"),
                        "paths": {
                            "markdown": result.file_storage.get("markdown_path"),
                            "json": result.file_storage.get("json_path")
                        }
                    }
                },
                "enhanced_markdown_preview": result.enhanced_markdown[:500] + "..." if len(result.enhanced_markdown) > 500 else result.enhanced_markdown,
                "error": result.error_message if not result.success else None
            })

    except Exception as e:
        logger.error(f"文档处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


async def _process_document_background(
    pipeline_service,
    file_content: bytes,
    filename: str,
    document_id: str,
    options: Dict[str, Any]
):
    """后台处理文档"""
    try:
        await pipeline_service.process_document(
            file_content=file_content,
            filename=filename,
            document_id=document_id,
            options=options
        )
    except Exception as e:
        logger.error(f"后台处理文档失败: {str(e)}")
        pipeline_status_store[document_id] = PipelineStatusResponse(
            document_id=document_id,
            status="failed",
            progress=0,
            current_step="处理失败",
            error=str(e)
        )


@router.get("/status/{document_id}", summary="查询处理状态")
async def get_pipeline_status(document_id: str):
    """
    查询文档处理状态

    返回处理进度和结果
    """
    status = pipeline_status_store.get(document_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"未找到文档 {document_id} 的处理记录")

    return {
        "document_id": status.document_id,
        "status": status.status,
        "progress": status.progress,
        "current_step": status.current_step,
        "result": status.result,
        "error": status.error
    }


@router.get("/result/{document_id}", summary="获取处理结果")
async def get_pipeline_result(document_id: str):
    """
    获取完整的处理结果

    包括增强Markdown、知识图谱、向量存储等所有结果
    """
    status = pipeline_status_store.get(document_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"未找到文档 {document_id} 的处理记录")

    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"文档处理尚未完成，当前状态: {status.status}"
        )

    # 返回完整结果（包括增强Markdown）
    return {
        "document_id": document_id,
        "status": status.status,
        "result": status.result
    }


@router.get("/markdown/{document_id}", summary="获取增强Markdown")
async def get_enhanced_markdown(document_id: str):
    """
    获取生成的增强Markdown

    返回完整的Markdown文档
    """
    # TODO: 从本地存储读取Markdown文件
    status = pipeline_status_store.get(document_id)

    if not status or status.status != "completed":
        raise HTTPException(status_code=404, detail="文档未找到或处理未完成")

    # 临时实现：从结果中返回
    # 生产环境应从文件系统读取
    return {
        "document_id": document_id,
        "markdown": status.result.get("enhanced_markdown", "") if status.result else ""
    }


@router.delete("/status/{document_id}", summary="清除处理记录")
async def clear_pipeline_status(document_id: str):
    """
    清除文档处理记录（释放内存）
    """
    if document_id in pipeline_status_store:
        del pipeline_status_store[document_id]
        return {"message": f"已清除文档 {document_id} 的处理记录"}
    else:
        raise HTTPException(status_code=404, detail="未找到处理记录")
