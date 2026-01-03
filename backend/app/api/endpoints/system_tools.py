"""
系统工具API端点
提供数据库验证、文档导出等系统维护功能
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.core.structured_logging import get_structured_logger

from app.tools.database_tools import DatabaseTools
from app.tools.document_exporter import DocumentExporter

logger = get_structured_logger(__name__)

router = APIRouter(prefix="/system-tools", tags=["系统工具"])


class DataIntegrityResponse(BaseModel):
    """数据完整性响应"""
    total_documents: int
    total_chunks: int
    total_entities: int
    status_distribution: Dict[str, int]


class ExportRequest(BaseModel):
    """导出请求"""
    start_id: Optional[int] = None
    end_id: Optional[int] = None
    document_ids: Optional[List[int]] = None
    save_to_disk: bool = True


@router.get("/data-integrity", response_model=DataIntegrityResponse)
async def check_data_integrity():
    """
    检查数据完整性

    返回：
    - 文档总数
    - 分块总数
    - 实体总数
    - 状态分布
    """
    try:
        tools = DatabaseTools()
        tools.connect()

        result = tools.check_data_integrity()
        tools.close()

        logger.info(f"数据完整性检查: {result}")
        return DataIntegrityResponse(**result)

    except Exception as e:
        logger.error(f"数据完整性检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync-status")
async def check_sync_status():
    """
    检查数据同步状态

    返回文档元数据与实际数据的不一致情况
    """
    try:
        tools = DatabaseTools()
        tools.connect()

        result = tools.verify_sync_status()
        tools.close()

        return result

    except Exception as e:
        logger.error(f"同步状态检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export-documents")
async def export_documents(request: ExportRequest, background_tasks: BackgroundTasks):
    """
    导出文档

    支持三种方式：
    1. 指定ID范围: start_id 和 end_id
    2. 指定ID列表: document_ids
    3. 导出所有文档: 不提供任何参数

    返回导出的文档数据
    """
    try:
        exporter = DocumentExporter()
        exporter.connect()

        if request.document_ids:
            # 导出指定ID列表
            results = exporter.export_documents(request.document_ids, request.save_to_disk)
        elif request.start_id and request.end_id:
            # 导出ID范围
            results = exporter.export_range(request.start_id, request.end_id, request.save_to_disk)
        else:
            # 导出所有文档
            result = exporter.export_missing_documents(request.save_to_disk)
            results = result['results']

        summary = exporter.get_export_summary()
        exporter.close()

        return {
            "status": "success",
            "summary": summary,
            "documents": results[:10],  # 只返回前10个作为示例
            "total_exported": len(results)
        }

    except Exception as e:
        logger.error(f"文档导出失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix-metadata")
async def fix_metadata_issues():
    """
    修复元数据问题

    自动修复文档元数据与实际数据的不一致
    """
    try:
        tools = DatabaseTools()
        tools.connect()

        result = tools.fix_metadata_issues()
        tools.close()

        return result

    except Exception as e:
        logger.error(f"元数据修复失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-stats")
async def get_system_stats():
    """
    获取系统统计信息

    包括：
    - 数据库状态
    - 存储使用情况
    - 文档处理统计
    """
    try:
        tools = DatabaseTools()
        tools.connect()

        integrity = tools.check_data_integrity()
        sync_status = tools.verify_sync_status()

        tools.close()

        return {
            "database": {
                "status": "healthy",
                "integrity": integrity,
                "sync": sync_status
            },
            "timestamp": "2025-12-23"
        }

    except Exception as e:
        logger.error(f"系统统计获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
