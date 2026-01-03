"""
统一 PDF 提取 API 端点
提供增强的 PDF 提取功能，包括：
1. 统一 PDF 提取服务（快速模式/精确模式）
2. 智能区域检测
3. 批量向量嵌入优化

这些功能完全兼容现有系统，不影响现有功能
"""

import os
import tempfile
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.core.structured_logging import get_structured_logger

from app.services.parsers.advanced.unified_pdf_extractor import unified_pdf_extractor
from app.services.parsers.advanced.smart_region_detector import smart_region_detector
from app.core.config import settings

logger = get_structured_logger(__name__)

router = APIRouter()


class ExtractionResponse(BaseModel):
    """提取响应"""
    success: bool
    message: str
    filename: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None


class RegionDetectionResponse(BaseModel):
    """区域检测响应"""
    success: bool
    message: str
    filename: Optional[str] = None
    regions: Optional[List[dict]] = None
    error: Optional[str] = None


@router.post("/extract/fast", response_model=ExtractionResponse)
async def extract_fast(file: UploadFile = File(...)):
    """
    快速模式提取

    使用 PyMuPDF4LLM 提取，特点：
    1. 页码标记：每页开头加 {{第X页}}
    2. 图片提取：提取 PDF 中的图片（base64）
    3. 完整页面截图

    - **file**: PDF 文件
    """
    temp_file = None
    try:
        logger.info(f"收到文件: {file.filename}")

        # 保存上传的文件到临时位置
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        # 执行提取
        result = await unified_pdf_extractor.extract_fast(
            file_path=temp_file.name,
            original_filename=file.filename
        )

        # 只返回文件名部分，不包含路径
        filename_only = Path(file.filename).name if file.filename else None

        return ExtractionResponse(
            success=True,
            message="快速提取成功",
            filename=filename_only,
            data=result
        )

    except Exception as e:
        logger.error(f"快速提取失败: {e}")
        import traceback
        traceback.print_exc()

        filename_only = Path(file.filename).name if file.filename else None

        return ExtractionResponse(
            success=False,
            message="快速提取失败",
            filename=filename_only,
            error=str(e)
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass


@router.post("/extract/accurate", response_model=ExtractionResponse)
async def extract_accurate(file: UploadFile = File(...)):
    """
    精确模式提取

    使用现有系统的 Qwen VL 模型提取，特点：
    1. 批量处理页面
    2. 结构化输出（markdown、表格、公式、图片描述）
    3. 详细统计（Token 使用、耗时）

    - **file**: PDF 文件
    """
    temp_file = None
    try:
        logger.info(f"收到文件: {file.filename}")

        # 保存上传的文件到临时位置
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        # 执行提取
        result = await unified_pdf_extractor.extract_accurate(
            file_path=temp_file.name,
            original_filename=file.filename
        )

        filename_only = Path(file.filename).name if file.filename else None

        return ExtractionResponse(
            success=True,
            message="精确提取成功",
            filename=filename_only,
            data=result
        )

    except Exception as e:
        logger.error(f"精确提取失败: {e}")
        import traceback
        traceback.print_exc()

        filename_only = Path(file.filename).name if file.filename else None

        return ExtractionResponse(
            success=False,
            message="精确提取失败",
            filename=filename_only,
            error=str(e)
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass


@router.post("/extract/auto", response_model=ExtractionResponse)
async def extract_auto(
    file: UploadFile = File(...),
    force_mode: Optional[str] = Form(None)
):
    """
    自动模式提取

    根据文件大小和页数自动选择提取模式：
    - 页数 > 20 或文件大小 > 50MB：使用快速模式
    - 否则：使用精确模式

    Args:
        file: PDF 文件
        force_mode: 强制使用的模式 ("fast" 或 "accurate")

    - **file**: PDF 文件
    - **force_mode**: 可选，强制使用指定模式
    """
    temp_file = None
    try:
        logger.info(f"收到文件: {file.filename}")

        # 保存上传的文件到临时位置
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        # 执行提取
        result = await unified_pdf_extractor.extract_with_auto_mode(
            file_path=temp_file.name,
            original_filename=file.filename,
            force_mode=force_mode
        )

        filename_only = Path(file.filename).name if file.filename else None

        return ExtractionResponse(
            success=True,
            message="自动模式提取成功",
            filename=filename_only,
            data=result
        )

    except Exception as e:
        logger.error(f"自动模式提取失败: {e}")
        import traceback
        traceback.print_exc()

        filename_only = Path(file.filename).name if file.filename else None

        return ExtractionResponse(
            success=False,
            message="自动模式提取失败",
            filename=filename_only,
            error=str(e)
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass


@router.post("/detect-regions", response_model=RegionDetectionResponse)
async def detect_regions(
    file: UploadFile = File(...),
    pages: Optional[str] = Form(None)
):
    """
    智能区域检测

    检测 PDF 中的智能区域（表格、图片、文本）

    Args:
        file: PDF 文件
        pages: 要检测的页码列表，逗号分隔（如 "1,2,3"），None 表示全部

    - **file**: PDF 文件
    - **pages**: 可选，指定要检测的页码
    """
    temp_file = None
    try:
        logger.info(f"收到文件: {file.filename}")

        # 保存上传的文件到临时位置
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        # 解析页码
        page_list = None
        if pages:
            try:
                page_list = [int(p.strip()) - 1 for p in pages.split(',')]  # 转换为 0-based
            except ValueError:
                raise HTTPException(status_code=400, detail="页码格式错误，应为逗号分隔的数字")

        # 执行区域检测
        regions = smart_region_detector.detect_regions(temp_file.name, page_list)

        # 转换为字典格式
        regions_dict = [
            {
                "region_type": r.region_type,
                "bbox": r.bbox,
                "page_num": r.page_num,
                "content": r.content[:500],  # 限制内容长度
                "confidence": r.confidence
            }
            for r in regions
        ]

        filename_only = Path(file.filename).name if file.filename else None

        return RegionDetectionResponse(
            success=True,
            message=f"区域检测成功，共检测到 {len(regions)} 个区域",
            filename=filename_only,
            regions=regions_dict
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"区域检测失败: {e}")
        import traceback
        traceback.print_exc()

        filename_only = Path(file.filename).name if file.filename else None

        return RegionDetectionResponse(
            success=False,
            message="区域检测失败",
            filename=filename_only,
            error=str(e)
        )
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass


@router.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "service": "unified-pdf-extraction",
        "features": {
            "fast_mode": True,
            "accurate_mode": True,
            "auto_mode": True,
            "region_detection": True,
            "batch_embedding": True
        },
        "models": {
            "multimodal": settings.qwen_multimodal_model,
            "embedding": settings.qwen_text_embedding_model
        },
        "timestamp": datetime.now().isoformat()
    }
