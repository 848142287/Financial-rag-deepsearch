"""
OCR 服务 API 端点

提供完整的文档OCR识别API（使用GLM-4.6V）：
1. 图片OCR识别
2. PDF文档OCR识别
3. Word/PPT文档OCR识别（自动转换）
4. 批量处理
"""

import os
import tempfile
import time
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.services.ocr_service import get_ocr_service
from app.services.ocr.document_converter import (
    get_document_converter,
    ConversionResult
)


# ==================== 请求/响应模型 ====================

class ImageOCRRequest(BaseModel):
    """图片OCR请求"""
    prompt: str = Field(
        default=None,
        description="OCR提示词（可选，默认使用GLM-4.6V内置提示词）"
    )


class PdfOCRRequest(BaseModel):
    """PDF OCR请求"""
    max_pages: Optional[int] = Field(default=None, description="最大处理页数")
    max_concurrent: int = Field(default=12, description="最大并发数")


class DocumentOCRRequest(BaseModel):
    """文档OCR请求（Word/PPT等）"""
    dpi: int = Field(default=144, description="转换DPI")
    max_pages: Optional[int] = Field(default=None, description="最大处理页数")
    auto_delete: bool = Field(default=True, description="处理完成后自动删除临时文件")


class OCRResponse(BaseModel):
    """OCR响应"""
    success: bool
    message: str
    text: str = ""
    page_count: int = 0
    processing_time: float = 0.0
    metadata: dict = {}


class BatchOCRResponse(BaseModel):
    """批量OCR响应"""
    success: bool
    message: str
    total_pages: int
    processed_pages: int
    full_text: str = ""
    processing_time: float = 0.0
    errors: List[str] = []


# ==================== 路由器 ====================

router = APIRouter()


# ==================== 健康检查 ====================

@router.get("/health", tags=["OCR服务"])
async def health_check():
    """健康检查端点"""
    try:
        ocr_service = get_ocr_service()
        return {
            "status": "healthy",
            "service": "GLM-4.6V OCR",
            "model": "glm-4.6v",
            "features": [
                "图片OCR识别（高精度）",
                "PDF文档OCR",
                "Word/PPT文档OCR（自动转换）",
                "批量并发处理",
                "内置优化的OCR提示词",
                "自动格式保持"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")


# ==================== 1. 图片OCR ====================

@router.post("/ocr/image", response_model=OCRResponse, tags=["OCR识别"])
async def ocr_image(
    file: UploadFile = File(..., description="图片文件"),
    prompt: str = Form(default=None)
):
    """
    图片OCR识别

    支持的格式：PNG, JPG, JPEG, BMP, GIF
    使用GLM-4.6V云端API进行高精度OCR识别
    """
    ocr_service = get_ocr_service()

    # 读取文件内容
    content = await file.read()

    try:
        start_time = time.time()

        # OCR识别
        result = await ocr_service.extract_text_from_image(
            content,
            prompt=prompt
        )

        processing_time = time.time() - start_time

        if not result.get('success'):
            return OCRResponse(
                success=False,
                message=result.get('error', 'OCR识别失败'),
                text="",
                processing_time=processing_time
            )

        return OCRResponse(
            success=True,
            message="OCR识别完成",
            text=result.get('text', ''),
            page_count=1,
            processing_time=processing_time,
            metadata={
                "model": result.get('model', 'glm-4.6v'),
                "confidence": result.get('confidence', 0.0),
                "filename": file.filename
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")


# ==================== 2. PDF OCR ====================

@router.post("/ocr/pdf", response_model=BatchOCRResponse, tags=["OCR识别"])
async def ocr_pdf(
    file: UploadFile = File(..., description="PDF文件"),
    max_pages: Optional[int] = Form(default=None),
    max_concurrent: int = Form(default=12)
):
    """
    PDF文档OCR识别

    支持扫描PDF和文本PDF的OCR识别
    使用GLM-4.6V云端API
    """
    ocr_service = get_ocr_service()

    # 读取PDF文件
    content = await file.read()

    try:
        start_time = time.time()

        # PDF批量OCR识别
        results = await ocr_service.batch_extract_from_pdf(
            content,
            pages=None,  # None表示全部页面
            max_concurrent=max_concurrent
        )

        processing_time = time.time() - start_time

        # 统计结果
        total_pages = len(results)
        processed_pages = sum(1 for r in results if r.get('success'))
        errors = [r.get('error', 'Unknown error') for r in results if not r.get('success')]
        full_text = "\n\n".join([r.get('text', '') for r in results if r.get('success')])

        return BatchOCRResponse(
            success=len(errors) == 0,
            message=f"PDF OCR完成，处理了{processed_pages}/{total_pages}页" +
                    (f"，{len(errors)}页失败" if errors else ""),
            total_pages=total_pages,
            processed_pages=processed_pages,
            full_text=full_text,
            processing_time=processing_time,
            errors=errors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF OCR失败: {str(e)}")


# ==================== 3. Word/PPT OCR ====================

@router.post("/ocr/document", response_model=BatchOCRResponse, tags=["OCR识别"])
async def ocr_document(
    file: UploadFile = File(..., description="文档文件（Word/PPT/Excel等）"),
    max_pages: Optional[int] = Form(default=None),
    auto_delete: bool = Form(default=True)
):
    """
    文档OCR识别（自动转换）

    支持格式：
    - Word (.docx, .doc)
    - PowerPoint (.pptx, .ppt)
    - Excel (.xlsx, .xls)
    - 纯文本 (.txt)
    - 图片 (.png, .jpg等)
    - PDF (.pdf)

    自动转换为PDF后进行OCR识别
    """
    ocr_service = get_ocr_service()
    converter = get_document_converter()

    # 保存临时文件
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
    try:
        content = await file.read()
        temp_input.write(content)
        temp_input_path = temp_input.name
    finally:
        temp_input.close()

    temp_pdf_path = None

    try:
        start_time = time.time()

        # 转换为PDF
        conversion_result: ConversionResult = converter.convert_to_pdf(
            temp_input_path
        )

        if not conversion_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"文档转换失败: {conversion_result.error_message}"
            )

        temp_pdf_path = conversion_result.output_path

        # 读取转换后的PDF
        with open(temp_pdf_path, 'rb') as f:
            pdf_content = f.read()

        # PDF OCR识别
        results = await ocr_service.batch_extract_from_pdf(
            pdf_content,
            pages=None,
            max_concurrent=12
        )

        processing_time = time.time() - start_time

        # 统计结果
        total_pages = len(results)
        processed_pages = sum(1 for r in results if r.get('success'))
        errors = [r.get('error', 'Unknown error') for r in results if not r.get('success')]
        full_text = "\n\n".join([r.get('text', '') for r in results if r.get('success')])

        return BatchOCRResponse(
            success=len(errors) == 0,
            message=f"文档OCR完成，处理了{processed_pages}/{total_pages}页" +
                    (f"，{len(errors)}页失败" if errors else ""),
            total_pages=total_pages,
            processed_pages=processed_pages,
            full_text=full_text,
            processing_time=processing_time,
            errors=errors
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档OCR失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_pdf_path and auto_delete and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)


# ==================== 4. 批量图片OCR ====================

@router.post("/ocr/batch-images", response_model=BatchOCRResponse, tags=["OCR识别"])
async def ocr_batch_images(
    files: List[UploadFile] = File(..., description="图片文件列表"),
    prompt: str = Form(default=None)
):
    """
    批量图片OCR识别

    一次上传多张图片进行OCR识别
    使用GLM-4.6V云端API
    """
    ocr_service = get_ocr_service()
    results = []
    errors = []
    start_time = time.time()

    for file in files:
        try:
            content = await file.read()
            result = await ocr_service.extract_text_from_image(
                content,
                prompt=prompt
            )

            if result.get('success'):
                result['metadata'] = result.get('metadata', {})
                result['metadata']['filename'] = file.filename
                results.append(result)
            else:
                errors.append(f"{file.filename}: {result.get('error', 'Unknown error')}")

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    processing_time = time.time() - start_time

    # 合并结果
    full_text = "\n\n".join([r.get('text', '') for r in results if r.get('success')])

    return BatchOCRResponse(
        success=len(errors) == 0,
        message=f"批量OCR完成，处理了{len(results)}/{len(files)}张图片" +
                (f"，{len(errors)}张失败" if errors else ""),
        total_pages=len(files),
        processed_pages=len(results),
        full_text=full_text,
        processing_time=processing_time,
        errors=errors
    )


# ==================== 5. 一体化处理 ====================

@router.post("/ocr/process", tags=["OCR处理"])
async def ocr_process_full(
    file: UploadFile = File(..., description="文档文件"),
    max_pages: Optional[int] = Form(default=None),
    max_concurrent: int = Form(default=12)
):
    """
    一体化OCR处理

    自动检测文件类型并选择最佳处理方案：
    1. PDF → 直接OCR
    2. Word/PPT → 转PDF → OCR
    3. 图片 → 直接OCR
    4. 其他 → 尝试转换

    使用GLM-4.6V云端API
    """
    ocr_service = get_ocr_service()
    converter = get_document_converter()

    # 读取文件
    content = await file.read()
    filename = file.filename.lower()
    start_time = time.time()

    try:
        if filename.endswith('.pdf'):
            # PDF直接OCR
            results = await ocr_service.batch_extract_from_pdf(
                content,
                pages=None,
                max_concurrent=max_concurrent
            )

            total_pages = len(results)
            processed_pages = sum(1 for r in results if r.get('success'))
            errors = [r.get('error') for r in results if not r.get('success')]
            full_text = "\n\n".join([r.get('text', '') for r in results if r.get('success')])

            return {
                "success": len(errors) == 0,
                "file_type": "pdf",
                "total_pages": total_pages,
                "processed_pages": processed_pages,
                "text": full_text,
                "processing_time": time.time() - start_time,
                "errors": errors
            }

        elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 图片直接OCR
            result = await ocr_service.extract_text_from_image(content)

            return {
                "success": result.get('success', False),
                "file_type": "image",
                "total_pages": 1,
                "processed_pages": 1 if result.get('success') else 0,
                "text": result.get('text', ''),
                "processing_time": time.time() - start_time,
                "errors": [result.get('error')] if not result.get('success') else []
            }

        else:
            # 其他格式：转换 → OCR
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
            try:
                temp_input.write(content)
                temp_input_path = temp_input.name
            finally:
                temp_input.close()

            try:
                conversion_result = converter.convert_to_pdf(temp_input_path)

                if not conversion_result.success:
                    return {
                        "success": False,
                        "error": conversion_result.error_message
                    }

                # 读取转换后的PDF
                with open(conversion_result.output_path, 'rb') as f:
                    pdf_content = f.read()

                results = await ocr_service.batch_extract_from_pdf(
                    pdf_content,
                    pages=None,
                    max_concurrent=max_concurrent
                )

                total_pages = len(results)
                processed_pages = sum(1 for r in results if r.get('success'))
                errors = [r.get('error') for r in results if not r.get('success')]
                full_text = "\n\n".join([r.get('text', '') for r in results if r.get('success')])

                return {
                    "success": len(errors) == 0,
                    "file_type": "document",
                    "converted_pdf": conversion_result.output_path,
                    "total_pages": total_pages,
                    "processed_pages": processed_pages,
                    "text": full_text,
                    "processing_time": time.time() - start_time,
                    "errors": errors
                }

            finally:
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if conversion_result.success and os.path.exists(conversion_result.output_path):
                    os.unlink(conversion_result.output_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
