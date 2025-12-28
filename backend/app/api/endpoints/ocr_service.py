"""
OCR 服务 API 端点

提供完整的文档OCR识别API：
1. 图片OCR识别
2. PDF文档OCR识别
3. Word/PPT文档OCR识别（自动转换）
4. 批量处理
"""

import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.ocr.deepseek_ocr_service import (
    get_deepseek_ocr_service,
    OCRResult,
    BatchOCRResult
)
from app.services.ocr.document_converter import (
    get_document_converter,
    ConversionResult
)


# ==================== 请求/响应模型 ====================

class ImageOCRRequest(BaseModel):
    """图片OCR请求"""
    prompt: str = Field(
        default="<image>\n<|grounding|>Convert the document to markdown.",
        description="OCR提示词"
    )


class PdfOCRRequest(BaseModel):
    """PDF OCR请求"""
    dpi: int = Field(default=144, description="转换DPI")
    max_pages: Optional[int] = Field(default=None, description="最大处理页数")
    batch_size: int = Field(default=4, description="批处理大小")
    use_batch: bool = Field(default=True, description="是否使用批处理")


class DocumentOCRRequest(BaseModel):
    """文档OCR请求（Word/PPT等）"""
    dpi: int = Field(default=144, description="转换DPI")
    max_pages: Optional[int] = Field(default=None, description="最大处理页数")
    auto_delete: bool = Field(default=True, description="处理完成后自动删除临时文件")


class OCRResponse(BaseModel):
    """OCR响应"""
    success: bool
    message: str
    markdown: str = ""
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
    full_markdown: str = ""
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
        ocr_service = get_deepseek_ocr_service()
        return {
            "status": "healthy",
            "service": "DeepSeek-OCR",
            "api_base": ocr_service.api_base,
            "timeout": ocr_service.timeout,
            "features": [
                "图片OCR识别",
                "PDF文档OCR",
                "Word/PPT文档OCR（自动转换）",
                "批量并发处理",
                "Markdown格式输出"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")


# ==================== 1. 图片OCR ====================

@router.post("/ocr/image", response_model=OCRResponse, tags=["OCR识别"])
async def ocr_image(
    file: UploadFile = File(..., description="图片文件"),
    prompt: str = Form(default="<image>\n<|grounding|>Convert the document to markdown.")
):
    """
    图片OCR识别

    支持的格式：PNG, JPG, JPEG, BMP, GIF
    """
    ocr_service = get_deepseek_ocr_service()

    # 保存临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    finally:
        temp_file.close()

    try:
        # OCR识别
        result: OCRResult = ocr_service.ocr_image(temp_file_path, prompt)

        if result.metadata.get("error"):
            raise HTTPException(status_code=500, detail=result.metadata["error"])

        return OCRResponse(
            success=True,
            message="OCR识别完成",
            markdown=result.markdown,
            text=result.text,
            page_count=1,
            processing_time=result.processing_time,
            metadata={
                "confidence": result.confidence,
                "filename": file.filename
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# ==================== 2. PDF OCR ====================

@router.post("/ocr/pdf", response_model=BatchOCRResponse, tags=["OCR识别"])
async def ocr_pdf(
    file: UploadFile = File(..., description="PDF文件"),
    dpi: int = Form(default=144),
    max_pages: Optional[int] = Form(default=None),
    batch_size: int = Form(default=4),
    use_batch: bool = Form(default=True)
):
    """
    PDF文档OCR识别

    支持扫描PDF和文本PDF的OCR识别
    """
    ocr_service = get_deepseek_ocr_service()

    # 保存临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    finally:
        temp_file.close()

    try:
        # PDF OCR识别
        if use_batch:
            result: BatchOCRResult = ocr_service.ocr_pdf_batch(
                temp_file_path,
                dpi=dpi,
                max_pages=max_pages,
                batch_size=batch_size
            )
        else:
            result: BatchOCRResult = ocr_service.ocr_pdf(
                temp_file_path,
                dpi=dpi,
                max_pages=max_pages
            )

        # 获取完整文本
        full_markdown = ocr_service.get_full_markdown(result)
        full_text = ocr_service.get_full_text(result)

        return BatchOCRResponse(
            success=result.success,
            message=f"PDF OCR完成，处理了{result.processed_pages}页" +
                    (f"，{len(result.errors)}页失败" if result.errors else ""),
            total_pages=result.total_pages,
            processed_pages=result.processed_pages,
            full_markdown=full_markdown,
            full_text=full_text,
            processing_time=result.total_time,
            errors=result.errors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF OCR失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# ==================== 3. Word/PPT OCR ====================

@router.post("/ocr/document", response_model=BatchOCRResponse, tags=["OCR识别"])
async def ocr_document(
    file: UploadFile = File(..., description="文档文件（Word/PPT/Excel等）"),
    dpi: int = Form(default=144),
    max_pages: Optional[int] = Form(default=None),
    batch_size: int = Form(default=4),
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
    """
    ocr_service = get_deepseek_ocr_service()
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

        # PDF OCR识别
        ocr_result: BatchOCRResult = ocr_service.ocr_pdf_batch(
            temp_pdf_path,
            dpi=dpi,
            max_pages=max_pages,
            batch_size=batch_size
        )

        # 获取完整文本
        full_markdown = ocr_service.get_full_markdown(ocr_result)
        full_text = ocr_service.get_full_text(ocr_result)

        return BatchOCRResponse(
            success=ocr_result.success,
            message=f"文档OCR完成，处理了{ocr_result.processed_pages}页" +
                    (f"，{len(ocr_result.errors)}页失败" if ocr_result.errors else ""),
            total_pages=ocr_result.total_pages,
            processed_pages=ocr_result.processed_pages,
            full_markdown=full_markdown,
            full_text=full_text,
            processing_time=ocr_result.total_time,
            errors=ocr_result.errors
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
    prompt: str = Form(default="<image>\n<|grounding|>Convert the document to markdown.")
):
    """
    批量图片OCR识别

    一次上传多张图片进行OCR识别
    """
    ocr_service = get_deepseek_ocr_service()
    results = []
    errors = []
    start_time = 0  # 临时占位

    for file in files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
        try:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        finally:
            temp_file.close()

        try:
            result = ocr_service.ocr_image(temp_file_path, prompt)
            result.metadata["filename"] = file.filename
            results.append(result)

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    # 合并结果
    full_markdown = "\n\n".join([r.markdown for r in results if r.markdown])
    full_text = "\n\n".join([r.text for r in results if r.text])

    return BatchOCRResponse(
        success=len(errors) == 0,
        message=f"批量OCR完成，处理了{len(results)}张图片" +
                (f"，{len(errors)}张失败" if errors else ""),
        total_pages=len(files),
        processed_pages=len(results),
        full_markdown=full_markdown,
        full_text=full_text,
        processing_time=sum(r.processing_time for r in results),
        errors=errors
    )


# ==================== 5. 一体化处理 ====================

@router.post("/ocr/process", tags=["OCR处理"])
async def ocr_process_full(
    file: UploadFile = File(..., description="文档文件"),
    dpi: int = Form(default=144),
    max_pages: Optional[int] = Form(default=None),
    batch_size: int = Form(default=4),
    output_format: str = Form(default="markdown")  # markdown, text, json
):
    """
    一体化OCR处理

    自动检测文件类型并选择最佳处理方案：
    1. PDF → 直接OCR
    2. Word/PPT → 转PDF → OCR
    3. 图片 → 直接OCR
    4. 其他 → 尝试转换
    """
    # 根据文件扩展名判断类型
    filename = file.filename.lower()
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")

    try:
        content = await file.read()
        temp_input.write(content)
        temp_input_path = temp_input.name
    finally:
        temp_input.close()

    try:
        if filename.endswith('.pdf'):
            # PDF直接OCR
            from app.services.ocr.deepseek_ocr_service import get_deepseek_ocr_service
            ocr_service = get_deepseek_ocr_service()
            result = ocr_service.ocr_pdf_batch(
                temp_input_path,
                dpi=dpi,
                max_pages=max_pages,
                batch_size=batch_size
            )

            full_markdown = ocr_service.get_full_markdown(result)
            full_text = ocr_service.get_full_text(result)

            return {
                "success": result.success,
                "file_type": "pdf",
                "total_pages": result.total_pages,
                "processed_pages": result.processed_pages,
                "markdown": full_markdown if output_format in ["markdown", "json"] else None,
                "text": full_text if output_format in ["text", "json"] else None,
                "processing_time": result.total_time,
                "errors": result.errors
            }

        else:
            # 其他格式：转换 → OCR
            converter = get_document_converter()
            conversion_result = converter.convert_to_pdf(temp_input_path)

            if not conversion_result.success:
                return {
                    "success": False,
                    "error": conversion_result.error_message
                }

            ocr_service = get_deepseek_ocr_service()
            result = ocr_service.ocr_pdf_batch(
                conversion_result.output_path,
                dpi=dpi,
                max_pages=max_pages,
                batch_size=batch_size
            )

            full_markdown = ocr_service.get_full_markdown(result)
            full_text = ocr_service.get_full_text(result)

            return {
                "success": result.success,
                "file_type": "document",
                "converted_pdf": conversion_result.output_path,
                "total_pages": result.total_pages,
                "processed_pages": result.processed_pages,
                "markdown": full_markdown if output_format in ["markdown", "json"] else None,
                "text": full_text if output_format in ["text", "json"] else None,
                "processing_time": result.total_time,
                "errors": result.errors
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
