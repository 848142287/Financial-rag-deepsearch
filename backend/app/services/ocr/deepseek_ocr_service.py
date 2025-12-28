"""
DeepSeek-OCR 服务封装

功能特点：
1. 通过 HTTP API 调用 DeepSeek-OCR 服务
2. 支持图片、PDF、Word、PPT 文档 OCR
3. 批量处理和并发控制
4. Markdown 格式输出
"""

import os
import io
import time
import base64
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str                           # 识别的文本
    markdown: str                       # Markdown格式
    confidence: float = 0.0             # 置信度
    page_num: int = 0                   # 页码
    processing_time: float = 0.0         # 处理时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class BatchOCRResult:
    """批量OCR结果"""
    success: bool
    total_pages: int
    processed_pages: int
    results: List[OCRResult] = field(default_factory=list)
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class DeepSeekOCRService:
    """
    DeepSeek-OCR 服务客户端

    通过 HTTP API 调用 DeepSeek-OCR vLLM 服务
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8002",
        timeout: int = 120,
        max_workers: int = 4
    ):
        """
        初始化 OCR 服务

        Args:
            api_base: DeepSeek-OCR API 地址
            timeout: 请求超时时间（秒）
            max_workers: 最大并发数
        """
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_workers = max_workers
        self.client = httpx.Client(timeout=timeout)

    def close(self):
        """关闭客户端"""
        if self.client:
            self.client.close()

    def _ocr_image_sync(
        self,
        image: Image.Image,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
    ) -> OCRResult:
        """
        对单张图片进行OCR识别（同步）

        Args:
            image: PIL Image对象
            prompt: OCR提示词

        Returns:
            OCRResult对象
        """
        start_time = time.time()

        try:
            # 将图片转换为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # 调用 OCR API
            response = self.client.post(
                f"{self.api_base}/ocr/image",
                files={"file": ("image.png", img_bytes, "image/png")},
                data={"prompt": prompt}
            )
            response.raise_for_status()

            result = response.json()
            processing_time = time.time() - start_time

            return OCRResult(
                text=result.get("text", ""),
                markdown=result.get("markdown", result.get("text", "")),
                confidence=result.get("confidence", 0.0),
                processing_time=processing_time,
                metadata=result.get("metadata", {})
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return OCRResult(
                text="",
                markdown="",
                confidence=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    def ocr_image(
        self,
        image_path: str,
        prompt: Optional[str] = None
    ) -> OCRResult:
        """
        识别单张图片

        Args:
            image_path: 图片路径
            prompt: 可选的OCR提示词

        Returns:
            OCRResult对象
        """
        if not os.path.exists(image_path):
            return OCRResult(
                text="",
                markdown="",
                confidence=0.0,
                metadata={"error": f"文件不存在: {image_path}"}
            )

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            default_prompt = "<image>\n<|grounding|>Convert the document to markdown."
            return self._ocr_image_sync(image, prompt or default_prompt)

        except Exception as e:
            return OCRResult(
                text="",
                markdown="",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def ocr_pdf(
        self,
        pdf_path: str,
        dpi: int = 144,
        max_pages: Optional[int] = None,
        prompt: Optional[str] = None
    ) -> BatchOCRResult:
        """
        识别PDF文档

        Args:
            pdf_path: PDF文件路径
            dpi: 转换图片的DPI（影响清晰度）
            max_pages: 最大处理页数（None表示全部）
            prompt: OCR提示词

        Returns:
            BatchOCRResult对象
        """
        start_time = time.time()
        results = []
        errors = []

        if not PYMUPDF_AVAILABLE:
            return BatchOCRResult(
                success=False,
                total_pages=0,
                processed_pages=0,
                errors=["PyMuPDF未安装，请安装: pip install pymupdf"]
            )

        if not os.path.exists(pdf_path):
            return BatchOCRResult(
                success=False,
                total_pages=0,
                processed_pages=0,
                errors=[f"文件不存在: {pdf_path}"]
            )

        try:
            # 打开PDF
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count

            if max_pages:
                total_pages = min(total_pages, max_pages)

            # 转换每一页为图片
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    img_bytes = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_bytes))

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # OCR识别
                    default_prompt = "<image>\n<|grounding|>Convert the document to markdown."
                    ocr_result = self._ocr_image_sync(image, prompt or default_prompt)
                    ocr_result.page_num = page_num + 1

                    results.append(ocr_result)

                except Exception as e:
                    errors.append(f"第{page_num + 1}页识别失败: {str(e)}")

            doc.close()

            total_time = time.time() - start_time

            return BatchOCRResult(
                success=len(errors) == 0,
                total_pages=total_pages,
                processed_pages=len(results),
                results=results,
                total_time=total_time,
                errors=errors
            )

        except Exception as e:
            total_time = time.time() - start_time
            return BatchOCRResult(
                success=False,
                total_pages=0,
                processed_pages=0,
                total_time=total_time,
                errors=[str(e)]
            )

    def ocr_pdf_batch(
        self,
        pdf_path: str,
        dpi: int = 144,
        max_pages: Optional[int] = None,
        batch_size: int = 4
    ) -> BatchOCRResult:
        """
        批量并发识别PDF文档

        Args:
            pdf_path: PDF文件路径
            dpi: 转换图片的DPI
            max_pages: 最大处理页数
            batch_size: 批处理大小

        Returns:
            BatchOCRResult对象
        """
        start_time = time.time()
        results = []
        errors = []

        if not PYMUPDF_AVAILABLE:
            return BatchOCRResult(
                success=False,
                total_pages=0,
                processed_pages=0,
                errors=["PyMuPDF未安装"]
            )

        try:
            # 打开PDF并转换为图片列表
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count

            if max_pages:
                total_pages = min(total_pages, max_pages)

            # 转换所有页为图片
            images = []
            for page_num in range(total_pages):
                page = doc[page_num]
                matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))

                if image.mode != "RGB":
                    image = image.convert("RGB")

                images.append((page_num, image))

            doc.close()

            # 批量并发OCR
            default_prompt = "<image>\n<|grounding|>Convert the document to markdown."

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for page_num, image in images:
                    future = executor.submit(self._ocr_image_sync, image, default_prompt)
                    futures.append((page_num, future))

                for page_num, future in futures:
                    try:
                        result = future.result(timeout=self.timeout)
                        result.page_num = page_num + 1
                        results.append(result)
                    except Exception as e:
                        errors.append(f"第{page_num + 1}页识别失败: {str(e)}")

            # 按页码排序
            results.sort(key=lambda x: x.page_num)

            total_time = time.time() - start_time

            return BatchOCRResult(
                success=len(errors) == 0,
                total_pages=total_pages,
                processed_pages=len(results),
                results=results,
                total_time=total_time,
                errors=errors
            )

        except Exception as e:
            total_time = time.time() - start_time
            return BatchOCRResult(
                success=False,
                total_pages=0,
                processed_pages=0,
                total_time=total_time,
                errors=[str(e)]
            )

    def get_full_markdown(self, result: BatchOCRResult) -> str:
        """
        获取完整的Markdown文本

        Args:
            result: BatchOCRResult对象

        Returns:
            完整的Markdown文本
        """
        markdown_parts = []

        for ocr_result in result.results:
            if ocr_result.markdown:
                markdown_parts.append(f"\n\n<!-- Page {ocr_result.page_num} -->\n\n")
                markdown_parts.append(ocr_result.markdown)

        return "\n".join(markdown_parts)

    def get_full_text(self, result: BatchOCRResult) -> str:
        """
        获取完整的纯文本

        Args:
            result: BatchOCRResult对象

        Returns:
            完整的纯文本
        """
        text_parts = []

        for ocr_result in result.results:
            if ocr_result.text:
                text_parts.append(f"\n\n--- Page {ocr_result.page_num} ---\n\n")
                text_parts.append(ocr_result.text)

        return "\n".join(text_parts)


# 创建全局实例（需要在配置中设置API地址）
_deepseek_ocr_service = None


def get_deepseek_ocr_service() -> DeepSeekOCRService:
    """获取DeepSeek-OCR服务实例"""
    global _deepseek_ocr_service
    if _deepseek_ocr_service is None:
        from app.core.config import settings
        api_base = getattr(settings, 'deepseek_ocr_api_base', 'http://localhost:8002')
        _deepseek_ocr_service = DeepSeekOCRService(api_base=api_base)
    return _deepseek_ocr_service
