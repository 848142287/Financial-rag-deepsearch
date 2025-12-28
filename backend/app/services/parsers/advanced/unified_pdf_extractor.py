"""
统一 PDF 提取服务
支持快速模式（PyMuPDF4LLM）和精确模式（VLM）
集成了 Multimodal_RAG 的提取功能，使用现有系统的配置
"""

import io
import base64
import re
import tempfile
import shutil
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    import pymupdf4llm
    import fitz  # PyMuPDF
    from PIL import Image
    from pdf2image import convert_from_bytes
except ImportError as e:
    print(f"警告: 缺少依赖库 {e}. 请安装: pip install pymupdf4llm pymupdf Pillow pdf2image")
    raise

from app.core.config import settings
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """提取结果数据类"""
    filename: str = ""
    markdown_content: str = ""
    tables: List[Dict[str, Any]] = None
    formulas: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    token_usage: Dict[str, int] = None
    time_cost: Dict[str, float] = None
    page_images: List[Any] = None
    per_page_results: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.formulas is None:
            self.formulas = []
        if self.metadata is None:
            self.metadata = {}
        if self.token_usage is None:
            self.token_usage = {}
        if self.time_cost is None:
            self.time_cost = {}
        if self.page_images is None:
            self.page_images = []
        if self.per_page_results is None:
            self.per_page_results = []


class UnifiedPDFExtractor:
    """统一 PDF 提取服务"""

    def __init__(self):
        self.default_dpi = 100
        self.default_pages_per_request = 1

    async def extract_fast(
        self,
        file_path: str,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        快速模式：使用 PyMuPDF4LLM 提取

        特点：
        1. 页码标记：在每页开头加 {{第X页}}，方便后续分页处理
        2. 图片提取：提取 PDF 中的图片（不是截图整页），返回 base64
        3. 给每页生成完整截图
        """
        logger.info("="*60)
        logger.info("快速模式提取 - 使用 PyMuPDF4LLM")
        logger.info("="*60)

        # 获取文件名
        filename = original_filename or Path(file_path).name

        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        temp_dir = Path(tempfile.mkdtemp())
        temp_images_dir = temp_dir / "images"
        temp_images_dir.mkdir(exist_ok=True)

        logger.info("正在提取 PDF 内容和图片...")

        try:
            # 使用 PyMuPDF4LLM 提取
            md_data = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,
                write_images=True,
                image_path=str(temp_images_dir),
                image_format="png",
                dpi=150
            )

            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            logger.info(f"文档共 {total_pages} 页")

            markdown_parts = []

            # 处理 markdown 内容
            if isinstance(md_data, list):
                for idx, page_data in enumerate(md_data):
                    page_num = idx + 1
                    if isinstance(page_data, dict):
                        text = page_data.get('text', '')
                    else:
                        text = str(page_data)

                    text = text.replace(str(temp_images_dir.absolute()), "images")
                    markdown_parts.append(f"{{{{第{page_num}页}}}}\n{text}\n")
            else:
                text = str(md_data)
                text = text.replace(str(temp_images_dir.absolute()), "images")

                if "-----" in text or "---" in text:
                    pages = text.split("-----") if "-----" in text else text.split("---")
                    for idx, page_text in enumerate(pages):
                        if page_text.strip():
                            page_num = idx + 1
                            markdown_parts.append(f"{{{{第{page_num}页}}}}\n{page_text.strip()}\n")
                else:
                    for page_num in range(1, total_pages + 1):
                        markdown_parts.append(f"{{{{第{page_num}页}}}}\n")
                    markdown_parts.append(text)

            # 收集提取的图片
            logger.info("正在收集提取的图片...")
            images_data = []

            for img_file in sorted(temp_images_dir.glob("*.png")):
                try:
                    img = Image.open(img_file)
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                    img_filename = img_file.name
                    page_num = 1
                    match = re.search(r'(\d+)', img_filename)
                    if match:
                        page_num = int(match.group(1))

                    images_data.append({
                        "filename": img_filename,
                        "base64": img_base64,
                        "page_num": page_num
                    })

                    logger.info(f"  ✓ {img_filename}")

                except Exception as e:
                    logger.error(f"处理图片失败 {img_file.name}: {e}")

            # 生成完整页面截图
            logger.info("正在生成页面完整截图...")
            for page_num in range(total_pages):
                page = doc[page_num]
                logger.info(f"  处理第 {page_num + 1}/{total_pages} 页")

                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))

                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                    filename = f"page_{page_num + 1}_full.png"
                    images_data.append({
                        "filename": filename,
                        "base64": img_base64,
                        "page_num": page_num + 1
                    })

                    # 在 markdown 中添加截图链接
                    markdown_parts.append(f"\n![{filename}](images/{filename})\n")
                    pix = None

                except Exception as e:
                    logger.error(f"截图失败: {e}")

            doc.close()

            final_markdown = "".join(markdown_parts)

            logger.info("="*60)
            logger.info("✓ 快速提取完成")
            logger.info(f"  - 页数: {total_pages}")
            logger.info(f"  - 图片数: {len(images_data)}")
            logger.info(f"  - Markdown长度: {len(final_markdown)} 字符")
            logger.info("="*60)

            return {
                "filename": filename,
                "markdown": final_markdown,
                "images": images_data,
                "metadata": {
                    "total_pages": total_pages,
                    "total_images": len(images_data),
                    "extraction_mode": "fast"
                }
            }

        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    async def extract_accurate(
        self,
        file_path: str,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        精确模式：使用现有系统的 Qwen VL 模型提取

        特点：
        1. 使用系统的 Qwen VL Plus 模型
        2. 批量处理页面
        3. 返回结构化的 JSON（markdown、表格、公式、图片描述）
        """
        logger.info("="*60)
        logger.info(f"精确模式提取 - 使用 Qwen VL ({settings.qwen_multimodal_model})")
        logger.info("="*60)

        filename = original_filename or Path(file_path).name

        # 使用系统的多模态提取器
        from .enhanced_llm_multimodal_extractor import EnhancedLLMMultimodalExtractor

        extractor = EnhancedLLMMultimodalExtractor(
            pages_per_request=self.default_pages_per_request
        )

        result = await extractor.extract_from_pdf(file_path, original_filename=filename)

        # 组装最终 markdown
        markdown_parts = []
        for page_result in result.per_page_results:
            page_num = page_result['page_num']
            page_markdown = page_result.get('markdown', '')

            # 添加页码标识符（用于分隔不同页面）
            markdown_parts.append(f"{{{{第{page_num}页}}}}\n{page_markdown}\n")

        final_markdown = "".join(markdown_parts)

        total_image_descriptions = sum(len(p.get('images', [])) for p in result.per_page_results)

        logger.info("="*60)
        logger.info("✓ 精确提取完成")
        logger.info(f"  - 页数: {result.metadata['total_pages']}")
        logger.info(f"  - 表格数: {result.metadata['total_tables']}")
        logger.info(f"  - 公式数: {result.metadata['total_formulas']}")
        logger.info(f"  - 图片描述: {total_image_descriptions} 个")
        logger.info(f"  - Token使用: {result.token_usage['total_tokens']:,}")
        logger.info(f"  - 耗时: {result.time_cost['total_time']}秒")
        logger.info(f"  - Markdown长度: {len(final_markdown)} 字符")
        logger.info("="*60)

        return {
            "filename": filename,
            "markdown": final_markdown,
            "images": [],  # 精确模式不返回图片 base64，因为图片描述已在 markdown 中
            "metadata": {
                "total_pages": result.metadata['total_pages'],
                "total_tables": result.metadata['total_tables'],
                "total_formulas": result.metadata['total_formulas'],
                "total_image_descriptions": total_image_descriptions,
                "token_usage": result.token_usage,
                "time_cost": result.time_cost,
                "extraction_mode": "accurate"
            }
        }

    async def extract_with_auto_mode(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
        force_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        自动模式：根据文件大小和页数自动选择提取模式

        Args:
            file_path: PDF 文件路径
            original_filename: 原始文件名
            force_mode: 强制使用的模式 ("fast" 或 "accurate")

        Returns:
            提取结果
        """
        if force_mode:
            logger.info(f"使用强制模式: {force_mode}")
            if force_mode == "fast":
                return await self.extract_fast(file_path, original_filename)
            else:
                return await self.extract_accurate(file_path, original_filename)

        # 自动判断
        doc = fitz.open(file_path)
        page_count = len(doc)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        doc.close()

        # 判断逻辑：页数 > 20 或文件大小 > 50MB 使用快速模式
        if page_count > 20 or file_size > 50:
            logger.info(f"自动选择快速模式 (页数: {page_count}, 文件大小: {file_size:.2f}MB)")
            return await self.extract_fast(file_path, original_filename)
        else:
            logger.info(f"自动选择精确模式 (页数: {page_count}, 文件大小: {file_size:.2f}MB)")
            return await self.extract_accurate(file_path, original_filename)


# 全局实例
unified_pdf_extractor = UnifiedPDFExtractor()
