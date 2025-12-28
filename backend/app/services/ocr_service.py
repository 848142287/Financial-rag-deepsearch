"""
OCR服务 - 使用Qwen-VL进行图像文本识别
支持扫描文档、图片中的文本提取
"""

import logging
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """OCR配置"""
    api_key: str = "sk-5233a3a4b1a24426b6846a432794bbe2"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-vl-max"  # 使用Qwen VL模型进行OCR
    timeout: int = 60
    max_retries: int = 3


class QwenOCRService:
    """
    基于Qwen-VL的OCR服务
    用于提取扫描文档和图片中的文本
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._client = None

    async def _get_client(self):
        """获取HTTP客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=self.config.timeout
            )
        return self._client

    async def extract_text_from_image(
        self,
        image_bytes: bytes,
        prompt: str = "请识别图片中的所有文字内容，包括标题、正文、表格等。请按原文的格式和结构输出识别结果。"
    ) -> Dict[str, Any]:
        """
        从图片中提取文本

        Args:
            image_bytes: 图片字节内容
            prompt: 提示词

        Returns:
            包含识别结果的字典
        """
        try:
            # 编码图片为base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            client = await self._get_client()

            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            for attempt in range(self.config.max_retries):
                try:
                    response = await client.post("/chat/completions", json=payload)
                    response.raise_for_status()
                    result = response.json()

                    extracted_text = result['choices'][0]['message']['content']

                    return {
                        'success': True,
                        'text': extracted_text,
                        'model': self.config.model,
                        'prompt_tokens': result.get('usage', {}).get('prompt_tokens', 0),
                        'completion_tokens': result.get('usage', {}).get('completion_tokens', 0),
                        'total_tokens': result.get('usage', {}).get('total_tokens', 0)
                    }

                except httpx.HTTPStatusError as e:
                    logger.warning(f"OCR请求失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                    else:
                        raise

        except Exception as e:
            logger.error(f"OCR文本提取失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }

    async def extract_text_from_pdf_page(
        self,
        pdf_bytes: bytes,
        page_num: int = 0
    ) -> Dict[str, Any]:
        """
        从PDF页面提取文本(先尝试PyPDF2,如果失败则使用OCR)

        智能优化: 如果PyPDF2能提取足够文本,跳过OCR

        Args:
            pdf_bytes: PDF文件字节
            page_num: 页码(从0开始)

        Returns:
            提取结果
        """
        import PyPDF2
        import io

        try:
            # 首先尝试使用PyPDF2提取文本
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            if page_num >= len(pdf_reader.pages):
                return {
                    'success': False,
                    'error': f'页码 {page_num} 超出范围 (共 {len(pdf_reader.pages)} 页)',
                    'text': ''
                }

            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # 智能判断: 如果文本质量足够高,跳过OCR
            # 降低阈值从100到50字符,提高跳过率
            if text and len(text.strip()) > 50:
                # 进一步检查文本质量(不是乱码)
                # 检查中文字符比例
                chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
                total_chars = len(text.strip())
                chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

                # 如果中文字符占比>10%或总字符>100,认为文本质量好
                if chinese_ratio > 0.1 or total_chars > 100:
                    logger.info(f"✅ 第 {page_num + 1} 页跳过OCR(PyPDF2提取: {len(text.strip())}字符)")
                    return {
                        'success': True,
                        'text': text,
                        'method': 'PyPDF2',
                        'confidence': 'high',
                        'skipped_ocr': True  # 标记跳过了OCR
                    }

            # 文本质量不足,需要OCR
            logger.info(f"🔄 第 {page_num + 1} 页文本质量不足,使用OCR")
            return await self._ocr_pdf_page(pdf_bytes, page_num)

        except Exception as e:
            logger.error(f"PDF页面文本提取失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }

    async def _ocr_pdf_page(
        self,
        pdf_bytes: bytes,
        page_num: int
    ) -> Dict[str, Any]:
        """
        将PDF页面转换为图片并使用OCR识别

        Args:
            pdf_bytes: PDF字节
            page_num: 页码

        Returns:
            OCR结果
        """
        try:
            # 将PDF页面转换为图片
            from pdf2image import convert_from_bytes

            # 转换指定页面
            images = await asyncio.to_thread(
                convert_from_bytes,
                pdf_bytes,
                dpi=200,
                first_page=page_num + 1,
                last_page=page_num + 1
            )

            if not images:
                return {
                    'success': False,
                    'error': f'无法转换PDF第 {page_num + 1} 页为图片',
                    'text': ''
                }

            # 获取第一张图片
            image = images[0]

            # 转换为字节
            from io import BytesIO
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            image_bytes = img_buffer.getvalue()

            # 使用OCR识别
            ocr_result = await self.extract_text_from_image(image_bytes)

            return {
                'success': ocr_result['success'],
                'text': ocr_result.get('text', ''),
                'method': 'OCR',
                'model': ocr_result.get('model', 'qwen-vl-max'),
                'error': ocr_result.get('error', '')
            }

        except ImportError:
            logger.warning("pdf2image未安装,无法进行PDF OCR")
            return {
                'success': False,
                'error': 'pdf2image未安装,请安装: pip install pdf2image',
                'text': ''
            }
        except Exception as e:
            logger.error(f"PDF OCR失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }

    async def extract_structured_text(
        self,
        image_bytes: bytes,
        structure_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        提取结构化文本(标题、段落、表格等)

        Args:
            image_bytes: 图片字节
            structure_types: 需要提取的结构类型

        Returns:
            结构化提取结果
        """
        if structure_types is None:
            structure_types = ['title', 'paragraph', 'table', 'list']

        prompt = f"""请分析这张图片并提取以下结构化内容:
{', '.join(structure_types)}

请以JSON格式返回结果,包含:
- titles: 标题列表(带层级)
- paragraphs: 段落列表
- tables: 表格数据
- lists: 列表项

如果某些内容不存在,返回空数组。"""

        result = await self.extract_text_from_image(image_bytes, prompt)

        if result['success']:
            # 尝试解析JSON响应
            try:
                import json
                structured_data = json.loads(result['text'])
                return {
                    'success': True,
                    'structured': structured_data,
                    'raw_text': result['text']
                }
            except json.JSONDecodeError:
                # JSON解析失败,返回原始文本
                return {
                    'success': True,
                    'structured': None,
                    'raw_text': result['text'],
                    'note': '无法解析为JSON,返回原始文本'
                }

        return result

    async def batch_extract_from_pdf(
        self,
        pdf_bytes: bytes,
        pages: Optional[List[int]] = None,
        max_concurrent: int = 12  # 默认从3提升到12
    ) -> List[Dict[str, Any]]:
        """
        批量从PDF提取文本(支持并发)

        Args:
            pdf_bytes: PDF字节
            pages: 要处理的页码列表(None表示全部)
            max_concurrent: 最大并发数(默认12，适配16 worker配置)

        Returns:
            每页的提取结果列表
        """
        import PyPDF2
        import io

        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        total_pages = len(pdf_reader.pages)

        if pages is None:
            pages = list(range(total_pages))

        # 创建并发任务
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(page_num):
            async with semaphore:
                return await self.extract_text_from_pdf_page(pdf_bytes, page_num)

        # 并发执行
        tasks = [extract_with_semaphore(page_num) for page_num in pages]
        results = await asyncio.gather(*tasks)

        logger.info(f"批量OCR完成: {len(results)} 页")
        return results


# 全局OCR服务实例
_ocr_service = None


def get_ocr_service() -> QwenOCRService:
    """获取OCR服务实例"""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = QwenOCRService()
    return _ocr_service
