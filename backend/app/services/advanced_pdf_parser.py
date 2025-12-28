"""
高级PDF解析服务
使用PyMuPDF4LLM进行高质量的PDF内容提取
支持文本、表格、图片的结构化提取
支持OCR结果缓存以提升性能
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import io
import re
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PDFParseConfig:
    """PDF解析配置"""
    extract_images: bool = True
    extract_tables: bool = True
    extract_markdown: bool = True
    extract_structured: bool = True
    page_range: Optional[Tuple[int, int]] = None  # (start, end)
    ocr_fallback: bool = True  # 当文本提取失败时使用OCR


class AdvancedPDFParser:
    """
    高级PDF解析器
    使用PyMuPDF4LLM提供更好的PDF内容提取
    """

    def __init__(self, config: Optional[PDFParseConfig] = None):
        self.config = config or PDFParseConfig()
        self._ocr_service = None
        self._redis_client = None

    def _get_ocr_service(self):
        """获取OCR服务"""
        if self._ocr_service is None:
            from app.services.ocr_service import get_ocr_service
            self._ocr_service = get_ocr_service()
        return self._ocr_service

    async def _get_redis_client(self):
        """获取Redis客户端"""
        if self._redis_client is None:
            try:
                import redis.asyncio as redis
                self._redis_client = await redis.Redis(
                    host='redis',
                    port=6379,
                    password='redis123456',
                    db=2,  # 使用独立的DB用于OCR缓存
                    decode_responses=False
                )
            except Exception as e:
                logger.warning(f"Redis连接失败，缓存功能将不可用: {e}")
                self._redis_client = False
        return None if self._redis_client is False else self._redis_client

    def _compute_file_hash(self, pdf_bytes: bytes) -> str:
        """计算文件内容的MD5哈希"""
        return hashlib.md5(pdf_bytes).hexdigest()

    async def _get_cached_ocr_result(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """从Redis获取缓存的OCR结果"""
        try:
            redis_client = await self._get_redis_client()
            if redis_client is None:
                return None

            cache_key = f"ocr_result:{file_hash}"
            cached_data = await redis_client.get(cache_key)

            if cached_data:
                result = json.loads(cached_data)
                logger.info(f"✅ 从Redis获取OCR缓存: {file_hash[:8]}...")
                return result
            else:
                return None

        except Exception as e:
            logger.warning(f"获取OCR缓存失败: {e}")
            return None

    async def _cache_ocr_result(self, file_hash: str, result: Dict[str, Any], ttl: int = 86400):
        """将OCR结果缓存到Redis"""
        try:
            redis_client = await self._get_redis_client()
            if redis_client is None:
                return

            cache_key = f"ocr_result:{file_hash}"
            cache_data = json.dumps(result, ensure_ascii=False)

            await redis_client.setex(cache_key, ttl, cache_data)
            logger.info(f"✅ OCR结果已缓存: {file_hash[:8]}... (TTL: {ttl}秒)")

        except Exception as e:
            logger.warning(f"缓存OCR结果失败: {e}")

    async def parse_pdf(self, pdf_bytes: bytes, filename: str = "") -> Dict[str, Any]:
        """
        完整的PDF解析

        Args:
            pdf_bytes: PDF文件字节
            filename: 文件名

        Returns:
            解析结果
        """
        result = {
            'success': False,
            'filename': filename,
            'pages_processed': 0,
            'content': {
                'raw_text': '',
                'markdown': '',
                'structured': {}
            },
            'images': [],
            'tables': [],
            'metadata': {},
            'errors': []
        }

        try:
            # 方法1: 尝试使用PyMuPDF4LLM
            pymupdf_result = await self._parse_with_pymupdf4llm(pdf_bytes, filename)

            if pymupdf_result['success']:
                # 检查文本质量
                text_length = len(pymupdf_result['content']['raw_text'])

                if text_length > 100:  # 有足够的文本
                    logger.info(f"PyMuPDF4LLM解析成功: {text_length} 字符")
                    result.update(pymupdf_result)

                    # 如果文本质量仍然不高,尝试OCR增强
                    if text_length < 500 and self.config.ocr_fallback:
                        logger.info("文本量较少,尝试OCR增强")
                        ocr_result = await self._ocr_enhance_pdf(pdf_bytes)
                        if ocr_result['success'] and len(ocr_result['content']['raw_text']) > text_length:
                            logger.info("OCR增强成功,使用OCR结果")
                            result.update(ocr_result)
                else:
                    # 文本太少,可能是扫描文档,使用OCR
                    logger.info("文本量极少,可能是扫描文档,使用OCR")
                    if self.config.ocr_fallback:
                        ocr_result = await self._ocr_enhance_pdf(pdf_bytes)
                        result.update(ocr_result)
                    else:
                        result.update(pymupdf_result)

            else:
                # PyMuPDF4LLM失败,检查是否需要fallback到OCR
                if pymupdf_result.get('fallback_to_ocr') and self.config.ocr_fallback:
                    logger.warning("PyMuPDF4LLM遇到内部bug,使用OCR作为fallback")
                    ocr_result = await self._ocr_enhance_pdf(pdf_bytes)
                    if ocr_result['success']:
                        result.update(ocr_result)
                    else:
                        # OCR也失败了,尝试PyPDF2 fallback
                        fallback_result = await self._parse_with_fallback(pdf_bytes)
                        result.update(fallback_result)
                else:
                    # 使用普通fallback方法
                    logger.warning("PyMuPDF4LLM解析失败,使用fallback方法")
                    fallback_result = await self._parse_with_fallback(pdf_bytes)
                    result.update(fallback_result)

            result['success'] = True

        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            result['errors'].append(str(e))

        return result

    async def _parse_with_pymupdf4llm(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """使用PyMuPDF4LLM解析PDF"""
        import tempfile
        import os

        temp_pdf_path = None

        try:
            # 尝试导入pymupdf4llm
            try:
                import pymupdf4llm
            except ImportError:
                logger.warning("pymupdf4llm未安装")
                return {
                    'success': False,
                    'error': 'pymupdf4llm未安装: pip install pymupdf4llm'
                }

            # PyMuPDF4LLM需要文件路径，不能使用BytesIO
            # 创建临时文件
            try:
                # 创建临时PDF文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_bytes)
                    temp_pdf_path = tmp_file.name

                logger.info(f"创建临时PDF文件: {temp_pdf_path}")

                # 提取为Markdown
                if self.config.extract_markdown:
                    try:
                        md_text = await asyncio.to_thread(
                            pymupdf4llm.to_markdown,
                            temp_pdf_path  # 传递文件路径而不是BytesIO
                        )

                        # 检查返回的文本是否有效
                        if not md_text or not isinstance(md_text, str):
                            logger.warning("PyMuPDF4LLM返回空文本，尝试fallback方法")
                            return {
                                'success': False,
                                'error': 'Empty or invalid markdown returned'
                            }

                        # 提取纯文本
                        raw_text = await asyncio.to_thread(
                            self._extract_text_from_markdown,
                            md_text
                        )

                        # 提取结构化内容
                        structured = await self._extract_structured_from_markdown(md_text)

                        return {
                            'success': True,
                            'content': {
                                'raw_text': raw_text,
                                'markdown': md_text,
                                'structured': structured
                            },
                            'method': 'pymupdf4llm',
                            'pages_processed': raw_text.count('\f') + 1
                        }

                    except NameError as e:
                        # 特殊处理PyMuPDF4LLM库的内部bug (如 "name 'item' is not defined")
                        if 'item' in str(e):
                            logger.warning(f"PyMuPDF4LLM库内部错误({e}), 这是已知的库bug, 将使用OCR作为fallback")
                            return {
                                'success': False,
                                'error': f'PyMuPDF4LLM内部错误: {str(e)}',
                                'fallback_to_ocr': True
                            }
                        else:
                            logger.error(f"PyMuPDF4LLM解析失败(NameError): {e}")
                            return {
                                'success': False,
                                'error': str(e)
                            }
                    except Exception as e:
                        logger.error(f"PyMuPDF4LLM解析失败: {e}")
                        return {
                            'success': False,
                            'error': str(e)
                        }
                else:
                    return {
                        'success': False,
                        'error': 'Markdown提取未启用'
                    }

            except Exception as e:
                logger.error(f"临时文件处理失败: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }

        except Exception as e:
            logger.error(f"PyMuPDF4LLM处理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

        finally:
            # 清理临时文件
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                    logger.info(f"已删除临时文件: {temp_pdf_path}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {e}")

    async def _parse_with_fallback(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Fallback解析方法"""
        import PyPDF2

        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            all_text = []
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(f"--- Page {i+1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"第 {i+1} 页提取失败: {e}")

            raw_text = "\n\n".join(all_text)

            return {
                'success': True,
                'content': {
                    'raw_text': raw_text,
                    'markdown': self._convert_to_markdown(raw_text),
                    'structured': {}
                },
                'method': 'PyPDF2_fallback',
                'pages_processed': len(pdf_reader.pages)
            }

        except Exception as e:
            logger.error(f"Fallback解析失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'content': {'raw_text': '', 'markdown': '', 'structured': {}}
            }

    async def _ocr_enhance_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """使用OCR增强PDF解析（带缓存）"""
        # 计算文件哈希用于缓存
        file_hash = self._compute_file_hash(pdf_bytes)

        # 尝试从缓存获取结果
        cached_result = await self._get_cached_ocr_result(file_hash)
        if cached_result:
            logger.info(f"✅ 使用OCR缓存，跳过API调用")
            return {
                'success': True,
                'content': cached_result['content'],
                'method': 'OCR (cached)',
                'pages_processed': cached_result['pages_processed'],
                'models_used': cached_result.get('models_used', ['qwen-vl-max']),
                'from_cache': True
            }

        # 缓存未命中，执行OCR
        ocr_service = self._get_ocr_service()

        try:
            logger.info("🔄 缓存未命中，执行并行OCR...")
            # 使用批量OCR，提高并发数充分利用16个worker
            ocr_results = await ocr_service.batch_extract_from_pdf(
                pdf_bytes,
                max_concurrent=12  # 从3提升到12，充分利用16个worker
            )

            # 合并所有页面的文本
            all_text = []
            for i, page_result in enumerate(ocr_results):
                if page_result['success'] and page_result['text']:
                    all_text.append(f"--- Page {i+1} ---\n{page_result['text']}")

            raw_text = "\n\n".join(all_text)

            result = {
                'success': True,
                'content': {
                    'raw_text': raw_text,
                    'markdown': self._convert_to_markdown(raw_text),
                    'structured': {}
                },
                'method': 'OCR',
                'pages_processed': len(ocr_results),
                'models_used': ['qwen-vl-max']
            }

            # 缓存结果（24小时TTL）
            await self._cache_ocr_result(file_hash, result, ttl=86400)

            return result

        except Exception as e:
            logger.error(f"OCR增强失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'content': {'raw_text': '', 'markdown': '', 'structured': {}}
            }

    def _extract_text_from_markdown(self, md_text: str) -> str:
        """从Markdown中提取纯文本"""
        # 移除Markdown格式标记
        text = md_text

        # 移除标题标记
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

        # 移除加粗/斜体
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)

        # 移除链接
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # 移除图片
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)

        return text.strip()

    def _convert_to_markdown(self, text: str) -> str:
        """将纯文本转换为简单的Markdown"""
        lines = text.split('\n')
        md_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                md_lines.append('')
                continue

            # 检测标题
            if self._is_heading(line):
                md_lines.append(f"\n{line}\n")
            else:
                md_lines.append(line)

        return "\n".join(md_lines)

    def _is_heading(self, text: str) -> bool:
        """检测是否为标题"""
        heading_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节篇]',
            r'^\d+\.\d+\s+\S',
            r'^[一二三四五六七八九十]+[、.]',
            r'^\d{1,2}[、.]',
        ]
        return any(re.match(pattern, text) for pattern in heading_patterns)

    async def _extract_structured_from_markdown(self, md_text: str) -> Dict[str, Any]:
        """从Markdown中提取结构化内容"""
        structured = {
            'titles': [],
            'sections': [],
            'tables': [],
            'lists': []
        }

        lines = md_text.split('\n')
        current_section = None
        title_path = []

        for line in lines:
            # 提取标题
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()

                title_info = {
                    'level': level,
                    'title': title,
                    'path': title_path.copy() + [title]
                }

                if level == 1:
                    structured['titles'].append(title_info)
                    title_path = [title]
                    current_section = title
                else:
                    structured['titles'].append(title_info)

            # 提取表格(简单的Markdown表格)
            elif '|' in line and line.count('|') >= 2:
                if not structured['tables'] or line.strip().startswith('|-'):
                    continue
                structured['tables'].append({
                    'content': line,
                    'section': current_section
                })

            # 提取列表
            elif line.strip().startswith(('-', '*', '•')):
                structured['lists'].append({
                    'item': line.strip()[1:].strip(),
                    'section': current_section
                })

        return structured


# 全局解析器实例
_pdf_parser = None


def get_pdf_parser() -> AdvancedPDFParser:
    """获取PDF解析器实例"""
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = AdvancedPDFParser()
    return _pdf_parser
