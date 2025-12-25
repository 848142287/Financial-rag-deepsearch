"""
Docling解析器
基于开源Docling项目的高级文档解析器
专门为文档解析优化，支持多种布局和格式
"""

import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ..base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)


class DoclingParser(BaseFileParser):
    """Docling文档解析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.parser_name = "Docling"
        self.supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.html', '.htm', '.md', '.odt', '.ods', '.odp']

        # Docling配置
        self.docling_path = self.config.get('docling_path', 'docling')
        self.output_dir = self.config.get('output_dir', './docling_output')
        self.backend = self.config.get('backend', 'auto')  # auto, pypdfium, pypdfium2
        self.extract_images = self.config.get('extract_images', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_figures = self.config.get('extract_figures', True)
        self.language = self.config.get('language', 'zh')

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        ext = file_extension or Path(file_path).suffix.lower()

        # 检查Docling是否可用
        if not self._is_docling_available():
            logger.warning("Docling not available")
            return False

        return ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """解析文件内容"""
        try:
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)

            # 构建Docling命令
            cmd = self._build_docling_command(file_path, **kwargs)

            # 执行解析
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get('timeout', 300),
                cwd=self.output_dir
            )

            if result.returncode != 0:
                error_msg = f"Docling parsing failed: {result.stderr}"
                logger.error(error_msg)
                return ParseResult(
                    content="",
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )

            # 解析输出结果
            content, metadata = await self._parse_docling_output(file_path)

            # 提取元数据
            file_info = self._extract_file_metadata(file_path)
            metadata.update(file_info)

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except subprocess.TimeoutExpired:
            error_msg = f"Docling parsing timeout for {file_path}"
            logger.error(error_msg)
            return ParseResult(
                content="",
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"Docling parsing error for {file_path}: {str(e)}"
            logger.error(error_msg)
            return ParseResult(
                content="",
                metadata={'error': str(e)},
                success=False,
                error_message=error_msg
            )

    def _is_docling_available(self) -> bool:
        """检查Docling是否可用"""
        try:
            result = subprocess.run(
                [self.docling_path, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _build_docling_command(self, file_path: str, **kwargs) -> List[str]:
        """构建Docling命令"""
        cmd = [self.docling_path, file_path]

        # 输出格式
        output_format = kwargs.get('output_format', 'markdown')
        cmd.extend(['--output', f'{output_format}'])

        # 输出目录
        cmd.extend(['--output-dir', self.output_dir])

        # 后端选择
        if self.backend != 'auto':
            cmd.extend(['--backend', self.backend])

        # 语言设置
        if self.language:
            cmd.extend(['--language', self.language])

        # 提取选项
        extract_options = []
        if self.extract_images:
            extract_options.append('images')
        if self.extract_tables:
            extract_options.append('tables')
        if self.extract_figures:
            extract_options.append('figures')

        if extract_options:
            cmd.extend(['--extract', ','.join(extract_options)])

        # 其他选项
        if kwargs.get('ocr_enabled', False):
            cmd.extend(['--ocr'])

        if kwargs.get('debug', False):
            cmd.extend(['--debug'])

        return cmd

    async def _parse_docling_output(self, file_path: str) -> tuple[str, Dict]:
        """解析Docling输出结果"""
        try:
            # Docling通常输出JSON或Markdown文件
            output_files = self._find_output_files(file_path)

            if not output_files:
                return "", {"error": "No output files found"}

            content_parts = []
            metadata = {
                'output_files': output_files,
                'parser_version': 'Docling',
                'backend': self.backend,
                'extract_images': self.extract_images,
                'extract_tables': self.extract_tables,
                'extract_figures': self.extract_figures,
                'language': self.language
            }

            # 解析Markdown/文本输出
            for output_file in output_files:
                if output_file.endswith(('.md', '.txt', '.html')):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content_parts.append(content)

                    # 提取资源信息
                    if self.extract_images or self.extract_figures:
                        assets_dir = os.path.join(self.output_dir, 'assets')
                        if os.path.exists(assets_dir):
                            metadata['extracted_assets'] = self._list_files(assets_dir)

                    # 提取表格数据
                    if self.extract_tables:
                        tables_file = self._find_tables_file(file_path)
                        if tables_file:
                            with open(tables_file, 'r', encoding='utf-8') as f:
                                tables_data = json.load(f)
                                metadata['tables_data'] = tables_data

                    # 解析JSON元数据
                    if output_file.endswith('.json'):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            metadata.update(json_data)

            content = '\n\n'.join(content_parts)
            return content, metadata

        except Exception as e:
            logger.error(f"Error parsing Docling output: {str(e)}")
            return "", {"error": str(e)}

    def _find_output_files(self, file_path: str) -> List[str]:
        """查找输出文件"""
        output_files = []

        # Docling通常创建与输入文件同名的输出文件
        base_name = Path(file_path).stem

        # 常见的输出文件扩展名
        extensions = ['.md', '.txt', '.json', '.html']

        # 在输出目录中查找
        for ext in extensions:
            output_file = os.path.join(self.output_dir, f"{base_name}{ext}")
            if os.path.exists(output_file):
                output_files.append(output_file)

        return output_files

    def _find_tables_file(self, file_path: str) -> Optional[str]:
        """查找表格数据文件"""
        base_name = Path(file_path).stem
        tables_file = os.path.join(self.output_dir, f"{base_name}_tables.json")

        return tables_file if os.path.exists(tables_file) else None

    def _list_files(self, directory: str) -> List[str]:
        """列出目录中的所有文件"""
        try:
            return [f.name for f in Path(directory).iterdir() if f.is_file()]
        except Exception:
            return []

    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取文件元数据"""
        try:
            file_info = Path(file_path).stat()
            return {
                'file_size': file_info.st_size,
                'file_modified': file_info.st_mtime,
                'file_created': file_info.st_ctime,
                'file_extension': Path(file_path).suffix.lower()
            }
        except Exception:
            return {}

    def get_parser_capabilities(self) -> Dict[str, Any]:
        """获取解析器能力信息"""
        return {
            'supports_ocr': True,  # Docling支持OCR
            'supports_image_extraction': self.extract_images,
            'supports_table_extraction': self.extract_tables,
            'supports_figure_extraction': self.extract_figures,
            'supported_backends': ['auto', 'pypdfium', 'pypdfium2'],
            'supported_languages': ['zh', 'en', 'ja', 'ko', 'auto'],
            'output_formats': ['markdown', 'text', 'json', 'html'],
            'docling_available': self._is_docling_available(),
            'docling_path': self.docling_path,
            'current_backend': self.backend
        }

    def get_supported_layouts(self) -> List[str]:
        """获取支持的布局类型"""
        return [
            'text',
            'title',
            'paragraph',
            'list',
            'table',
            'figure',
            'caption',
            'header',
            'footer',
            'page_number',
            'footnote'
        ]

    def detect_document_structure(self, file_path: str) -> Dict[str, Any]:
        """检测文档结构"""
        try:
            if not self.can_parse(file_path):
                return {"error": "Unsupported file type"}

            # 使用Docling的分析功能
            cmd = [self.docling_path, file_path, '--analyze', '--output', 'json']

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.output_dir
            )

            if result.returncode == 0:
                # 解析分析结果
                analysis_file = os.path.join(self.output_dir, f"{Path(file_path.stem}_analysis.json")
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        return json.load(f)

            return {"error": "Failed to analyze document structure"}

        except Exception as e:
            logger.error(f"Error detecting document structure: {str(e)}")
            return {"error": str(e)}