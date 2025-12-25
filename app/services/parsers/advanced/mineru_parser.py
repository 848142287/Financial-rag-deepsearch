"""
MinerU解析器
基于开源MinerU项目的高级文档解析器
支持多种文档格式的解析，包括PDF、Word、Excel等
"""

import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)


class MinerUParser(BaseFileParser):
    """MinerU文档解析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.parser_name = "MinerU"
        self.supported_extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.html', '.htm', '.md', '.txt']

        # MinerU配置
        self.mineru_path = self.config.get('mineru_path', 'mineru')
        self.output_dir = self.config.get('output_dir', './mineru_output')
        self.extract_images = self.config.get('extract_images', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.ocr_enabled = self.config.get('ocr_enabled', True)
        self.language = self.config.get('language', 'zh')

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        ext = file_extension or Path(file_path).suffix.lower()

        # 检查MinerU是否可用
        if not self._is_mineru_available():
            logger.warning("MinerU not available")
            return False

        return ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """解析文件内容"""
        try:
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)

            # 构建MinerU命令
            cmd = self._build_mineru_command(file_path, **kwargs)

            # 执行解析
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get('timeout', 300),
                cwd=self.output_dir
            )

            if result.returncode != 0:
                error_msg = f"MinerU parsing failed: {result.stderr}"
                logger.error(error_msg)
                return ParseResult(
                    content="",
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )

            # 解析输出结果
            content, metadata = await self._parse_mineru_output(file_path)

            # 提取元数据
            file_info = self._extract_file_metadata(file_path)
            metadata.update(file_info)

            return ParseResult(
                content=content,
                metadata=metadata,
                success=True
            )

        except subprocess.TimeoutExpired:
            error_msg = f"MinerU parsing timeout for {file_path}"
            logger.error(error_msg)
            return ParseResult(
                content="",
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"MinerU parsing error for {file_path}: {str(e)}"
            logger.error(error_msg)
            return ParseResult(
                content="",
                metadata={'error': str(e)},
                success=False,
                error_message=error_msg
            )

    def _is_mineru_available(self) -> bool:
        """检查MinerU是否可用"""
        try:
            result = subprocess.run(
                [self.mineru_path, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _build_mineru_command(self, file_path: str, **kwargs) -> List[str]:
        """构建MinerU命令"""
        cmd = [self.mineru_path]

        # 基本参数
        cmd.extend([
            file_path,
            '--output', self.output_dir,
            '--format', kwargs.get('format', 'markdown')
        ])

        # OCR配置
        if self.ocr_enabled:
            cmd.extend(['--ocr'])
            if self.language:
                cmd.extend(['--language', self.language])

        # 图像提取
        if self.extract_images:
            cmd.extend(['--extract_images'])

        # 表格提取
        if self.extract_tables:
            cmd.extend(['--extract_tables'])

        # 应用自定义配置
        custom_config = kwargs.get('mineru_config')
        if custom_config:
            cmd.extend(['--config', custom_config])

        return cmd

    async def _parse_mineru_output(self, file_path: str) -> tuple[str, Dict]:
        """解析MinerU输出结果"""
        try:
            # 查找输出文件
            output_files = self._find_output_files(file_path)

            if not output_files:
                return "", {"error": "No output files found"}

            content_parts = []
            metadata = {
                'output_files': output_files,
                'parser_version': 'MinerU',
                'extract_images': self.extract_images,
                'extract_tables': self.extract_tables,
                'ocr_enabled': self.ocr_enabled
            }

            # 解析Markdown输出
            for output_file in output_files:
                if output_file.endswith('.md'):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                        content_parts.append(md_content)

                    # 提取图像信息
                    if self.extract_images:
                        images_dir = os.path.join(self.output_dir, 'images')
                        if os.path.exists(images_dir):
                            metadata['extracted_images'] = self._list_files(images_dir)

                    # 提取表格信息
                    if self.extract_tables:
                        tables_dir = os.path.join(self.output_dir, 'tables')
                        if os.path.exists(tables_dir):
                            metadata['extracted_tables'] = self._list_files(tables_dir)

            # 解析JSON元数据（如果存在）
            json_file = self._find_metadata_file(file_path)
            if json_file:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_metadata = json.load(f)
                    metadata.update(json_metadata)

            content = '\n\n'.join(content_parts)
            return content, metadata

        except Exception as e:
            logger.error(f"Error parsing MinerU output: {str(e)}")
            return "", {"error": str(e)}

    def _find_output_files(self, file_path: str) -> List[str]:
        """查找输出文件"""
        output_files = []

        # 常见的输出文件扩展名
        extensions = ['.md', '.txt', '.json', '.html']

        # 在输出目录中查找
        for ext in extensions:
            pattern = os.path.join(self.output_dir, f"*{ext}")
            output_files.extend(Path(self.output_dir).glob(pattern))

        return sorted([str(f) for f in output_files])

    def _find_metadata_file(self, file_path: str) -> Optional[str]:
        """查找元数据文件"""
        # 尝试不同的元数据文件名
        metadata_patterns = [
            f"{Path(file_path).stem}_metadata.json",
            "metadata.json",
            "document_info.json"
        ]

        for pattern in metadata_patterns:
            metadata_file = os.path.join(self.output_dir, pattern)
            if os.path.exists(metadata_file):
                return metadata_file

        return None

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
            'supports_ocr': self.ocr_enabled,
            'supports_image_extraction': self.extract_images,
            'supports_table_extraction': self.extract_tables,
            'supported_languages': ['zh', 'en', 'ja', 'ko', 'auto'],
            'output_formats': ['markdown', 'text', 'json', 'html'],
            'mineru_available': self._is_mineru_available(),
            'mineru_path': self.mineru_path
        }