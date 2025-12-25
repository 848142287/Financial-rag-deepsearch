"""
LibreOffice文档转换器
支持各种Office文档格式的转换
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import asyncio

from .base_converter import BaseConverter, ConversionResult, ConversionStatus

logger = logging.getLogger(__name__)


class LibreOfficeConverter(BaseConverter):
    """LibreOffice转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # LibreOffice可执行文件路径
        self.libreoffice_path = self.config.get(
            'libreoffice_path',
            self._find_libreoffice()
        )

        # 转换参数
        self.timeout = self.config.get('timeout', 300)  # 5分钟超时
        self.headless = self.config.get('headless', True)

        if not self.libreoffice_path:
            raise RuntimeError("未找到LibreOffice安装")

    def _find_libreoffice(self) -> Optional[str]:
        """查找LibreOffice安装路径"""
        possible_paths = [
            '/usr/bin/libreoffice',
            '/usr/bin/soffice',
            '/Applications/LibreOffice.app/Contents/MacOS/soffice',
            'C:\\Program Files\\LibreOffice\\program\\soffice.exe',
            'C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 尝试在PATH中查找
        try:
            result = subprocess.run(['which', 'libreoffice'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        return None

    async def convert(
        self,
        input_path: str,
        output_format: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ConversionResult:
        """转换文档"""
        start_time = time.time()

        try:
            # 验证输入文件
            if not self.validate_input_file(input_path):
                return ConversionResult(
                    status=ConversionStatus.FAILED,
                    error_message="输入文件验证失败"
                )

            # 生成输出路径
            output_path = self.generate_output_path(input_path, output_format, output_path)

            # 准备转换命令
            cmd = self._build_command(input_path, output_format, output_path, **kwargs)

            logger.info(f"开始转换: {input_path} -> {output_format}")

            # 执行转换
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # 等待转换完成
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            conversion_time = time.time() - start_time

            if process.returncode == 0 and os.path.exists(output_path):
                logger.info(f"转换成功完成: {output_path}")

                return ConversionResult(
                    status=ConversionStatus.COMPLETED,
                    output_path=output_path,
                    output_format=output_format,
                    metadata=self.get_file_metadata(output_path),
                    conversion_time=conversion_time,
                    file_size=os.path.getsize(output_path)
                )
            else:
                error_msg = stderr.decode('utf-8') if stderr else "未知错误"
                logger.error(f"转换失败: {error_msg}")

                return ConversionResult(
                    status=ConversionStatus.FAILED,
                    error_message=error_msg,
                    conversion_time=conversion_time
                )

        except asyncio.TimeoutError:
            logger.error(f"转换超时: {self.timeout}秒")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                error_message=f"转换超时 ({self.timeout}秒)",
                conversion_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"转换异常: {str(e)}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                error_message=str(e),
                conversion_time=time.time() - start_time
            )

    def _build_command(
        self,
        input_path: str,
        output_format: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """构建转换命令"""
        cmd = [self.libreoffice_path]

        # 添加headless参数
        if self.headless:
            cmd.extend(['--headless'])

        # 添加转换参数
        cmd.extend(['--convert-to', output_format])

        # 添加输出目录
        if output_path:
            output_dir = os.path.dirname(output_path)
            cmd.extend(['--outdir', output_dir])
        else:
            cmd.extend(['--outdir', self.temp_dir])

        # 添加输入文件
        cmd.append(input_path)

        return cmd

    def get_supported_input_formats(self) -> List[str]:
        """获取支持的输入格式"""
        return [
            '.doc', '.docx',  # Word文档
            '.xls', '.xlsx',  # Excel表格
            '.ppt', '.pptx',  # PowerPoint演示文稿
            '.odt', '.ods', '.odp', '.odg',  # OpenDocument格式
            '.rtf',  # Rich Text Format
            '.html', '.htm',  # HTML文档
            '.txt',  # 纯文本
            '.csv',  # CSV文件
            '.xml',  # XML文档
        ]

    def get_supported_output_formats(self) -> List[str]:
        """获取支持的输出格式"""
        return [
            'pdf',  # PDF文档
            'docx',  # Word文档
            'xlsx',  # Excel表格
            'pptx',  # PowerPoint演示文稿
            'txt',  # 纯文本
            'html',  # HTML文档
            'rtf',  # Rich Text Format
            'odt',  # OpenDocument Text
            'ods',  # OpenDocument Spreadsheet
            'odp',  # OpenDocument Presentation
            'csv',  # CSV文件
            'xml',  # XML文档
        ]

    async def batch_convert(
        self,
        input_files: List[str],
        output_format: str,
        output_dir: Optional[str] = None
    ) -> List[ConversionResult]:
        """批量转换文档"""
        results = []

        for input_file in input_files:
            output_path = None
            if output_dir:
                filename = Path(input_file).stem + f".{output_format}"
                output_path = os.path.join(output_dir, filename)

            result = await self.convert(input_file, output_format, output_path)
            results.append(result)

        return results

    async def get_conversion_info(self, input_path: str) -> Dict[str, Any]:
        """获取文档转换信息"""
        if not os.path.exists(input_path):
            return {"error": "文件不存在"}

        file_ext = Path(input_path).suffix.lower()

        return {
            "file_path": input_path,
            "file_extension": file_ext,
            "supported_input": file_ext in self.get_supported_input_formats(),
            "available_formats": self.get_supported_output_formats(),
            "file_metadata": self.get_file_metadata(input_path)
        }