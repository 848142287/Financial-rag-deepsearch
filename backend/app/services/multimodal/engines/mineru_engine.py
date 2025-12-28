"""
Mineru引擎集成
提供基于Mineru的高质量文档解析能力
"""

import os
import asyncio
import logging
import tempfile
import subprocess
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MineruEngine:
    """Mineru解析引擎"""

    def __init__(self):
        """初始化Mineru引擎"""
        self.engine_name = "mineru"
        self.temp_dir = tempfile.mkdtemp(prefix="mineru_")

        # Mineru配置
        self.config = {
            "extract_images": True,
            "extract_tables": True,
            "extract_formulas": True,
            "ocr_enabled": True,
            "language": "zh,en",  # 支持中英文
            "dpi": 300,  # 高分辨率
            "output_format": "json"
        }

        logger.info("Mineru引擎初始化完成")

    async def parse_document(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        使用Mineru解析文档

        Args:
            file_path: 文件路径
            output_dir: 输出目录

        Returns:
            解析结果
        """
        try:
            logger.info(f"开始Mineru解析: {file_path}")

            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 准备输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 执行Mineru解析
            result = await self._execute_mineru(file_path, output_dir)

            # 处理解析结果
            processed_result = await self._process_mineru_result(result, output_dir)

            logger.info(f"Mineru解析完成: {len(processed_result.get('content_blocks', []))}个内容块")
            return processed_result

        except Exception as e:
            logger.error(f"Mineru解析失败: {str(e)}")
            raise

    async def _execute_mineru(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """执行Mineru解析命令"""
        try:
            # 构建Mineru命令
            cmd = [
                "python", "-m", "magic_pdf.pipe",
                "--pdf_path", file_path,
                "--output_dir", output_dir,
                "--extract_images", str(self.config["extract_images"]).lower(),
                "--extract_tables", str(self.config["extract_tables"]).lower(),
                "--extract_formulas", str(self.config["extract_formulas"]).lower(),
                "--ocr_enabled", str(self.config["ocr_enabled"]).lower(),
                "--language", self.config["language"],
                "--dpi", str(self.config["dpi"]),
                "--output_format", self.config["output_format"]
            ]

            # 执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
            )

            stdout, stderr = await process.communicate()

            # 检查执行结果
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "未知错误"
                raise RuntimeError(f"Mineru执行失败: {error_msg}")

            # 读取输出文件
            output_file = os.path.join(output_dir, "result.json")
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            else:
                # 如果没有JSON输出，尝试解析其他输出文件
                result = await self._parse_mineru_output_files(output_dir)

            return result

        except Exception as e:
            logger.error(f"Mineru命令执行失败: {str(e)}")
            raise

    async def _process_mineru_result(self, raw_result: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """处理Mineru解析结果"""
        try:
            processed_result = {
                "engine": self.engine_name,
                "total_pages": 0,
                "content_blocks": [],
                "images": [],
                "tables": [],
                "formulas": [],
                "metadata": {}
            }

            # 处理页面内容
            if "pages" in raw_result:
                processed_result["total_pages"] = len(raw_result["pages"])

                for page_idx, page_data in enumerate(raw_result["pages"]):
                    page_number = page_idx + 1

                    # 处理文本块
                    if "texts" in page_data:
                        for text_block in page_data["texts"]:
                            block = {
                                "id": f"text_{page_number}_{len(processed_result['content_blocks'])}",
                                "content_type": "text",
                                "content": text_block.get("text", ""),
                                "bbox": text_block.get("bbox"),
                                "page_number": page_number,
                                "confidence": text_block.get("confidence", 1.0),
                                "source_engine": self.engine_name
                            }
                            processed_result["content_blocks"].append(block)

                    # 处理图片
                    if "images" in page_data:
                        for img_block in page_data["images"]:
                            # 保存图片文件
                            img_path = await self._save_image(img_block, output_dir, page_number)

                            image_info = {
                                "id": f"image_{page_number}_{len(processed_result['images'])}",
                                "content_type": "image",
                                "file_path": img_path,
                                "description": img_block.get("description", ""),
                                "bbox": img_block.get("bbox"),
                                "page_number": page_number,
                                "confidence": img_block.get("confidence", 1.0),
                                "metadata": img_block.get("metadata", {})
                            }
                            processed_result["images"].append(image_info)

                            # 添加到内容块
                            content_block = {
                                "id": image_info["id"],
                                "content_type": "image",
                                "content": image_info["description"],
                                "bbox": image_info["bbox"],
                                "page_number": page_number,
                                "confidence": image_info["confidence"],
                                "source_engine": self.engine_name,
                                "file_path": img_path
                            }
                            processed_result["content_blocks"].append(content_block)

                    # 处理表格
                    if "tables" in page_data:
                        for table_block in page_data["tables"]:
                            table_data = await self._process_table(table_block, output_dir, page_number)

                            table_info = {
                                "id": f"table_{page_number}_{len(processed_result['tables'])}",
                                "content_type": "table",
                                "html": table_data["html"],
                                "text": table_data["text"],
                                "bbox": table_block.get("bbox"),
                                "page_number": page_number,
                                "confidence": table_block.get("confidence", 1.0),
                                "metadata": table_block.get("metadata", {})
                            }
                            processed_result["tables"].append(table_info)

                            # 添加到内容块
                            content_block = {
                                "id": table_info["id"],
                                "content_type": "table",
                                "content": table_info["text"],
                                "bbox": table_info["bbox"],
                                "page_number": page_number,
                                "confidence": table_info["confidence"],
                                "source_engine": self.engine_name
                            }
                            processed_result["content_blocks"].append(content_block)

                    # 处理公式
                    if "formulas" in page_data:
                        for formula_block in page_data["formulas"]:
                            formula_data = await self._process_formula(formula_block, output_dir, page_number)

                            formula_info = {
                                "id": f"formula_{page_number}_{len(processed_result['formulas'])}",
                                "content_type": "formula",
                                "latex": formula_data["latex"],
                                "text": formula_data["text"],
                                "bbox": formula_block.get("bbox"),
                                "page_number": page_number,
                                "confidence": formula_block.get("confidence", 1.0),
                                "metadata": formula_block.get("metadata", {})
                            }
                            processed_result["formulas"].append(formula_info)

                            # 添加到内容块
                            content_block = {
                                "id": formula_info["id"],
                                "content_type": "formula",
                                "content": formula_info["text"],
                                "bbox": formula_info["bbox"],
                                "page_number": page_number,
                                "confidence": formula_info["confidence"],
                                "source_engine": self.engine_name,
                                "latex": formula_info["latex"]
                            }
                            processed_result["content_blocks"].append(content_block)

            # 添加元数据
            processed_result["metadata"] = {
                "engine": self.engine_name,
                "config": self.config,
                "output_dir": output_dir,
                "processing_time": 0  # 将在调用方计算
            }

            return processed_result

        except Exception as e:
            logger.error(f"Mineru结果处理失败: {str(e)}")
            raise

    async def _save_image(self, img_block: Dict[str, Any], output_dir: str, page_number: int) -> str:
        """保存图片文件"""
        try:
            import base64
            from PIL import Image
            import io

            # 创建图片目录
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # 处理图片数据
            img_data = img_block.get("data")
            if img_data and isinstance(img_data, str):
                # Base64编码的图片
                if img_data.startswith("data:image"):
                    # 去掉前缀
                    img_data = img_data.split(",", 1)[1]

                image_bytes = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(image_bytes))

                # 保存图片
                img_filename = f"page_{page_number}_img_{len(os.listdir(images_dir))}.png"
                img_path = os.path.join(images_dir, img_filename)
                image.save(img_path)

                return img_path
            else:
                # 如果没有图片数据，返回空路径
                return ""

        except Exception as e:
            logger.warning(f"保存图片失败: {str(e)}")
            return ""

    async def _process_table(self, table_block: Dict[str, Any], output_dir: str, page_number: int) -> Dict[str, Any]:
        """处理表格数据"""
        try:
            # 提取表格数据
            table_data = table_block.get("data", [])
            if not table_data:
                return {"html": "", "text": ""}

            # 生成HTML表格
            html = "<table border='1'>\n"
            for row_idx, row in enumerate(table_data):
                html += "  <tr>\n"
                for cell in row:
                    cell_tag = "th" if row_idx == 0 else "td"
                    html += f"    <{cell_tag}>{cell}</{cell_tag}>\n"
                html += "  </tr>\n"
            html += "</table>"

            # 生成文本表格
            text_rows = []
            for row in table_data:
                text_rows.append(" | ".join(str(cell) for cell in row))
            text = "\n".join(text_rows)

            return {"html": html, "text": text}

        except Exception as e:
            logger.warning(f"处理表格失败: {str(e)}")
            return {"html": "", "text": str(table_block.get("data", ""))}

    async def _process_formula(self, formula_block: Dict[str, Any], output_dir: str, page_number: int) -> Dict[str, Any]:
        """处理公式数据"""
        try:
            latex = formula_block.get("latex", "")
            text = formula_block.get("text", latex)  # 如果没有文本，使用LaTeX

            return {"latex": latex, "text": text}

        except Exception as e:
            logger.warning(f"处理公式失败: {str(e)}")
            return {"latex": "", "text": str(formula_block)}

    async def _parse_mineru_output_files(self, output_dir: str) -> Dict[str, Any]:
        """解析Mineru输出文件"""
        try:
            result = {
                "engine": self.engine_name,
                "total_pages": 0,
                "content_blocks": [],
                "metadata": {}
            }

            # 查找所有输出文件
            output_files = []
            for file in os.listdir(output_dir):
                if file.endswith(('.json', '.txt', '.md')):
                    output_files.append(file)

            # 解析每个文件
            for file in output_files:
                file_path = os.path.join(output_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file.endswith('.json'):
                            data = json.load(f)
                            if isinstance(data, list):
                                result["content_blocks"].extend(data)
                            else:
                                result.update(data)
                        else:
                            # 文本文件作为文本块处理
                            content = f.read()
                            block = {
                                "id": f"text_{file}",
                                "content_type": "text",
                                "content": content,
                                "page_number": 1,
                                "confidence": 1.0,
                                "source_engine": self.engine_name
                            }
                            result["content_blocks"].append(block)
                except Exception as e:
                    logger.warning(f"解析输出文件失败 {file}: {str(e)}")

            return result

        except Exception as e:
            logger.error(f"解析Mineru输出文件失败: {str(e)}")
            return {"engine": self.engine_name, "content_blocks": [], "metadata": {}}

    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Mineru引擎临时文件已清理")
        except Exception as e:
            logger.error(f"Mineru引擎清理失败: {str(e)}")

    def __del__(self):
        """析构函数"""
        self.cleanup()