"""
VLM精确提取解析器
使用现有系统的视觉语言模型进行高精度PDF解析

特点：
- 使用系统已配置的多模态模型（如通义千问VL）
- 自动识别表格、公式、图片
- 结构化Markdown输出
- 适用于复杂文档

依赖：使用现有的VLM集成模块
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from io import BytesIO
import base64
import json

from ..base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image未安装，请运行: pip install pdf2image")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow未安装，请运行: pip install Pillow")


class VLMPreciseParser(BaseFileParser):
    """
    VLM精确提取解析器

    使用系统现有的视觉语言模型进行高精度PDF解析

    特点：
    - 高精度表格识别
    - 公式提取和LaTeX转换
    - 图片内容描述
    - 复杂版式处理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.parser_name = "VLMPrecise"
        self.supported_extensions = ['.pdf']

        # VLM配置（使用现有模型）
        self.vlm_model_name = self.config.get('vlm_model_name', 'qwen-vl-max')
        self.model_manager = self.config.get('model_manager', None)

        # PDF处理配置
        self.dpi = self.config.get('dpi', 150)
        self.max_image_size = self.config.get('max_image_size', 2000)
        self.pages_per_request = self.config.get('pages_per_request', 1)

        # 输出配置
        self.output_dir = self.config.get('output_dir', './vlm_output')
        self.save_images = self.config.get('save_images', True)

        # 检查依赖
        self._check_dependencies()

    def _check_dependencies(self):
        """检查必要的依赖是否安装"""
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image未安装，部分功能将不可用")
        if not PIL_AVAILABLE:
            logger.warning("Pillow未安装，部分功能将不可用")

    def can_parse(self, file_path: str, file_extension: str = None) -> bool:
        """检查是否能解析指定文件"""
        ext = file_extension or Path(file_path).suffix.lower()

        if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
            logger.warning("VLM parser dependencies not available")
            return False

        return ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """解析文件内容"""
        try:
            # 验证文件
            valid, error_msg = self.validate_file(file_path)
            if not valid:
                return ParseResult(
                    content="",
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )

            start_time = time.time()

            # 检查VLM模型是否可用
            if not self._is_vlm_available():
                error_msg = "VLM model not available. Please configure model_manager"
                return ParseResult(
                    content="",
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )

            # 创建输出目录
            output_path = Path(self.output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            images_dir = output_path / "images"
            images_dir.mkdir(exist_ok=True, parents=True)

            # 将PDF转换为图片
            logger.info(f"Converting PDF to images: {file_path}")
            images = await self._pdf_to_images(file_path)

            if not images:
                error_msg = "Failed to convert PDF to images"
                return ParseResult(
                    content="",
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )

            logger.info(f"Converted {len(images)} pages to images")

            # 使用VLM分析每一页
            markdown_content = []
            all_tables = []
            all_formulas = []
            all_images = []

            # 分批处理页面
            for i in range(0, len(images), self.pages_per_request):
                batch_images = images[i:i + self.pages_per_request]
                page_nums = list(range(i + 1, i + len(batch_images) + 1))

                logger.info(f"Processing pages {page_nums}")

                try:
                    # VLM分析
                    batch_result = await self._analyze_pages_with_vlm(
                        batch_images,
                        page_nums,
                        images_dir if self.save_images else None
                    )

                    markdown_content.append(batch_result['markdown'])
                    all_tables.extend(batch_result['tables'])
                    all_formulas.extend(batch_result['formulas'])
                    all_images.extend(batch_result['images'])

                except Exception as e:
                    logger.error(f"Failed to analyze pages {page_nums}: {e}")
                    # 添加占位符
                    for page_num in page_nums:
                        markdown_content.append(f"\n---\n\n## 第{page_num}页（处理失败）\n\n")

            # 合并所有内容
            final_markdown = "\n".join(markdown_content)

            # 生成元数据
            metadata = {
                'total_pages': len(images),
                'total_tables': len(all_tables),
                'total_formulas': len(all_formulas),
                'total_images': len(all_images),
                'parser': 'VLM',
                'model_name': self.vlm_model_name,
                'images_dir': str(images_dir) if self.save_images else None
            }

            parse_time = time.time() - start_time
            metadata['parse_time'] = parse_time

            logger.info(f"VLM parsing completed in {parse_time:.2f}s")

            return ParseResult(
                content=final_markdown,
                metadata=metadata,
                success=True,
                parse_time=parse_time
            )

        except Exception as e:
            error_msg = f"VLM parsing error for {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ParseResult(
                content="",
                metadata={'error': str(e)},
                success=False,
                error_message=error_msg
            )

    async def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """将PDF转换为图片列表"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()

            images = convert_from_bytes(
                pdf_content,
                dpi=self.dpi,
                fmt='jpeg'
            )

            return images

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

    async def _analyze_pages_with_vlm(
        self,
        images: List[Image.Image],
        page_nums: List[int],
        images_dir: Optional[Path]
    ) -> Dict[str, Any]:
        """使用VLM分析页面"""

        # 转换图片为base64
        image_base64_list = []
        for img in images:
            # 调整大小
            if img.width > self.max_image_size or img.height > self.max_image_size:
                img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)

            # 转换为base64
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            image_base64_list.append(img_base64)

        # 构建prompt
        page_range = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])
        prompt = self._build_extraction_prompt(page_range, len(page_nums))

        # 调用VLM模型
        try:
            # 使用现有的VLM集成
            from app.services.multimodal.vlm_integration import ImageInput, BaseVLMModel

            # 获取VLM模型（从model_manager）
            vlm_model = None
            if self.model_manager:
                vlm_model = self.model_manager.models.get(self.vlm_model_name)

            if not vlm_model:
                logger.warning(f"VLM model {self.vlm_model_name} not found, using fallback method")
                # 使用简化的提取方法
                return await self._fallback_extraction(images, page_nums, images_dir)

            # 构建图像输入
            image_input = ImageInput(base64_data=image_base64_list[0] if len(image_base64_list) == 1 else None)

            # 调用VLM分析
            result = await vlm_model.analyze_image(
                image_input,
                question=prompt
            )

            # 解析结果
            return self._parse_vlm_result(result, page_nums)

        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            # 回退到简单提取
            return await self._fallback_extraction(images, page_nums, images_dir)

    def _build_extraction_prompt(self, page_range: str, total_pages: int) -> str:
        """构建提取提示词"""
        return f"""请分析这些PDF页面（第{page_range}页，共{total_pages}页）并提取信息：

1. **Markdown内容**：
   - 识别所有标题层级
   - 保持段落结构和格式
   - 表格用Markdown表格语法
   - 按页面顺序组织

2. **表格提取**：
   - 提取所有表格（表头+数据）
   - 标注所在页码

3. **公式提取**：
   - 提取数学公式
   - 使用LaTeX格式

4. **图片描述**：
   - 描述所有图像内容

请以JSON格式返回，包含pages（markdown内容）、tables、formulas、images字段。
如果无法返回JSON，请直接返回Markdown格式的内容。"""

    def _parse_vlm_result(self, vlm_result, page_nums: List[int]) -> Dict[str, Any]:
        """解析VLM返回的结果"""
        try:
            # 尝试解析JSON
            if vlm_result.text_description:
                try:
                    result_data = json.loads(vlm_result.text_description)
                    return {
                        'markdown': result_data.get('text', ''),
                        'tables': result_data.get('tables', []),
                        'formulas': result_data.get('formulas', []),
                        'images': result_data.get('images', [])
                    }
                except json.JSONDecodeError:
                    # 不是JSON格式，直接作为markdown
                    return {
                        'markdown': vlm_result.text_description,
                        'tables': [],
                        'formulas': [],
                        'images': []
                    }

            return {
                'markdown': '',
                'tables': [],
                'formulas': [],
                'images': []
            }

        except Exception as e:
            logger.error(f"Failed to parse VLM result: {e}")
            return {
                'markdown': '',
                'tables': [],
                'formulas': [],
                'images': []
            }

    async def _fallback_extraction(
        self,
        images: List[Image.Image],
        page_nums: List[int],
        images_dir: Optional[Path]
    ) -> Dict[str, Any]:
        """回退的简单提取方法"""
        markdown_parts = []

        for i, (img, page_num) in enumerate(zip(images, page_nums)):
            # 保存图片
            if images_dir:
                img_path = images_dir / f"page_{page_num}.jpg"
                img.save(img_path, 'JPEG', quality=95)

            # 添加占位符
            markdown_parts.append(f"\n## 第{page_num}页\n")
            if images_dir:
                markdown_parts.append(f"![Page {page_num}](images/page_{page_num}.jpg)\n")

        return {
            'markdown': "\n".join(markdown_parts),
            'tables': [],
            'formulas': [],
            'images': []
        }

    def _is_vlm_available(self) -> bool:
        """检查VLM是否可用"""
        try:
            from app.services.multimodal.vlm_integration import BaseVLMModel
            return True
        except ImportError:
            return False

    def get_parser_info(self) -> Dict[str, Any]:
        """获取解析器信息"""
        info = super().get_parser_info()
        info.update({
            'library_version': 'VLM-based',
            'available': self._is_vlm_available() and PDF2IMAGE_AVAILABLE and PIL_AVAILABLE,
            'features': {
                'markdown_output': True,
                'image_extraction': True,
                'table_extraction': True,
                'formula_extraction': True,
                'high_precision': True,
                'complex_layouts': True
            },
            'performance': {
                'avg_time_per_page': '2-5s',
                'recommended_for': 'Complex documents, academic papers, financial reports'
            },
            'requires': 'VLM model configuration'
        })
        return info
