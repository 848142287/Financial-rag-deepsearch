"""
增强的 LLM 多模态提取器
使用现有系统的 Qwen VL 模型进行 PDF 信息提取
集成了 Multimodal_RAG 的提取功能，兼容现有配置
"""

import io
import base64
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from pdf2image import convert_from_bytes
from PIL import Image
import logging

from app.core.config import settings
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """提取结果数据类"""
    markdown_content: str
    tables: List[Dict[str, Any]]
    formulas: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    token_usage: Dict[str, int]
    time_cost: Dict[str, float]
    page_images: List[Any]
    per_page_results: List[Dict[str, Any]]


class EnhancedLLMMultimodalExtractor:
    """
    增强的 PDF 多模态信息抽取器

    使用现有系统的 Qwen VL 模型，无需新增 LLM 配置
    """

    def __init__(self, pages_per_request: int = 1):
        self.pages_per_request = pages_per_request
        self.dpi = 100

        # 统计信息
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.pdf_convert_time = 0
        self.api_call_time = 0
        self.total_time = 0

        logger.info(f"初始化增强的 LLM 多模态提取器")
        logger.info(f"  使用模型: {settings.qwen_multimodal_model}")
        logger.info(f"  每次处理页数: {pages_per_request}")

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """将 PDF 转换为图片列表"""
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        images = convert_from_bytes(pdf_content, dpi=self.dpi)
        return images

    def image_to_base64(self, image: Image.Image, max_size: int = 2000) -> str:
        """将 PIL Image 转换为 base64 字符串，并压缩图片"""
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _get_extraction_prompt(self, page_range: str, total_pages: int) -> str:
        """生成提取指令的 prompt"""
        return f"""【重要】请直接分析图片内容并返回 JSON，不要说"我无法提取"或给出任何解释。

分析这些 PDF 页面（第{page_range}页，共{total_pages}页）：

1. **Markdown 内容**：
   - 识别所有标题层级（# ## ###）
   - 保持段落结构和格式
   - 保留列表、引用
   - 表格用 Markdown 表格语法
   - 忽略页眉和页脚内容，不要将其识别为标题或正文
   - 按页面顺序组织，页间用 --- 分隔

2. **表格提取**：
   - 提取所有表格（表头+数据）
   - 标注所在页码

3. **公式提取**：
   - 提取所有数学公式（行内+独立）
   - 使用 LaTeX 格式
   - 标注所在页码

4. **图片描述**：
   - 描述所有非表格、非公式的图像内容
   - 包括图表、示意图、照片等
   - 输出其视觉内容、含义及上下文作用
   - 标注所在页码

**输出格式要求（必须严格遵守）：**
- 直接返回纯 JSON 对象
- 不要用 ```json 或 ``` 包裹
- 不要添加任何解释文字

**JSON 结构：**
{{
    "pages": [
        {{
            "page_num": 页码(整数),
            "markdown": "该页完整markdown内容",
            "page_title": "主标题或空字符串"
        }}
    ],
    "tables": [
        {{
            "page": 页码(整数),
            "id": "表格1",
            "caption": "表格标题或空字符串",
            "content": "markdown表格",
            "data": [["单元格1", "单元格2"]]
        }}
    ],
    "formulas": [
        {{
            "page": 页码(整数),
            "id": "公式1",
            "latex": "LaTeX公式",
            "type": "inline或display",
            "context": "公式前后文本"
        }}
    ],
    "images": [
        {{
            "page": 页码(整数),
            "id": "图片1",
            "description": "图片的详细描述",
            "type": "chart/graph/photo/diagram/etc",
            "context": "图片出现的上下文"
        }}
    ]
}}

现在请分析图片并直接返回上述 JSON 结构："""

    async def call_multimodal_api(
        self,
        image_base64_list: List[str],
        page_nums: List[int],
        total_pages: int
    ) -> Dict[str, Any]:
        """
        调用多模态 API

        使用现有系统的 Qwen VL 模型
        """
        start_time = time.time()
        page_range = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])

        prompt = self._get_extraction_prompt(page_range, total_pages)

        # 构建消息内容
        content_items = [{"type": "text", "text": prompt}]

        # 添加图片
        for img_base64 in image_base64_list:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的 PDF 文档信息提取助手。你必须严格按照用户要求的 JSON 格式返回结果，不要添加任何解释性文字或 markdown 代码块标记。"
            },
            {"role": "user", "content": content_items}
        ]

        try:
            # 使用现有系统的 llm_service
            response = await llm_service.chat_completion(
                messages=messages,
                model=settings.qwen_multimodal_model,
                temperature=0.1,
                max_tokens=4096,
                use_qwen=True  # 使用 Qwen 模型
            )

            # 提取 token 使用信息
            if 'usage' in response:
                prompt_tokens = response['usage']['prompt_tokens']
                completion_tokens = response['usage']['completion_tokens']
                total_tokens = response['usage']['total_tokens']

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens

                logger.info(f"  第{page_range}页 Token: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")

            content = response.get('content', '')

            api_time = time.time() - start_time
            logger.info(f"  第{page_range}页 耗时: {api_time:.2f}秒")

            return self._parse_response_content(content, page_nums)

        except Exception as e:
            logger.error(f"多模态 API 调用失败: {type(e).__name__}: {e}")
            raise

    def _parse_response_content(self, content: str, page_nums: List[int]) -> Dict[str, Any]:
        """解析 API 响应内容"""
        try:
            # 清理内容
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            logger.error(f"原始内容前200字符: {content[:200]}...")

            # 检查是否是模型拒绝提取的回复
            refusal_keywords = ["unable to", "cannot", "can't", "sorry", "i'm not able", "无法"]
            if any(keyword in content.lower() for keyword in refusal_keywords):
                logger.warning(f"模型拒绝提取内容，页面 {page_nums} 将标记为处理失败")
                return {
                    "pages": [
                        {
                            "page_num": num,
                            "markdown": f"## 第{num}页\n**提取失败：模型拒绝处理此页面**\n{content[:500]}",
                            "page_title": f"第{num}页（提取失败）"
                        } for num in page_nums
                    ],
                    "tables": [],
                    "formulas": [],
                    "images": []
                }

            # 如果不是拒绝，尝试将原始内容作为 markdown
            logger.warning(f"将原始响应作为 markdown 内容保存")
            return {
                "pages": [
                    {
                        "page_num": num,
                        "markdown": content,
                        "page_title": f"第{num}页"
                    } for num in page_nums
                ],
                "tables": [],
                "formulas": [],
                "images": []
            }

    async def extract_from_pdf(
        self,
        pdf_path: str,
        original_filename: Optional[str] = None
    ) -> ExtractionResult:
        """从 PDF 文件中提取完整信息"""
        overall_start = time.time()

        filename = original_filename or Path(pdf_path).name
        logger.info(f"开始处理 PDF: {pdf_path}")
        logger.info("="*60)

        # 1. 转换 PDF 为图片
        logger.info("\n[步骤1] 正在将 PDF 转换为图片...")
        convert_start = time.time()
        images = self.pdf_to_images(pdf_path)
        self.pdf_convert_time = time.time() - convert_start
        total_pages = len(images)
        logger.info(f"✓ 转换完成: 共 {total_pages} 页 (耗时: {self.pdf_convert_time:.2f}秒)")

        # 2. 分组并逐批分析
        logger.info(f"\n[步骤2] 开始 API 调用分析 (每次处理{self.pages_per_request}页)...")
        api_start = time.time()

        all_markdown = []
        all_tables = []
        all_formulas = []
        all_images = []
        page_titles = []
        per_page_results = []

        # 将页面分组
        page_groups = []
        for i in range(0, total_pages, self.pages_per_request):
            end_idx = min(i + self.pages_per_request, total_pages)
            page_group = {
                'images': images[i:end_idx],
                'page_nums': list(range(i + 1, end_idx + 1))
            }
            page_groups.append(page_group)

        logger.info(f"已分为 {len(page_groups)} 个批次")

        # 顺序执行 API 调用
        results = []
        for idx, group in enumerate(page_groups):
            logger.info(f"\n处理批次 {idx + 1}/{len(page_groups)}:")
            image_base64_list = [self.image_to_base64(img) for img in group['images']]

            try:
                result = await self.call_multimodal_api(
                    image_base64_list,
                    group['page_nums'],
                    total_pages
                )
                results.append(result)
                logger.info(f"批次 {idx + 1} 完成")
            except Exception as e:
                logger.error(f"批次 {idx + 1} 失败: {e}")
                results.append(Exception(str(e)))

            # 批次间延迟
            if idx < len(page_groups) - 1:
                await asyncio.sleep(1)

        self.api_call_time = time.time() - api_start
        logger.info(f"\n✓ API 调用完成 (总耗时: {self.api_call_time:.2f}秒)")

        # 3. 整合结果
        logger.info(f"\n[步骤3] 整合提取结果...")

        for batch_idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批次 {batch_idx + 1} 处理失败: {result}")
                page_group = page_groups[batch_idx]
                for page_num in page_group['page_nums']:
                    if len(all_markdown) > 0:
                        all_markdown.append("\n---\n")
                    all_markdown.append(f"## 第{page_num}页（处理失败）\n")

                    per_page_results.append({
                        "page_num": page_num,
                        "status": "failed",
                        "error": str(result),
                        "markdown": f"## 第{page_num}页（处理失败）\n",
                        "page_title": "",
                        "tables": [],
                        "formulas": [],
                        "images": []
                    })
                continue

            pages_data = result.get('pages', [])
            page_group = page_groups[batch_idx]
            expected_page_nums = page_group['page_nums']

            for i, page_num in enumerate(expected_page_nums):
                if len(all_markdown) > 0:
                    all_markdown.append("\n---\n")

                if i < len(pages_data):
                    page_data = pages_data[i]
                    markdown = page_data.get('markdown', '')
                    page_title = page_data.get('page_title', '')

                    if page_title:
                        page_titles.append(page_title)
                    all_markdown.append(markdown)

                    page_tables = [t for t in result.get('tables', []) if t.get('page') == page_num]
                    page_formulas = [f for f in result.get('formulas', []) if f.get('page') == page_num]
                    page_images = [img for img in result.get('images', []) if img.get('page') == page_num]

                    per_page_results.append({
                        "page_num": page_num,
                        "status": "success",
                        "markdown": markdown,
                        "page_title": page_title,
                        "tables": page_tables,
                        "formulas": page_formulas,
                        "images": page_images
                    })

                    logger.info(f"✓ 第{page_num}页内容已拼接")
                else:
                    logger.warning(f"第 {page_num} 页数据缺失")
                    all_markdown.append(f"## 第{page_num}页（数据缺失）\n")

                    per_page_results.append({
                        "page_num": page_num,
                        "status": "missing",
                        "markdown": f"## 第{page_num}页（数据缺失）\n",
                        "page_title": "",
                        "tables": [],
                        "formulas": [],
                        "images": []
                    })

            for table in result.get('tables', []):
                all_tables.append(table)

            for formula in result.get('formulas', []):
                all_formulas.append(formula)

            for image in result.get('images', []):
                all_images.append(image)

        final_markdown = ''.join(all_markdown)

        if page_titles:
            document_title = page_titles[0]
            final_markdown = f"# {document_title}\n{final_markdown}"

        self.total_time = time.time() - overall_start

        token_usage = {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens
        }

        time_cost = {
            "pdf_convert_time": round(self.pdf_convert_time, 2),
            "api_call_time": round(self.api_call_time, 2),
            "total_time": round(self.total_time, 2)
        }

        metadata = {
            "total_pages": total_pages,
            "total_tables": len(all_tables),
            "total_formulas": len(all_formulas),
            "total_images": len(all_images),
            "page_titles": page_titles
        }

        logger.info("\n" + "="*60)
        logger.info("✓ 提取完成！")
        logger.info("="*60)
        logger.info(f"总页数: {metadata['total_pages']}")
        logger.info(f"表格数: {metadata['total_tables']}")
        logger.info(f"公式数: {metadata['total_formulas']}")
        logger.info(f"图片数: {metadata['total_images']}")
        logger.info(f"Token: {token_usage['total_tokens']:,}")
        logger.info(f"耗时: {time_cost['total_time']}秒")
        logger.info("="*60)

        return ExtractionResult(
            markdown_content=final_markdown,
            tables=all_tables,
            formulas=all_formulas,
            metadata=metadata,
            token_usage=token_usage,
            time_cost=time_cost,
            page_images=images,
            per_page_results=per_page_results
        )


# 全局实例
enhanced_llm_multimodal_extractor = EnhancedLLMMultimodalExtractor()
