"""
Qwen-VL引擎 - 真实实现
使用阿里云DashScope API进行多模态分析
"""

import base64
import asyncio
from typing import Dict, Any, List, Optional
from app.core.structured_logging import get_structured_logger
from app.core.config import settings
import httpx

logger = get_structured_logger(__name__)

class QwenVLEngine:
    """Qwen-VL引擎 - 支持OCR和多模态分析"""

    def __init__(self):
        self.api_key = settings.qwen_vl_api_key
        self.base_url = settings.qwen_vl_base_url
        self.ocr_model = settings.qwen_vl_ocr_model
        self.vl_max_model = settings.qwen_vl_max_model
        self.vl_plus_model = settings.qwen_vl_plus_model
        self.max_tokens = settings.qwen_vl_max_tokens
        self.temperature = settings.qwen_vl_temperature

        # 初始化HTTP客户端
        self.client = None

        logger.info("=" * 80)
        logger.info("🎨 Qwen-VL多模态引擎初始化")
        logger.info(f"  - API Base URL: {self.base_url}")
        logger.info(f"  - OCR模型: {self.ocr_model}")
        logger.info(f"  - VL-Plus模型: {self.vl_plus_model}")
        logger.info(f"  - VL-Max模型: {self.vl_max_model}")
        logger.info("=" * 80)

    async def _get_client(self):
        """获取HTTP客户端（懒加载）"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=120.0
            )
        return self.client

    async def close(self):
        """关闭HTTP客户端"""
        if self.client:
            await self.client.aclose()
            self.client = None

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def parse_with_ocr(
        self,
        file_path: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        使用Qwen-VL-OCR进行文字识别

        Args:
            file_path: 图片或PDF文件路径
            document_id: 文档ID

        Returns:
            OCR识别结果
        """
        logger.info(f"🔍 开始OCR识别: {file_path}")

        try:
            # 编码图片
            image_base64 = self._encode_image(file_path)

            # 构造请求
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的OCR识别助手。请准确识别图片中的所有文字内容，保持原有的排版格式。"
                },
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
                            "text": "请识别这张图片中的所有文字内容，尽可能准确地提取所有可见文本。"
                        }
                    ]
                }
            ]

            # 调用API
            client = await self._get_client()
            response = await client.post(
                f"/chat/completions",
                json={
                    "model": self.ocr_model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )

            response.raise_for_status()
            result = response.json()

            # 提取识别结果
            text_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            logger.info(f"✅ OCR识别完成，识别文字长度: {len(text_content)}")

            return {
                'text_blocks': [{
                    'text': text_content,
                    'confidence': 0.95,
                    'bbox': None
                }],
                'full_text': text_content,
                'metadata': {
                    'engine': 'qwen_vl_ocr',
                    'model': self.ocr_model,
                    'file_path': file_path,
                    'document_id': document_id
                },
                'usage': result.get('usage', {})
            }

        except Exception as e:
            logger.error(f"❌ OCR识别失败: {str(e)}")
            return {
                'text_blocks': [],
                'error': str(e),
                'metadata': {'engine': 'qwen_vl_ocr', 'status': 'failed'}
            }

    async def parse_with_vl_max(
        self,
        file_path: str,
        document_id: str,
        analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """
        使用Qwen-VL-Max进行深度多模态分析

        Args:
            file_path: 图片文件路径
            document_id: 文档ID
            analysis_type: 分析类型 (general, chart, table, formula)

        Returns:
            多模态分析结果
        """
        logger.info(f"🎨 开始深度多模态分析: {file_path} (类型: {analysis_type})")

        try:
            # 编码图片
            image_base64 = self._encode_image(file_path)

            # 根据分析类型构造不同的提示词
            prompts = {
                "general": "请详细描述这张图片的内容，包括：1) 主要元素和对象 2) 文字内容 3) 布局结构 4) 任何图表或表格",
                "chart": "请分析这张图表，提供：1) 图表类型（柱状图、折线图、饼图等）2) 数据轴标签和数值 3) 图表标题 4) 关键趋势或结论",
                "table": "请提取这张表格的所有数据，以结构化的方式返回每一行每一列的内容",
                "formula": "请识别并转写这个数学公式，使用LaTeX格式表示"
            }

            system_prompt = {
                "general": "你是一个专业的多模态内容分析助手，擅长理解图片中的复杂场景和内容。",
                "chart": "你是一个专业的数据图表分析专家，擅长识别和分析各种类型的图表。",
                "table": "你是一个专业的表格数据提取专家，擅长从图片中准确提取表格数据。",
                "formula": "你是一个专业的数学公式识别专家，擅长识别和转写各种数学公式。"
            }

            prompt = prompts.get(analysis_type, prompts["general"])
            sys_prompt = system_prompt.get(analysis_type, system_prompt["general"])

            # 构造请求
            messages = [
                {
                    "role": "system",
                    "content": sys_prompt
                },
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

            # 调用API
            client = await self._get_client()
            response = await client.post(
                f"/chat/completions",
                json={
                    "model": self.vl_max_model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )

            response.raise_for_status()
            result = response.json()

            # 提取分析结果
            analysis_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            logger.info(f"✅ 深度多模态分析完成，分析内容长度: {len(analysis_content)}")

            return {
                'analysis_results': [{
                    'type': analysis_type,
                    'content': analysis_content,
                    'confidence': 0.90
                }],
                'full_analysis': analysis_content,
                'metadata': {
                    'engine': 'qwen_vl_max',
                    'model': self.vl_max_model,
                    'file_path': file_path,
                    'document_id': document_id,
                    'analysis_type': analysis_type
                },
                'usage': result.get('usage', {})
            }

        except Exception as e:
            logger.error(f"❌ 深度多模态分析失败: {str(e)}")
            return {
                'analysis_results': [],
                'error': str(e),
                'metadata': {'engine': 'qwen_vl_max', 'status': 'failed'}
            }

    async def analyze_images_batch(
        self,
        image_paths: List[str],
        document_id: str,
        analysis_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        批量分析多张图片

        Args:
            image_paths: 图片路径列表
            document_id: 文档ID
            analysis_type: 分析类型

        Returns:
            批量分析结果列表
        """
        logger.info(f"🎨 开始批量分析 {len(image_paths)} 张图片")

        # 并发分析所有图片
        tasks = [
            self.parse_with_vl_max(img_path, document_id, analysis_type)
            for img_path in image_paths
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"图片 {i+1} 分析失败: {str(result)}")
                processed_results.append({
                    'error': str(result),
                    'metadata': {'image_path': image_paths[i], 'status': 'failed'}
                })
            else:
                processed_results.append(result)

        logger.info(f"✅ 批量分析完成，成功: {sum(1 for r in processed_results if 'error' not in r)}/{len(image_paths)}")

        return processed_results