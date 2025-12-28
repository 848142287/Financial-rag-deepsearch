"""
多模态文档分析服务
提供OCR识别、图片描述、图表分析、公式解释等多模态分析能力
"""

import logging
import base64
import io
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available")


@dataclass
class ImageAnalysisResult:
    """图片分析结果"""
    image_type: str  # 'image', 'chart', 'formula', 'table'
    description: str  # 详细描述
    ocr_text: Optional[str] = None  # OCR识别的文本
    chart_info: Optional[Dict[str, Any]] = None  # 图表特有信息
    formula_info: Optional[Dict[str, Any]] = None  # 公式特有信息
    confidence: float = 0.0
    analysis_time: Optional[float] = None


@dataclass
class ChartAnalysisResult:
    """图表分析结果"""
    chart_type: str  # 柱状图、折线图、饼图等
    x_axis_meaning: str  # 横坐标意义
    y_axis_meaning: str  # 纵坐标意义
    data_summary: str  # 数据统计意义
    trend: str  # 变化趋势
    key_insights: List[str]  # 关键洞察


class MultimodalDocumentAnalyzer:
    """多模态文档分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多模态分析器

        Args:
            config: 配置参数
                - ocr_enabled: 是否启用OCR
                - ocr_engine: OCR引擎 ('deepseek', 'tesseract')
                - qwen_vl_api_key: Qwen VL API密钥
                - qwen_vl_api_url: Qwen VL API地址
                - qwen_model: 模型名称 (默认 'qwen-vl-max')
                - timeout: 超时时间（秒）
        """
        self.config = config or {}
        self.ocr_enabled = self.config.get('ocr_enabled', True)
        self.ocr_engine = self.config.get('ocr_engine', 'deepseek')
        self.qwen_vl_api_key = self.config.get('qwen_vl_api_key', '')
        self.qwen_vl_api_url = self.config.get(
            'qwen_vl_api_url',
            'https://dashscope.aliyuncs.com/compatible-mode/v1'
        )
        self.qwen_model = self.config.get('qwen_model', 'qwen-vl-max')
        self.timeout = self.config.get('timeout', 60)

        # 初始化HTTP客户端
        if HTTPX_AVAILABLE:
            self.http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    'Authorization': f'Bearer {self.qwen_vl_api_key}',
                    'Content-Type': 'application/json'
                }
            )
        else:
            self.http_client = None
            logger.warning("httpx not available, VL features will be limited")

    async def analyze_image(
        self,
        image_path: str,
        image_type: str = 'auto'
    ) -> ImageAnalysisResult:
        """
        分析图片（通用图片、图表、公式）

        Args:
            image_path: 图片路径
            image_type: 图片类型 ('auto', 'image', 'chart', 'formula', 'table')

        Returns:
            ImageAnalysisResult: 分析结果
        """
        import time
        start_time = time.time()

        try:
            # 1. 读取图片
            image_base64 = self._encode_image(image_path)
            if not image_base64:
                return ImageAnalysisResult(
                    image_type='unknown',
                    description='Failed to encode image',
                    confidence=0.0
                )

            # 2. 自动检测图片类型
            if image_type == 'auto':
                image_type = await self._detect_image_type(image_base64)

            # 3. OCR识别
            ocr_text = None
            if self.ocr_enabled:
                ocr_text = await self._ocr_recognize(image_path)

            # 4. 根据类型进行深度分析
            if image_type == 'chart':
                description, chart_info = await self._analyze_chart(image_base64, ocr_text)
                result = ImageAnalysisResult(
                    image_type='chart',
                    description=description,
                    ocr_text=ocr_text,
                    chart_info=chart_info,
                    confidence=0.85,
                    analysis_time=time.time() - start_time
                )
            elif image_type == 'formula':
                description, formula_info = await self._analyze_formula(image_base64, ocr_text)
                result = ImageAnalysisResult(
                    image_type='formula',
                    description=description,
                    ocr_text=ocr_text,
                    formula_info=formula_info,
                    confidence=0.85,
                    analysis_time=time.time() - start_time
                )
            elif image_type == 'table':
                description = await self._analyze_table(image_base64, ocr_text)
                result = ImageAnalysisResult(
                    image_type='table',
                    description=description,
                    ocr_text=ocr_text,
                    confidence=0.80,
                    analysis_time=time.time() - start_time
                )
            else:  # 普通图片
                description = await self._describe_general_image(image_base64, ocr_text)
                result = ImageAnalysisResult(
                    image_type='image',
                    description=description,
                    ocr_text=ocr_text,
                    confidence=0.80,
                    analysis_time=time.time() - start_time
                )

            return result

        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            return ImageAnalysisResult(
                image_type='unknown',
                description=f'Analysis failed: {str(e)}',
                confidence=0.0,
                analysis_time=time.time() - start_time
            )

    async def _detect_image_type(self, image_base64: str) -> str:
        """检测图片类型"""
        # 使用Qwen VL进行类型检测
        prompt = """请分析这张图片，判断它是以下哪种类型：
1. chart（图表：柱状图、折线图、饼图、散点图等）
2. formula（数学公式、化学方程式等）
3. table（表格）
4. image（普通图片）

请只返回类型名称，不要其他内容。"""

        try:
            response = await self._call_qwen_vl(image_base64, prompt)
            response_lower = response.lower()

            if 'chart' in response_lower or '图表' in response_lower or '图' in response_lower:
                return 'chart'
            elif 'formula' in response_lower or '公式' in response_lower or 'equation' in response_lower:
                return 'formula'
            elif 'table' in response_lower or '表格' in response_lower:
                return 'table'
            else:
                return 'image'
        except:
            return 'image'

    async def _analyze_chart(
        self,
        image_base64: str,
        ocr_text: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """深度分析图表"""
        prompt = f"""请详细分析这张图表，提供以下信息：

1. 图表类型（柱状图、折线图、饼图、散点图等）
2. 横坐标（X轴）代表的意义
3. 纵坐标（Y轴）代表的意义
4. 图表标题和图例说明
5. 数据的主要统计意义（最大值、最小值、平均值、总体趋势等）
6. 数据的变化趋势（上升、下降、波动、稳定等）
7. 关键洞察和异常点

{f"OCR识别的文本：{ocr_text}" if ocr_text else ""}

请用结构化的方式回答，尽量详细。"""

        try:
            response = await self._call_qwen_vl(image_base64, prompt)

            # 解析响应，提取结构化信息
            chart_info = self._parse_chart_analysis(response)

            return response, chart_info
        except Exception as e:
            logger.error(f"Failed to analyze chart: {e}")
            return f"图表分析失败: {str(e)}", {}

    async def _analyze_formula(
        self,
        image_base64: str,
        ocr_text: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """深度分析公式"""
        prompt = f"""请详细分析这个公式/方程式，提供以下信息：

1. 公式的标准表达（LaTeX格式如果可能）
2. 公式中每个变量的含义
3. 公式的物理意义或数学意义
4. 公式的应用场景
5. 相关的重要性质或变换

{f"OCR识别的文本：{ocr_text}" if ocr_text else ""}

请用清晰易懂的语言解释这个公式。"""

        try:
            response = await self._call_qwen_vl(image_base64, prompt)

            formula_info = {
                'latex': self._extract_latex(response),
                'variables': self._extract_variables(response),
                'meaning': self._extract_meaning(response)
            }

            return response, formula_info
        except Exception as e:
            logger.error(f"Failed to analyze formula: {e}")
            return f"公式分析失败: {str(e)}", {}

    async def _analyze_table(
        self,
        image_base64: str,
        ocr_text: Optional[str]
    ) -> str:
        """分析表格"""
        prompt = f"""请详细分析这张表格，提供以下信息：

1. 表格的整体结构描述（行数、列数）
2. 表头内容
3. 关键数据和总结
4. 数据的趋势或规律

{f"OCR识别的文本：{ocr_text}" if ocr_text else ""}

请用清晰的方式描述表格内容。"""

        try:
            return await self._call_qwen_vl(image_base64, prompt)
        except Exception as e:
            logger.error(f"Failed to analyze table: {e}")
            return f"表格分析失败: {str(e)}"

    async def _describe_general_image(
        self,
        image_base64: str,
        ocr_text: Optional[str]
    ) -> str:
        """描述普通图片"""
        prompt = f"""请详细描述这张图片的内容，包括：

1. 图片的主要对象和场景
2. 图片的细节信息
3. 图片中包含的文字（如果有）

{f"OCR识别的文本：{ocr_text}" if ocr_text else ""}

请用自然流畅的语言描述。"""

        try:
            return await self._call_qwen_vl(image_base64, prompt)
        except Exception as e:
            logger.error(f"Failed to describe image: {e}")
            return f"图片描述失败: {str(e)}"

    async def _ocr_recognize(self, image_path: str) -> Optional[str]:
        """OCR识别图片中的文字"""
        if not self.ocr_enabled:
            return None

        try:
            # 导入OCR服务
            from app.services.ocr_service import ocr_service

            # 使用OCR服务识别
            result = await ocr_service.recognize_text(image_path)

            if result and result.get('success'):
                return result.get('text', '')
            return None

        except ImportError:
            logger.warning("OCR service not available")
            return None
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return None

    async def _call_qwen_vl(self, image_base64: str, prompt: str) -> str:
        """调用Qwen VL API"""
        if not self.http_client:
            raise Exception("HTTP client not available")

        try:
            payload = {
                "model": self.qwen_model,
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

            response = await self.http_client.post(
                f"{self.qwen_vl_api_url}/chat/completions",
                json=payload
            )

            response.raise_for_status()
            result = response.json()

            # 提取生成的文本
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "No response from VL model"

        except Exception as e:
            logger.error(f"Qwen VL API call failed: {e}")
            raise

    def _encode_image(self, image_path: str) -> Optional[str]:
        """将图片编码为base64"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None

    def _parse_chart_analysis(self, response: str) -> Dict[str, Any]:
        """解析图表分析结果"""
        chart_info = {
            'chart_type': 'unknown',
            'x_axis': '',
            'y_axis': '',
            'data_summary': '',
            'trend': '',
            'key_insights': []
        }

        # 简单的文本解析（可以后续增强）
        lines = response.split('\n')
        for line in lines:
            if '图表类型' in line or '类型' in line:
                chart_info['chart_type'] = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            elif '横坐标' in line or 'X轴' in line:
                chart_info['x_axis'] = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            elif '纵坐标' in line or 'Y轴' in line:
                chart_info['y_axis'] = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()
            elif '趋势' in line:
                chart_info['trend'] = line.split('：')[-1].strip() if '：' in line else line.split(':')[-1].strip()

        return chart_info

    def _extract_latex(self, response: str) -> str:
        """提取LaTeX公式"""
        # 查找 $$...$$ 或 $...$ 模式
        latex_pattern = r'\$\$([^\$]+)\$\$|\$([^\$]+)\$'
        matches = re.findall(latex_pattern, response)
        if matches:
            return matches[0][0] if matches[0][0] else matches[0][1]
        return ""

    def _extract_variables(self, response: str) -> List[str]:
        """提取变量说明"""
        variables = []
        # 查找变量定义模式
        var_pattern = r'(\w+)\s*[：:]\s*([^\n]+)'
        matches = re.findall(var_pattern, response)
        for var, meaning in matches:
            variables.append(f"{var}: {meaning}")
        return variables

    def _extract_meaning(self, response: str) -> str:
        """提取公式含义"""
        # 查找意义相关段落
        meaning_patterns = ['意义', '表示', '含义', 'meaning']
        for pattern in meaning_patterns:
            if pattern in response:
                idx = response.index(pattern)
                return response[idx:idx+200]  # 提取后续200字符
        return ""

    async def batch_analyze_images(
        self,
        image_paths: List[str],
        max_concurrent: int = 5
    ) -> List[ImageAnalysisResult]:
        """批量分析图片"""
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(image_path: str) -> ImageAnalysisResult:
            async with semaphore:
                return await self.analyze_image(image_path)

        tasks = [analyze_with_semaphore(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze {image_paths[i]}: {result}")
                processed_results.append(ImageAnalysisResult(
                    image_type='unknown',
                    description=f'Error: {str(result)}',
                    confidence=0.0
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def close(self):
        """关闭HTTP客户端"""
        if self.http_client:
            await self.http_client.aclose()


# 全局单例
_multimodal_analyzer: Optional[MultimodalDocumentAnalyzer] = None


def get_multimodal_analyzer(config: Optional[Dict[str, Any]] = None) -> MultimodalDocumentAnalyzer:
    """获取多模态分析器单例"""
    global _multimodal_analyzer
    if _multimodal_analyzer is None:
        _multimodal_analyzer = MultimodalDocumentAnalyzer(config)
    return _multimodal_analyzer
