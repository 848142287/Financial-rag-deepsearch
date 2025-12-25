"""
GLM-4.6V大模型服务
集成智谱AI的GLM-4.6V模型，提供多模态分析和金融文档理解能力
"""

import asyncio
import base64
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import httpx
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GLMConfig:
    """GLM服务配置"""
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    api_key: str = "44596ec77769435f83d815244ccfc2b8.HB8v1H2cQPmRRbgl"
    model: str = "glm-4v-plus"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens: int = 8000


class GLMService:
    """GLM-4.6V服务类"""

    def __init__(self, config: Optional[GLMConfig] = None):
        self.config = config or GLMConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送API请求"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    endpoint,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"GLM API HTTP error (attempt {attempt + 1}): {e}")
                if e.response.status_code == 429:  # Rate limit
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

            except Exception as e:
                logger.error(f"GLM API request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise

        raise Exception(f"GLM API request failed after {self.config.max_retries} attempts")

    async def _encode_image_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    async def generate_summary(self, text: str, max_length: int = 500) -> str:
        """生成文本摘要"""
        prompt = f"""
        请为以下金融文档内容生成一个简洁的摘要，重点突出：
        1. 主要观点和结论
        2. 关键数据和指标
        3. 重要趋势和变化
        4. 投资建议或风险提示

        摘要长度控制在{max_length}字以内，使用专业但易懂的语言。

        文档内容：
        {text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": min(max_length * 2, self.config.max_tokens)
        }

        try:
            response = await self._make_request("chat/completions", payload)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"摘要生成失败: {str(e)}"

    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """分析图片内容"""
        try:
            # 检查是否为图表
            is_chart = await self._detect_chart(image_path)

            if is_chart:
                return await self._analyze_chart_detailed(image_path)
            else:
                return await self._analyze_general_image(image_path)

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                "description": f"图片分析失败: {str(e)}",
                "error": str(e)
            }

    async def _detect_chart(self, image_path: str) -> bool:
        """检测是否为图表"""
        try:
            image_base64 = await self._encode_image_base64(image_path)

            prompt = """
            请判断这张图片是否为金融图表（如股票K线图、柱状图、饼图、折线图等）。
            回答"是"或"否"即可。
            """

            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 10
            }

            response = await self._make_request("chat/completions", payload)
            answer = response["choices"][0]["message"]["content"].strip()
            return "是" in answer

        except Exception as e:
            logger.error(f"Error detecting chart: {e}")
            return False

    async def _analyze_chart_detailed(self, image_path: str) -> Dict[str, Any]:
        """详细分析图表"""
        image_base64 = await self._encode_image_base64(image_path)

        prompt = """
        请详细分析这张金融图表，提供以下信息：
        1. 图表类型（如K线图、柱状图、折线图、饼图等）
        2. 主要数据和数值
        3. 趋势分析（上升、下降、震荡等）
        4. 关键转折点或重要事件
        5. 技术指标（如果有）
        6. 投资启示

        请用结构化的JSON格式返回分析结果。
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": self.config.max_tokens
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            # 尝试解析JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "chart_type": "未知",
                    "description": content,
                    "trend_analysis": "无法解析",
                    "key_points": [],
                    "investment_insights": "需要人工分析"
                }

        except Exception as e:
            logger.error(f"Error analyzing chart: {e}")
            return {
                "error": str(e),
                "chart_type": "分析失败",
                "description": "图表分析出现错误"
            }

    async def _analyze_general_image(self, image_path: str) -> Dict[str, Any]:
        """分析一般图片"""
        image_base64 = await self._encode_image_base64(image_path)

        prompt = """
        请描述这张图片的内容，重点关注：
        1. 图片中的主要对象和场景
        2. 文字信息（如果有）
        3. 与金融或投资相关的元素
        4. 可能的用途或含义

        请用简洁明了的语言进行描述。
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            return {
                "description": content,
                "image_type": "general",
                "analyzed_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing general image: {e}")
            return {
                "error": str(e),
                "description": "图片分析失败"
            }

    async def analyze_chart(self, image_path: str) -> Dict[str, Any]:
        """分析图表（简化接口）"""
        return await self._analyze_chart_detailed(image_path)

    async def extract_text_from_image(self, image_path: str) -> str:
        """从图片中提取文字（OCR）"""
        image_base64 = await self._encode_image_base64(image_path)

        prompt = """
        请提取图片中的所有文字内容，保持原有的格式和结构。
        如果有表格或列表，请保持其结构。
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": self.config.max_tokens
        }

        try:
            response = await self._make_request("chat/completions", payload)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return f"文字提取失败: {str(e)}"

    async def analyze_table(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析表格数据"""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        # 将表格转换为文本格式
        table_text = "表格标题：\n"
        table_text += " | ".join(headers) + "\n"
        table_text += "-" * (len(" | ".join(headers))) + "\n"

        for row in rows:
            table_text += " | ".join(str(cell) for cell in row) + "\n"

        prompt = f"""
        请分析以下表格数据，提供：
        1. 表格主要内容和用途说明
        2. 关键数据和趋势
        3. 数据变化的模式
        4. 重要发现或异常点
        5. 对投资决策的启示

        表格数据：
        {table_text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": self.config.max_tokens
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            return {
                "table_summary": content,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing table: {e}")
            return {
                "error": str(e),
                "table_summary": "表格分析失败"
            }

    async def analyze_table_trends(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析表格趋势"""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        # 简化数据，只取前10行避免token过多
        sample_rows = rows[:10] if len(rows) > 10 else rows

        table_text = "表格数据（样本）：\n"
        table_text += " | ".join(headers) + "\n"
        for row in sample_rows:
            table_text += " | ".join(str(cell) for cell in row) + "\n"

        prompt = f"""
        请分析表格数据的趋势，重点关注：
        1. 数值列的增长或下降趋势
        2. 环比和同比变化
        3. 关键指标的走势
        4. 异常波动点
        5. 未来趋势预测

        表格数据：
        {table_text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": self.config.max_tokens
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            return {
                "trend_analysis": content,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing table trends: {e}")
            return {
                "error": str(e),
                "trend_analysis": "趋势分析失败"
            }

    async def extract_table_metrics(self, table_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取表格关键指标"""
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        table_text = "表格数据：\n"
        table_text += " | ".join(headers) + "\n"
        for row in rows[:5]:  # 只取前5行
            table_text += " | ".join(str(cell) for cell in row) + "\n"

        prompt = f"""
        请从表格中提取关键指标和数值，以JSON格式返回：
        {{
            "metrics": [
                {{
                    "name": "指标名称",
                    "value": "数值",
                    "unit": "单位",
                    "trend": "趋势（上升/下降/稳定）"
                }}
            ]
        }}

        表格数据：
        {table_text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            # 尝试解析JSON
            try:
                result = json.loads(content)
                return result.get("metrics", [])
            except json.JSONDecodeError:
                return [{
                    "name": "解析失败",
                    "value": content,
                    "unit": "N/A",
                    "trend": "N/A"
                }]

        except Exception as e:
            logger.error(f"Error extracting table metrics: {e}")
            return []

    async def explain_formula(self, formula: str) -> str:
        """解释数学公式"""
        prompt = f"""
        请解释以下数学或金融公式的含义：
        1. 公式的数学定义
        2. 各个变量的含义
        3. 公式的应用场景
        4. 计算步骤说明
        5. 实际例子

        公式：{formula}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1500
        }

        try:
            response = await self._make_request("chat/completions", payload)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error explaining formula: {e}")
            return f"公式解释失败: {str(e)}"

    async def extract_formula_variables(self, formula: str) -> List[Dict[str, str]]:
        """提取公式变量"""
        prompt = f"""
        请从公式中提取所有变量和参数，以JSON格式返回：
        {{
            "variables": [
                {{
                    "symbol": "变量符号",
                    "name": "变量名称",
                    "description": "变量描述"
                }}
            ]
        }}

        公式：{formula}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            try:
                result = json.loads(content)
                return result.get("variables", [])
            except json.JSONDecodeError:
                return []

        except Exception as e:
            logger.error(f"Error extracting formula variables: {e}")
            return []

    async def breakdown_formula_steps(self, formula: str) -> List[str]:
        """分解公式计算步骤"""
        prompt = f"""
        请将以下公式的计算过程分解为步骤，以JSON格式返回：
        {{
            "steps": [
                "步骤1的描述",
                "步骤2的描述",
                ...
            ]
        }}

        公式：{formula}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            try:
                result = json.loads(content)
                return result.get("steps", [])
            except json.JSONDecodeError:
                return [content]

        except Exception as e:
            logger.error(f"Error breaking down formula steps: {e}")
            return []

    async def extract_key_points(self, text: str) -> List[str]:
        """提取关键要点"""
        prompt = f"""
        请从以下文本中提取3-5个关键要点，每个要点不超过50字：
        {text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            # 尝试分割为要点列表
            points = [point.strip() for point in content.split('\n') if point.strip()]
            return points[:5]  # 最多返回5个要点

        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """分析情感倾向"""
        prompt = f"""
        请分析以下金融文本的情感倾向，返回JSON格式：
        {{
            "sentiment": "positive/negative/neutral",
            "confidence": 0.95,
            "score": 0.75,
            "reasoning": "分析原因"
        }}

        文本：{text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "score": 0.0,
                    "reasoning": content
                }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "score": 0.0,
                "error": str(e)
            }

    async def extract_financial_entities(self, text: str) -> List[Dict[str, str]]:
        """提取金融实体"""
        prompt = f"""
        请从文本中提取金融相关实体，以JSON格式返回：
        {{
            "entities": [
                {{
                    "name": "实体名称",
                    "type": "company/stock/person/indicator/concept",
                    "description": "简短描述"
                }}
            ]
        }}

        文本：{text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            try:
                result = json.loads(content)
                return result.get("entities", [])
            except json.JSONDecodeError:
                return []

        except Exception as e:
            logger.error(f"Error extracting financial entities: {e}")
            return []

    async def analyze_mixed_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """分析混合内容"""
        prompt = f"""
        请分析以下包含多种元素的文档内容，提供综合分析：
        1. 内容主题和类型
        2. 关键信息点
        3. 数据和分析结论
        4. 图表和表格的含义
        5. 整体洞察

        内容：{content}
        元数据：{json.dumps(metadata, ensure_ascii=False)}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": self.config.max_tokens
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            return {
                "comprehensive_analysis": content,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing mixed content: {e}")
            return {
                "error": str(e),
                "comprehensive_analysis": "混合内容分析失败"
            }

    async def generate_document_insights(self, analyses: List[Dict]) -> str:
        """生成文档洞察"""
        # 合并所有分析结果
        combined_text = "\n".join([
            f"块{i+1}: {analysis.get('summary', '')}"
            for i, analysis in enumerate(analyses)
        ])

        prompt = f"""
        基于以下各部分的分析结果，生成整个文档的综合洞察：
        1. 核心主题和观点
        2. 重要数据和发现
        3. 关键结论和建议
        4. 投资启示

        分析结果：
        {combined_text}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }

        try:
            response = await self._make_request("chat/completions", payload)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating document insights: {e}")
            return f"洞察生成失败: {str(e)}"

    async def extract_document_trends(self, analyses: List[Dict]) -> Dict[str, Any]:
        """提取文档趋势"""
        prompt = f"""
        从以下分析结果中识别重要趋势和模式：
        {json.dumps(analyses, ensure_ascii=False)}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            return {
                "trends": content,
                "extracted_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error extracting document trends: {e}")
            return {
                "trends": "趋势提取失败",
                "error": str(e)
            }

    async def identify_risks_and_opportunities(self, analyses: List[Dict]) -> Dict[str, Any]:
        """识别风险和机会"""
        prompt = f"""
        从以下分析结果中识别投资风险和机会：
        {json.dumps(analyses, ensure_ascii=False)}

        返回格式：
        {{
            "risks": ["风险1", "风险2"],
            "opportunities": ["机会1", "机会2"],
            "recommendations": ["建议1", "建议2"]
        }}
        """

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }

        try:
            response = await self._make_request("chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "risks": [],
                    "opportunities": [],
                    "recommendations": [content]
                }

        except Exception as e:
            logger.error(f"Error identifying risks and opportunities: {e}")
            return {
                "risks": [],
                "opportunities": [],
                "recommendations": [],
                "error": str(e)
            }


# 全局服务实例
glm_service = GLMService()


async def get_glm_service() -> GLMService:
    """获取GLM服务实例"""
    return glm_service