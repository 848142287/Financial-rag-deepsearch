"""
真实的Qwen多模态大模型服务
集成阿里云的Qwen3-VL-Plus、Qwen-VL-OCR、Qwen2.5-VL-Embedding等模型
"""

import asyncio
import base64
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json
import requests
from PIL import Image
import io
from app.services.minio_service import MinIOService

try:
    import dashscope
    from dashscope import Generation, MultiModalConversation, MultiModalEmbedding, TextEmbedding
    from http import HTTPStatus
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("dashscope SDK not installed, falling back to HTTP API")

logger = logging.getLogger(__name__)

# MinIO 服务实例（延迟初始化）
_minio_service = None

def get_minio_service():
    """获取 MinIO 服务实例"""
    global _minio_service
    if _minio_service is None:
        _minio_service = MinIOService()
    return _minio_service


@dataclass
class RealQwenConfig:
    """真实Qwen服务配置 - 强制使用高级多模态模型"""
    api_key: str = "sk-5233a3a4b1a24426b6846a432794bbe2"
    # DashScope SDK 使用原生 API，不需要 compatible-mode
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    multimodal_model: str = "qwen-vl-plus"  # 多模态模型（用于图像+文本分析）
    text_model: str = "qwen-plus"  # 纯文本模型（用于实体提取、文本生成等）
    ocr_model: str = "qwen-vl-ocr-latest"  # OCR专用模型
    embedding_model: str = "text-embedding-v4"  # 多模态嵌入模型
    text_embedding_model: str = "text-embedding-v4"  # 纯文本嵌入模型
    rerank_model: str = "qwen3-rerank"  # 重排序模型
    timeout: int = 120  # 增加超时时间
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens: int = 8000
    # 强制启用所有高级功能
    enable_image_analysis: bool = True
    enable_chart_analysis: bool = True
    enable_formula_extraction: bool = True
    enable_entity_extraction: bool = True


class RealQwenService:
    """真实Qwen多模态服务类"""

    def __init__(self, config: Optional[RealQwenConfig] = None):
        if config is None:
            config = RealQwenConfig()
        self.config = config

        # 初始化dashscope
        if DASHSCOPE_AVAILABLE:
            dashscope.api_key = self.config.api_key
            logger.info("使用DashScope SDK")
        else:
            logger.warning("DashScope SDK未安装，使用HTTP API")

    async def analyze_document_multimodal(self, file_content: bytes, filename: str, sections: List[Dict]) -> Dict[str, Any]:
        """使用qwen-vl-plus进行多模态文档分析"""
        logger.info(f"使用{self.config.multimodal_model}进行多模态分析...")

        try:
            if DASHSCOPE_AVAILABLE:
                return await self._analyze_with_sdk(file_content, filename, sections)
            else:
                return await self._analyze_with_http(file_content, filename, sections)
        except Exception as e:
            logger.error(f"多模态分析失败: {e}")
            return self._get_fallback_analysis(sections)

    async def _analyze_with_sdk(self, file_content: bytes, filename: str, sections: List[Dict]) -> Dict[str, Any]:
        """使用DashScope SDK进行分析 - 使用OCR方式提取图片文本"""
        analysis_results = {
            'model_used': f"{self.config.multimodal_model}+{self.config.ocr_model}",
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': '',
            'sections_analysis': [],
            'images_found': [],
            'charts_found': [],
            'formulas_found': []
        }

        # 使用OCR服务提取图片文本
        try:
            from app.services.ocr_service import get_ocr_service
            ocr_service = get_ocr_service()

            import fitz  # PyMuPDF
            pdf_document = fitz.open(stream=file_content, filetype="pdf")

            all_text = ""
            for page_num in range(min(len(pdf_document), 10)):  # 限制处理前10页
                page = pdf_document[page_num]

                # 提取文本
                text = page.get_text()
                all_text += f"\n\n--- 第 {page_num + 1} 页 ---\n\n{text}"

                # 转换页面为图片进行OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")

                # 使用OCR提取图片中的文本
                try:
                    logger.info(f"使用{self.config.ocr_model}对第{page_num + 1}页进行OCR...")

                    # 调用OCR服务提取文本
                    ocr_result = await ocr_service.extract_text_from_image(
                        image_bytes=img_data,
                        prompt="""请识别图片中的所有文字内容，包括标题、正文、表格、图表等。
请保持原文的格式和结构，按逻辑输出识别结果。

请特别注意：
1. 如果有表格，请用表格格式呈现
2. 如果有图表，请描述图表的内容和数据
3. 如果有公式，请用文字描述公式
4. 保持段落结构"""
                    )

                    if ocr_result['success']:
                        ocr_text = ocr_result['text']
                        logger.info(f"✅ OCR成功，提取了 {len(ocr_text)} 字符")

                        # 使用LLM分析OCR提取的文本
                        analysis_prompt = f"""基于以下OCR识别的文本内容，请提供结构化分析：

识别的文本内容：
{ocr_text[:2000]}

请以JSON格式返回分析结果：
{{
  "title": "页面主标题",
  "summary": "内容摘要（100字以内）",
  "key_points": ["要点1", "要点2", "要点3"],
  "has_images": true/false,
  "has_charts": true/false,
  "has_formulas": true/false,
  "content_types": ["表格", "图表", "公式", "正文"],
  "entities": ["实体1", "实体2"],
  "key_data": ["关键数据1", "关键数据2"]
}}"""

                        messages = [
                            {
                                "role": "user",
                                "content": analysis_prompt
                            }
                        ]

                        # 使用文本生成模型分析OCR结果
                        response = Generation.call(
                            model="qwen-plus",  # 使用纯文本模型分析OCR结果
                            messages=messages,
                            temperature=0.3,
                            max_tokens=1500
                        )

                        if response.status_code == 200:
                            result_text = response.output.text
                            try:
                                # 清理可能的markdown标记
                                if result_text.startswith('```json'):
                                    result_text = result_text[7:]
                                elif result_text.startswith('```'):
                                    result_text = result_text[3:]
                                if result_text.endswith('```'):
                                    result_text = result_text[:-3]
                                result_text = result_text.strip()

                                result_json = json.loads(result_text)

                                # 构建分析结果
                                section_analysis = {
                                    'page': page_num + 1,
                                    'title': result_json.get('title', f'第 {page_num + 1} 页'),
                                    'summary': result_json.get('summary', ocr_text[:200]),
                                    'key_points': result_json.get('key_points', []),
                                    'has_images': result_json.get('has_images', False),
                                    'has_charts': result_json.get('has_charts', False),
                                    'has_formulas': result_json.get('has_formulas', False),
                                    'ocr_text': ocr_text,
                                    'content_types': result_json.get('content_types', [])
                                }

                                analysis_results['sections_analysis'].append(section_analysis)

                                # 记录图片、图表、公式信息
                                if section_analysis['has_images']:
                                    analysis_results['images_found'].append({
                                        'page': page_num + 1,
                                        'description': f"检测到图片内容 (OCR识别)"
                                    })

                                if section_analysis['has_charts']:
                                    analysis_results['charts_found'].append({
                                        'page': page_num + 1,
                                        'analysis': "检测到图表或数据 (OCR识别)"
                                    })

                                if section_analysis['has_formulas']:
                                    analysis_results['formulas_found'].append({
                                        'page': page_num + 1,
                                        'explanation': "检测到数学公式 (OCR识别)"
                                    })

                                logger.info(f"✅ 第 {page_num + 1} 页分析完成")

                            except json.JSONDecodeError:
                                logger.warning(f"无法解析第 {page_num + 1} 页的分析结果，使用原始OCR文本")
                                # 使用OCR文本作为结果
                                section_analysis = {
                                    'page': page_num + 1,
                                    'title': f'第 {page_num + 1} 页',
                                    'summary': ocr_text[:200],
                                    'key_points': [],
                                    'has_images': False,
                                    'has_charts': False,
                                    'has_formulas': False,
                                    'ocr_text': ocr_text
                                }
                                analysis_results['sections_analysis'].append(section_analysis)
                        else:
                            logger.error(f"分析OCR结果失败: {response.status_code}")
                            # 直接使用OCR文本
                            section_analysis = {
                                'page': page_num + 1,
                                'title': f'第 {page_num + 1} 页',
                                'summary': ocr_text[:200],
                                'key_points': [],
                                'has_images': False,
                                'has_charts': False,
                                'has_formulas': False,
                                'ocr_text': ocr_text
                            }
                            analysis_results['sections_analysis'].append(section_analysis)
                    else:
                        logger.error(f"OCR失败: {ocr_result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"第 {page_num + 1} 页OCR处理失败: {e}")
                    # 使用纯文本作为回退
                    section_analysis = {
                        'page': page_num + 1,
                        'title': f'第 {page_num + 1} 页',
                        'summary': text[:200] if text else '',
                        'key_points': [],
                        'has_images': False,
                        'has_charts': False,
                        'has_formulas': False,
                        'extracted_text': text
                    }
                    if text.strip():
                        analysis_results['sections_analysis'].append(section_analysis)

            pdf_document.close()

            # 生成整体摘要
            if all_text:
                summary_messages = [
                    {
                        "role": "user",
                        "content": f"""请为以下文档内容生成一个简洁的摘要（不超过200字）：

{all_text[:2000]}...

摘要应该包含：
1. 文档主题
2. 主要内容
3. 关键结论"""
                    }
                ]

                summary_response = Generation.call(
                    model=self.config.multimodal_model,
                    messages=summary_messages,
                    temperature=0.3,
                    max_tokens=300
                )

                if summary_response.status_code == 200:
                    analysis_results['summary'] = summary_response.output.text

        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            return await self._analyze_text_only(sections)

        return analysis_results

    async def _analyze_text_only(self, sections: List[Dict]) -> Dict[str, Any]:
        """纯文本分析（回退方案）"""
        analysis_results = {
            'model_used': self.config.multimodal_model,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': '这是一个PDF文档的文本分析',
            'sections_analysis': [],
            'images_found': [],
            'charts_found': [],
            'formulas_found': []
        }

        for section in sections[:10]:
            section_analysis = {
                'page': section.get('page', 1),
                'title': section.get('title', '未命名章节'),
                'summary': section.get('content', '')[:300],
                'key_points': [],
                'has_images': False,
                'has_charts': False,
                'has_formulas': False
            }

            # 简单检测
            content = section.get('content', '')
            if '图' in content or 'image' in content.lower():
                section_analysis['has_images'] = True
                analysis_results['images_found'].append({
                    'page': section.get('page', 1),
                    'description': '检测到图片内容'
                })

            if '表' in content or 'chart' in content.lower() or '数据' in content:
                section_analysis['has_charts'] = True
                analysis_results['charts_found'].append({
                    'page': section.get('page', 1),
                    'analysis': '检测到图表或数据'
                })

            if any(word in content for word in ['公式', 'equation', 'Σ', '∑', '∫', '±']):
                section_analysis['has_formulas'] = True
                analysis_results['formulas_found'].append({
                    'page': section.get('page', 1),
                    'explanation': '检测到数学公式'
                })

            analysis_results['sections_analysis'].append(section_analysis)

        return analysis_results

    async def _analyze_with_http(self, file_content: bytes, filename: str, sections: List[Dict]) -> Dict[str, Any]:
        """使用HTTP API进行分析"""
        # 实现HTTP API调用逻辑
        logger.info("使用HTTP API进行分析")
        return await self._analyze_text_only(sections)

    def _get_fallback_analysis(self, sections: List[Dict]) -> Dict[str, Any]:
        """获取回退分析结果"""
        return {
            'model_used': self.config.multimodal_model,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': '文档分析摘要',
            'sections_analysis': [],
            'images_found': [],
            'charts_found': [],
            'formulas_found': []
        }

    async def extract_entities_relationships(self, text_content: str) -> tuple[List[Dict], List[Dict]]:
        """使用qwen-vl-plus提取实体关系 - 优化版（上下文缩减+缓存）"""
        logger.info(f"🔍 开始提取实体关系... (文本长度: {len(text_content)} 字符)")

        try:
            # 【优化1】检查Redis缓存
            import hashlib
            import redis.asyncio as redis

            text_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
            cache_key = f"entity_extraction:{text_hash}"

            try:
                redis_client = await redis.Redis(
                    host='redis',
                    port=6379,
                    password='redis123456',
                    db=3,  # 使用DB3用于实体缓存
                    decode_responses=False
                )

                # 尝试从缓存获取
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    import json
                    cached_result = json.loads(cached_data)
                    logger.info(f"✅ 使用缓存的实体关系 (hash: {text_hash[:8]}...)")
                    return cached_result['entities'], cached_result['relationships']

            except Exception as e:
                logger.warning(f"Redis缓存检查失败: {e}")

            # 【优化2】智能上下文缩减 - 只处理关键部分
            max_length = 3000  # 从6000减少到3000（减少50%处理时间）

            if len(text_content) > max_length:
                # 智能截取策略：优先保留开头和结尾，跳过中间重复内容
                text_start = text_content[:1500]  # 前1500字符（通常包含重要信息）
                text_end = text_content[-1500:] if len(text_content) > 3000 else ""  # 后1500字符

                # 组合关键部分
                text_to_process = text_start + "\n...\n" + text_end if text_end else text_start

                logger.info(f"📏 文本智能缩减: {len(text_content)} → {len(text_to_process)} 字符 (保留{len(text_to_process)/len(text_content)*100:.1f}%)")
            else:
                text_to_process = text_content

            # 改进的提示词 - 更具体的指令，强调关系提取
            prompt = f"""你是一个专业的金融文档实体和关系识别专家。请从以下文本中仔细提取实体和它们之间的关系。

文本内容：
{text_to_process}

请按以下要求提取：

1. **实体类型**：
   - 公司/机构（如：寒武纪、华为、英伟达、海光、中科院）
   - 产品/技术（如：AI芯片、CPU、GPU、思元590、A100、昇腾910）
   - 人物/角色（如：CEO、专家、分析师）
   - 数值/指标（如：1600亿元、100%、2023年）
   - 地理位置（如：中国、美国）

2. **关系类型**（重要！必须提取关系）：
   - 生产关系：A公司生产B产品（例：英伟达生产A100芯片）
   - 竞争关系：A与B在市场上竞争（例：英伟达与AMD竞争）
   - 合作关系：A与B有合作关系（例：华为与寒武纪合作）
   - 所属关系：A属于B类别或位于B地
   - 对比关系：A达到B数值或具有B特征

3. **提取示例**：
   文本："英伟达生产A100 GPU芯片，与寒武纪的思元590竞争"
   实体：[英伟达, A100, GPU芯片, 寒武纪, 思元590]
   关系：
   - [英伟达, 生产, A100]
   - [英伟达, 竞争, 寒武纪]
   - [寒武纪, 生产, 思元590]

请严格按以下JSON格式返回：
{{
  "entities": [
    {{"name": "英伟达", "type": "公司", "confidence": 0.95}},
    {{"name": "A100", "type": "产品", "confidence": 0.95}}
  ],
  "relationships": [
    {{"from_entity": "英伟达", "to_entity": "A100", "type": "生产", "confidence": 0.9}},
    {{"from_entity": "英伟达", "to_entity": "寒武纪", "type": "竞争", "confidence": 0.85}}
  ]
}}

重要提示：
- 必须提取实体之间的关系，这是最重要的任务
- 如果没有明确的关系，至少提取"提及"关系（在同一句话中出现的实体）
- 只返回JSON，不要有其他文字
- 如果找不到任何实体，返回 {{"entities": [], "relationships": []}}"""

            messages = [
                {
                    "role": "system",
                    "content": "你是专业的金融文档分析专家，擅长提取实体和关系。必须返回纯JSON格式。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # 调用模型 - 使用纯文本模型进行实体提取
            response = Generation.call(
                model=self.config.text_model,
                messages=messages,
                temperature=0.05,  # 降低温度
                max_tokens=3000,    # 增加输出长度
                result_format='message'  # 确保消息格式
            )

            if response.status_code == 200:
                # 修复NoneType错误: 检查response.output.text是否为None
                result_text = response.output.text if response.output.text else ""

                if result_text:
                    result_text = result_text.strip()

                    # 清理可能的markdown代码块标记
                    if result_text.startswith('```json'):
                        result_text = result_text[7:]
                    elif result_text.startswith('```'):
                        result_text = result_text[3:]
                    if result_text.endswith('```'):
                        result_text = result_text[:-3]
                    result_text = result_text.strip()

                try:
                    result = json.loads(result_text) if result_text else {'entities': [], 'relationships': []}
                    entities = result.get('entities', [])
                    relationships = result.get('relationships', [])

                    # 验证和清理数据
                    valid_entities = []
                    valid_relationships = []

                    for entity in entities:
                        if isinstance(entity, dict) and 'name' in entity and entity['name']:
                            name = str(entity['name']).strip() if entity['name'] else None
                            if name:
                                entity_type = entity.get('type', 'UNKNOWN') or 'UNKNOWN'
                                description = entity.get('description') or ''
                                valid_entities.append({
                                    'name': name,
                                    'type': str(entity_type).strip(),
                                    'description': str(description).strip(),
                                    'confidence': float(entity.get('confidence', 0.7))
                                })

                    for rel in relationships:
                        if isinstance(rel, dict):
                            # 支持多种字段名：from_entity/to_entity 或 source/target
                            from_entity = rel.get('from_entity') or rel.get('source')
                            to_entity = rel.get('to_entity') or rel.get('target')
                            rel_type = rel.get('type') or rel.get('relation', 'RELATED_TO')

                            if from_entity and to_entity:
                                valid_relationships.append({
                                    'from_entity': str(from_entity).strip(),
                                    'to_entity': str(to_entity).strip(),
                                    'type': str(rel_type).strip(),
                                    'description': rel.get('description', ''),
                                    'confidence': float(rel.get('confidence', 0.7))
                                })

                    logger.info(f"✅ 提取到 {len(valid_entities)} 个实体和 {len(valid_relationships)} 个关系")

                    # 【优化3】缓存提取结果
                    try:
                        cache_data = {
                            'entities': valid_entities,
                            'relationships': valid_relationships,
                            'text_length': len(text_content),
                            'timestamp': datetime.now().isoformat()
                        }
                        await redis_client.setex(
                            cache_key,
                            86400,  # 缓存24小时
                            json.dumps(cache_data, ensure_ascii=False)
                        )
                        logger.info(f"✅ 实体关系已缓存 (hash: {text_hash[:8]}...)")
                    except Exception as cache_err:
                        logger.warning(f"缓存存储失败: {cache_err}")

                    return valid_entities, valid_relationships

                except json.JSONDecodeError as je:
                    logger.warning(f"JSON解析失败: {je}, 尝试备用方案")
                    # 备用方案: 使用正则表达式提取实体
                    return await self._extract_entities_fallback(text_to_process)

            else:
                logger.error(f"❌ API调用失败: {response.status_code}, {response.message}")
                logger.error(f"请求模型: {self.config.multimodal_model}")
                logger.error(f"失败原因可能: URL配置错误或模型不可用")
                logger.error(f"详细错误: {response}")
                logger.warning("将使用备用方案提取实体和关系...")
                return await self._extract_entities_fallback(text_to_process)

        except Exception as e:
            logger.error(f"实体关系提取失败: {e}")
            return await self._extract_entities_fallback(text_content)

    async def _extract_entities_fallback(self, text_content: str) -> tuple[List[Dict], List[Dict]]:
        """备用实体提取方案 - 增强版，包含简单关系提取"""
        import re

        logger.info("使用备用方案提取实体和关系...")

        entities = []
        relationships = []

        try:
            # 提取公司/机构名称
            company_patterns = [
                r'英伟达|华为|寒武纪|海光|中科院|阿里|腾讯|百度|字节|美团|京东|小米|OPPO|vivo',
                r'\w+[科技公司|集团|证券|银行|保险]{1,3}',
                r'OpenAI|Anthropic|Google|Microsoft|Apple|Meta|Amazon'
            ]

            # 提取产品名称
            product_patterns = [
                r'[A-Z0-9]+-[A-Z0-9]+',  # 如 A100, H100
                r'思元\d+[A-Z]*',
                r'昇腾\d+[A-Z]*',
                r'AI\s*芯片|CPU|GPU|DCU|CUDA|CANN|ChatGPT|GPT'
            ]

            # 提取数值指标
            value_patterns = [
                r'\d+[.]\d+\s*(?:亿元|万元|%|倍|TFlops)',
                r'\d{4}年',
                r'\d+\s*[个项台]'
            ]

            all_patterns = company_patterns + product_patterns + value_patterns

            for pattern in all_patterns:
                matches = re.findall(pattern, text_content)
                for match in matches:
                    if len(str(match)) >= 2:  # 过滤掉太短的
                        entities.append({
                            'name': str(match),
                            'type': self._guess_entity_type(str(match)),
                            'description': f"从文本中提取",
                            'confidence': 0.6
                        })

            # 去重实体
            seen = set()
            unique_entities = []
            for entity in entities:
                if entity['name'] not in seen:
                    seen.add(entity['name'])
                    unique_entities.append(entity)

            # 简单关系提取：基于句子中共现的实体
            logger.info("开始提取简单关系...")

            # 分割文本为句子
            sentences = re.split(r'[。！？；;.\n]', text_content)

            # 为每个句子中的实体创建"提及"关系
            entity_names = [e['name'] for e in unique_entities]

            for sentence in sentences:
                if len(sentence) < 10:  # 跳过太短的句子
                    continue

                # 找出这个句子中出现的实体
                entities_in_sentence = []
                for entity_name in entity_names:
                    if entity_name in sentence:
                        entities_in_sentence.append(entity_name)

                # 如果句子中有2个或更多实体，创建关系
                if len(entities_in_sentence) >= 2:
                    for i in range(len(entities_in_sentence) - 1):
                        from_entity = entities_in_sentence[i]
                        to_entity = entities_in_sentence[i + 1]

                        # 避免重复
                        rel_key = f"{from_entity}->{to_entity}"
                        if not any(r.get('relation_key') == rel_key for r in relationships):
                            relationships.append({
                                'from_entity': from_entity,
                                'to_entity': to_entity,
                                'type': '提及',
                                'description': f"在同一句子中出现: {sentence[:50]}...",
                                'confidence': 0.5,
                                'relation_key': rel_key
                            })

            logger.info(f"✅ 备用方案提取到 {len(unique_entities)} 个实体和 {len(relationships)} 个关系")
            return unique_entities[:20], relationships[:15]  # 限制数量

        except Exception as e:
            logger.error(f"备用实体提取失败: {e}")
            return [], []

    def _guess_entity_type(self, name: str) -> str:
        """根据名称猜测实体类型"""
        import re
        if '公司' in name or '集团' in name or '证券' in name or '银行' in name:
            return '公司'
        elif any(x in name for x in ['芯片', 'CPU', 'GPU', 'DCU', 'GPT', 'CUDA', 'CANN']):
            return '产品'
        elif re.search(r'\d+[.]\d+|%|年', name):
            return '数值'
        else:
            return 'UNKNOWN'

    async def generate_embeddings_multimodal(self, texts: List[str], images: Optional[List[bytes]] = None) -> List[List[float]]:
        """使用qwen2.5-vl-embedding生成多模态嵌入"""
        logger.info(f"使用{self.config.embedding_model}生成多模态嵌入...")

        embeddings = []

        try:
            if DASHSCOPE_AVAILABLE and images:
                # 多模态嵌入
                for i, text in enumerate(texts[:10]):  # 限制处理数量
                    input_data = [{'text': text}]

                    # 如果有图片，添加图片
                    if i < len(images):
                        # 将图片转换为base64
                        img_base64 = base64.b64encode(images[i]).decode()
                        input_data.append({'image': img_base64})

                    resp = MultiModalEmbedding.call(
                        model=self.config.embedding_model,
                        input=input_data
                    )

                    if resp.status_code == HTTPStatus.OK:
                        embedding = resp.output['embeddings'][0]['embedding']
                        embeddings.append(embedding)
                    else:
                        logger.error(f"多模态嵌入生成失败: {resp}")
                        # 使用文本嵌入作为回退
                        text_embedding = await self._generate_text_embedding(text)
                        embeddings.append(text_embedding)
            else:
                # 纯文本嵌入
                for text in texts[:10]:
                    embedding = await self._generate_text_embedding(text)
                    embeddings.append(embedding)

        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            # 返回零向量
            for _ in texts[:10]:
                embeddings.append([0.0] * 1024)  # Qwen2.5-VL-Embedding维度

        return embeddings

    async def _generate_text_embedding(self, text: str) -> List[float]:
        """生成文本嵌入（使用text-embedding-v4）"""
        try:
            if DASHSCOPE_AVAILABLE:
                resp = TextEmbedding.call(
                    model=self.config.text_embedding_model,
                    input=text
                )

                if resp.status_code == HTTPStatus.OK:
                    return resp.output['embeddings'][0]['embedding']

            # HTTP API回退
            url = f"{self.config.base_url}/embeddings"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.config.text_embedding_model,
                "input": text
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result['data'][0]['embedding']

        except Exception as e:
            logger.error(f"文本嵌入生成失败: {e}")

        # 返回零向量
        return [0.0] * 1024  # Qwen2.5-VL-Embedding维度

    async def rerank_documents(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        """使用qwen3-rerank进行文档重排序"""
        logger.info(f"使用{self.config.rerank_model}进行文档重排序...")

        try:
            # 准备请求
            url = f"https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.config.rerank_model,
                "input": {
                    "query": query,
                    "documents": documents[:10]  # 限制文档数量
                },
                "parameters": {
                    "return_documents": True,
                    "top_n": top_n,
                    "instruct": "Given a query, retrieve relevant passages that answer the query."
                }
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                if 'output' in result and 'results' in result['output']:
                    ranked_docs = result['output']['results']
                    logger.info(f"重排序完成，返回 {len(ranked_docs)} 个文档")
                    return ranked_docs

            logger.error(f"重排序失败: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"文档重排序失败: {e}")

        # 回退方案：返回原始顺序
        return [{'index': i, 'document': doc, 'relevance_score': 1.0}
                for i, doc in enumerate(documents[:top_n])]

    async def extract_formulas(self, text: str) -> List[Dict]:
        """使用qwen-vl-plus提取和解释公式"""
        logger.info("提取数学公式...")

        try:
            prompt = f"""请从以下文本中提取所有数学公式，并解释其含义：

{text_content[:1500]}

请以JSON格式返回：
{{
  "formulas": [
    {{
      "formula": "公式表达式",
      "explanation": "公式含义解释",
      "variables": ["变量说明"],
      "context": "公式上下文"
    }}
  ]
}}"""

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = Generation.call(
                model=self.config.text_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )

            if response.status_code == 200:
                result_text = response.output.text
                try:
                    result = json.loads(result_text)
                    formulas = result.get('formulas', [])
                    logger.info(f"提取到 {len(formulas)} 个公式")
                    return formulas
                except json.JSONDecodeError:
                    logger.error("公式提取结果解析失败")

        except Exception as e:
            logger.error(f"公式提取失败: {e}")

        return []

    async def analyze_images(self, image_data: bytes, context: str = "") -> Dict:
        """使用qwen-vl-ocr-latest分析图片"""
        logger.info("使用qwen-vl-ocr-latest分析图片...")

        try:
            # 检查图片大小
            img_size_mb = len(image_data) / (1024 * 1024)
            if img_size_mb > 8:
                logger.error(f"Image too large ({img_size_mb:.2f}MB)")
                return {"description": "图片太大，无法处理"}

            # 转换为 Base64 格式
            img_base64 = base64.b64encode(image_data).decode()
            image_url = f"data:image/jpeg;base64,{img_base64}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": image_url,  # 使用 Base64 格式
                        },
                        {
                            "text": f"""请分析这张图片内容：
{context}

请描述：
1. 图片的主要内容
2. 识别的文字信息
3. 图片中的数据或图表
4. 图片的意义

请以JSON格式返回：{{"description": "图片描述", "text_content": "识别的文字", "data_analysis": "数据分析", "significance": "图片意义"}}"""
                        }
                    ]
                }
            ]

            response = MultiModalConversation.call(
                model=self.config.ocr_model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )

            if response.status_code == 200:
                result_text = response.output.text
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"description": result_text}

        except Exception as e:
            logger.error(f"图片分析失败: {e}")

        return {"description": "图片分析失败"}

    async def analyze_charts(self, text: str, page: int = 1) -> List[Dict]:
        """分析图表数据和趋势"""
        logger.info("分析图表数据...")

        try:
            prompt = f"""请分析以下文本中的图表数据：

{text[:1000]}

请识别：
1. 图表类型（柱状图、折线图、饼图等）
2. 数据规律
3. 趋势分析
4. 统计意义

请以JSON格式返回：
{{
  "charts": [
    {{
      "type": "图表类型",
      "data_pattern": "数据规律描述",
      "trend": "趋势分析",
      "statistical_significance": "统计意义",
      "insights": ["关键洞察"]
    }}
  ]
}}"""

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = Generation.call(
                model=self.config.text_model,
                messages=messages,
                temperature=0.2,
                max_tokens=1200
            )

            if response.status_code == 200:
                result_text = response.output.text
                try:
                    result = json.loads(result_text)
                    charts = result.get('charts', [])
                    # 添加页码信息
                    for chart in charts:
                        chart['page'] = page
                    return charts
                except json.JSONDecodeError:
                    logger.error("图表分析结果解析失败")

        except Exception as e:
            logger.error(f"图表分析失败: {e}")

        return []