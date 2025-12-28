"""
语义单元标注器 - Semantic Unit Annotator
为文档段落添加语义类型标签，提升检索精度
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SemanticType(Enum):
    """语义类型枚举"""
    ARGUMENT = "论述段落"  # 阐述观点、理论分析
    DATA_STATEMENT = "数据陈述"  # 引用数据、统计信息
    CASE_DESCRIPTION = "案例描述"  # 具体案例、实例说明
    DEFINITION = "定义说明"  # 概念定义、术语解释
    CONCLUSION = "结论总结"  # 总结性、概括性内容
    SUGGESTION = "建议措施"  # 建议、措施、行动计划
    QUESTION = "问题提出"  # 提出问题、指出矛盾
    BACKGROUND = "背景介绍"  # 背景信息、历史沿革
    UNKNOWN = "未知类型"


@dataclass
class SemanticAnnotation:
    """语义标注结果"""
    semantic_type: SemanticType
    importance: str  # 高/中/低
    keywords: List[str]
    summary: str
    confidence: float  # 0-1


class SemanticUnitAnnotator:
    """语义单元标注器"""

    # 语义类型识别规则
    SEMANTIC_PATTERNS = {
        SemanticType.DATA_STATEMENT: [
            r'\d+[%。，,]',  # 数字+百分比/标点
            r'(增长|下降|上升|减少|增加).{0,20}\d+',  # 趋势+数字
            r'(总计|合计|平均|占比).{0,30}\d+',  # 统计词汇+数字
            r'(数据|统计|图表|报告).*(显示|表明|显示)',
        ],
        SemanticType.DEFINITION: [
            r'^.{0,50}(是指|定义为|意思是|即|所谓)',
            r'^.{0,50}(概念|定义|术语).{0,30}是',
        ],
        SemanticType.CONCLUSION: [
            r'^(综上|总之|因此|所以|由此可见|综上所述)',
            r'(结论|总结|小结|概括).{0,20}(:|：)',
            r'^(结果|最终)',
        ],
        SemanticType.SUGGESTION: [
            r'^(建议|提议|推荐|应该|需要)',
            r'(措施|行动|方案|计划).{0,20}(:|：)',
            r'(优化|改进|提升).{0,30}',
        ],
        SemanticType.QUESTION: [
            r'^(如何|怎么|怎样|是否|为什么|何)',
            r'\?|？',  # 问号
            r'(问题|疑问|困惑|不解)',
        ],
        SemanticType.CASE_DESCRIPTION: [
            r'^(例如|比如|案例|实例|举例)',
            r'(以.{2,10}为例)',
        ],
        SemanticType.BACKGROUND: [
            r'^(背景|历史|起源|发展|演变)',
            r'^(在.{2,20}背景下|过去|以前|历史上)',
        ]
    }

    # 关键词权重（用于重要性判断）
    IMPORTANCE_KEYWORDS = {
        'high': [
            '关键', '核心', '重要', '主要', '首要', '根本', '基础',
            '战略', '重大', '显著', '突出', '明显', '重点', '核心',
            'critical', 'key', 'important', 'major', 'primary', 'fundamental'
        ],
        'medium': [
            '一定', '相关', '比较', '较为', '相当', '基本',
            'some', 'related', 'relatively', 'fairly'
        ]
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化标注器

        Args:
            config: 配置字典
                - use_llm: 是否使用LLM增强 (默认False)
                - llm_client: LLM客户端 (可选)
        """
        self.config = config or {}
        self.use_llm = self.config.get('use_llm', False)
        self.llm_client = self.config.get('llm_client', None)

        # 编译正则表达式
        self.compiled_patterns = {}
        for sem_type, patterns in self.SEMANTIC_PATTERNS.items():
            self.compiled_patterns[sem_type] = [
                re.compile(pattern) for pattern in patterns
            ]

    def annotate_paragraph(self, paragraph: str) -> SemanticAnnotation:
        """
        标注单个段落

        Args:
            paragraph: 段落文本

        Returns:
            SemanticAnnotation: 标注结果
        """
        if not paragraph or not isinstance(paragraph, str):
            return SemanticAnnotation(
                semantic_type=SemanticType.UNKNOWN,
                importance="低",
                keywords=[],
                summary="",
                confidence=0.0
            )

        # 基于规则识别
        semantic_type = self._identify_semantic_type(paragraph)
        importance = self._assess_importance(paragraph)
        keywords = self._extract_keywords(paragraph)
        summary = self._generate_summary(paragraph)
        confidence = self._calculate_confidence(paragraph, semantic_type)

        return SemanticAnnotation(
            semantic_type=semantic_type,
            importance=importance,
            keywords=keywords,
            summary=summary,
            confidence=confidence
        )

    def _identify_semantic_type(self, text: str) -> SemanticType:
        """识别语义类型"""
        text_lower = text.lower()

        # 检查每种模式
        type_scores = {}
        for sem_type, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text):
                    score += 1
            if score > 0:
                type_scores[sem_type] = score

        # 返回得分最高的类型
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]

        # 默认为论述段落
        return SemanticType.ARGUMENT

    def _assess_importance(self, text: str) -> str:
        """评估重要性"""
        text_lower = text.lower()

        # 检查高重要性关键词
        for keyword in self.IMPORTANCE_KEYWORDS['high']:
            if keyword in text_lower:
                return "高"

        # 检查中等重要性关键词
        for keyword in self.IMPORTANCE_KEYWORDS['medium']:
            if keyword in text_lower:
                return "中"

        # 根据段落长度判断（长段落通常更重要）
        if len(text) > 200:
            return "中"

        return "低"

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取：提取数字百分比和重要词汇
        keywords = []

        # 1. 提取数字和百分比
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        keywords.extend(numbers[:3])

        # 2. 提取引号内容（可能是术语）
        quoted = re.findall(r'["\']([^"\']{2,10})["\']', text)
        keywords.extend(quoted[:2])

        # 3. 如果关键词不够，提取长词
        if len(keywords) < max_keywords:
            words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)  # 中文词汇
            # 去重并限制数量
            unique_words = list(set(words))[:max_keywords - len(keywords)]
            keywords.extend(unique_words)

        return keywords[:max_keywords]

    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """生成摘要"""
        # 简单截取前几句
        sentences = re.split(r'[。！？.!?]', text)

        summary = ""
        for sentence in sentences:
            summary += sentence.strip() + "。"
            if len(summary) >= max_length:
                break

        return summary.strip()[:max_length]

    def _calculate_confidence(self, text: str, semantic_type: SemanticType) -> float:
        """计算置信度"""
        # 如果是明确匹配的类型，置信度高
        if semantic_type != SemanticType.ARGUMENT and semantic_type != SemanticType.UNKNOWN:
            return 0.8

        # 如果是默认类型，置信度较低
        return 0.5

    async def annotate_with_llm(self, paragraph: str) -> SemanticAnnotation:
        """
        使用LLM进行增强标注

        Args:
            paragraph: 段落文本

        Returns:
            SemanticAnnotation: 标注结果
        """
        if not self.llm_client or not self.use_llm:
            # 回退到规则标注
            return self.annotate_paragraph(paragraph)

        try:
            prompt = f"""请分析以下文本段落的语义类型和重要性。

文本内容:
{paragraph}

请以JSON格式返回分析结果，格式如下:
{{
  "semantic_type": "论述段落/数据陈述/案例描述/定义说明/结论总结/建议措施/问题提出/背景介绍",
  "importance": "高/中/低",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "summary": "一句话总结（不超过50字）"
}}

只返回JSON，不要其他内容。"""

            # 调用LLM（这里需要根据实际的LLM客户端调整）
            # response = await self.llm_client.generate(prompt)
            # result = json.loads(response)

            # 模拟LLM响应
            result = {
                "semantic_type": "论述段落",
                "importance": "中",
                "keywords": ["分析", "研究", "发现"],
                "summary": "该段落阐述了研究发现"
            }

            # 映射语义类型
            type_mapping = {
                "论述段落": SemanticType.ARGUMENT,
                "数据陈述": SemanticType.DATA_STATEMENT,
                "案例描述": SemanticType.CASE_DESCRIPTION,
                "定义说明": SemanticType.DEFINITION,
                "结论总结": SemanticType.CONCLUSION,
                "建议措施": SemanticType.SUGGESTION,
                "问题提出": SemanticType.QUESTION,
                "背景介绍": SemanticType.BACKGROUND
            }

            semantic_type = type_mapping.get(result.get("semantic_type"), SemanticType.ARGUMENT)

            return SemanticAnnotation(
                semantic_type=semantic_type,
                importance=result.get("importance", "中"),
                keywords=result.get("keywords", []),
                summary=result.get("summary", ""),
                confidence=0.9  # LLM标注置信度较高
            )

        except Exception as e:
            logger.error(f"LLM annotation failed: {e}")
            # 回退到规则标注
            return self.annotate_paragraph(paragraph)

    async def annotate_document(
        self,
        elements: List[Any],
        content_attr: str = 'content'
    ) -> tuple:
        """
        标注文档中所有段落

        Args:
            elements: 文档元素列表
            content_attr: 内容属性名

        Returns:
            (annotated_elements, statistics): (标注后的元素列表, 统计信息)
        """
        statistics = {
            'total_paragraphs': 0,
            'annotated_paragraphs': 0,
            'type_distribution': {},
            'importance_distribution': {'高': 0, '中': 0, '低': 0}
        }

        for element in elements:
            try:
                elem_type = getattr(element, 'element_type', None)
                content = getattr(element, content_attr, None)

                if elem_type == 'paragraph' and content:
                    statistics['total_paragraphs'] += 1

                    # 进行标注
                    if self.use_llm:
                        annotation = await self.annotate_with_llm(content)
                    else:
                        annotation = self.annotate_paragraph(content)

                    # 添加标注到元素元数据
                    if hasattr(element, 'metadata'):
                        element.metadata['semantic_annotation'] = {
                            'semantic_type': annotation.semantic_type.value,
                            'importance': annotation.importance,
                            'keywords': annotation.keywords,
                            'summary': annotation.summary,
                            'confidence': annotation.confidence
                        }

                    statistics['annotated_paragraphs'] += 1

                    # 更新统计
                    sem_type = annotation.semantic_type.value
                    statistics['type_distribution'][sem_type] = \
                        statistics['type_distribution'].get(sem_type, 0) + 1
                    statistics['importance_distribution'][annotation.importance] += 1

            except Exception as e:
                logger.error(f"Error annotating element: {e}")

        return elements, statistics

    def get_annotation_summary(self, statistics: Dict[str, Any]) -> str:
        """获取标注摘要"""
        summary = f"""
语义标注统计摘要:
  总段落数: {statistics['total_paragraphs']}
  已标注数: {statistics['annotated_paragraphs']}
  标注率: {statistics['annotated_paragraphs']/statistics['total_paragraphs']*100:.1f}%

语义类型分布:
"""
        for sem_type, count in sorted(statistics['type_distribution'].items(),
                                      key=lambda x: x[1], reverse=True):
            ratio = count / statistics['total_paragraphs'] * 100 if statistics['total_paragraphs'] > 0 else 0
            summary += f"  - {sem_type}: {count} ({ratio:.1f}%)\n"

        summary += "\n重要性分布:\n"
        for importance, count in statistics['importance_distribution'].items():
            ratio = count / statistics['total_paragraphs'] * 100 if statistics['total_paragraphs'] > 0 else 0
            summary += f"  - {importance}: {count} ({ratio:.1f}%)\n"

        return summary
