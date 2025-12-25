"""
多模态内容解析器
统一处理文本、图片、表格、公式等多种内容类型
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import base64
import io

from .base import BaseFileParser, ParseResult
from ..multimodal.image_analyzer import ImageAnalyzer
from ..multimodal.table_extractor import TableExtractor
from ..multimodal.formula_parser import FormulaParser
from ..multimodal.chart_detector import ChartDetector

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"
    CHART = "chart"
    CODE = "code"
    DIAGRAM = "diagram"


@dataclass
class MultimodalContent:
    """多模态内容结构"""
    content_type: ContentType
    text: str
    metadata: Dict[str, Any]
    position: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    cross_reference: Optional[str] = None


class MultimodalParser(BaseFileParser):
    """多模态内容解析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.parser_name = "MultimodalParser"
        self.supported_extensions = [
            '.pdf', '.docx', '.doc', '.pptx', '.ppt',
            '.xlsx', '.xls', '.csv', '.txt', '.md',
            '.html', '.htm', '.png', '.jpg', '.jpeg',
            '.tiff', '.tif', '.gif', '.bmp'
        ]

        # 初始化各个子解析器
        self.image_analyzer = ImageAnalyzer(config or {})
        self.table_extractor = TableExtractor(config or {})
        self.formula_parser = FormulaParser(config or {})
        self.chart_detector = ChartDetector(config or {})

        # 内容分类器
        self.content_classifiers = {
            ContentType.IMAGE: self._classify_image,
            ContentType.TABLE: self._classify_table,
            ContentType.FORMULA: self._classify_formula,
            ContentType.CHART: self._classify_chart,
            ContentType.CODE: self._classify_code,
            ContentType.DIAGRAM: self._classify_diagram
        }

    def can_parse(self, file_path: str) -> bool:
        """检查是否可以解析该文件"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions

    async def parse(self, file_path: str, **kwargs) -> ParseResult:
        """解析文件并提取多模态内容"""
        try:
            # 使用基础解析器获取初始内容
            from .adapter import ParserAdapter
            adapter = ParserAdapter()
            base_result = await adapter.parse(file_path)

            # 分析和分类内容
            multimodal_contents = []
            text_chunks = base_result.text.split('\n\n')

            for i, chunk in enumerate(text_chunks):
                if chunk.strip():
                    content_type = self._classify_content(chunk)

                    if content_type == ContentType.TEXT:
                        # 普通文本
                        multimodal_contents.append(MultimodalContent(
                            content_type=content_type,
                            text=chunk.strip(),
                            metadata={'chunk_index': i},
                            position={'page': kwargs.get('page', 1), 'index': i}
                        ))
                    else:
                        # 多模态内容，需要进一步分析
                        analyzed_content = await self._analyze_multimodal_content(
                            chunk, content_type, i, **kwargs
                        )
                        multimodal_contents.extend(analyzed_content)

            # 提取实体和关系
            entities, relations = await self._extract_knowledge(multimodal_contents)

            # 处理跨页内容
            cross_page_contents = await self._process_cross_page_content(
                multimodal_contents, **kwargs
            )

            return ParseResult(
                text=base_result.text,
                metadata={
                    **base_result.metadata,
                    'multimodal_contents': [
                        {
                            'type': content.content_type.value,
                            'text': content.text,
                            'metadata': content.metadata,
                            'position': content.position,
                            'confidence': content.confidence,
                            'cross_reference': content.cross_reference
                        }
                        for content in cross_page_contents
                    ],
                    'entities': entities,
                    'relations': relations,
                    'content_statistics': self._generate_statistics(cross_page_contents)
                },
                tables=base_result.tables,
                images=base_result.images
            )

        except Exception as e:
            logger.error(f"多模态解析失败 {file_path}: {str(e)}")
            raise

    def _classify_content(self, text: str) -> ContentType:
        """对文本内容进行分类"""
        # 检查是否为表格
        if self._is_table_content(text):
            return ContentType.TABLE

        # 检查是否为公式
        if self._is_formula_content(text):
            return ContentType.FORMULA

        # 检查是否为代码
        if self._is_code_content(text):
            return ContentType.CODE

        # 检查是否包含图表信息
        if self._is_chart_content(text):
            return ContentType.CHART

        # 检查是否为图表说明
        if self._is_diagram_content(text):
            return ContentType.DIAGRAM

        # 默认为文本
        return ContentType.TEXT

    def _is_table_content(self, text: str) -> bool:
        """判断是否为表格内容"""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False

        # 检查是否有分隔符
        separators = ['|', '\t', ',', ';']
        has_separator = any('|' in line or '\t' in line for line in lines[:3])

        # 检查是否有表头
        header_line = lines[0]
        content_lines = lines[1:3] if len(lines) > 1 else []

        if has_separator and content_lines:
            return True

        # 检查数字列是否对齐
        if content_lines:
            for line in content_lines:
                numbers = re.findall(r'\d+\.?\d*', line)
                if len(numbers) >= 2:
                    return True

        return False

    def _is_formula_content(self, text: str) -> bool:
        """判断是否为公式内容"""
        formula_indicators = [
            r'\$.*?\$',  # LaTeX公式
            r'\\[a-zA-Z]+\{.*?\}',  # LaTeX命令
            r'∑\s*.*?\s*=',  # 求和
            r'∫\s*.*?\s*dx',  # 积分
            r'√\s*\(',
            r'π\s*[+\-*/=]',
            r'[a-zA-Z]\s*[+\-*/]\s*[a-zA-Z]\s*=',
            r'ROE\s*=',
            r'ROA\s*=',
            r'PE\s*=',
            r'P/E\s*=',
        ]

        for pattern in formula_indicators:
            if re.search(pattern, text):
                return True

        return False

    def _is_code_content(self, text: str) -> bool:
        """判断是否为代码内容"""
        code_indicators = [
            r'function\s+\w+\s*\(',
            r'class\s+\w+\s*:',
            r'def\s+\w+\s*\(',
            r'import\s+\w+',
            r'#include\s*<',
            r'public\s+class\s+\w+',
            r'<script\s*>',
            r'SELECT\s+.*\s+FROM',
            r'CREATE\s+TABLE',
        ]

        for pattern in code_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # 检查是否有代码块标记
        if '```' in text or '{{{' in text:
            return True

        return False

    def _is_chart_content(self, text: str) -> bool:
        """判断是否为图表内容"""
        chart_keywords = [
            '柱状图', '折线图', '饼图', '散点图', '雷达图',
            'bar chart', 'line chart', 'pie chart', 'scatter plot',
            '图表显示', '如图所示', 'Figure', 'Chart', 'Graph',
            'X轴', 'Y轴', '横坐标', '纵坐标',
            '同比增长', '环比增长', '占比', '百分比'
        ]

        text_lower = text.lower()
        for keyword in chart_keywords:
            if keyword in text_lower:
                return True

        return False

    def _is_diagram_content(self, text: str) -> bool:
        """判断是否为图表说明"""
        diagram_keywords = [
            '流程图', '架构图', '组织架构', '系统架构',
            'flowchart', 'architecture', 'diagram',
            '框图', '示意图', '拓扑图',
            '数据流向', '业务流程'
        ]

        text_lower = text.lower()
        for keyword in diagram_keywords:
            if keyword in text_lower:
                return True

        return False

    async def _analyze_multimodal_content(
        self,
        text: str,
        content_type: ContentType,
        chunk_index: int,
        **kwargs
    ) -> List[MultimodalContent]:
        """分析多模态内容"""
        contents = []

        if content_type == ContentType.IMAGE:
            # 分析图片内容
            image_analysis = await self.image_analyzer.analyze(text)
            contents.append(MultimodalContent(
                content_type=content_type,
                text=text,
                metadata={
                    'analysis': image_analysis,
                    'image_type': image_analysis.get('type', 'unknown')
                },
                position={'page': kwargs.get('page', 1), 'index': chunk_index},
                confidence=image_analysis.get('confidence', 0.8)
            ))

        elif content_type == ContentType.TABLE:
            # 提取表格结构
            table_data = await self.table_extractor.extract(text)
            contents.append(MultimodalContent(
                content_type=content_type,
                text=text,
                metadata={
                    'table_data': table_data,
                    'rows': len(table_data.get('data', [])),
                    'columns': len(table_data.get('headers', [])),
                    'table_type': table_data.get('type', 'standard')
                },
                position={'page': kwargs.get('page', 1), 'index': chunk_index},
                confidence=table_data.get('confidence', 0.9)
            ))

        elif content_type == ContentType.FORMULA:
            # 解析公式
            formula_data = await self.formula_parser.parse(text)
            contents.append(MultimodalContent(
                content_type=content_type,
                text=text,
                metadata={
                    'formula': formula_data,
                    'formula_type': formula_data.get('type', 'math'),
                    'variables': formula_data.get('variables', []),
                    'category': formula_data.get('category', 'general')
                },
                position={'page': kwargs.get('page', 1), 'index': chunk_index},
                confidence=formula_data.get('confidence', 0.85)
            ))

        elif content_type == ContentType.CHART:
            # 检测图表类型
            chart_data = await self.chart_detector.detect(text)
            contents.append(MultimodalContent(
                content_type=content_type,
                text=text,
                metadata={
                    'chart_type': chart_data.get('type', 'unknown'),
                    'chart_data': chart_data.get('data', {}),
                    'description': chart_data.get('description', '')
                },
                position={'page': kwargs.get('page', 1), 'index': chunk_index},
                confidence=chart_data.get('confidence', 0.75)
            ))

        elif content_type == ContentType.CODE:
            # 分析代码
            contents.append(MultimodalContent(
                content_type=content_type,
                text=text,
                metadata={
                    'language': self._detect_language(text),
                    'lines': len(text.split('\n'))
                },
                position={'page': kwargs.get('page', 1), 'index': chunk_index},
                confidence=0.95
            ))

        elif content_type == ContentType.DIAGRAM:
            # 处理图表说明
            contents.append(MultimodalContent(
                content_type=content_type,
                text=text,
                metadata={
                    'diagram_type': self._detect_diagram_type(text),
                    'elements': self._extract_diagram_elements(text)
                },
                position={'page': kwargs.get('page', 1), 'index': chunk_index},
                confidence=0.7
            ))

        return contents

    def _detect_language(self, code: str) -> str:
        """检测代码语言"""
        if 'def ' in code and 'import ' in code:
            return 'python'
        elif 'function' in code or 'var ' in code:
            return 'javascript'
        elif 'public class' in code:
            return 'java'
        elif 'SELECT' in code.upper():
            return 'sql'
        elif '#include' in code:
            return 'cpp'
        else:
            return 'unknown'

    def _detect_diagram_type(self, text: str) -> str:
        """检测图表类型"""
        if '流程' in text or 'flow' in text.lower():
            return 'flowchart'
        elif '架构' in text or 'architecture' in text.lower():
            return 'architecture'
        elif '组织' in text or 'organization' in text.lower():
            return 'organization'
        else:
            return 'diagram'

    def _extract_diagram_elements(self, text: str) -> List[str]:
        """提取图表元素"""
        elements = []
        # 简单的元素提取逻辑
        element_patterns = [
            r'(\w+)\s*[-=]>\s*(\w+)',  # A -> B
            r'(\w+)\s*\[(.*?)\]',       # A[描述]
            r'\((.*?)\)',               # 括号内的内容
        ]

        for pattern in element_patterns:
            matches = re.findall(pattern, text)
            elements.extend([str(m) for m in matches])

        return list(set(elements))

    async def _extract_knowledge(
        self,
        contents: List[MultimodalContent]
    ) -> Tuple[List[Dict], List[Dict]]:
        """从多模态内容中提取知识"""
        entities = []
        relations = []

        for content in contents:
            # 从文本中提取实体
            if content.content_type in [ContentType.TEXT, ContentType.TABLE]:
                text_entities = self._extract_text_entities(content.text)
                for entity in text_entities:
                    entities.append({
                        'text': entity,
                        'type': self._classify_entity_type(entity),
                        'source': content.content_type.value,
                        'position': content.position,
                        'confidence': content.confidence * 0.8
                    })

            # 从表格中提取关系
            if content.content_type == ContentType.TABLE:
                table_relations = self._extract_table_relations(content.metadata.get('table_data', {}))
                relations.extend(table_relations)

            # 从公式中提取关系
            if content.content_type == ContentType.FORMULA:
                formula_relations = self._extract_formula_relations(content.metadata.get('formula', {}))
                relations.extend(formula_relations)

        return entities, relations

    def _extract_text_entities(self, text: str) -> List[str]:
        """从文本中提取实体"""
        entities = []

        # 提取数字和单位
        numbers = re.findall(r'\d+\.?\d*\s*(?:万元|亿元|元|%|倍|倍|万|亿)', text)
        entities.extend(numbers)

        # 提取公司名称
        company_pattern = r'[A-Za-z\u4e00-\u9fa5]+(?:公司|集团|企业|股份|有限|科技|工业|商业)'
        companies = re.findall(company_pattern, text)
        entities.extend(companies)

        # 提取时间
        time_pattern = r'\d{4}年|\d{1,2}月|\d{1,2}日|Q[1-4]|上半年|下半年'
        times = re.findall(time_pattern, text)
        entities.extend(times)

        return entities

    def _classify_entity_type(self, entity: str) -> str:
        """分类实体类型"""
        if re.match(r'\d+\.?\d*\s*(?:万元|亿元|元)', entity):
            return 'financial_amount'
        elif re.match(r'\d+\.?\d*%?', entity):
            return 'ratio'
        elif '公司' in entity or '集团' in entity:
            return 'company'
        elif re.match(r'\d{4}年|Q[1-4]', entity):
            return 'time_period'
        else:
            return 'unknown'

    def _extract_table_relations(self, table_data: Dict) -> List[Dict]:
        """从表格数据中提取关系"""
        relations = []
        if not table_data.get('data'):
            return relations

        headers = table_data.get('headers', [])
        rows = table_data.get('data', [])

        for row in rows:
            if len(row) > 1:
                relations.append({
                    'subject': row[0],
                    'predicate': headers[1] if len(headers) > 1 else 'has_value',
                    'object': row[1],
                    'source': 'table'
                })

        return relations

    def _extract_formula_relations(self, formula_data: Dict) -> List[Dict]:
        """从公式中提取关系"""
        relations = []
        formula = formula_data.get('expression', '')
        variables = formula_data.get('variables', [])

        if '=' in formula:
            parts = formula.split('=')
            if len(parts) == 2:
                relations.append({
                    'subject': parts[0].strip(),
                    'predicate': 'is_calculated_as',
                    'object': parts[1].strip(),
                    'source': 'formula'
                })

        return relations

    async def _process_cross_page_content(
        self,
        contents: List[MultimodalContent],
        **kwargs
    ) -> List[MultimodalContent]:
        """处理跨页内容"""
        # 简单的跨页内容检测逻辑
        processed_contents = []

        for i, content in enumerate(contents):
            # 检查是否有跨页标记
            cross_ref = self._detect_cross_page_reference(content.text, i, contents)
            if cross_ref:
                content.cross_reference = cross_ref

            processed_contents.append(content)

        return processed_contents

    def _detect_cross_page_reference(
        self,
        text: str,
        current_index: int,
        all_contents: List[MultimodalContent]
    ) -> Optional[str]:
        """检测跨页引用"""
        # 检查是否有"续表"、"续上页"等标记
        if re.search(r'续表|续上页|continued|see.*page', text, re.IGNORECASE):
            # 简单返回前一个内容的ID
            if current_index > 0:
                return f"cross_page_ref_{current_index - 1}"

        return None

    def _generate_statistics(self, contents: List[MultimodalContent]) -> Dict:
        """生成内容统计信息"""
        stats = {
            'total_contents': len(contents),
            'content_types': {},
            'confidence_avg': 0.0,
            'cross_page_count': 0
        }

        total_confidence = 0
        for content in contents:
            # 统计内容类型
            content_type = content.content_type.value
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1

            # 累计置信度
            total_confidence += content.confidence

            # 统计跨页内容
            if content.cross_reference:
                stats['cross_page_count'] += 1

        if contents:
            stats['confidence_avg'] = total_confidence / len(contents)

        return stats