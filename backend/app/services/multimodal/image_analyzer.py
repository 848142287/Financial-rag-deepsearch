"""
图片内容分析器
分析图片类型、提取图片中的文本、识别图表等
"""

import logging
import re
from typing import Dict, Any, List, Optional
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """图片内容分析器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_types = {
            'chart': ['柱状图', '折线图', '饼图', 'bar chart', 'line chart', 'pie chart'],
            'table': ['表格', '数据表', 'table'],
            'document': ['文档', '页面', '截图', 'screenshot'],
            'logo': ['logo', '标志', '图标'],
            'diagram': ['流程图', '架构图', '流程', 'flowchart', 'diagram'],
            'natural': ['风景', '照片', '图片', 'image', 'photo']
        }

    async def analyze(self, image_info: str) -> Dict[str, Any]:
        """分析图片内容"""
        try:
            # 提取图片描述文本
            description = self._extract_description(image_info)

            # 分析图片类型
            image_type = self._classify_image_type(description, image_info)

            # 提取关键信息
            keywords = self._extract_keywords(description)

            # 判断是否为财务相关图片
            is_financial = self._is_financial_image(description, image_type)

            return {
                'type': image_type,
                'description': description,
                'keywords': keywords,
                'is_financial': is_financial,
                'confidence': self._calculate_confidence(description, image_type),
                'extracted_data': self._extract_structured_data(description, image_type)
            }

        except Exception as e:
            logger.error(f"图片分析失败: {str(e)}")
            return {
                'type': 'unknown',
                'description': '',
                'keywords': [],
                'is_financial': False,
                'confidence': 0.0,
                'extracted_data': {}
            }

    def _extract_description(self, image_info: str) -> str:
        """提取图片描述文本"""
        # 从图片信息中提取描述性文本
        # 这里简化处理，实际应该使用OCR或图片描述模型

        # 查找常见的描述模式
        desc_patterns = [
            r'图\s*\d+[:：]\s*(.*?)(?:\n|$)',
            r'Figure\s*\d+[:：]\s*(.*?)(?:\n|$)',
            r'图表[:：]\s*(.*?)(?:\n|$)',
            r'截图[:：]\s*(.*?)(?:\n|$)',
            r'如上图所示[:：]\s*(.*?)(?:\n|$)',
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, image_info, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 如果没有找到明确的描述，返回整个文本
        return image_info.strip()

    def _classify_image_type(self, description: str, image_info: str) -> str:
        """分类图片类型"""
        text = (description + ' ' + image_info).lower()

        # 按优先级检查各种类型
        type_priorities = [
            'chart',
            'table',
            'document',
            'diagram',
            'logo',
            'natural'
        ]

        for img_type in type_priorities:
            keywords = self.image_types.get(img_type, [])
            if any(keyword in text for keyword in keywords):
                return img_type

        return 'unknown'

    def _extract_keywords(self, description: str) -> List[str]:
        """提取关键词"""
        keywords = []

        # 财务相关关键词
        financial_keywords = [
            '收入', '利润', '营收', '成本', '费用', '资产', '负债',
            'revenue', 'profit', 'cost', 'asset', 'liability',
            '增长率', '同比', '环比', '占比', '百分比',
            'growth rate', 'ratio', 'percentage'
        ]

        # 时间相关关键词
        time_keywords = [
            '2023', '2024', '2025', 'Q1', 'Q2', 'Q3', 'Q4',
            '一季度', '二季度', '三季度', '四季度',
            '上半年', '下半年', '年度', '月度'
        ]

        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', description)

        # 合并所有关键词
        all_keywords = financial_keywords + time_keywords + numbers

        for keyword in all_keywords:
            if keyword.lower() in description.lower():
                keywords.append(keyword)

        return list(set(keywords))

    def _is_financial_image(self, description: str, image_type: str) -> bool:
        """判断是否为财务相关图片"""
        financial_indicators = [
            '财务', '收入', '利润', '资产', '负债', '现金流',
            'financial', 'revenue', 'profit', 'asset', 'cash flow',
            '财报', '年报', '季报', '业绩', '估值'
        ]

        text = (description + ' ' + image_type).lower()
        return any(indicator in text for indicator in financial_indicators)

    def _calculate_confidence(self, description: str, image_type: str) -> float:
        """计算置信度"""
        confidence = 0.5

        # 根据描述长度调整
        if len(description) > 50:
            confidence += 0.2

        # 根据图片类型调整
        if image_type in ['chart', 'table']:
            confidence += 0.2
        elif image_type == 'unknown':
            confidence -= 0.2

        # 根据关键词数量调整
        keywords = self._extract_keywords(description)
        if len(keywords) > 3:
            confidence += 0.1

        return min(max(confidence, 0.0), 1.0)

    def _extract_structured_data(self, description: str, image_type: str) -> Dict[str, Any]:
        """提取结构化数据"""
        data = {}

        if image_type == 'chart':
            data = self._extract_chart_data(description)
        elif image_type == 'table':
            data = self._extract_table_data(description)
        elif image_type == 'financial':
            data = self._extract_financial_data(description)

        return data

    def _extract_chart_data(self, description: str) -> Dict[str, Any]:
        """提取图表数据"""
        data = {}

        # 识别图表类型
        chart_types = {
            'bar': ['柱状图', '条形图', 'bar chart'],
            'line': ['折线图', '曲线图', 'line chart'],
            'pie': ['饼图', '扇形图', 'pie chart'],
            'scatter': ['散点图', 'scatter plot']
        }

        for chart_type, keywords in chart_types.items():
            if any(keyword in description.lower() for keyword in keywords):
                data['chart_type'] = chart_type
                break

        # 提取数据点
        numbers = re.findall(r'(\d+\.?\d*)', description)
        if len(numbers) > 1:
            data['data_points'] = [float(n) for n in numbers[:10]]  # 限制前10个数据点

        # 提取标签
        labels = re.findall(r'([A-Za-z\u4e00-\u9fa5]+)[：:]\s*\d+\.?\d*', description)
        if labels:
            data['labels'] = [label.split('：')[0].split(':')[0] for label in labels[:10]]

        return data

    def _extract_table_data(self, description: str) -> Dict[str, Any]:
        """提取表格数据"""
        data = {}

        # 识别表头
        header_pattern = r'([^|\n\t]+)[|\t](?:[^|\n\t]+[|\t])*'
        headers = re.findall(header_pattern, description.split('\n')[0] if '\n' in description else description)
        if headers:
            data['headers'] = headers[0].split('|') if '|' in headers[0] else headers[0].split('\t')

        # 识别数据行
        rows = description.split('\n')
        data_rows = []
        for row in rows[1:]:  # 跳过表头
            if '|' in row or '\t' in row:
                data_rows.append(row.split('|') if '|' in row else row.split('\t'))

        if data_rows:
            data['rows'] = data_rows[:10]  # 限制前10行

        return data

    def _extract_financial_data(self, description: str) -> Dict[str, Any]:
        """提取财务数据"""
        data = {}

        # 提取金额
        amount_pattern = r'(\d+\.?\d*)\s*(万元|亿元|元)'
        amounts = re.findall(amount_pattern, description)
        if amounts:
            data['amounts'] = [{'value': float(a[0]), 'unit': a[1]} for a in amounts]

        # 提取百分比
        percentage_pattern = r'(\d+\.?\d*)%'
        percentages = re.findall(percentage_pattern, description)
        if percentages:
            data['percentages'] = [float(p) for p in percentages]

        # 提取增长率
        growth_pattern = r'(?:增长|增长率为|涨幅|涨幅为)\s*(\d+\.?\d*)%'
        growth_rates = re.findall(growth_pattern, description)
        if growth_rates:
            data['growth_rates'] = [float(g) for g in growth_rates]

        return data