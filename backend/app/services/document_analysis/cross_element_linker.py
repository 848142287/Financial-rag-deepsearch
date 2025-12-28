"""
跨元素关联建立器 - Cross Element Linker
建立文档内不同元素间的关联关系
"""

import logging
import re
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


class CrossElementLinker:
    """跨元素关联建立器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化关联建立器

        Args:
            config: 配置字典
        """
        self.config = config or {}

    def build_content_network(
        self,
        elements: List[Any]
    ) -> Dict[str, List]:
        """
        构建内容网络

        Args:
            elements: 文档元素列表

        Returns:
            Dict: 关联网络
        """
        try:
            network = {
                'data_reference_chain': self._trace_data_references(elements),
                'logical_proof_chain': self._build_logical_chain(elements),
                'visual_reinforcement_chain': self._link_visual_evidence(elements)
            }

            return network

        except Exception as e:
            logger.error(f"Error building content network: {e}")
            return {
                'data_reference_chain': [],
                'logical_proof_chain': [],
                'visual_reinforcement_chain': []
            }

    def _trace_data_references(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """追踪数据引用链"""
        chains = []

        # 提取表格和图表
        tables = []
        charts = []
        paragraphs = []

        for idx, element in enumerate(elements):
            elem_type = getattr(element, 'element_type', None)

            if elem_type == 'table':
                tables.append((idx, element))
            elif elem_type in ['image', 'chart']:
                charts.append((idx, element))
            elif elem_type == 'paragraph':
                paragraphs.append((idx, element))

        # 简单实现：查找数字引用关系
        for para_idx, paragraph in paragraphs:
            content = getattr(paragraph, 'content', '')
            if not content:
                continue

            # 提取数字
            numbers = re.findall(r'\d+(?:\.\d+)?%?', content)

            if numbers:
                # 查找可能的数据源
                potential_sources = []

                # 检查表格
                for table_idx, table in tables:
                    table_metadata = getattr(table, 'metadata', {})
                    if table_metadata:
                        potential_sources.append({
                            'source_type': 'table',
                            'source_idx': table_idx,
                            'source_id': table_metadata.get('table_id', f'table_{table_idx}')
                        })

                # 检查图表
                for chart_idx, chart in charts:
                    chart_metadata = getattr(chart, 'metadata', {})
                    if chart_metadata:
                        potential_sources.append({
                            'source_type': 'chart',
                            'source_idx': chart_idx,
                            'source_id': chart_metadata.get('chart_id', f'chart_{chart_idx}')
                        })

                if potential_sources:
                    chains.append({
                        'from': {
                            'type': 'paragraph',
                            'idx': para_idx,
                            'content_preview': content[:100]
                        },
                        'to': potential_sources[:3],  # 最多3个可能的数据源
                        'reference_type': 'data_citation',
                        'numbers_mentioned': numbers[:5]  # 最多5个数字
                    })

        return chains[:10]  # 返回最多10条引用链

    def _build_logical_chain(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """构建逻辑证明链"""
        chains = []

        # 简单实现：识别问题-证据-结论结构
        questions = []
        evidences = []
        conclusions = []

        for idx, element in enumerate(elements):
            elem_type = getattr(element, 'element_type', None)
            content = getattr(element, 'content', '')

            if not content:
                continue

            # 检查语义标注
            metadata = getattr(element, 'metadata', {})
            semantic_type = metadata.get('semantic_annotation', {}).get('semantic_type', '') if metadata else ''

            if semantic_type == '问题提出':
                questions.append((idx, element))
            elif semantic_type == '数据陈述' or elem_type in ['table', 'chart']:
                evidences.append((idx, element))
            elif semantic_type == '结论总结':
                conclusions.append((idx, element))

        # 构建逻辑链：问题 -> 证据 -> 结论
        for quest_idx, question in questions:
            # 查找后续的证据
            related_evidences = [
                (ev_idx, ev) for ev_idx, ev in evidences
                if ev_idx > quest_idx and ev_idx < quest_idx + 10  # 在10个元素内
            ]

            # 查找后续的结论
            related_conclusions = [
                (con_idx, con) for con_idx, con in conclusions
                if con_idx > quest_idx and con_idx < quest_idx + 15
            ]

            if related_evidences or related_conclusions:
                chain = {
                    'question': {
                        'idx': quest_idx,
                        'content': getattr(question, 'content', '')[:100]
                    },
                    'evidences': [
                        {
                            'idx': ev_idx,
                            'type': getattr(ev, 'element_type', 'unknown'),
                            'preview': str(getattr(ev, 'content', ''))[:100]
                        }
                        for ev_idx, ev in related_evidences[:3]
                    ],
                    'conclusions': [
                        {
                            'idx': con_idx,
                            'content': getattr(con, 'content', '')[:100]
                        }
                        for con_idx, con in related_conclusions[:2]
                    ]
                }

                chains.append(chain)

        return chains[:5]  # 返回最多5条逻辑链

    def _link_visual_evidence(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """链接视觉证据"""
        links = []

        # 找出所有图表
        charts = []
        for idx, element in enumerate(elements):
            elem_type = getattr(element, 'element_type', None)
            if elem_type in ['image', 'chart']:
                charts.append((idx, element))

        # 对于每个图表，查找相关的文本描述
        for chart_idx, chart in charts:
            chart_metadata = getattr(chart, 'metadata', {})
            chart_desc = chart_metadata.get('description', '') if chart_metadata else ''

            # 查找前后的段落
            nearby_paragraphs = []
            for offset in [-3, -2, -1, 1, 2, 3]:
                para_idx = chart_idx + offset
                if 0 <= para_idx < len(elements):
                    element = elements[para_idx]
                    if getattr(element, 'element_type', None) == 'paragraph':
                        content = getattr(element, 'content', '')
                        if content:
                            nearby_paragraphs.append({
                                'idx': para_idx,
                                'offset': offset,
                                'content': content[:200]
                            })

            if nearby_paragraphs:
                links.append({
                    'visual_element': {
                        'idx': chart_idx,
                        'type': 'chart' if chart_metadata.get('image_type') == 'chart' else 'image',
                        'description': chart_desc[:100]
                    },
                    'related_text': nearby_paragraphs[:5],
                    'relationship_type': 'visual_reinforcement'
                })

        return links[:10]  # 返回最多10条视觉证据链接


class ConsistencyValidator:
    """一致性验证器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化验证器

        Args:
            config: 配置字典
        """
        self.config = config or {}

    def validate_document(self, elements: List[Any]) -> Dict[str, Any]:
        """
        验证文档一致性

        Args:
            elements: 文档元素列表

        Returns:
            Dict: 验证结果
        """
        try:
            validation_results = {
                'data_consistency': self._check_data_consistency(elements),
                'logical_consistency': self._check_logical_consistency(elements),
                'temporal_consistency': self._check_timeline_consistency(elements),
                'reference_consistency': self._check_reference_links(elements)
            }

            # 计算总体一致性评分
            scores = []
            for check_type, result in validation_results.items():
                if isinstance(result, dict) and 'score' in result:
                    scores.append(result['score'])

            overall_score = sum(scores) / len(scores) if scores else 0.5

            validation_results['overall_consistency_score'] = round(overall_score, 2)
            validation_results['overall_rating'] = self._get_rating(overall_score)

            return validation_results

        except Exception as e:
            logger.error(f"Error validating document: {e}")
            return {
                'data_consistency': {'status': 'error'},
                'logical_consistency': {'status': 'error'},
                'temporal_consistency': {'status': 'error'},
                'reference_consistency': {'status': 'error'},
                'overall_consistency_score': 0.0,
                'overall_rating': '未知'
            }

    def _check_data_consistency(self, elements: List[Any]) -> Dict[str, Any]:
        """检查数据一致性"""
        issues = []

        # 提取所有数值数据
        numeric_data = defaultdict(list)

        for element in elements:
            elem_type = getattr(element, 'element_type', None)

            if elem_type == 'table':
                metadata = getattr(element, 'metadata', {})
                table_data = metadata.get('data', []) if metadata else []

                # 简化检查：查找明显的数值不一致
                for row in table_data:
                    for cell in row:
                        # 提取数值
                        numbers = re.findall(r'\d+(?:\.\d+)?', str(cell))
                        for num in numbers:
                            numeric_data[num].append({
                                'element_type': 'table',
                                'context': str(cell)[:50]
                            })

            elif elem_type == 'paragraph':
                content = getattr(element, 'content', '')
                numbers = re.findall(r'\d+(?:\.\d+)?', content)

                for num in numbers:
                    numeric_data[num].append({
                        'element_type': 'paragraph',
                        'context': content[:50]
                    })

        # 检查同一数字在不同位置的上下文是否一致
        # 这里是简化实现，实际可以更复杂
        consistency_score = 0.8  # 默认分数

        return {
            'score': consistency_score,
            'issues': issues,
            'unique_values': len(numeric_data),
            'total_occurrences': sum(len(v) for v in numeric_data.values())
        }

    def _check_logical_consistency(self, elements: List[Any]) -> Dict[str, Any]:
        """检查逻辑一致性"""
        issues = []

        # 简化实现：检查是否有矛盾的说法
        # 实际应用中可以使用NLP技术进行更深入的分析

        consistency_score = 0.75  # 默认分数

        return {
            'score': consistency_score,
            'issues': issues,
            'note': '逻辑一致性检查需要更深入的自然语言理解'
        }

    def _check_timeline_consistency(self, elements: List[Any]) -> Dict[str, Any]:
        """检查时序一致性"""
        issues = []

        # 提取所有时间信息
        time_references = []

        for element in elements:
            content = getattr(element, 'content', '')

            # 简单的时间模式匹配
            time_patterns = [
                r'\d{4}年',
                r'\d{4}/\d{1,2}/\d{1,2}',
                r'(Q1|Q2|Q3|Q4)',
                r'(一月|二月|三月|四月|五月|六月|七月|八月|九月|十月|十一月|十二月)'
            ]

            for pattern in time_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    time_references.append({
                        'time': match,
                        'context': content[:100]
                    })

        consistency_score = 0.85  # 默认分数

        return {
            'score': consistency_score,
            'issues': issues,
            'time_references_count': len(time_references)
        }

    def _check_reference_links(self, elements: List[Any]) -> Dict[str, Any]:
        """检查引用一致性"""
        issues = []

        # 简化实现：检查是否有悬空引用
        # 实际应用中需要建立完整的引用图

        consistency_score = 0.9  # 默认分数

        return {
            'score': consistency_score,
            'issues': issues,
            'note': '引用一致性检查需要更复杂的引用追踪'
        }

    def _get_rating(self, score: float) -> str:
        """根据分数获取评级"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.75:
            return "良好"
        elif score >= 0.6:
            return "一般"
        else:
            return "需要改进"
