"""
质量检测器
检测内容质量和完整性
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class QualityIssue:
    """质量问题"""
    issue_type: str
    severity: str  # low, medium, high, critical
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class QualityReport:
    """质量报告"""
    overall_score: float
    quality_level: QualityLevel
    issues: List[QualityIssue]
    metrics: Dict[str, Any]
    recommendations: List[str]


class QualityDetector:
    """质量检测器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.7,
            QualityLevel.FAIR: 0.5,
            QualityLevel.POOR: 0.0
        }

        self.quality_checks = {
            'completeness': self._check_completeness,
            'accuracy': self._check_accuracy,
            'consistency': self._check_consistency,
            'clarity': self._check_clarity,
            'structure': self._check_structure,
            'relevance': self._check_relevance
        }

    async def detect_quality(self, content: str, content_type: str = 'text', context: Optional[Dict] = None) -> QualityReport:
        """检测内容质量"""
        try:
            issues = []
            metrics = {}
            recommendations = []

            # 执行各项质量检查
            for check_name, check_func in self.quality_checks.items():
                check_issues, check_metrics = check_func(content, content_type, context)
                issues.extend(check_issues)
                metrics[check_name] = check_metrics

            # 计算综合分数
            overall_score = self._calculate_overall_score(metrics)

            # 确定质量等级
            quality_level = self._determine_quality_level(overall_score)

            # 生成建议
            recommendations = self._generate_recommendations(issues, metrics)

            return QualityReport(
                overall_score=overall_score,
                quality_level=quality_level,
                issues=issues,
                metrics=metrics,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"质量检测失败: {str(e)}")
            return QualityReport(
                overall_score=0.0,
                quality_level=QualityLevel.POOR,
                issues=[QualityIssue(
                    issue_type='error',
                    severity='critical',
                    description=f'质量检测失败: {str(e)}'
                )],
                metrics={},
                recommendations=[]
            )

    def _check_completeness(self, content: str, content_type: str, context: Optional[Dict]) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """检查完整性"""
        issues = []
        metrics = {}

        # 检查内容长度
        if len(content) < 10:
            issues.append(QualityIssue(
                issue_type='content_too_short',
                severity='high',
                description='内容过短，可能缺少重要信息',
                suggestion='添加更多详细内容'
            ))

        # 检查句子完整性
        sentences = re.split(r'[。！？.!?]', content)
        incomplete_sentences = sum(1 for s in sentences if len(s.strip()) > 0 and len(s.strip()) < 5)

        if incomplete_sentences > len(sentences) * 0.3:
            issues.append(QualityIssue(
                issue_type='incomplete_sentences',
                severity='medium',
                description=f'检测到{incomplete_sentences}个不完整句子',
                suggestion='检查并补充完整的句子'
            ))

        # 检查表格完整性
        if content_type == 'table':
            table_completeness = self._check_table_completeness(content)
            metrics['table_completeness'] = table_completeness

            if table_completeness < 0.8:
                issues.append(QualityIssue(
                    issue_type='incomplete_table',
                    severity='high',
                    description='表格数据不完整',
                    suggestion='补充缺失的表格数据'
                ))

        # 检查公式完整性
        if content_type == 'formula':
            formula_completeness = self._check_formula_completeness(content)
            metrics['formula_completeness'] = formula_completeness

            if formula_completeness < 0.9:
                issues.append(QualityIssue(
                    issue_type='incomplete_formula',
                    severity='high',
                    description='公式不完整',
                    suggestion='检查公式语法'
                ))

        metrics['content_length'] = len(content)
        metrics['sentence_count'] = len([s for s in sentences if s.strip()])
        metrics['incomplete_sentences'] = incomplete_sentences

        return issues, metrics

    def _check_accuracy(self, content: str, content_type: str, context: Optional[Dict]) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """检查准确性"""
        issues = []
        metrics = {}

        # 检查数字一致性
        numbers = re.findall(r'(\d+\.?\d*)', content)
        if numbers:
            # 检查是否有明显异常的数字
            anomalies = self._detect_number_anomalies([float(n) for n in numbers])
            if anomalies:
                issues.append(QualityIssue(
                    issue_type='number_anomaly',
                    severity='medium',
                    description=f'检测到异常数字: {anomalies}',
                    suggestion='核实数据的准确性'
                ))

        # 检查财务指标合理性
        financial_issues = self._check_financial_accuracy(content)
        issues.extend(financial_issues)

        # 检查时间逻辑
        time_issues = self._check_temporal_accuracy(content)
        issues.extend(time_issues)

        metrics['number_count'] = len(numbers)
        metrics['anomaly_count'] = len(anomalies) if anomalies else 0

        return issues, metrics

    def _check_consistency(self, content: str, content_type: str, context: Optional[Dict]) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """检查一致性"""
        issues = []
        metrics = {}

        # 检查术语一致性
        term_inconsistencies = self._check_term_consistency(content)
        issues.extend(term_inconsistencies)

        # 检查格式一致性
        format_inconsistencies = self._check_format_consistency(content, content_type)
        issues.extend(format_inconsistencies)

        # 检查数据格式一致性
        data_format_issues = self._check_data_format_consistency(content)
        issues.extend(data_format_issues)

        metrics['term_consistency_score'] = 1.0 - len(term_inconsistencies) / 10  # 假设最多10个术语
        metrics['format_consistency_score'] = 1.0 - len(format_inconsistencies) / 5

        return issues, metrics

    def _check_clarity(self, content: str, content_type: str, context: Optional[Dict]) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """检查清晰度"""
        issues = []
        metrics = {}

        # 检查句子长度
        sentences = re.split(r'[。！？.!?]', content)
        long_sentences = [s for s in sentences if len(s.strip()) > 100]

        if long_sentences:
            issues.append(QualityIssue(
                issue_type='long_sentences',
                severity='low',
                description=f'检测到{len(long_sentences)}个长句子，影响可读性',
                suggestion='将长句子分解为多个短句'
            ))

        # 检查专业术语使用
        jargon_density = self._calculate_jargon_density(content)
        if jargon_density > 0.3:
            issues.append(QualityIssue(
                issue_type='high_jargon_density',
                severity='medium',
                description='专业术语密度过高，可能影响理解',
                suggestion='添加术语解释或使用更通俗的表达'
            ))

        # 检查模糊表达
        ambiguous_phrases = self._detect_ambiguous_phrases(content)
        if ambiguous_phrases:
            issues.append(QualityIssue(
                issue_type='ambiguous_expressions',
                severity='low',
                description=f'检测到模糊表达: {ambiguous_phrases[:3]}',
                suggestion='使用更精确的表达方式'
            ))

        metrics['avg_sentence_length'] = sum(len(s) for s in sentences) / max(len(sentences), 1)
        metrics['jargon_density'] = jargon_density
        metrics['ambiguous_phrase_count'] = len(ambiguous_phrases)

        return issues, metrics

    def _check_structure(self, content: str, content_type: str, context: Optional[Dict]) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """检查结构"""
        issues = []
        metrics = {}

        if content_type == 'text':
            # 检查段落结构
            paragraphs = content.split('\n\n')
            if len(paragraphs) == 1 and len(content) > 200:
                issues.append(QualityIssue(
                    issue_type='no_paragraphs',
                    severity='medium',
                    description='长文本缺少段落结构',
                    suggestion='添加适当的段落分隔'
                ))

            # 检查逻辑结构
            structure_score = self._evaluate_logical_structure(content)
            metrics['logical_structure_score'] = structure_score

        elif content_type == 'table':
            # 检查表格结构
            structure_score = self._evaluate_table_structure(content)
            metrics['table_structure_score'] = structure_score

        return issues, metrics

    def _check_relevance(self, content: str, content_type: str, context: Optional[Dict]) -> Tuple[List[QualityIssue], Dict[str, Any]]:
        """检查相关性"""
        issues = []
        metrics = {}

        # 检查内容相关性（需要上下文）
        if context and 'domain' in context:
            domain = context['domain']
            relevance_score = self._calculate_domain_relevance(content, domain)
            metrics['domain_relevance'] = relevance_score

            if relevance_score < 0.5:
                issues.append(QualityIssue(
                    issue_type='low_relevance',
                    severity='medium',
                    description=f'内容与{domain}领域相关性较低',
                    suggestion='添加更多领域相关内容'
                ))

        return issues, metrics

    def _check_table_completeness(self, table_content: str) -> float:
        """检查表格完整性"""
        # 计算非空单元格比例
        cells = re.findall(r'\|([^|]*)\|', table_content)
        if not cells:
            return 0.0

        non_empty_cells = sum(1 for cell in cells if cell.strip())
        return non_empty_cells / len(cells)

    def _check_formula_completeness(self, formula_content: str) -> float:
        """检查公式完整性"""
        # 检查括号匹配
        open_brackets = formula_content.count('(') + formula_content.count('{')
        close_brackets = formula_content.count(')') + formula_content.count('}')

        if open_brackets == close_brackets:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(open_brackets - close_brackets) / max(open_brackets, 1))

    def _detect_number_anomalies(self, numbers: List[float]) -> List[float]:
        """检测异常数字"""
        anomalies = []
        if len(numbers) < 3:
            return anomalies

        # 简单的异常检测：偏离均值3个标准差
        mean = sum(numbers) / len(numbers)
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        std_dev = variance ** 0.5

        for num in numbers:
            if abs(num - mean) > 3 * std_dev:
                anomalies.append(num)

        return anomalies

    def _check_financial_accuracy(self, content: str) -> List[QualityIssue]:
        """检查财务数据准确性"""
        issues = []

        # 检查百分比合理性
        percentages = re.findall(r'(\d+\.?\d*)%', content)
        for pct_str in percentages:
            pct = float(pct_str)
            if pct > 1000:  # 大多数情况下百分比不应超过1000%
                issues.append(QualityIssue(
                    issue_type='unrealistic_percentage',
                    severity='high',
                    description=f'异常百分比: {pct}%',
                    suggestion='核实百分比数据的准确性'
                ))

        # 检查增长率合理性
        growth_patterns = [
            r'增长(\d+\.?\d*)%',
            r'增幅(\d+\.?\d*)%',
            r'上涨(\d+\.?\d*)%'
        ]

        for pattern in growth_patterns:
            matches = re.findall(pattern, content)
            for growth_str in matches:
                growth = float(growth_str)
                if abs(growth) > 1000:  # 增长率通常不会超过1000%
                    issues.append(QualityIssue(
                        issue_type='unrealistic_growth',
                        severity='medium',
                        description=f'异常增长率: {growth}%',
                        suggestion='核实增长率数据'
                    ))

        return issues

    def _check_temporal_accuracy(self, content: str) -> List[QualityIssue]:
        """检查时间逻辑准确性"""
        issues = []

        # 提取时间点
        time_points = re.findall(r'(\d{4})年', content)
        if len(time_points) > 1:
            years = sorted(int(y) for y in time_points)
            # 检查是否有未来时间（假设当前是2024年）
            current_year = 2024
            future_years = [y for y in years if y > current_year]

            if future_years:
                issues.append(QualityIssue(
                    issue_type='future_reference',
                    severity='medium',
                    description=f'提及未来时间: {future_years}',
                    suggestion='确认是否为预测数据'
                ))

        return issues

    def _check_term_consistency(self, content: str) -> List[QualityIssue]:
        """检查术语一致性"""
        issues = []

        # 常见的不一致术语
        inconsistent_terms = {
            'ROE': ['净资产收益率', '股东权益回报率'],
            'ROA': ['资产收益率', '总资产回报率'],
            'PE': ['市盈率', '本益比']
        }

        for term, alternatives in inconsistent_terms.items():
            # 检查是否混用了多个表达
            used_count = sum(1 for alt in [term] + alternatives if alt in content)
            if used_count > 1:
                issues.append(QualityIssue(
                    issue_type='term_inconsistency',
                    severity='low',
                    description=f'术语不一致: {term} 及其变体同时出现',
                    suggestion=f'统一使用 {term}'
                ))

        return issues

    def _check_format_consistency(self, content: str, content_type: str) -> List[QualityIssue]:
        """检查格式一致性"""
        issues = []

        # 检查数字格式
        number_formats = []
        for match in re.finditer(r'\d+\.?\d*', content):
            num = match.group()
            if '.' in num:
                number_formats.append(len(num.split('.')[1]))

        if number_formats and len(set(number_formats)) > 2:
            issues.append(QualityIssue(
                issue_type='inconsistent_number_format',
                severity='low',
                description='小数位数不一致',
                suggestion='统一数字格式'
            ))

        return issues

    def _check_data_format_consistency(self, content: str) -> List[QualityIssue]:
        """检查数据格式一致性"""
        issues = []

        # 检查日期格式
        date_formats = []
        date_patterns = [
            (r'\d{4}年\d{1,2}月\d{1,2}日', 'YYYY年MM月DD日'),
            (r'\d{4}-\d{1,2}-\d{1,2}', 'YYYY-MM-DD'),
            (r'\d{4}/\d{1,2}/\d{1,2}', 'YYYY/MM/DD')
        ]

        for pattern, format_name in date_patterns:
            if re.search(pattern, content):
                date_formats.append(format_name)

        if len(date_formats) > 2:
            issues.append(QualityIssue(
                issue_type='inconsistent_date_format',
                severity='low',
                description='日期格式不统一',
                suggestion='统一使用一种日期格式'
            ))

        return issues

    def _calculate_jargon_density(self, content: str) -> float:
        """计算专业术语密度"""
        # 财务专业术语列表
        jargon_terms = [
            '资产', '负债', '权益', '收入', '成本', '利润', '现金流',
            'ROE', 'ROA', 'PE', 'PB', 'EBITDA', '毛利率', '净利率',
            '资产负债率', '流动比率', '速动比率'
        ]

        words = content.split()
        jargon_count = sum(1 for word in words if any(term in word for term in jargon_terms))

        return jargon_count / max(len(words), 1)

    def _detect_ambiguous_phrases(self, content: str) -> List[str]:
        """检测模糊表达"""
        ambiguous_patterns = [
            r'大概', '可能', '也许', '左右', '差不多',
            'about', 'approximately', 'around', 'roughly'
        ]

        ambiguous_phrases = []
        for pattern in ambiguous_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            ambiguous_phrases.extend(matches)

        return ambiguous_phrases

    def _evaluate_logical_structure(self, content: str) -> float:
        """评估逻辑结构"""
        score = 0.5  # 基础分数

        # 检查是否有标题
        if re.search(r'^#+\s', content, re.MULTILINE):
            score += 0.2

        # 检查是否有列表
        if re.search(r'^\s*[-*+]\s', content, re.MULTILINE):
            score += 0.1

        # 检查是否有结论性语句
        if any(keyword in content for keyword in ['总之', '综上所述', 'conclusion', 'summary']):
            score += 0.2

        return min(score, 1.0)

    def _evaluate_table_structure(self, table_content: str) -> float:
        """评估表格结构"""
        # 检查是否有表头
        has_header = bool(re.search(r'\|.*\|', table_content))
        # 检查是否有分隔线
        has_separator = bool(re.search(r'\|[-\s\|]+\|', table_content))

        score = 0.5
        if has_header:
            score += 0.3
        if has_separator:
            score += 0.2

        return min(score, 1.0)

    def _calculate_domain_relevance(self, content: str, domain: str) -> float:
        """计算领域相关性"""
        domain_keywords = {
            'financial': [
                '财务', '会计', '报表', '资产', '负债', '利润', '收入',
                'financial', 'accounting', 'report', 'asset', 'liability'
            ],
            'technical': [
                '技术', '算法', '系统', '架构', '开发', '代码',
                'technology', 'algorithm', 'system', 'architecture', 'development'
            ]
        }

        keywords = domain_keywords.get(domain.lower(), [])
        if not keywords:
            return 0.5  # 未知领域返回中等相关性

        word_count = len(content.split())
        keyword_count = sum(1 for keyword in keywords if keyword in content.lower())

        return min(keyword_count / max(word_count / 100, 1), 1.0)

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """计算综合分数"""
        # 各项权重
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.15,
            'clarity': 0.15,
            'structure': 0.1,
            'relevance': 0.1
        }

        total_score = 0
        for check_name, weight in weights.items():
            check_metrics = metrics.get(check_name, {})

            # 提取分数（简化处理）
            if check_name == 'completeness':
                score = min(1.0, check_metrics.get('content_length', 0) / 100)
            elif check_name == 'accuracy':
                score = 1.0 - min(check_metrics.get('anomaly_count', 0) / 10, 1.0)
            elif check_name in ['consistency', 'clarity']:
                score = check_metrics.get(f'{check_name}_consistency_score', 0.8)
            else:
                score = 0.8  # 默认分数

            total_score += score * weight

        return total_score

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """确定质量等级"""
        for level, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return QualityLevel.POOR

    def _generate_recommendations(self, issues: List[QualityIssue], metrics: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 根据问题类型生成建议
        issue_types = [issue.issue_type for issue in issues]

        if 'content_too_short' in issue_types:
            recommendations.append('增加内容的详细程度，确保信息完整')

        if 'incomplete_table' in issue_types:
            recommendations.append('补充表格中缺失的数据')

        if 'number_anomaly' in issue_types:
            recommendations.append('核实并修正异常的数值数据')

        if 'term_inconsistency' in issue_types:
            recommendations.append('统一专业术语的使用')

        if 'long_sentences' in issue_types:
            recommendations.append('简化长句，提高可读性')

        # 根据指标生成建议
        if metrics.get('completeness', {}).get('completeness_score', 1.0) < 0.7:
            recommendations.append('提高内容完整性，避免信息缺失')

        if metrics.get('accuracy', {}).get('accuracy_score', 1.0) < 0.7:
            recommendations.append('仔细核对数据准确性')

        # 如果没有明显问题，给出通用建议
        if not recommendations:
            recommendations.append('内容质量良好，继续保持')

        return recommendations[:5]  # 最多返回5个建议