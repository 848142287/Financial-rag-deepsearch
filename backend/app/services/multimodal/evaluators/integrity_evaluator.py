"""
语义完整性评估器
评估解析结果的语义完整性
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from collections import defaultdict

from ..core.multimodal_parser import ParsedDocument, ContentBlock, ContentType, Chapter

logger = logging.getLogger(__name__)


class IntegrityIssue(Enum):
    """完整性问题类型"""
    MISSING_CONTENT = "missing_content"
    BROKEN_STRUCTURE = "broken_structure"
    LOW_QUALITY = "low_quality"
    MISSING_REFERENCES = "missing_references"
    INCONSISTENT_FORMAT = "inconsistent_format"
    ORPHANED_CONTENT = "orphaned_content"
    DUPLICATE_CONTENT = "duplicate_content"


@dataclass
class IntegrityIssueDetail:
    """完整性问题详情"""
    issue_type: IntegrityIssue
    severity: float  # 0.0 - 1.0
    description: str
    location: str  # 位置描述
    suggestions: List[str] = field(default_factory=list)
    affected_blocks: List[str] = field(default_factory=list)


@dataclass
class IntegrityReport:
    """完整性评估报告"""
    overall_score: float  # 0.0 - 1.0
    issues: List[IntegrityIssueDetail]
    completeness_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    structure_metrics: Dict[str, float]
    recommendations: List[str]


class IntegrityEvaluator:
    """语义完整性评估器"""

    def __init__(self):
        """初始化完整性评估器"""
        # 评估权重
        self.evaluation_weights = {
            "content_completeness": 0.3,
            "structure_integrity": 0.25,
            "content_quality": 0.25,
            "semantic_coherence": 0.2
        }

        # 完整性阈值
        self.thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "fair": 0.5,
            "poor": 0.3
        }

        # 内容类型预期比例
        self.expected_content_ratios = {
            ContentType.TEXT: 0.5,      # 50% 文本
            ContentType.IMAGE: 0.15,    # 15% 图片
            ContentType.TABLE: 0.15,    # 15% 表格
            ContentType.FORMULA: 0.05,  # 5% 公式
            ContentType.HEADER: 0.05,   # 5% 标题
            ContentType.FOOTER: 0.05,   # 5% 页脚
        }

        # 语义连贯性模式
        self.coherence_patterns = {
            "chapter_continuity": [
                r"本章小结", r"本章总结", r"下章", r"前文",
                r"previous chapter", r"next chapter", r"summary"
            ],
            "reference_patterns": [
                r"如表\d+", r"如图\d+", r"公式\d+", r"Eq\.",
                r"see Table", r"see Figure", r"see Equation"
            ],
            "flow_indicators": [
                r"首先", r"其次", r"然后", r"最后", r"综上所述",
                r"First", r"Second", r"Third", r"Finally", r"In conclusion"
            ]
        }

        logger.info("完整性评估器初始化完成")

    async def evaluate_integrity(self, document: ParsedDocument) -> float:
        """
        评估文档的语义完整性

        Args:
            document: 解析后的文档

        Returns:
            完整性评分 (0.0 - 1.0)
        """
        try:
            logger.info(f"开始评估文档完整性: {document.document_id}")

            # 生成完整性报告
            report = await self.generate_integrity_report(document)

            # 记录主要问题
            if report.issues:
                critical_issues = [issue for issue in report.issues if issue.severity > 0.7]
                if critical_issues:
                    logger.warning(f"发现{len(critical_issues)}个严重完整性问题")

            logger.info(f"完整性评估完成，评分: {report.overall_score:.3f}")
            return report.overall_score

        except Exception as e:
            logger.error(f"完整性评估失败: {str(e)}")
            return 0.0

    async def generate_integrity_report(self, document: ParsedDocument) -> IntegrityReport:
        """生成详细的完整性评估报告"""
        try:
            issues = []

            # 评估内容完整性
            completeness_score, completeness_issues = await self._evaluate_content_completeness(document)
            issues.extend(completeness_issues)

            # 评估结构完整性
            structure_score, structure_issues = await self._evaluate_structure_integrity(document)
            issues.extend(structure_issues)

            # 评估内容质量
            quality_score, quality_issues = await self._evaluate_content_quality(document)
            issues.extend(quality_issues)

            # 评估语义连贯性
            coherence_score, coherence_issues = await self._evaluate_semantic_coherence(document)
            issues.extend(coherence_issues)

            # 计算综合评分
            overall_score = (
                completeness_score * self.evaluation_weights["content_completeness"] +
                structure_score * self.evaluation_weights["structure_integrity"] +
                quality_score * self.evaluation_weights["content_quality"] +
                coherence_score * self.evaluation_weights["semantic_coherence"]
            )

            # 生成详细指标
            completeness_metrics = {
                "content_density": self._calculate_content_density(document),
                "type_distribution": self._evaluate_type_distribution(document),
                "coverage_ratio": self._calculate_coverage_ratio(document)
            }

            quality_metrics = {
                "avg_confidence": self._calculate_avg_confidence(document),
                "high_quality_ratio": self._calculate_high_quality_ratio(document),
                "low_confidence_count": len([b for b in document.content_blocks if b.confidence < 0.5])
            }

            structure_metrics = {
                "chapter_hierarchy_score": self._evaluate_chapter_hierarchy(document.chapters),
                "content_chapter_alignment": self._evaluate_content_chapter_alignment(document),
                "structural_completeness": self._evaluate_structural_completeness(document)
            }

            # 生成建议
            recommendations = self._generate_recommendations(issues, overall_score)

            report = IntegrityReport(
                overall_score=overall_score,
                issues=sorted(issues, key=lambda x: x.severity, reverse=True),
                completeness_metrics=completeness_metrics,
                quality_metrics=quality_metrics,
                structure_metrics=structure_metrics,
                recommendations=recommendations
            )

            return report

        except Exception as e:
            logger.error(f"生成完整性报告失败: {str(e)}")
            return IntegrityReport(
                overall_score=0.0,
                issues=[IntegrityIssueDetail(
                    IntegrityIssue.MISSING_CONTENT,
                    1.0,
                    f"评估过程出错: {str(e)}",
                    "整体文档",
                    ["请检查解析流程"]
                )],
                completeness_metrics={},
                quality_metrics={},
                structure_metrics={},
                recommendations=["请重新运行解析流程"]
            )

    async def _evaluate_content_completeness(self, document: ParsedDocument) -> Tuple[float, List[IntegrityIssueDetail]]:
        """评估内容完整性"""
        issues = []
        score = 1.0

        # 检查内容块数量
        total_blocks = len(document.content_blocks)
        if total_blocks == 0:
            issues.append(IntegrityIssueDetail(
                IntegrityIssue.MISSING_CONTENT,
                1.0,
                "文档中没有提取到任何内容块",
                "整个文档",
                ["检查文件格式", "确认解析引擎正常工作"]
            ))
            return 0.0, issues

        # 检查每页是否有内容
        pages_with_content = set(block.page_number for block in document.content_blocks)
        expected_pages = set(range(1, document.total_pages + 1))
        missing_pages = expected_pages - pages_with_content

        if missing_pages:
            severity = len(missing_pages) / document.total_pages
            issues.append(IntegrityIssueDetail(
                IntegrityIssue.MISSING_CONTENT,
                severity,
                f"页面 {sorted(missing_pages)} 缺少内容",
                f"页面: {sorted(missing_pages)}",
                ["重新解析缺失页面", "检查页面是否为空白"]
            ))
            score -= severity * 0.2

        # 检查内容类型分布
        type_distribution = defaultdict(int)
        for block in document.content_blocks:
            type_distribution[block.content_type] += 1

        # 评估类型分布的合理性
        for content_type, expected_ratio in self.expected_content_ratios.items():
            actual_count = type_distribution.get(content_type, 0)
            actual_ratio = actual_count / total_blocks if total_blocks > 0 else 0

            if actual_ratio < expected_ratio * 0.5:  # 低于预期50%
                issues.append(IntegrityIssueDetail(
                    IntegrityIssue.MISSING_CONTENT,
                    0.3,
                    f"{content_type.value} 类型内容过少 (实际: {actual_ratio:.1%}, 预期: {expected_ratio:.1%})",
                    f"内容类型: {content_type.value}",
                    ["检查解析配置", "调整内容识别参数"]
                ))
                score -= 0.1

        # 检查内容长度异常
        for block in document.content_blocks:
            if len(block.content.strip()) < 10:
                issues.append(IntegrityIssueDetail(
                    IntegrityIssue.LOW_QUALITY,
                    0.2,
                    f"内容块过短: '{block.content[:20]}...'",
                    f"块ID: {block.id}, 页面: {block.page_number}",
                    ["合并相邻内容块", "检查OCR识别质量"]
                ))

        return max(0.0, score), issues

    async def _evaluate_structure_integrity(self, document: ParsedDocument) -> Tuple[float, List[IntegrityIssueDetail]]:
        """评估结构完整性"""
        issues = []
        score = 1.0

        # 检查章节结构
        if not document.chapters:
            issues.append(IntegrityIssueDetail(
                IntegrityIssue.BROKEN_STRUCTURE,
                0.4,
                "文档缺少章节结构",
                "整个文档",
                ["使用结构分析功能", "检查标题识别参数"]
            ))
            score -= 0.4

        else:
            # 检查章节层次
            levels = [chapter.level for chapter in document.chapters]
            if levels != sorted(levels):
                issues.append(IntegrityIssueDetail(
                    IntegrityIssue.BROKEN_STRUCTURE,
                    0.3,
                    "章节层次结构不连续",
                    "章节结构",
                    ["重新分析章节结构", "检查章节标题格式"]
                ))
                score -= 0.3

            # 检查章节内容覆盖
            for chapter in document.chapters:
                if not chapter.blocks:
                    issues.append(IntegrityIssueDetail(
                        IntegrityIssue.ORPHANED_CONTENT,
                        0.2,
                        f"章节 '{chapter.title}' 没有分配内容块",
                        f"章节: {chapter.id}",
                        ["检查内容分配逻辑", "重新运行结构分析"]
                    ))
                    score -= 0.1

        # 检查页面连续性
        content_pages = sorted(set(block.page_number for block in document.content_blocks))
        if len(content_pages) > 1:
            gaps = []
            for i in range(len(content_pages) - 1):
                if content_pages[i + 1] - content_pages[i] > 1:
                    gaps.append((content_pages[i], content_pages[i + 1]))

            if gaps:
                issues.append(IntegrityIssueDetail(
                    IntegrityIssue.BROKEN_STRUCTURE,
                    0.2,
                    f"页面内容不连续: {gaps}",
                    "页面结构",
                    ["检查缺失页面", "确认页面范围正确"]
                ))
                score -= 0.2

        return max(0.0, score), issues

    async def _evaluate_content_quality(self, document: ParsedDocument) -> Tuple[float, List[IntegrityIssueDetail]]:
        """评估内容质量"""
        issues = []
        score = 1.0

        if not document.content_blocks:
            return 0.0, []

        # 计算平均置信度
        confidences = [block.confidence for block in document.content_blocks if block.confidence is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)

            if avg_confidence < 0.6:
                issues.append(IntegrityIssueDetail(
                    IntegrityIssue.LOW_QUALITY,
                    0.4,
                    f"整体内容质量偏低 (平均置信度: {avg_confidence:.2f})",
                    "整体文档",
                    ["调整解析参数", "使用更高质量的识别引擎"]
                ))
                score -= 0.4
            elif avg_confidence < 0.8:
                issues.append(IntegrityIssueDetail(
                    IntegrityIssue.LOW_QUALITY,
                    0.2,
                    f"内容质量一般 (平均置信度: {avg_confidence:.2f})",
                    "整体文档",
                    ["考虑提高质量阈值", "检查低质量内容块"]
                ))
                score -= 0.2

        # 检查低置信度内容块
        low_confidence_blocks = [
            block for block in document.content_blocks
            if block.confidence and block.confidence < 0.5
        ]

        if low_confidence_blocks:
            severity = len(low_confidence_blocks) / len(document.content_blocks)
            issues.append(IntegrityIssueDetail(
                IntegrityIssue.LOW_QUALITY,
                severity,
                f"发现 {len(low_confidence_blocks)} 个低置信度内容块",
                f"多页面 (共{len(low_confidence_blocks)}个)",
                ["重新处理低质量页面", "使用备用解析方法"]
            ))
            score -= severity * 0.3

        # 检查重复内容
        content_hashes = {}
        duplicate_blocks = []

        for block in document.content_blocks:
            content_hash = self._generate_content_hash(block.content)
            if content_hash in content_hashes:
                duplicate_blocks.append(block)
            else:
                content_hashes[content_hash] = block

        if duplicate_blocks:
            severity = len(duplicate_blocks) / len(document.content_blocks)
            issues.append(IntegrityIssueDetail(
                IntegrityIssue.DUPLICATE_CONTENT,
                severity,
                f"发现 {len(duplicate_blocks)} 个重复内容块",
                "多页面",
                ["去重处理", "检查解析逻辑"]
            ))
            score -= severity * 0.2

        return max(0.0, score), issues

    async def _evaluate_semantic_coherence(self, document: ParsedDocument) -> Tuple[float, List[IntegrityIssueDetail]]:
        """评估语义连贯性"""
        issues = []
        score = 1.0

        # 检查内容连贯性模式
        all_content = " ".join(block.content for block in document.content_blocks)

        # 检查章节连续性
        continuity_patterns = self.coherence_patterns["chapter_continuity"]
        missing_continuity = []

        for pattern in continuity_patterns:
            if not re.search(pattern, all_content, re.IGNORECASE):
                missing_continuity.append(pattern)

        if missing_continuity:
            issues.append(IntegrityIssueDetail(
                IntegrityIssue.BROKEN_STRUCTURE,
                0.2,
                "缺少章节连续性标记",
                "文档结构",
                ["检查章节标题识别", "完善结构分析"]
            ))
            score -= 0.2

        # 检查引用完整性
        reference_patterns = self.coherence_patterns["reference_patterns"]
        found_references = set()
        referenced_items = set()

        for pattern in reference_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            for match in matches:
                found_references.add(match)
                # 提取引用的项目编号
                number_match = re.search(r'\d+', match)
                if number_match:
                    referenced_items.add(number_match.group())

        # 验证引用的图表公式是否存在
        tables_count = len([b for b in document.content_blocks if b.content_type == ContentType.TABLE])
        figures_count = len([b for b in document.content_blocks if b.content_type in [ContentType.IMAGE, ContentType.FIGURE]])
        formulas_count = len([b for b in document.content_blocks if b.content_type == ContentType.FORMULA])

        if len(found_references) > 0:
            expected_refs = tables_count + figures_count + formulas_count
            if expected_refs < len(found_references):
                issues.append(IntegrityIssueDetail(
                    IntegrityIssue.MISSING_REFERENCES,
                    0.3,
                    f"引用的项目数 ({len(found_references)}) 超过实际项目数 ({expected_refs})",
                    "文档引用",
                    ["检查解析准确性", "验证引用完整性"]
                ))
                score -= 0.3

        # 检查流程指示词
        flow_patterns = self.coherence_patterns["flow_indicators"]
        flow_found = any(re.search(pattern, all_content, re.IGNORECASE) for pattern in flow_patterns)

        if not flow_found and len(document.content_blocks) > 10:
            issues.append(IntegrityIssueDetail(
                IntegrityIssue.INCONSISTENT_FORMAT,
                0.1,
                "文档缺少逻辑流程指示词",
                "文档流程",
                ["检查内容识别", "优化内容组织"]
            ))
            score -= 0.1

        return max(0.0, score), issues

    def _calculate_content_density(self, document: ParsedDocument) -> float:
        """计算内容密度"""
        if not document.content_blocks or document.total_pages == 0:
            return 0.0

        total_content_length = sum(len(block.content) for block in document.content_blocks)
        return total_content_length / (document.total_pages * 1000)  # 每页千字符数

    def _evaluate_type_distribution(self, document: ParsedDocument) -> float:
        """评估内容类型分布"""
        if not document.content_blocks:
            return 0.0

        type_counts = defaultdict(int)
        for block in document.content_blocks:
            type_counts[block.content_type] += 1

        total_blocks = len(document.content_blocks)
        distribution_score = 0.0

        for content_type, expected_ratio in self.expected_content_ratios.items():
            actual_ratio = type_counts.get(content_type, 0) / total_blocks
            # 计算与预期比例的差异
            diff = abs(actual_ratio - expected_ratio)
            distribution_score += (1 - diff) * self.expected_content_ratios[content_type]

        return distribution_score

    def _calculate_coverage_ratio(self, document: ParsedDocument) -> float:
        """计算页面覆盖率"""
        if document.total_pages == 0:
            return 0.0

        pages_with_content = set(block.page_number for block in document.content_blocks)
        return len(pages_with_content) / document.total_pages

    def _calculate_avg_confidence(self, document: ParsedDocument) -> float:
        """计算平均置信度"""
        confidences = [block.confidence for block in document.content_blocks if block.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _calculate_high_quality_ratio(self, document: ParsedDocument) -> float:
        """计算高质量内容比例"""
        if not document.content_blocks:
            return 0.0

        high_quality_count = sum(
            1 for block in document.content_blocks
            if block.confidence and block.confidence >= 0.8
        )
        return high_quality_count / len(document.content_blocks)

    def _evaluate_chapter_hierarchy(self, chapters: List[Chapter]) -> float:
        """评估章节层次结构"""
        if not chapters:
            return 0.0

        # 检查层次是否合理
        levels = [chapter.level for chapter in chapters]
        level_gaps = 0

        for i in range(1, len(levels)):
            if abs(levels[i] - levels[i-1]) > 1:
                level_gaps += 1

        return 1.0 - (level_gaps / len(levels))

    def _evaluate_content_chapter_alignment(self, document: ParsedDocument) -> float:
        """评估内容与章节对齐"""
        if not document.chapters or not document.content_blocks:
            return 1.0

        aligned_blocks = 0
        for block in document.content_blocks:
            if block.chapter_id:
                aligned_blocks += 1

        return aligned_blocks / len(document.content_blocks)

    def _evaluate_structural_completeness(self, document: ParsedDocument) -> float:
        """评估结构完整性"""
        score = 1.0

        # 检查是否有标题
        has_headers = any(block.content_type == ContentType.HEADER for block in document.content_blocks)
        if not has_headers and len(document.content_blocks) > 5:
            score -= 0.2

        # 检查是否有页脚（可选）
        # has_footers = any(block.content_type == ContentType.FOOTER for block in document.content_blocks)

        return max(0.0, score)

    def _generate_content_hash(self, content: str) -> str:
        """生成内容哈希"""
        # 简化实现：去除空白字符后计算哈希
        normalized = re.sub(r'\s+', '', content.lower())
        return str(hash(normalized))

    def _generate_recommendations(self, issues: List[IntegrityIssueDetail], overall_score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于评分的总体建议
        if overall_score < 0.3:
            recommendations.append("文档解析质量较差，建议重新运行解析流程")
        elif overall_score < 0.5:
            recommendations.append("文档解析存在较多问题，建议检查解析配置")
        elif overall_score < 0.7:
            recommendations.append("文档解析质量一般，可考虑优化某些参数")
        else:
            recommendations.append("文档解析质量良好")

        # 基于具体问题的建议
        issue_types = set(issue.issue_type for issue in issues)

        if IntegrityIssue.MISSING_CONTENT in issue_types:
            recommendations.append("补充缺失的内容，特别是表格和图片")

        if IntegrityIssue.BROKEN_STRUCTURE in issue_types:
            recommendations.append("修复文档结构问题，确保章节层次正确")

        if IntegrityIssue.LOW_QUALITY in issue_types:
            recommendations.append("提高内容识别质量，考虑使用更高精度的设置")

        if IntegrityIssue.MISSING_REFERENCES in issue_types:
            recommendations.append("检查和修复引用关系，确保图表公式引用正确")

        # 去重建议
        unique_recommendations = list(set(recommendations))
        return sorted(unique_recommendations, key=len, reverse=True)  # 长建议在前

    def get_integrity_level(self, score: float) -> str:
        """获取完整性等级"""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        elif score >= self.thresholds["good"]:
            return "good"
        elif score >= self.thresholds["fair"]:
            return "fair"
        elif score >= self.thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"

    def export_report(self, report: IntegrityReport, output_path: str):
        """导出完整性评估报告"""
        try:
            export_data = {
                "overall_score": report.overall_score,
                "integrity_level": self.get_integrity_level(report.overall_score),
                "issues": [
                    {
                        "type": issue.issue_type.value,
                        "severity": issue.severity,
                        "description": issue.description,
                        "location": issue.location,
                        "suggestions": issue.suggestions,
                        "affected_blocks": issue.affected_blocks
                    }
                    for issue in report.issues
                ],
                "completeness_metrics": report.completeness_metrics,
                "quality_metrics": report.quality_metrics,
                "structure_metrics": report.structure_metrics,
                "recommendations": report.recommendations,
                "evaluation_weights": self.evaluation_weights
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"完整性评估报告已导出: {output_path}")

        except Exception as e:
            logger.error(f"导出报告失败: {str(e)}")
            raise