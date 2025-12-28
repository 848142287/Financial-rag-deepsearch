"""
数据平衡管理器
负责平衡各类别文档数据分布，确保数据集的多样性和均衡性
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class DocumentCategory(Enum):
    """文档类别枚举"""
    ANNUAL_REPORT = "年报"
    QUARTERLY_REPORT = "季报"
    RESEARCH_REPORT = "研究报告"
    NEWS_ANALYSIS = "新闻分析"
    POLICY_DOCUMENT = "政策文件"
    INDUSTRY_ANALYSIS = "行业分析"
    COMPANY_PROFILE = "公司概况"
    MARKET_DATA = "市场数据"
    REGULATORY_FILING = "监管文件"
    INVESTMENT_STRATEGY = "投资策略"


class DocumentSource(Enum):
    """文档来源枚举"""
    SEC = "美国证监会"
    CSRC = "中国证监会"
    SSE = "上交所"
    SZSE = "深交所"
    WIND = "万得"
    BLOOMBERG = "彭博"
    REUTERS = "路透"
    THOMSON_REUTERS = "汤姆森路透"
    MEDIA = "媒体"
    OTHER = "其他"


@dataclass
class DocumentMetadata:
    """文档元数据"""
    document_id: str
    category: DocumentCategory
    source: DocumentSource
    file_path: str
    file_size: int
    page_count: int
    upload_time: str
    quality_score: float
    tags: List[str]
    language: str = "zh"


@dataclass
class CategoryStats:
    """类别统计信息"""
    category: DocumentCategory
    count: int
    total_size: int
    avg_quality: float
    sources: Dict[str, int]
    last_updated: str


@dataclass
class BalanceConfig:
    """平衡配置"""
    target_distribution: Dict[DocumentCategory, float]  # 目标分布百分比
    min_documents_per_category: int  # 每类最小文档数
    max_documents_per_category: int  # 每类最大文档数
    quality_threshold: float  # 质量阈值
    balance_strategy: str  # 平衡策略: "undersample", "oversample", "hybrid"


class DataBalancer:
    """数据平衡器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据平衡器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.documents: Dict[str, DocumentMetadata] = {}
        self.category_stats: Dict[DocumentCategory, CategoryStats] = {}

        # 默认平衡配置
        self.default_config = BalanceConfig(
            target_distribution={
                DocumentCategory.ANNUAL_REPORT: 0.20,
                DocumentCategory.QUARTERLY_REPORT: 0.15,
                DocumentCategory.RESEARCH_REPORT: 0.15,
                DocumentCategory.NEWS_ANALYSIS: 0.10,
                DocumentCategory.POLICY_DOCUMENT: 0.10,
                DocumentCategory.INDUSTRY_ANALYSIS: 0.10,
                DocumentCategory.COMPANY_PROFILE: 0.05,
                DocumentCategory.MARKET_DATA: 0.05,
                DocumentCategory.REGULATORY_FILING: 0.05,
                DocumentCategory.INVESTMENT_STRATEGY: 0.05
            },
            min_documents_per_category=10,
            max_documents_per_category=1000,
            quality_threshold=0.6,
            balance_strategy="hybrid"
        )

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")

        return {}

    async def add_document(self, metadata: DocumentMetadata) -> bool:
        """
        添加文档到平衡器

        Args:
            metadata: 文档元数据

        Returns:
            是否添加成功
        """
        try:
            # 检查质量
            if metadata.quality_score < self.default_config.quality_threshold:
                logger.warning(f"文档质量不达标: {metadata.document_id}, 质量分数: {metadata.quality_score}")
                return False

            # 检查类别容量
            category_count = sum(1 for doc in self.documents.values() if doc.category == metadata.category)
            if category_count >= self.default_config.max_documents_per_category:
                logger.warning(f"类别已达上限: {metadata.category.value}, 当前数量: {category_count}")
                return False

            # 添加文档
            self.documents[metadata.document_id] = metadata

            # 更新统计
            await self._update_category_stats(metadata.category)

            logger.info(f"文档添加成功: {metadata.document_id}, 类别: {metadata.category.value}")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    async def remove_document(self, document_id: str) -> bool:
        """
        移除文档

        Args:
            document_id: 文档ID

        Returns:
            是否移除成功
        """
        try:
            if document_id in self.documents:
                category = self.documents[document_id].category
                del self.documents[document_id]
                await self._update_category_stats(category)
                logger.info(f"文档移除成功: {document_id}")
                return True
            else:
                logger.warning(f"文档不存在: {document_id}")
                return False

        except Exception as e:
            logger.error(f"移除文档失败: {str(e)}")
            return False

    async def _update_category_stats(self, category: DocumentCategory):
        """更新类别统计"""
        try:
            category_docs = [doc for doc in self.documents.values() if doc.category == category]

            if not category_docs:
                if category in self.category_stats:
                    del self.category_stats[category]
                return

            # 计算统计信息
            count = len(category_docs)
            total_size = sum(doc.file_size for doc in category_docs)
            avg_quality = sum(doc.quality_score for doc in category_docs) / count

            # 统计来源
            sources = Counter(doc.source.value for doc in category_docs)

            self.category_stats[category] = CategoryStats(
                category=category,
                count=count,
                total_size=total_size,
                avg_quality=avg_quality,
                sources=dict(sources),
                last_updated=pd.Timestamp.now().isoformat()
            )

        except Exception as e:
            logger.error(f"更新类别统计失败: {str(e)}")

    async def get_balance_status(self) -> Dict[str, Any]:
        """获取数据平衡状态"""
        try:
            total_docs = len(self.documents)
            if total_docs == 0:
                return {
                    "total_documents": 0,
                    "balance_score": 0.0,
                    "categories": {},
                    "recommendations": ["数据集为空，需要添加更多文档"]
                }

            # 计算当前分布
            current_distribution = {}
            for category, stats in self.category_stats.items():
                current_distribution[category] = stats.count / total_docs

            # 计算平衡分数
            balance_score = self._calculate_balance_score(current_distribution)

            # 生成建议
            recommendations = self._generate_recommendations(current_distribution)

            # 构建结果
            result = {
                "total_documents": total_docs,
                "balance_score": balance_score,
                "categories": {
                    category.value: {
                        "count": stats.count,
                        "percentage": stats.count / total_docs * 100,
                        "target_percentage": self.default_config.target_distribution.get(category, 0) * 100,
                        "avg_quality": stats.avg_quality,
                        "sources": stats.sources
                    }
                    for category, stats in self.category_stats.items()
                },
                "target_distribution": {
                    category.value: percentage * 100
                    for category, percentage in self.default_config.target_distribution.items()
                },
                "recommendations": recommendations
            }

            return result

        except Exception as e:
            logger.error(f"获取平衡状态失败: {str(e)}")
            return {
                "error": str(e),
                "total_documents": 0,
                "balance_score": 0.0
            }

    def _calculate_balance_score(self, current_distribution: Dict[DocumentCategory, float]) -> float:
        """计算平衡分数 (0-1)"""
        try:
            if not current_distribution:
                return 0.0

            # 计算与目标分布的差异
            total_diff = 0.0
            target_dist = self.default_config.target_distribution

            for category in DocumentCategory:
                current = current_distribution.get(category, 0.0)
                target = target_dist.get(category, 0.0)
                total_diff += abs(current - target)

            # 归一化到0-1范围 (最大差异为2.0，因为所有差异之和最大为200%)
            balance_score = max(0.0, 1.0 - total_diff / 2.0)

            return round(balance_score, 3)

        except Exception as e:
            logger.error(f"计算平衡分数失败: {str(e)}")
            return 0.0

    def _generate_recommendations(self, current_distribution: Dict[DocumentCategory, float]) -> List[str]:
        """生成平衡建议"""
        recommendations = []
        target_dist = self.default_config.target_distribution
        min_docs = self.default_config.min_documents_per_category

        try:
            for category in DocumentCategory:
                current = current_distribution.get(category, 0.0)
                target = target_dist.get(category, 0.0)
                current_count = sum(1 for doc in self.documents.values() if doc.category == category)

                # 检查是否需要增加文档
                if current_count < min_docs:
                    recommendations.append(
                        f"{category.value}类文档数量不足({current_count}<{min_docs})，建议添加更多{category.value}文档"
                    )
                elif current < target - 0.05:  # 低于目标5%以上
                    shortage = int(len(self.documents) * (target - current))
                    recommendations.append(
                        f"{category.value}类文档占比偏低({current:.1%}<{target:.1%})，建议增加约{shortage}份文档"
                    )
                elif current > target + 0.05:  # 高于目标5%以上
                    excess = int(len(self.documents) * (current - target))
                    recommendations.append(
                        f"{category.value}类文档占比偏高({current:.1%}>{target:.1%})，考虑移除约{excess}份低质量文档"
                    )

            # 检查整体质量
            if self.documents:
                avg_quality = sum(doc.quality_score for doc in self.documents.values()) / len(self.documents)
                if avg_quality < self.default_config.quality_threshold:
                    recommendations.append(
                        f"整体文档质量偏低({avg_quality:.2f}<{self.default_config.quality_threshold})，建议添加更高质量的文档"
                    )

        except Exception as e:
            logger.error(f"生成建议失败: {str(e)}")
            recommendations.append("生成建议时出现错误")

        return recommendations

    async def auto_balance(self) -> Dict[str, Any]:
        """自动平衡数据集"""
        try:
            balance_status = await self.get_balance_status()
            actions = []

            current_distribution = {}
            for category, stats in self.category_stats.items():
                current_distribution[category] = stats.count / len(self.documents)

            target_dist = self.default_config.target_distribution
            strategy = self.default_config.balance_strategy

            # 执行平衡策略
            if strategy == "undersample":
                actions.extend(await self._undersample(current_distribution, target_dist))
            elif strategy == "oversample":
                actions.extend(await self._oversample(current_distribution, target_dist))
            else:  # hybrid
                actions.extend(await self._hybrid_balance(current_distribution, target_dist))

            # 重新计算平衡状态
            new_status = await self.get_balance_status()

            result = {
                "balance_strategy": strategy,
                "actions_taken": actions,
                "before_balance": balance_status,
                "after_balance": new_status,
                "improvement": new_status["balance_score"] - balance_status["balance_score"]
            }

            logger.info(f"自动平衡完成，平衡分数提升: {result['improvement']:.3f}")
            return result

        except Exception as e:
            logger.error(f"自动平衡失败: {str(e)}")
            return {"error": str(e), "actions_taken": []}

    async def _undersample(self, current_dist: Dict[DocumentCategory, float],
                          target_dist: Dict[DocumentCategory, float]) -> List[str]:
        """下采样平衡"""
        actions = []

        try:
            for category in DocumentCategory:
                current = current_dist.get(category, 0.0)
                target = target_dist.get(category, 0.0)

                if current > target + 0.05:  # 高于目标5%以上
                    # 获取该类别的文档，按质量排序
                    category_docs = [
                        doc for doc in self.documents.values()
                        if doc.category == category
                    ]
                    category_docs.sort(key=lambda x: x.quality_score)

                    # 计算需要移除的数量
                    total_docs = len(self.documents)
                    target_count = int(total_docs * target)
                    remove_count = len(category_docs) - target_count

                    # 移除低质量文档
                    for i in range(min(remove_count, len(category_docs))):
                        doc_id = category_docs[i].document_id
                        await self.remove_document(doc_id)
                        actions.append(f"移除低质量文档: {doc_id} ({category.value})")

        except Exception as e:
            logger.error(f"下采样失败: {str(e)}")
            actions.append(f"下采样失败: {str(e)}")

        return actions

    async def _oversample(self, current_dist: Dict[DocumentCategory, float],
                         target_dist: Dict[DocumentCategory, float]) -> List[str]:
        """上采样平衡（仅记录需要添加的文档）"""
        actions = []

        try:
            for category in DocumentCategory:
                current = current_dist.get(category, 0.0)
                target = target_dist.get(category, 0.0)

                if current < target - 0.05:  # 低于目标5%以上
                    # 计算需要添加的数量
                    total_docs = len(self.documents)
                    target_count = int(total_docs * target)
                    current_count = sum(1 for doc in self.documents.values() if doc.category == category)
                    add_count = target_count - current_count

                    actions.append(
                        f"需要添加{add_count}份{category.value}类文档以达到平衡分布"
                    )

        except Exception as e:
            logger.error(f"上采样失败: {str(e)}")
            actions.append(f"上采样失败: {str(e)}")

        return actions

    async def _hybrid_balance(self, current_dist: Dict[DocumentCategory, float],
                             target_dist: Dict[DocumentCategory, float]) -> List[str]:
        """混合平衡策略"""
        # 先执行下采样，再记录上采样需求
        actions = []
        actions.extend(await self._undersample(current_dist, target_dist))

        # 重新计算分布
        new_current_dist = {}
        total_docs = len(self.documents)
        if total_docs > 0:
            for category in DocumentCategory:
                count = sum(1 for doc in self.documents.values() if doc.category == category)
                new_current_dist[category] = count / total_docs

        actions.extend(await self._oversample(new_current_dist, target_dist))
        return actions

    async def export_analysis(self, output_path: str):
        """导出数据分析结果"""
        try:
            balance_status = await self.get_balance_status()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(balance_status, f, ensure_ascii=False, indent=2)

            logger.info(f"数据分析结果已导出到: {output_path}")

        except Exception as e:
            logger.error(f"导出分析结果失败: {str(e)}")

    def get_quality_summary(self) -> Dict[str, Any]:
        """获取数据质量摘要"""
        try:
            if not self.documents:
                return {
                    "total_documents": 0,
                    "avg_quality": 0.0,
                    "quality_distribution": {},
                    "low_quality_count": 0
                }

            qualities = [doc.quality_score for doc in self.documents.values()]
            avg_quality = sum(qualities) / len(qualities)

            # 质量分布
            quality_ranges = {
                "优秀 (0.9-1.0)": 0,
                "良好 (0.7-0.9)": 0,
                "一般 (0.5-0.7)": 0,
                "较差 (0.3-0.5)": 0,
                "很差 (0.0-0.3)": 0
            }

            for q in qualities:
                if q >= 0.9:
                    quality_ranges["优秀 (0.9-1.0)"] += 1
                elif q >= 0.7:
                    quality_ranges["良好 (0.7-0.9)"] += 1
                elif q >= 0.5:
                    quality_ranges["一般 (0.5-0.7)"] += 1
                elif q >= 0.3:
                    quality_ranges["较差 (0.3-0.5)"] += 1
                else:
                    quality_ranges["很差 (0.0-0.3)"] += 1

            low_quality_count = sum(1 for q in qualities if q < self.default_config.quality_threshold)

            return {
                "total_documents": len(self.documents),
                "avg_quality": round(avg_quality, 3),
                "min_quality": min(qualities),
                "max_quality": max(qualities),
                "quality_distribution": quality_ranges,
                "low_quality_count": low_quality_count,
                "quality_threshold": self.default_config.quality_threshold
            }

        except Exception as e:
            logger.error(f"获取质量摘要失败: {str(e)}")
            return {"error": str(e)}


# 全局数据平衡器实例
data_balancer = DataBalancer()