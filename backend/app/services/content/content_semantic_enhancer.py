"""
内容语义增强服务
对文档内容进行深度语义分析和增强
"""
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from langchain_core.documents import Document

from app.services.parsers.advanced.enhanced_metadata_extractor import (
    ChunkMetadataExtraction,
    DocumentMetadataExtraction
)

logger = logging.getLogger(__name__)


class SemanticCategory(str, Enum):
    """语义类别"""
    FINANCIAL_STATEMENT = "financial_statement"   # 财务报表
    INVESTMENT_ADVICE = "investment_advice"         # 投资建议
    MARKET_ANALYSIS = "market_analysis"             # 市场分析
    RISK_ASSESSMENT = "risk_assessment"             # 风险评估
    BUSINESS_NEWS = "business_news"                 # 商业新闻
    RESEARCH_REPORT = "research_report"             # 研究报告


@dataclass
class SemanticEnhancement:
    """语义增强结果"""
    chunk_id: str
    document_id: str

    # 基础信息
    original_content: str = ""
    semantic_category: str = ""

    # 增强内容
    enhanced_summary: str = ""
    structured_info: Dict[str, Any] = field(default_factory=dict)
    key_insights: List[str] = field(default_factory=list)

    # 语义标签
    semantic_tags: Set[str] = field(default_factory=set)
    business_domain: str = ""
    industry_tags: List[str] = field(default_factory=list)

    # 情感和倾向
    sentiment: str = ""          # positive, negative, neutral
    sentiment_score: float = 0.0  # -1.0 to 1.0
    risk_level: str = ""          # high, medium, low
    opportunity_level: str = ""   # high, medium, low

    # 质量指标
    information_density: float = 0.0
    clarity_score: float = 0.0

    # 时间信息
    temporal_info: Dict[str, str] = field(default_factory=dict)

    # 处理元数据
    processing_time: float = 0.0
    model_used: str = ""


class ContentSemanticEnhancer:
    """内容语义增强器"""

    def __init__(self):
        # 行业关键词
        self.industry_keywords = self._load_industry_keywords()

        # 语义规则
        self.semantic_rules = self._load_semantic_rules()

    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """加载行业关键词"""
        return {
            "banking": ["银行", "存款", "贷款", "利率", "储蓄", "信贷"],
            "securities": ["证券", "股票", "基金", "债券", "IPO", "上市"],
            "insurance": ["保险", "保费", "理赔", "承保", "保单"],
            "real_estate": ["房地产", "房价", "楼市", "住房", "物业"],
            "technology": ["科技", "互联网", "AI", "人工智能", "芯片", "半导体"],
            "energy": ["能源", "石油", "天然气", "电力", "新能源", "光伏"],
            "healthcare": ["医疗", "医药", "生物", "疫苗", "医院"],
            "manufacturing": ["制造", "工厂", "生产", "制造业", "产能"],
        }

    def _load_semantic_rules(self) -> Dict[str, List[str]]:
        """加载语义规则"""
        return {
            "risk_keywords": ["风险", "下跌", "亏损", "违约", "危机", "警告"],
            "opportunity_keywords": ["增长", "上涨", "盈利", "突破", "机会", "扩张"],
            "financial_statement_keywords": ["年报", "财报", "营业收入", "净利润", "资产", "负债"],
        }

    async def enhance_content(
        self,
        document_analysis_result: Any
    ) -> List[SemanticEnhancement]:
        """
        增强文档内容的语义信息

        Args:
            document_analysis_result: DocumentAnalysisResult

        Returns:
            SemanticEnhancement 列表
        """
        start_time = datetime.now()

        try:
            document_id = document_analysis_result.document_id
            chunks = document_analysis_result.chunks
            chunks_metadata = document_analysis_result.chunks_metadata

            enhancements = []

            for chunk, metadata in zip(chunks, chunks_metadata):
                chunk_id = chunk.metadata.get('chunk_id', f"{document_id}_chunk_{chunks.index(chunk)}")

                enhancement = SemanticEnhancement(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    original_content=chunk.page_content
                )

                # 1. 分类语义类别
                enhancement.semantic_category = self._classify_semantic_category(
                    chunk.page_content,
                    metadata
                )

                # 2. 生成增强摘要
                enhancement.enhanced_summary = self._generate_enhanced_summary(metadata)

                # 3. 提取结构化信息
                enhancement.structured_info = self._extract_structured_info(metadata)

                # 4. 提取关键洞察
                enhancement.key_insights = self._extract_key_insights(metadata)

                # 5. 添加语义标签
                enhancement.semantic_tags = self._generate_semantic_tags(
                    chunk.page_content,
                    metadata
                )

                # 6. 识别业务领域和行业
                enhancement.business_domain, enhancement.industry_tags = self._identify_domain_and_industry(
                    chunk.page_content
                )

                # 7. 分析情感倾向
                enhancement.sentiment = metadata.sentiment
                enhancement.sentiment_score = self._calculate_sentiment_score(
                    chunk.page_content,
                    metadata
                )

                # 8. 评估风险和机会水平
                enhancement.risk_level = self._assess_risk_level(
                    chunk.page_content,
                    enhancement.sentiment
                )
                enhancement.opportunity_level = self._assess_opportunity_level(
                    chunk.page_content,
                    enhancement.sentiment
                )

                # 9. 计算质量指标
                enhancement.information_density = self._calculate_information_density(chunk.page_content)
                enhancement.clarity_score = self._calculate_clarity_score(
                    chunk.page_content,
                    metadata
                )

                # 10. 提取时间信息
                enhancement.temporal_info = self._extract_temporal_info(
                    chunk.page_content,
                    metadata
                )

                enhancements.append(enhancement)

            processing_time = (datetime.now() - start_time).total_seconds()

            # 更新处理时间
            for enhancement in enhancements:
                enhancement.processing_time = processing_time / len(enhancements)

            logger.info(
                f"内容语义增强完成: {document_id}, "
                f"{len(enhancements)} 个chunk, "
                f"耗时: {processing_time:.2f}s"
            )

            return enhancements

        except Exception as e:
            logger.error(f"内容语义增强失败: {e}")
            return []

    def _classify_semantic_category(
        self,
        content: str,
        metadata: ChunkMetadataExtraction
    ) -> str:
        """分类语义类别"""
        # 根据关键词分类
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in self.semantic_rules["financial_statement_keywords"]):
            return SemanticCategory.FINANCIAL_STATEMENT.value

        # 根据摘要推断
        if metadata.summary:
            if "投资" in metadata.summary or "建议" in metadata.summary:
                return SemanticCategory.INVESTMENT_ADVICE.value
            elif "市场" in metadata.summary or "行业" in metadata.summary:
                return SemanticCategory.MARKET_ANALYSIS.value
            elif "风险" in metadata.summary:
                return SemanticCategory.RISK_ASSESSMENT.value

        return SemanticCategory.BUSINESS_NEWS.value

    def _generate_enhanced_summary(self, metadata: ChunkMetadataExtraction) -> str:
        """生成增强摘要"""
        summary_parts = []

        if metadata.summary:
            summary_parts.append(metadata.summary)

        if metadata.key_points:
            top_key_points = sorted(
                metadata.key_points,
                key=lambda kp: 3 if kp.importance == "high" else 1,
                reverse=True
            )[:3]
            summary_parts.append("关键点: " + "; ".join([kp.point for kp in top_key_points]))

        return " | ".join(summary_parts)

    def _extract_structured_info(self, metadata: ChunkMetadataExtraction) -> Dict[str, Any]:
        """提取结构化信息"""
        structured = {
            "key_points_count": len(metadata.key_points),
            "high_importance_count": sum(1 for kp in metadata.key_points if kp.importance == "high"),
            "tables_count": len(metadata.tables),
            "topics_count": len(metadata.topics),
            "language": metadata.language,
            "sentiment": metadata.sentiment,
        }

        # 提取关键统计数据
        if metadata.tables:
            structured["table_info"] = [
                {
                    "title": table.title,
                    "row_count": len(table.rows),
                    "column_count": len(table.headers)
                }
                for table in metadata.tables
            ]

        # 提取主题详情
        if metadata.topics:
            structured["topics"] = [
                {
                    "topic": topic.topic,
                    "relevance": round(topic.relevance_score, 2),
                    "keywords": topic.keywords
                }
                for topic in metadata.topics
            ]

        return structured

    def _extract_key_insights(self, metadata: ChunkMetadataExtraction) -> List[str]:
        """提取关键洞察"""
        insights = []

        # 从高重要性关键点中提取洞察
        high_importance_kps = [
            kp for kp in metadata.key_points
            if kp.importance == "high"
        ]

        for kp in high_importance_kps[:5]:
            insights.append(kp.point)

        # 从表格中提取洞察
        for table in metadata.tables[:2]:
            if table.summary:
                insights.append(f"表格洞察: {table.summary}")

        return insights

    def _generate_semantic_tags(
        self,
        content: str,
        metadata: ChunkMetadataExtraction
    ) -> Set[str]:
        """生成语义标签"""
        tags = set()

        # 从主题生成标签
        for topic in metadata.topics:
            if topic.relevance_score > 0.7:
                tags.add(topic.topic)
                tags.update(topic.keywords)

        # 从情感生成标签
        if metadata.sentiment != "neutral":
            tags.add(metadata.sentiment)

        # 根据关键点生成标签
        for kp in metadata.key_points:
            # 提取关键词
            words = kp.point.split()
            for word in words:
                if len(word) >= 2:
                    tags.add(word)

        return tags

    def _identify_domain_and_industry(self, content: str) -> Tuple[str, List[str]]:
        """识别业务领域和行业"""
        content_lower = content.lower()

        # 识别行业
        industries = []
        for industry, keywords in self.industry_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                industries.append(industry)

        # 识别业务领域
        business_domain = "general"
        if "银行" in content_lower or "金融" in content_lower:
            business_domain = "financial_services"
        elif "科技" in content_lower or "互联网" in content_lower:
            business_domain = "technology"
        elif "制造" in content_lower or "工厂" in content_lower:
            business_domain = "manufacturing"

        return business_domain, list(set(industries))

    def _calculate_sentiment_score(
        self,
        content: str,
        metadata: ChunkMetadataExtraction
    ) -> float:
        """计算情感分数"""
        sentiment = metadata.sentiment

        if sentiment == "positive":
            return 0.7
        elif sentiment == "negative":
            return -0.7
        else:
            return 0.0

    def _assess_risk_level(
        self,
        content: str,
        sentiment: str
    ) -> str:
        """评估风险水平"""
        content_lower = content.lower()

        risk_count = sum(1 for keyword in self.semantic_rules["risk_keywords"] if keyword in content_lower)

        if risk_count >= 3:
            return "high"
        elif risk_count >= 1:
            return "medium"
        else:
            return "low"

    def _assess_opportunity_level(
        self,
        content: str,
        sentiment: str
    ) -> str:
        """评估机会水平"""
        content_lower = content.lower()

        opp_count = sum(1 for keyword in self.semantic_rules["opportunity_keywords"] if keyword in content_lower)

        if opp_count >= 3:
            return "high"
        elif opp_count >= 1:
            return "medium"
        else:
            return "low"

    def _calculate_information_density(self, content: str) -> float:
        """计算信息密度"""
        # 简化实现：基于关键词密度
        words = content.split()
        if not words:
            return 0.0

        # 统计有意义的词汇（长度>=2的中文或英文单词）
        meaningful_words = sum(1 for word in words if len(word) >= 2)

        return min(1.0, meaningful_words / len(words) * 2)

    def _calculate_clarity_score(
        self,
        content: str,
        metadata: ChunkMetadataExtraction
    ) -> float:
        """计算清晰度分数"""
        score = 0.5  # 基础分

        # 有摘要加分
        if metadata.summary and len(metadata.summary) > 20:
            score += 0.2

        # 有关键点加分
        if metadata.key_points:
            score += 0.1

        # 结构化信息（表格）加分
        if metadata.tables:
            score += 0.1

        # 主题明确加分
        if metadata.topics:
            avg_relevance = sum(t.relevance_score for t in metadata.topics) / len(metadata.topics)
            score += avg_relevance * 0.1

        return min(1.0, score)

    def _extract_temporal_info(
        self,
        content: str,
        metadata: ChunkMetadataExtraction
    ) -> Dict[str, str]:
        """提取时间信息"""
        import re

        temporal = {}

        # 提取年份
        years = re.findall(r'(20\d{2})', content)
        if years:
            temporal["years"] = list(set(years))

        # 提取季度
        quarters = re.findall(r'(第?[一二三四]季度|Q[1-4])', content)
        if quarters:
            temporal["quarters"] = list(set(quarters))

        # 提取月份
        months = re.findall(r'(\d{1,2})月', content)
        if months:
            temporal["months"] = list(set(months))

        return temporal


# 全局实例
content_semantic_enhancer = ContentSemanticEnhancer()
