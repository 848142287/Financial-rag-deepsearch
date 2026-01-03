"""
优化的Embedding策略 - 金融领域适配
专门针对券商研报的embedding优化
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class DocumentType(Enum):
    """文档类型枚举"""
    BROKER_REPORT = "broker_report"  # 券商研报
    FINANCIAL_REPORT = "financial_report"  # 财务报告
    MARKET_RESEARCH = "market_research"  # 行业研究
    COMPANY_REPORT = "company_report"  # 公司报告
    GENERAL = "general"  # 通用文档


@dataclass
class ChunkingConfig:
    """分块配置"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_tables: bool = True
    preserve_formulas: bool = True
    section_aware: bool = True


@dataclass
class FinanceConfig:
    """金融领域配置"""
    # 券商名称列表
    brokers: List[str] = None

    # 金融实体类型
    entity_types: Dict[str, List[str]] = None

    # 财务指标模式
    financial_patterns: Dict[str, str] = None

    # 报告章节类型
    report_sections: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.brokers is None:
            self.brokers = [
                "中信证券", "国泰君安", "华泰证券", "招商证券",
                "海通证券", "广发证券", "中金公司", "申万宏源",
                "银河证券", "东方证券", "光大证券", "安信证券",
                "长江证券", "兴业证券", "国信证券"
            ]

        if self.entity_types is None:
            self.entity_types = {
                "STOCK": [
                    r"[0-9]{6}\.[A-Z]{2}",  # 股票代码 000001.SZ
                    r"[0-9]{6}",  # 纯数字代码
                    r"[A-Z]{4}-[0-9]{5}",  # 港股代码
                ],
                "FINANCIAL_RATIO": [
                    r"(PE|PB|ROE|ROA|EPS|BPS|毛利率|净利率|负债率)[0-9.]*%?",
                    r"(市盈率|市净率|净资产收益率|总资产收益率|每股收益)[0-9.]*%?",
                ],
                "MARKET_INDEX": [
                    r"(上证指数|深证成指|创业板指|沪深300|中证500|科创50)",
                    r"(恒生指数|国企指数|红筹指数|纳斯达克|标普500|道琼斯)",
                ],
                "FINANCIAL_TERM": [
                    r"(营业收入|净利润|归母净利润|扣非净利润)",
                    r"(资产负债表|利润表|现金流量表)",
                    r"(经营性现金流|投资性现金流|筹资性现金流)",
                    r"(毛利率|净利率|期间费用率|销售费用率|管理费用率)",
                    r"(资产负债率|流动比率|速动比率)",
                    r"(市盈率|市净率|市销率|股息率)",
                    r"(同比|环比|年均复合增长率)",
                ],
                "TIME_PERIOD": [
                    r"(20[0-9]{2})[年\-]([1-9]|1[0-2])[月\-]([1-3]?[0-9])日?",
                    r"([1-4]?[0-9]季度|[一二三四]季度)",
                    r"(20[0-9]{2}年上半年|20[0-9]{2}年下半年)",
                    r"(过去[一二三四五六七八九十]+年|近[一二三四五六七八九十]+年)",
                ],
                "QUANT_STRATEGY": [
                    r"(均线策略|动量策略|趋势跟踪|均值回归)",
                    r"(阿尔法策略|贝塔策略|对冲策略|套利策略)",
                    r"(量化选股|多因子模型|机器学习)",
                ]
            }

        if self.financial_patterns is None:
            self.financial_patterns = {
                "percentage": r"([0-9]+\.?[0-9]*)%",
                "amount": r"([0-9]+\.?[0-9]*)(万亿千百)?元",
                "growth_rate": r"增长([0-9]+\.?[0-9]*)%",
                "ratio": r"([0-9]+\.?[0-9]*):1",
            }

        if self.report_sections is None:
            self.report_sections = {
                "investment_advice": ["投资建议", "配置建议", "投资策略", "建议", "评级"],
                "risk_warning": ["风险提示", "风险因素", "风险", "风险警示"],
                "financial_data": ["财务", "业绩", "盈利", "营收", "利润", "财报"],
                "industry_analysis": ["行业分析", "行业", "产业链", "市场", "赛道"],
                "company_analysis": ["公司", "企业", "标的", "公司概况"],
                "valuation": ["估值", "定价", "目标价", "PE", "PB", "DCF", "相对估值"],
                "technical_analysis": ["技术", "工艺", "产品", "技术路线"],
                "market_performance": ["市场表现", "股价", "涨跌幅", "走势", "行情"],
            }


class OptimizedEmbeddingStrategy:
    """
    优化的Embedding策略

    功能：
    1. 文档类型检测
    2. 金融实体增强
    3. 自适应分块
    4. 元数据增强
    """

    def __init__(self, config: Optional[FinanceConfig] = None):
        self.config = config or FinanceConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """预编译正则表达式"""
        self.compiled_patterns = {}

        for entity_type, patterns in self.config.entity_types.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern) for pattern in patterns
            ]

    def detect_document_type(
        self,
        filename: str = "",
        content_preview: str = ""
    ) -> DocumentType:
        """
        检测文档类型

        Args:
            filename: 文件名
            content_preview: 内容预览（前500字符）

        Returns:
            DocumentType枚举值
        """
        content = content_preview.lower()
        filename_lower = filename.lower()

        # 检测券商研报
        broker_keywords = [
            "投资评级", "目标价", "投资建议", "风险提示",
            "盈利预测", "估值模型", "券商", "证券"
        ]

        # 检测财务报告
        financial_report_keywords = [
            "资产负债表", "利润表", "现金流量表",
            "财务报表", "审计报告", "附注"
        ]

        # 检测行业研究
        market_research_keywords = [
            "行业深度", "行业研究", "市场空间",
            "竞争格局", "产业链分析"
        ]

        # 检测公司报告
        company_report_keywords = [
            "公司深度", "公司研究", "上市公司",
            "年报", "半年度报告", "季度报告"
        ]

        # 检查文件名
        if any(keyword in filename_lower for keyword in ["研报", "研究报告", "深度"]):
            if sum(1 for kw in broker_keywords if kw in content) >= 2:
                return DocumentType.BROKER_REPORT

        # 检查内容
        broker_score = sum(1 for kw in broker_keywords if kw in content)
        financial_score = sum(1 for kw in financial_report_keywords if kw in content)
        market_score = sum(1 for kw in market_research_keywords if kw in content)
        company_score = sum(1 for kw in company_report_keywords if kw in content)

        # 得分最高的类型
        scores = {
            DocumentType.BROKER_REPORT: broker_score,
            DocumentType.FINANCIAL_REPORT: financial_score,
            DocumentType.MARKET_RESEARCH: market_score,
            DocumentType.COMPANY_REPORT: company_score,
        }

        max_score = max(scores.values())
        if max_score >= 2:
            return max(scores, key=scores.get)

        return DocumentType.GENERAL

    def get_adaptive_chunk_size(
        self,
        document_type: DocumentType
    ) -> Tuple[int, int]:
        """
        获取自适应chunk大小

        Args:
            document_type: 文档类型

        Returns:
            (chunk_size, chunk_overlap) 元组
        """
        config_map = {
            DocumentType.BROKER_REPORT: (600, 75),  # 研报较长，需要保持章节完整
            DocumentType.FINANCIAL_REPORT: (500, 50),  # 财报需要精确数据
            DocumentType.MARKET_RESEARCH: (700, 100),  # 行业研究需要更多上下文
            DocumentType.COMPANY_REPORT: (600, 75),
            DocumentType.GENERAL: (512, 50),
        }

        return config_map.get(document_type, (512, 50))

    def extract_financial_entities(
        self,
        text: str
    ) -> Dict[str, List[str]]:
        """
        提取金融实体

        Args:
            text: 输入文本

        Returns:
            实体类型到实体列表的映射
        """
        entities = {}

        for entity_type, patterns in self.compiled_patterns.items():
            entities[entity_type] = []

            for pattern in patterns:
                matches = pattern.findall(text)
                entities[entity_type].extend(matches)

            # 去重
            entities[entity_type] = list(set(entities[entity_type]))

        return entities

    def identify_section_type(
        self,
        section_title: str
    ) -> Optional[str]:
        """
        识别章节类型

        Args:
            section_title: 章节标题

        Returns:
            章节类型或None
        """
        title_lower = section_title.lower()

        for section_type, keywords in self.config.report_sections.items():
            if any(keyword in title_lower for keyword in keywords):
                return section_type

        return None

    def build_enhanced_chunk(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        构建增强的chunk文本

        增强：
        1. 添加文档类型前缀
        2. 添加券商名称
        3. 添加章节上下文
        4. 突出金融实体

        Args:
            text: 原始文本
            metadata: 元数据

        Returns:
            增强后的文本
        """
        parts = []

        # 1. 文档类型前缀
        doc_type = metadata.get("document_type", "")
        if doc_type:
            type_map = {
                "broker_report": "[券商研报]",
                "financial_report": "[财务报告]",
                "market_research": "[行业研究]",
                "company_report": "[公司报告]",
            }
            parts.append(type_map.get(doc_type, ""))

        # 2. 券商名称
        broker = metadata.get("broker", "")
        if broker:
            parts.append(f"[{broker}]")

        # 3. 章节上下文
        section = metadata.get("section", "")
        if section:
            section_type = self.identify_section_type(section)
            if section_type:
                parts.append(f"[{section}]")

        title = metadata.get("title", "")
        if title:
            parts.append(f"[{title}]")

        # 4. 日期信息
        date = metadata.get("date", "")
        if date:
            parts.append(f"[{date}]")

        # 5. 提取并标记金融实体
        entities = self.extract_financial_entities(text)
        entity_markers = []

        for entity_type, entity_list in entities.items():
            if entity_list:
                # 取前3个实体作为示例
                sample = entity_list[:3]
                entity_markers.append(f"{entity_type}:{','.join(sample)}")

        if entity_markers:
            parts.append(f"[金融实体:{';'.join(entity_markers)}]")

        # 6. 表格特殊标记
        content_type = metadata.get("content_type", "")
        if content_type == "table":
            parts.append("[表格数据]")

        # 组装
        prefix = " ".join(filter(None, parts))

        if prefix:
            return f"{prefix}\n\n{text}"
        else:
            return text

    def enhance_metadata_for_broker_report(
        self,
        metadata: Dict[str, Any],
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        为券商研报增强元数据

        Args:
            metadata: 原始元数据
            filename: 文件名

        Returns:
            增强后的元数据
        """
        enhanced = metadata.copy()

        # 从文件名提取券商名称
        if filename:
            for broker in self.config.brokers:
                if broker in filename:
                    enhanced["broker"] = broker
                    break

        # 检测报告类型
        if "评级" in filename or "rating" in filename.lower():
            enhanced["report_type"] = "rating_report"
        elif "深度" in filename or "deep" in filename.lower():
            enhanced["report_type"] = "deep_research"
        elif "点评" in filename or "comment" in filename.lower():
            enhanced["report_type"] = "event_comment"
        else:
            enhanced["report_type"] = "general"

        # 添加文档类型
        doc_type = self.detect_document_type(filename, metadata.get("content_preview", ""))
        enhanced["document_type"] = doc_type.value

        # 添加建议的自适应chunk参数
        chunk_size, chunk_overlap = self.get_adaptive_chunk_size(doc_type)
        enhanced["adaptive_chunk_size"] = chunk_size
        enhanced["adaptive_chunk_overlap"] = chunk_overlap

        return enhanced


# 便捷函数
def create_embedding_strategy() -> OptimizedEmbeddingStrategy:
    """创建优化的embedding策略"""
    return OptimizedEmbeddingStrategy()
