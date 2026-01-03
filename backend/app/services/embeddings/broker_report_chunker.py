"""
券商研报专用分块器
基于金融知识增强的chunking策略
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class BrokerReportSectionType(str, Enum):
    """券商研报章节类型"""
    OVERVIEW = "overview"  # 报告概述
    INVESTMENT_ADVICE = "investment_advice"  # 投资建议
    RISK_WARNING = "risk_warning"  # 风险提示
    FINANCIAL_DATA = "financial_data"  # 财务数据
    INDUSTRY_ANALYSIS = "industry_analysis"  # 行业分析
    COMPANY_ANALYSIS = "company_analysis"  # 公司分析
    VALUATION = "valuation"  # 估值分析
    MARKET_PERFORMANCE = "market_performance"  # 市场表现
    TECHNICAL_ANALYSIS = "technical_analysis"  # 技术分析
    GENERAL = "general"  # 通用内容


@dataclass
class BrokerReportChunk:
    """券商研报chunk"""
    content: str  # chunk内容
    chunk_id: str  # chunk唯一标识
    section_type: BrokerReportSectionType  # 章节类型
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    entities: List[str] = field(default_factory=list)  # 包含的实体
    metrics: List[str] = field(default_factory=list)  # 包含的指标
    priority: float = 0.5  # 优先级（用于检索排序）


@dataclass
class ChunkingResult:
    """分块结果"""
    chunks: List[BrokerReportChunk]
    stats: Dict[str, Any]


class BrokerReportChunker:
    """
    券商研报专用分块器

    特点：
    1. 识别研报典型章节结构
    2. 保持表格完整性
    3. 保持财务指标完整性
    4. 智能处理图表描述
    5. 优先级标记（投资建议、风险提示等）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_chunk_size = self.config.get("chunk_size", 600)
        self.default_overlap = self.config.get("overlap", 75)
        self._compile_section_patterns()
        self._load_financial_patterns()

    def _compile_section_patterns(self):
        """编译章节识别模式"""
        self.section_patterns = {
            BrokerReportSectionType.INVESTMENT_ADVICE: [
                r'(投资建议|配置建议|投资策略|评级建议|建议|评级)',
                r'(Investment Advice|Recommendation|Rating)',
            ],
            BrokerReportSectionType.RISK_WARNING: [
                r'(风险提示|风险因素|风险|风险警示)',
                r'(Risk Warning|Risk Factors|Risk)',
            ],
            BrokerReportSectionType.FINANCIAL_DATA: [
                r'(财务数据|财务分析|业绩|盈利|营收|利润)',
                r'(Financial Data|Financial Analysis|Earnings)',
            ],
            BrokerReportSectionType.INDUSTRY_ANALYSIS: [
                r'(行业分析|行业深度|行业研究|产业链|市场空间|赛道)',
                r'(Industry Analysis|Sector Analysis|Market Overview)',
            ],
            BrokerReportSectionType.COMPANY_ANALYSIS: [
                r'(公司分析|公司深度|公司概况|公司简介|标的)',
                r'(Company Analysis|Company Overview|Profile)',
            ],
            BrokerReportSectionType.VALUATION: [
                r'(估值分析|估值|定价|目标价|相对估值|绝对估值)',
                r'(Valuation|Pricing|Target Price|Fair Value)',
            ],
            BrokerReportSectionType.MARKET_PERFORMANCE: [
                r'(市场表现|股价表现|涨跌幅|走势|行情)',
                r'(Market Performance|Stock Price|Price Performance)',
            ],
            BrokerReportSectionType.TECHNICAL_ANALYSIS: [
                r'(技术分析|技术指标|技术面)',
                r'(Technical Analysis|Technical Indicators)',
            ],
        }

    def _load_financial_patterns(self):
        """加载金融模式"""
        # 表格模式
        self.table_patterns = [
            r'\|[^\n]+\|[^\n]*\|',  # Markdown表格
            r'(<table[^>]*>.*?</table>)',  # HTML表格
        ]

        # 财务指标模式（不应分割）
        self.financial_indicator_patterns = [
            r'(PE|PB|ROE|ROA|EPS)\s*[:：]\s*[0-9.]+',
            r'(市盈率|市净率|毛利率|净利率)\s*[:：]\s*[0-9.]+%?',
            r'(目标价)\s*[:：]\s*[0-9.]+\s*元',
        ]

        # 时间周期模式（不应分割）
        self.time_period_patterns = [
            r'(20\d{2}年[一二三四]?季度?)',
            r'([过去|近]\d+[年季度])',
        ]

    def chunk_report(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        对券商研报进行分块

        Args:
            text: 研报全文
            metadata: 元数据

        Returns:
            ChunkingResult
        """
        metadata = metadata or {}

        # 第一步：识别章节结构
        sections = self._identify_sections(text)

        # 第二步：按章节分块
        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(
                section,
                chunk_index,
                metadata
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # 第三步：后处理（去重、质量检查）
        chunks = self._post_process_chunks(chunks)

        # 统计
        stats = {
            "total_chunks": len(chunks),
            "by_section_type": {},
            "avg_chunk_size": sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
        }

        for chunk in chunks:
            section_type = chunk.section_type.value
            stats["by_section_type"][section_type] = stats["by_section_type"].get(section_type, 0) + 1

        logger.info(
            f"研报分块完成: {stats['total_chunks']} 个chunks, "
            f"平均大小: {stats['avg_chunk_size']:.0f} 字符"
        )

        return ChunkingResult(chunks=chunks, stats=stats)

    def _identify_sections(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """识别章节结构"""
        sections = []
        lines = text.split('\n')

        current_section = {
            "type": BrokerReportSectionType.OVERVIEW,
            "title": "报告概述",
            "content": [],
            "start_line": 0,
        }

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 检测章节标题
            section_type = self._detect_section_type(stripped)

            if section_type:
                # 保存当前章节
                if current_section["content"]:
                    sections.append(current_section)

                # 开始新章节
                current_section = {
                    "type": section_type,
                    "title": stripped,
                    "content": [],
                    "start_line": i,
                }
            else:
                # 累积内容
                current_section["content"].append(line)

        # 保存最后一个章节
        if current_section["content"]:
            sections.append(current_section)

        # 合并内容为字符串
        for section in sections:
            section["content"] = "\n".join(section["content"]).strip()

        return sections

    def _detect_section_type(
        self,
        line: str
    ) -> Optional[BrokerReportSectionType]:
        """检测章节类型"""
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return section_type

        # 检测标题格式（一级标题）
        if re.match(r'^#{1,2}\s', line) or re.match(r'^[一二三四五六七八九十]+[、.]\s', line):
            return BrokerReportSectionType.GENERAL

        return None

    def _chunk_section(
        self,
        section: Dict[str, Any],
        start_index: int,
        metadata: Dict[str, Any]
    ) -> List[BrokerReportChunk]:
        """对单个章节进行分块"""
        section_type = section["type"]
        content = section["content"]
        title = section["title"]

        chunks = []

        # 根据章节类型确定chunk大小
        chunk_size, overlap = self._get_adaptive_chunk_size(section_type)

        # 提取表格（保持完整）
        tables = self._extract_tables(content)

        # 移除表格，处理剩余文本
        text_without_tables = content
        for table in tables:
            text_without_tables = text_without_tables.replace(table, "")

        # 分块文本
        text_chunks = self._split_text(
            text_without_tables,
            chunk_size,
            overlap
        )

        chunk_index = start_index

        # 第一个chunk：章节标题 + 概述
        if text_chunks:
            first_chunk = BrokerReportChunk(
                content=f"【{title}】\n\n{text_chunks[0]}",
                chunk_id=f"chunk_{chunk_index}",
                section_type=section_type,
                metadata={
                    **metadata,
                    "section_title": title,
                    "is_first_chunk": True,
                },
                priority=self._calculate_priority(section_type, is_first=True),
            )
            chunks.append(first_chunk)
            chunk_index += 1

        # 其余文本chunks
        for i, text_chunk in enumerate(text_chunks[1:], 1):
            chunk = BrokerReportChunk(
                content=text_chunk,
                chunk_id=f"chunk_{chunk_index}",
                section_type=section_type,
                metadata={
                    **metadata,
                    "section_title": title,
                    "chunk_number": i + 1,
                },
                priority=self._calculate_priority(section_type),
            )
            chunks.append(chunk)
            chunk_index += 1

        # 表格chunks（保持完整）
        for table in tables:
            chunk = BrokerReportChunk(
                content=f"【表格数据】\n\n{table}",
                chunk_id=f"chunk_{chunk_index}",
                section_type=section_type,
                metadata={
                    **metadata,
                    "section_title": title,
                    "content_type": "table",
                },
                priority=self._calculate_priority(section_type) * 1.2,  # 表格优先级稍高
            )
            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def _get_adaptive_chunk_size(
        self,
        section_type: BrokerReportSectionType
    ) -> Tuple[int, int]:
        """获取自适应chunk大小"""
        # 投资建议和风险提示使用较小的chunk（重要性高）
        if section_type in [
            BrokerReportSectionType.INVESTMENT_ADVICE,
            BrokerReportSectionType.RISK_WARNING,
        ]:
            return (400, 50)

        # 财务数据和估值分析使用中等chunk
        elif section_type in [
            BrokerReportSectionType.FINANCIAL_DATA,
            BrokerReportSectionType.VALUATION,
        ]:
            return (500, 60)

        # 行业分析和公司分析可以使用较大chunk
        elif section_type in [
            BrokerReportSectionType.INDUSTRY_ANALYSIS,
            BrokerReportSectionType.COMPANY_ANALYSIS,
        ]:
            return (700, 100)

        # 默认
        return (self.default_chunk_size, self.default_overlap)

    def _calculate_priority(
        self,
        section_type: BrokerReportSectionType,
        is_first: bool = False
    ) -> float:
        """计算chunk优先级"""
        base_priority = {
            BrokerReportSectionType.INVESTMENT_ADVICE: 1.0,
            BrokerReportSectionType.RISK_WARNING: 0.95,
            BrokerReportSectionType.VALUATION: 0.85,
            BrokerReportSectionType.FINANCIAL_DATA: 0.8,
            BrokerReportSectionType.COMPANY_ANALYSIS: 0.7,
            BrokerReportSectionType.INDUSTRY_ANALYSIS: 0.65,
            BrokerReportSectionType.MARKET_PERFORMANCE: 0.6,
            BrokerReportSectionType.TECHNICAL_ANALYSIS: 0.55,
            BrokerReportSectionType.OVERVIEW: 0.75,
            BrokerReportSectionType.GENERAL: 0.5,
        }

        priority = base_priority.get(section_type, 0.5)

        # 首个chunk优先级提升
        if is_first:
            priority *= 1.1

        return min(1.0, priority)

    def _extract_tables(
        self,
        text: str
    ) -> List[str]:
        """提取表格"""
        tables = []

        for pattern in self.table_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            tables.extend(matches)

        return tables

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """分割文本"""
        chunks = []

        # 按段落分割
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检查是否包含财务指标（不分割）
            if self._contains_financial_indicator(para):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.append(para)
                continue

            # 检查是否包含时间周期（不分割）
            if self._contains_time_period(para):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.append(para)
                continue

            # 累积或创建新chunk
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)

                    # 添加重叠
                    if overlap > 0:
                        sentences = current_chunk.split('。')
                        if len(sentences) > 1:
                            overlap_text = '。'.join(sentences[-2:])
                            current_chunk = overlap_text + "。" + para
                        else:
                            current_chunk = para
                    else:
                        current_chunk = para
                else:
                    # 单个段落太长，需要分割
                    sub_chunks = self._split_long_paragraph(para, chunk_size)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # 最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)

        return [c.strip() for c in chunks if c.strip()]

    def _contains_financial_indicator(
        self,
        text: str
    ) -> bool:
        """检查是否包含财务指标"""
        for pattern in self.financial_indicator_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _contains_time_period(
        self,
        text: str
    ) -> bool:
        """检查是否包含时间周期"""
        for pattern in self.time_period_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _split_long_paragraph(
        self,
        text: str,
        max_length: int
    ) -> List[str]:
        """分割长段落"""
        chunks = []
        sentences = re.split(r'([。！？；\n])', text)

        current_chunk = ""

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')

            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _post_process_chunks(
        self,
        chunks: List[BrokerReportChunk]
    ) -> List[BrokerReportChunk]:
        """后处理chunks"""
        # 去重（基于内容相似度）
        unique_chunks = []

        seen_hashes = set()
        for chunk in chunks:
            content_hash = hash(chunk.content[:100])  # 使用前100字符的hash

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)

        # 提取实体和指标（简化版）
        for chunk in unique_chunks:
            chunk.entities = self._extract_entities_simple(chunk.content)
            chunk.metrics = self._extract_metrics_simple(chunk.content)

        return unique_chunks

    def _extract_entities_simple(
        self,
        text: str
    ) -> List[str]:
        """简单提取实体（股票代码、公司名称等）"""
        entities = []

        # 股票代码
        stock_codes = re.findall(r'\b[0-9]{6}\.[A-Z]{2}\b', text)
        entities.extend(stock_codes)

        # 公司名称（简化：提取中文大词）
        companies = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:股份)?有限公司?', text)
        entities.extend(companies[:3])  # 限制数量

        return list(set(entities))

    def _extract_metrics_simple(
        self,
        text: str
    ) -> List[str]:
        """简单提取指标"""
        metrics = []

        # 查找所有 "指标: 值" 模式
        patterns = [
            r'(PE|PB|ROE|ROA)\s*[:：]\s*[0-9.]+',
            r'(市盈率|市净率|毛利率|净利率)\s*[:：]\s*[0-9.]+%?',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            metrics.extend(matches)

        return list(set(metrics))

    async def chunk_reports_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[ChunkingResult]:
        """
        批量分块研报

        Args:
            texts: 研报文本列表
            metadata_list: 元数据列表

        Returns:
            ChunkingResult列表
        """
        results = []

        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list else None
            result = self.chunk_report(text, metadata)
            results.append(result)

        return results


# 全局实例
_broker_report_chunker: Optional[BrokerReportChunker] = None


def get_broker_report_chunker() -> BrokerReportChunker:
    """获取全局券商研报分块器"""
    global _broker_report_chunker
    if _broker_report_chunker is None:
        _broker_report_chunker = BrokerReportChunker()
    return _broker_report_chunker
