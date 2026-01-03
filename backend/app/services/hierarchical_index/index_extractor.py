"""
分层索引抽取服务
从文档中抽取三层索引：文档摘要、章节、片段
"""

import re
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.structured_logging import get_structured_logger
from app.schemas.hierarchical_index import (
    DocumentSummaryIndex,
    ChapterIndex,
    ChunkIndex,
    HierarchicalIndex,
    ChunkType
)
from app.services.chunking.smart_chunking_service import SmartChunkingService, ChunkingStrategy

logger = get_structured_logger(__name__)


class HierarchicalIndexExtractor:
    """
    分层索引抽取器

    功能：
    1. 抽取文档摘要索引（整体摘要、关键词、实体、主题）
    2. 抽取章节索引（层级结构、章节摘要）
    3. 抽取片段索引（智能分块、保持上下文）
    """

    def __init__(self):
        """初始化抽取器"""
        self.chunking_service = SmartChunkingService(
            default_strategy=ChunkingStrategy.INTELLIGENT
        )

        # 章节标题模式（支持多种格式）
        self.section_patterns = [
            r'^#+\s+(.+)$',  # Markdown标题
            r'^第[一二三四五六七八九十百]+章\s+(.+)$',  # 第X章
            r'^第\d+章\s+(.+)$',  # 第1章
            r'^\d+\.\s+(.+)$',  # 1. 标题
            r'^[一二三四五六七八九十百]+、\s*(.+)$',  # 一、标题
            r'^[（(]\s*[一二三四五六七八九十百]+\s*[)）]\s*(.+)$',  # （一）标题
        ]

        logger.info("分层索引抽取器初始化完成")

    async def extract_hierarchical_index(
        self,
        document_id: str,
        markdown_content: str,
        deepseek_summary: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> HierarchicalIndex:
        """
        抽取完整的分层索引

        Args:
            document_id: 文档ID
            markdown_content: Markdown格式的文档内容
            deepseek_summary: Deepseek深度汇总结果
            **kwargs: 额外参数

        Returns:
            HierarchicalIndex: 分层索引结构
        """
        start_time = datetime.now()

        logger.info(f"📚 开始抽取文档 {document_id} 的分层索引")

        try:
            # 1. 抽取文档摘要索引
            logger.info("  📝 抽取文档摘要索引...")
            document_summary = await self._extract_document_summary(
                document_id,
                markdown_content,
                deepseek_summary
            )

            # 2. 抽取章节索引
            logger.info("  📑 抽取章节索引...")
            chapters = await self._extract_chapters(
                document_id,
                markdown_content
            )

            # 3. 抽取片段索引
            logger.info("  ✂️ 抽取片段索引...")
            chunks = await self._extract_chunks(
                document_id,
                markdown_content,
                chapters
            )

            # 4. 建立章节和片段的关联关系
            logger.info("  🔗 建立关联关系...")
            self._link_chunks_to_chapters(chunks, chapters)

            # 5. 构建完整的分层索引
            hierarchical_index = HierarchicalIndex(
                document_id=document_id,
                document_summary=document_summary,
                chapters=chapters,
                chunks=chunks,
                total_chapters=len(chapters),
                total_chunks=len(chunks),
                created_at=datetime.now(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            logger.info(
                f"✅ 分层索引抽取完成！"
                f"摘要: 1, 章节: {len(chapters)}, 片段: {len(chunks)}, "
                f"耗时: {hierarchical_index.processing_time:.2f}秒"
            )

            return hierarchical_index

        except Exception as e:
            logger.error(f"❌ 分层索引抽取失败: {str(e)}", exc_info=True)
            raise

    async def _extract_document_summary(
        self,
        document_id: str,
        markdown_content: str,
        deepseek_summary: Optional[Dict[str, Any]] = None
    ) -> DocumentSummaryIndex:
        """
        抽取文档摘要索引

        Args:
            document_id: 文档ID
            markdown_content: 文档内容
            deepseek_summary: Deepseek汇总结果

        Returns:
            DocumentSummaryIndex: 文档摘要索引
        """
        # 提取关键词
        keywords = self._extract_keywords_from_text(markdown_content, top_k=20)

        # 提取实体（简单版本：识别常见的金融实体）
        entities = self._extract_entities(markdown_content)

        # 提取主题
        topics = self._extract_topics(markdown_content)

        # 生成摘要（优先使用Deepseek汇总，否则使用规则生成）
        if deepseek_summary and deepseek_summary.get("status") == "success":
            summary_text = self._generate_summary_from_deepseek(deepseek_summary)
        else:
            summary_text = self._generate_summary_by_rules(markdown_content)

        # 统计信息
        doc_length = len(markdown_content)
        section_count = len(self._extract_sections_from_markdown(markdown_content))

        return DocumentSummaryIndex(
            document_id=document_id,
            summary_text=summary_text,
            keywords=keywords,
            entities=entities,
            topics=topics,
            metadata={
                "source": "hierarchical_extractor",
                "has_deepseek_summary": deepseek_summary is not None
            },
            doc_length=doc_length,
            section_count=section_count,
            chunk_count=0,  # 稍后更新
            created_at=datetime.now()
        )

    async def _extract_chapters(
        self,
        document_id: str,
        markdown_content: str
    ) -> List[ChapterIndex]:
        """
        抽取章节索引

        Args:
            document_id: 文档ID
            markdown_content: 文档内容

        Returns:
            List[ChapterIndex]: 章节索引列表
        """
        sections = self._extract_sections_from_markdown(markdown_content)
        chapters = []

        for idx, section in enumerate(sections):
            chapter_id = f"{document_id}_ch_{idx:03d}"

            # 生成章节摘要
            summary = self._generate_section_summary(section["content"])

            # 提取章节关键词
            keywords = self._extract_keywords_from_text(
                section["content"],
                top_k=5
            )

            chapter = ChapterIndex(
                chapter_id=chapter_id,
                document_id=document_id,
                title=section["title"],
                level=section["level"],
                summary=summary,
                keywords=keywords,
                parent_chapter_id=None,  # 稍后计算
                child_chapter_ids=[],
                start_char=section.get("start_char", 0),
                end_char=section.get("end_char", 0),
                chunk_count=0,  # 稍后更新
                created_at=datetime.now()
            )

            chapters.append(chapter)

        # 计算层级关系
        self._calculate_chapter_hierarchy(chapters)

        return chapters

    async def _extract_chunks(
        self,
        document_id: str,
        markdown_content: str,
        chapters: List[ChapterIndex]
    ) -> List[ChunkIndex]:
        """
        抽取片段索引

        Args:
            document_id: 文档ID
            markdown_content: 文档内容
            chapters: 章节列表

        Returns:
            List[ChunkIndex]: 片段索引列表
        """
        # 使用智能分块服务
        chunk_results = self.chunking_service.chunk_document(
            text=markdown_content,
            strategy=ChunkingStrategy.INTELLIGENT,
            target_chunk_size=800,
            max_chunk_size=1500
        )

        chunks = []

        for chunk_result in chunk_results:
            chunk_id = f"{document_id}_chk_{chunk_result['index']:03d}"

            # 确定片段类型
            chunk_type = self._determine_chunk_type(chunk_result)

            # 查找所属章节
            chapter_id = self._find_chapter_for_chunk(
                chunk_result,
                chapters
            )

            chunk = ChunkIndex(
                chunk_id=chunk_id,
                document_id=document_id,
                chapter_id=chapter_id,
                content=chunk_result["text"],
                chunk_type=chunk_type,
                chunk_index=chunk_result["index"],
                start_char=chunk_result.get("start_pos", 0),
                end_char=chunk_result.get("end_pos", 0),
                metadata=chunk_result.get("metadata", {}),
                created_at=datetime.now()
            )

            chunks.append(chunk)

        return chunks

    def _extract_sections_from_markdown(
        self,
        markdown_content: str
    ) -> List[Dict[str, Any]]:
        """
        从Markdown中提取章节结构

        Args:
            markdown_content: Markdown内容

        Returns:
            List[Dict]: 章节列表
        """
        sections = []
        lines = markdown_content.split('\n')

        current_section = {
            "title": "概述",
            "level": 0,
            "content": [],
            "start_char": 0
        }

        current_char_pos = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline

            # 检测标题
            is_title = False
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # 保存上一个章节
                    if current_section["content"]:
                        current_section["content"] = '\n'.join(current_section["content"])
                        current_section["end_char"] = current_char_pos
                        sections.append(current_section)

                    # 创建新章节
                    level = self._get_title_level(line)
                    title = match.group(1) if match.groups() else line.strip('#').strip()

                    current_section = {
                        "title": title,
                        "level": level,
                        "content": [],
                        "start_char": current_char_pos
                    }
                    is_title = True
                    break

            if not is_title:
                current_section["content"].append(line)

            current_char_pos += line_length

        # 保存最后一个章节
        if current_section["content"]:
            current_section["content"] = '\n'.join(current_section["content"])
            current_section["end_char"] = current_char_pos
            sections.append(current_section)

        return sections

    def _get_title_level(self, title_line: str) -> int:
        """获取标题层级"""
        if title_line.startswith('#'):
            return len(title_line) - len(title_line.lstrip('#'))
        elif '第' in title_line and '章' in title_line:
            return 1
        elif re.match(r'^\d+\.\s+', title_line):
            return 2
        elif re.match(r'^[一二三四五六七八九十百]+、', title_line):
            return 2
        elif re.match(r'^[（(]\s*[一二三四五六七八九十百]+\s*[)）]', title_line):
            return 3
        return 1

    def _generate_section_summary(self, content: str) -> str:
        """生成章节摘要（前200字）"""
        # 简单版本：取前面的内容
        clean_content = re.sub(r'\s+', ' ', content.strip())
        if len(clean_content) > 200:
            return clean_content[:200] + "..."
        return clean_content

    def _extract_keywords_from_text(
        self,
        text: str,
        top_k: int = 10
    ) -> List[str]:
        """
        从文本中提取关键词

        Args:
            text: 文本内容
            top_k: 返回前K个关键词

        Returns:
            List[str]: 关键词列表
        """
        from collections import Counter

        # 提取中文词组（2-4个字）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)

        # 过滤常见停用词
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '但', '等', '及', '对'}
        filtered_words = [w for w in chinese_words if w not in stopwords]

        # 统计词频
        word_freq = Counter(filtered_words)

        # 返回高频词
        keywords = [word for word, freq in word_freq.most_common(top_k)
                   if freq >= 2]

        return keywords

    def _extract_entities(self, text: str) -> List[str]:
        """
        提取实体（简单版本）

        识别常见的金融实体：
        - 公司名（XXX公司、XXX集团）
        - 股票代码（000001）
        - 人名模式
        """
        entities = []

        # 提取公司名
        company_pattern = r'([\u4e00-\u9fa5]{2,6})(公司|集团|股份有限公司|有限公司)'
        companies = re.findall(company_pattern, text)
        for company in companies:
            entities.append(company[0] + company[1])

        # 提取股票代码
        stock_pattern = r'\d{6}'
        stocks = re.findall(stock_pattern, text)
        entities.extend(stocks[:5])  # 最多5个股票代码

        return list(set(entities))[:20]  # 最多20个实体

    def _extract_topics(self, text: str) -> List[str]:
        """提取主题（基于关键词聚类）"""
        # 简化版本：预定义的金融主题检测
        topic_keywords = {
            "投资分析": ["投资", "收益", "回报", "风险", "资产配置"],
            "财务分析": ["营收", "利润", "现金流", "负债", "财务"],
            "行业研究": ["行业", "市场", "竞争", "趋势", "前景"],
            "公司研究": ["公司", "业务", "产品", "管理", "战略"],
            "宏观经济": ["经济", "政策", "增长", "通胀", "利率"]
        }

        matched_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                matched_topics.append(topic)

        return matched_topics

    def _generate_summary_from_deepseek(
        self,
        deepseek_summary: Dict[str, Any]
    ) -> str:
        """从Deepseek汇总生成摘要"""
        enhanced_summary = deepseek_summary.get("enhanced_summary", "")

        if enhanced_summary and len(enhanced_summary) > 50:
            # 清理格式
            summary = re.sub(r'\s+', ' ', enhanced_summary.strip())
            if len(summary) > 500:
                summary = summary[:500] + "..."
            return summary

        # 回退到规则提取
        rule_based = deepseek_summary.get("rule_based_summary", {})
        sections = rule_based.get("sections", [])
        if sections:
            return f"本文档包含{len(sections)}个主要章节，" + \
                   f"主要讨论{', '.join([s['title'] for s in sections[:3]])}"

        return "文档摘要生成中..."

    def _generate_summary_by_rules(self, markdown_content: str) -> str:
        """基于规则生成摘要"""
        sections = self._extract_sections_from_markdown(markdown_content)

        if not sections:
            return "无法生成摘要"

        # 取前3个章节
        top_sections = sections[:3]
        section_titles = [s["title"] for s in top_sections]

        summary = f"本文档共{len(sections)}个章节，"
        summary += f"主要内容包括：{', '.join(section_titles)}"

        return summary

    def _determine_chunk_type(self, chunk_result: Dict[str, Any]) -> ChunkType:
        """确定片段类型"""
        metadata = chunk_result.get("metadata", {})
        chunk_type_str = metadata.get("type", "text")

        if chunk_type_str == "table":
            return ChunkType.TABLE
        elif chunk_type_str == "list":
            return ChunkType.LIST
        elif chunk_type_str == "image":
            return ChunkType.IMAGE
        elif chunk_type_str == "mixed":
            return ChunkType.MIXED
        else:
            return ChunkType.TEXT

    def _find_chapter_for_chunk(
        self,
        chunk_result: Dict[str, Any],
        chapters: List[ChapterIndex]
    ) -> Optional[str]:
        """查找片段所属的章节"""
        start_pos = chunk_result.get("start_pos", 0)

        # 找到包含该位置的章节
        for chapter in chapters:
            if chapter.start_char <= start_pos <= chapter.end_char:
                return chapter.chapter_id

        return None

    def _calculate_chapter_hierarchy(self, chapters: List[ChapterIndex]):
        """计算章节的层级关系"""
        for i, current_chapter in enumerate(chapters):
            # 查找父章节
            for j in range(i - 1, -1, -1):
                prev_chapter = chapters[j]
                if prev_chapter.level < current_chapter.level:
                    current_chapter.parent_chapter_id = prev_chapter.chapter_id
                    prev_chapter.child_chapter_ids.append(current_chapter.chapter_id)
                    break

    def _link_chunks_to_chapters(
        self,
        chunks: List[ChunkIndex],
        chapters: List[ChapterIndex]
    ):
        """建立片段和章节的关联关系"""
        # 统计每个章节的片段数量
        chapter_chunk_count = {}

        for chunk in chunks:
            if chunk.chapter_id:
                chapter_chunk_count[chunk.chapter_id] = \
                    chapter_chunk_count.get(chunk.chapter_id, 0) + 1

        # 更新章节的片段计数
        for chapter in chapters:
            chapter.chunk_count = chapter_chunk_count.get(chapter.chapter_id, 0)

        # 更新文档摘要的片段总数
        # （注意：这需要在外部调用时设置）


# 全局单例
_index_extractor = None


def get_hierarchical_index_extractor() -> HierarchicalIndexExtractor:
    """获取分层索引抽取器单例"""
    global _index_extractor
    if _index_extractor is None:
        _index_extractor = HierarchicalIndexExtractor()
    return _index_extractor
