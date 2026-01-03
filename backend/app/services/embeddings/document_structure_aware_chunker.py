"""
文档结构感知的分块器
基于券商研报的章节结构进行智能分块
"""
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class Section:
    """文档章节"""
    title: str
    level: int  # 1=一级标题, 2=二级标题, etc.
    content: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = None


class DocumentStructureAwareChunker:
    """文档结构感知的分块器"""

    def __init__(self):
        # 章节标题模式
        self.section_patterns = {
            1: [  # 一级标题
                r'^#{1,2}\s+(.+)$',  # Markdown #
                r'^[一二三四五六七八九十]+[、.]\s*(.+)$',  # 中文数字
                r'^\d+[、.]\s*(.+)$',  # 阿拉伯数字
            ],
            2: [  # 二级标题
                r'^#{3,4}\s+(.+)$',
                r'^[（(]\d+[)）]\s*(.+)$',
                r'^[一二三四五六七八九十]+[、.]\s*[^一二三四五六七八九十]{3,}$',
            ],
            3: [  # 三级标题
                r'^#{5,6}\s+(.+)$',
                r'^\d+\.\d+\s*(.+)$',
            ]
        }

        # 研报特有章节
        self.report_sections = {
            '投资建议': ['投资建议', '配置建议', '投资策略', '建议'],
            '风险提示': ['风险提示', '风险因素', '风险', '风险警示'],
            '财务数据': ['财务', '业绩', '盈利', '营收', '利润'],
            '行业分析': ['行业分析', '行业', '产业链', '市场'],
            '公司分析': ['公司', '企业', '标的'],
            '估值分析': ['估值', '定价', '目标价', 'PE', 'PB'],
            '技术分析': ['技术', '工艺', '产品', '技术路线'],
            '市场表现': ['市场表现', '股价', '涨跌幅', '走势'],
        }

    def parse_document_structure(
        self,
        text: str
    ) -> List[Section]:
        """
        解析文档结构

        Args:
            text: 文档全文

        Returns:
            章节列表
        """
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        current_pos = 0

        for i, line in enumerate(lines):
            line = line.rstrip()
            line_pos = current_pos
            current_pos += len(line) + 1  # +1 for newline

            # 检测章节标题
            level, title = self._detect_section_level(line)
            if level and title:
                # 保存当前章节
                if current_section:
                    current_section.end_pos = line_pos
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                # 创建新章节
                current_section = Section(
                    title=title,
                    level=level,
                    content="",
                    start_pos=line_pos,
                    end_pos=0,
                    metadata=self._extract_section_metadata(title, level)
                )
                current_content = []
            else:
                # 累积内容
                if current_section:
                    current_content.append(line)
                else:
                    # 文档开头的内容（标题之前）
                    if line.strip():
                        # 创建默认章节
                        current_section = Section(
                            title="文档概述",
                            level=0,
                            content="",
                            start_pos=0,
                            end_pos=0,
                            metadata={"type": "overview"}
                        )
                        current_content.append(line)

        # 保存最后一个章节
        if current_section:
            current_section.end_pos = current_pos
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        logger.info(f"解析文档结构，发现 {len(sections)} 个章节")
        return sections

    def _detect_section_level(self, line: str) -> Tuple[int, str]:
        """
        检测章节级别和标题

        Returns:
            (level, title) 或 (None, None)
        """
        line = line.strip()

        # 按级别检查
        for level in [1, 2, 3]:
            for pattern in self.section_patterns[level]:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    title = match.group(1).strip()
                    if len(title) > 2:  # 有意义的标题
                        return level, title

        return None, None

    def _extract_section_metadata(
        self,
        title: str,
        level: int
    ) -> Dict[str, Any]:
        """提取章节元数据"""
        metadata = {
            "level": level,
            "type": "general"
        }

        # 检测章节类型
        for section_type, keywords in self.report_sections.items():
            for keyword in keywords:
                if keyword in title:
                    metadata["type"] = section_type
                    metadata["keyword"] = keyword
                    return metadata

        return metadata

    def create_structure_aware_chunks(
        self,
        sections: List[Section],
        max_chunk_size: int = 500,
        min_chunk_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        基于结构创建chunks

        Args:
            sections: 章节列表
            max_chunk_size: 最大chunk大小
            min_chunk_size: 最小chunk大小

        Returns:
            chunk列表
        """
        chunks = []
        chunk_id = 0

        for section in sections:
            # 章节标题单独作为一个chunk
            title_chunk = {
                "content": f"【章节标题】{section.title}",
                "metadata": {
                    "chunk_type": "section_title",
                    "section_level": section.level,
                    "section_type": section.metadata.get("type", "general")
                },
                "chunk_id": f"section_title_{chunk_id}"
            }
            chunks.append(title_chunk)
            chunk_id += 1

            # 章节概述（第一段）
            first_para = self._extract_first_paragraph(section.content)
            if first_para and len(first_para) > min_chunk_size:
                summary_chunk = {
                    "content": f"【{section.title}概述】{first_para}",
                    "metadata": {
                        "chunk_type": "section_summary",
                        "section_title": section.title,
                        "section_level": section.level
                    },
                    "chunk_id": f"section_summary_{chunk_id}"
                }
                chunks.append(summary_chunk)
                chunk_id += 1

            # 章节内容分块
            content_chunks = self._split_section_content(
                section.content,
                section.title,
                max_chunk_size,
                min_chunk_size
            )
            chunks.extend(content_chunks)
            chunk_id += len(content_chunks)

        logger.info(f"基于文档结构创建了 {len(chunks)} 个chunks")
        return chunks

    def _extract_first_paragraph(self, content: str) -> str:
        """提取第一段"""
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # 有意义的第一段
                return para
        return ""

    def _split_section_content(
        self,
        content: str,
        section_title: str,
        max_size: int,
        min_size: int
    ) -> List[Dict[str, Any]]:
        """分割章节内容"""
        chunks = []

        # 按段落分割
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果单个段落超过max_size，需要分割
            if len(para) > max_size:
                if current_chunk:
                    chunks.append(self._create_content_chunk(
                        current_chunk, section_title, len(chunks)
                    ))
                    current_chunk = ""

                # 分割长段落
                sub_chunks = self._split_long_paragraph(para, max_size)
                chunks.extend(sub_chunks)
            else:
                # 累积段落
                if len(current_chunk) + len(para) > max_size:
                    if current_chunk:
                        chunks.append(self._create_content_chunk(
                            current_chunk, section_title, len(chunks)
                        ))
                        current_chunk = ""
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

        # 最后一个chunk
        if current_chunk:
            chunks.append(self._create_content_chunk(
                current_chunk, section_title, len(chunks)
            ))

        return chunks

    def _split_long_paragraph(
        self,
        text: str,
        max_size: int
    ) -> List[Dict[str, Any]]:
        """分割长段落"""
        chunks = []
        sentences = re.split(r'[。！？；\n]', text)
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > max_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "metadata": {"chunk_type": "content_part"},
                        "chunk_id": f"long_para_{len(chunks)}"
                    })
                    current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append({
                "content": current_chunk,
                "metadata": {"chunk_type": "content_part"},
                "chunk_id": f"long_para_{len(chunks)}"
            })

        return chunks

    def _create_content_chunk(
        self,
        content: str,
        section_title: str,
        index: int
    ) -> Dict[str, Any]:
        """创建内容chunk"""
        return {
            "content": content,
            "metadata": {
                "chunk_type": "section_content",
                "section_title": section_title,
                "chunk_index": index
            },
            "chunk_id": f"content_{index}"
        }


# 全局实例
document_structure_aware_chunker = DocumentStructureAwareChunker()
