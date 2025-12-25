"""
金融文档智能切分器
基于金融研报章节结构的专业切分策略
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import jieba
import logging

from .document_enhancer import financial_document_enhancer

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """文档片段"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    section: str
    sub_section: str = ""
    content_type: str = "text"
    importance_score: float = 1.0
    word_count: int = 0
    entities: List[Dict] = field(default_factory=list)
    financial_metrics: List[Dict] = field(default_factory=list)
    position: Dict[int, int] = field(default_factory=dict)


class FinancialDocumentChunker:
    """金融文档智能切分器"""

    def __init__(self):
        # 金融研报标准章节
        self.standard_sections = [
            ("投资要点", "investment_highlights", 0.9),
            ("核心观点", "core_viewpoint", 0.85),
            ("公司概况", "company_overview", 0.7),
            ("行业分析", "industry_analysis", 0.8),
            ("财务分析", "financial_analysis", 0.95),
            ("盈利预测", "earnings_forecast", 0.9),
            ("估值分析", "valuation_analysis", 0.85),
            ("风险提示", "risk_warning", 0.75),
            ("投资建议", "investment_recommendation", 0.9),
        ]

        # 章节权重映射
        self.section_weights = {eng: weight for _, eng, weight in self.standard_sections}

        # 不同章节的chunking策略
        self.chunking_strategies = {
            "financial_analysis": self._chunk_financial_analysis,
            "default": self._chunk_default,
        }

        # 默认参数
        self.default_chunk_size = 500
        self.max_chunk_size = 800
        self.chunk_overlap = 50

    def chunk_document(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """切分金融文档"""
        try:
            # 1. 文档增强处理
            enhanced_doc = financial_document_enhancer.enhance_document(text, metadata)

            # 2. 识别章节
            sections = self._identify_sections(enhanced_doc["normalized_text"])

            # 3. 按章节切分
            all_chunks = []
            for section_name, section_content, start_idx in sections:
                section_chunks = self._chunk_section(
                    section_content,
                    section_name,
                    enhanced_doc,
                    start_idx
                )
                all_chunks.extend(section_chunks)

            # 4. 为每个chunk生成唯一ID和丰富元数据
            final_chunks = []
            for i, chunk in enumerate(all_chunks):
                enriched_chunk = self._enrich_chunk_metadata(
                    chunk, i, enhanced_doc, len(all_chunks)
                )
                final_chunks.append(enriched_chunk)

            logger.info(f"文档切分完成，生成 {len(final_chunks)} 个片段")
            return final_chunks

        except Exception as e:
            logger.error(f"文档切分失败: {str(e)}")
            # 返回默认切分结果
            return self._fallback_chunking(text, metadata)

    def _identify_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """识别文档章节"""
        sections = []

        lines = text.split('\n')
        current_section = ("header", "header", 0.5)
        current_content = []
        current_start = 0

        for line_num, line in enumerate(lines):
            # 检查是否是章节标题
            detected_section = None
            for chinese_name, english_name, weight in self.standard_sections:
                # 多种可能的标题格式
                patterns = [
                    rf"^{chinese_name}[：:]\s*$",
                    rf"【{chinese_name}】\s*$",
                    rf"^§\s*{chinese_name}\s*$",
                    rf"^\d+[\.、]\s*{chinese_name}\s*$",
                ]

                for pattern in patterns:
                    if re.match(pattern, line.strip()):
                        detected_section = (chinese_name, english_name, weight)
                        break

                if detected_section:
                    break

            if detected_section and current_content:
                # 保存当前章节
                section_text = '\n'.join(current_content)
                sections.append((
                    detected_section[1],  # english_name
                    section_text,
                    current_start
                ))

                # 开始新章节
                current_section = detected_section
                current_content = [line]
                current_start = sum(len(l) + 1 for l in lines[:line_num])
            else:
                current_content.append(line)

        # 添加最后一个章节
        if current_content:
            section_text = '\n'.join(current_content)
            sections.append((
                current_section[1],
                section_text,
                current_start
            ))

        return sections

    def _chunk_section(self, content: str, section_name: str,
                      enhanced_doc: Dict, start_idx: int) -> List[DocumentChunk]:
        """切分单个章节"""
        # 选择切分策略
        chunk_strategy = self.chunking_strategies.get(
            section_name,
            self.chunking_strategies["default"]
        )

        # 执行切分
        section_chunks = chunk_strategy(content, section_name, enhanced_doc)

        # 添加全局偏移量
        for chunk in section_chunks:
            if hasattr(chunk, 'local_start'):
                chunk.position["global_start"] = start_idx + chunk.local_start
                chunk.position["global_end"] = start_idx + chunk.local_end
            else:
                chunk.position = {"global_start": start_idx, "global_end": start_idx + len(content)}

        return section_chunks

    def _chunk_financial_analysis(self, content: str, section_name: str,
                                 enhanced_doc: Dict) -> List[DocumentChunk]:
        """财务分析章节的特殊切分"""
        chunks = []

        # 识别财务分析子章节
        sub_sections = self._identify_financial_sub_sections(content)

        for sub_section_name, sub_content in sub_sections:
            # 按段落切分，但保持表格完整
            paragraphs = re.split(r'\n\s*\n', sub_content)

            current_chunk = []
            current_length = 0

            for para in paragraphs:
                para_length = len(para)

                # 判断是否是表格
                is_table = self._is_likely_table(para)

                if is_table:
                    # 表格单独作为一个chunk
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            '\n'.join(current_chunk),
                            section_name,
                            sub_section_name,
                            "table",
                            enhanced_doc
                        ))
                        current_chunk = []
                        current_length = 0

                    # 表格chunk
                    chunks.append(self._create_chunk(
                        para,
                        section_name,
                        sub_section_name,
                        "table",
                        enhanced_doc
                    ))
                elif current_length + para_length > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            '\n'.join(current_chunk),
                            section_name,
                            sub_section_name,
                            "text",
                            enhanced_doc
                        ))
                    current_chunk = [para]
                    current_length = para_length
                else:
                    current_chunk.append(para)
                    current_length += para_length

            # 处理最后一个chunk
            if current_chunk:
                chunks.append(self._create_chunk(
                    '\n'.join(current_chunk),
                    section_name,
                    sub_section_name,
                    "text",
                    enhanced_doc
                ))

        return chunks

    def _identify_financial_sub_sections(self, content: str) -> List[Tuple[str, str]]:
        """识别财务分析子章节"""
        sub_sections = [
            ("盈利能力分析", ["盈利能力", "利润分析", "盈利状况"]),
            ("偿债能力分析", ["偿债能力", "负债分析", "财务杠杆"]),
            ("运营能力分析", ["运营能力", "资产周转", "运营效率"]),
            ("成长能力分析", ["成长能力", "增长分析", "发展前景"]),
            ("现金流分析", ["现金流", "现金流量", "现金状况"]),
        ]

        identified = []
        lines = content.split('\n')

        for sub_name, keywords in sub_sections:
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in keywords):
                    # 找到子章节开始
                    start_idx = i
                    # 寻找子章节结束
                    end_idx = len(lines)

                    for j in range(i + 1, len(lines)):
                        # 检查是否是其他子章节的开始
                        is_other_sub = False
                        for other_name, other_keywords in sub_sections:
                            if other_name != sub_name:
                                if any(kw in lines[j] for kw in other_keywords):
                                    is_other_sub = True
                                    break

                        if is_other_sub or j == len(lines) - 1:
                            end_idx = j
                            break

                    sub_content = '\n'.join(lines[start_idx:end_idx])
                    identified.append((sub_name, sub_content))
                    break

        return identified

    def _chunk_default(self, content: str, section_name: str,
                      enhanced_doc: Dict) -> List[DocumentChunk]:
        """默认切分策略"""
        chunks = []

        # 按句子分割
        sentences = re.split(r'[。！？；\n]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.default_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        '。'.join(current_chunk),
                        section_name,
                        "",
                        "text",
                        enhanced_doc
                    ))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # 处理最后一个chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                '。'.join(current_chunk),
                section_name,
                "",
                "text",
                enhanced_doc
            ))

        return chunks

    def _is_likely_table(self, text: str) -> bool:
        """判断文本是否是表格"""
        # 简单的表格识别规则
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False

        # 检查是否包含数字和单位
        has_numbers = bool(re.search(r'\d+\.?\d*', text))
        has_units = any(unit in text for unit in ['%', '亿元', '万元', '倍', '元'])

        # 检查列对齐
        if len(lines) > 2:
            # 计算每行的分隔符数量
            separators_count = [line.count('|') + line.count('\t') for line in lines[:3]]
            if all(count > 0 for count in separators_count):
                return True

        return has_numbers and has_units and len(lines) >= 3

    def _create_chunk(self, text: str, section: str, sub_section: str,
                      content_type: str, enhanced_doc: Dict) -> DocumentChunk:
        """创建文档片段"""
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:16]

        # 提取chunk中的实体和指标
        chunk_entities = []
        chunk_metrics = []

        # 从文档级别的实体和指标中筛选
        for entity in enhanced_doc.get("entities", []):
            if entity.start_pos >= 0 and entity.end_pos <= len(text):
                chunk_entities.append({
                    "text": entity.text,
                    "type": entity.type,
                    "confidence": entity.confidence
                })

        for metric in enhanced_doc.get("financial_metrics", []):
            if metric["position"][0] >= 0 and metric["position"][1] <= len(text):
                chunk_metrics.append(metric)

        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata={
                "document_hash": enhanced_doc.get("document_hash"),
                "section": section,
                "sub_section": sub_section,
                "content_type": content_type,
                "word_count": len(jieba.lcut(text)),
                "has_financial_data": len(chunk_metrics) > 0,
                "has_company_mention": any(e["type"] == "COMPANY" for e in chunk_entities),
            },
            section=section,
            sub_section=sub_section,
            content_type=content_type,
            importance_score=self.section_weights.get(section, 0.5),
            word_count=len(jieba.lcut(text)),
            entities=chunk_entities,
            financial_metrics=chunk_metrics
        )

    def _enrich_chunk_metadata(self, chunk: DocumentChunk, chunk_index: int,
                              enhanced_doc: Dict, total_chunks: int) -> DocumentChunk:
        """丰富chunk元数据"""
        # 更新chunk ID
        chunk.chunk_id = f"{enhanced_doc.get('document_hash', 'unknown')}_{chunk_index:04d}"

        # 添加额外元数据
        chunk.metadata.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "processed_at": datetime.now().isoformat(),
            "chunking_version": "1.0",
        })

        # 继承文档元数据
        if enhanced_doc.get("metadata"):
            chunk.metadata.update({
                k: v for k, v in enhanced_doc["metadata"].items()
                if k in ["report_type", "company", "stock_code",
                        "analyst", "institution", "publish_date"]
            })

        return chunk

    def _fallback_chunking(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """备用切分策略"""
        # 简单按固定长度切分
        chunks = []
        chunk_size = self.default_chunk_size
        overlap = self.chunk_overlap

        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    metadata={
                        "section": "unknown",
                        "content_type": "text",
                        "word_count": len(jieba.lcut(chunk_text)),
                        "fallback_chunking": True,
                        "processed_at": datetime.now().isoformat()
                    },
                    section="unknown",
                    content_type="text"
                ))

        return chunks


# 全局实例
financial_document_chunker = FinancialDocumentChunker()