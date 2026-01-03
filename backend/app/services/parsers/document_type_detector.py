"""
统一文档类型检测器
整合所有文档类型检测逻辑到单一模块
"""
from enum import Enum
from pathlib import Path
from typing import Optional


class DocumentType(Enum):
    """文档类型枚举"""
    FINANCIAL_REPORT = "financial_report"
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_DOC = "technical_doc"
    CONTRACT = "contract"
    SIMPLE_DOC = "simple_doc"
    UNKNOWN = "unknown"


class DocumentTypeDetector:
    """统一的文档类型检测器"""

    # 文档类型关键词映射
    TYPE_KEYWORDS = {
        DocumentType.FINANCIAL_REPORT: [
            '财务', '报表', '年报', '季报', '半年报', '财务报告',
            'financial', 'report', 'annual', 'quarterly', 'earnings'
        ],
        DocumentType.ACADEMIC_PAPER: [
            '论文', '研究', '学术', '期刊', 'research', 'paper',
            'journal', 'academic', 'thesis', 'study'
        ],
        DocumentType.CONTRACT: [
            '合同', '协议', 'agreement', 'contract', '条款', 'clause'
        ],
        DocumentType.TECHNICAL_DOC: [
            '技术', '手册', '说明', '规范', 'technical', 'manual',
            'specification', 'guide', 'tutorial'
        ]
    }

    # 文件扩展名映射
    EXTENSION_MAP = {
        '.pdf': [DocumentType.FINANCIAL_REPORT, DocumentType.ACADEMIC_PAPER, DocumentType.CONTRACT],
        '.doc': [DocumentType.SIMPLE_DOC],
        '.docx': [DocumentType.SIMPLE_DOC],
        '.txt': [DocumentType.SIMPLE_DOC],
        '.md': [DocumentType.TECHNICAL_DOC],
    }

    @classmethod
    def detect(
        cls,
        file_path: str,
        content: Optional[str] = None
    ) -> DocumentType:
        """
        检测文档类型

        Args:
            file_path: 文件路径
            content: 文件内容（可选，用于更精确的检测）

        Returns:
            DocumentType: 检测到的文档类型
        """
        filename = Path(file_path).name.lower()
        file_ext = Path(file_path).suffix.lower()

        # 1. 基于文件扩展名的快速检测
        if file_ext in cls.EXTENSION_MAP:
            possible_types = cls.EXTENSION_MAP[file_ext]

            # 如果扩展名唯一确定类型，直接返回
            if len(possible_types) == 1:
                return possible_types[0]

        # 2. 基于文件名的关键词检测
        for doc_type, keywords in cls.TYPE_KEYWORDS.items():
            if any(keyword in filename for keyword in keywords):
                return doc_type

        # 3. 基于内容的深度检测（如果提供了内容）
        if content:
            return cls._detect_from_content(content)

        # 4. 默认返回简单文档类型
        return DocumentType.SIMPLE_DOC

    @classmethod
    def _detect_from_content(cls, content: str) -> DocumentType:
        """基于内容检测文档类型"""
        content_lower = content.lower()

        # 检查财务报告特征
        financial_indicators = [
            '营业收入', '净利润', '资产负债表', '现金流量表',
            'revenue', 'net income', 'balance sheet', 'cash flow'
        ]
        if any(indicator in content_lower for indicator in financial_indicators):
            return DocumentType.FINANCIAL_REPORT

        # 检查学术论文特征
        academic_indicators = [
            '摘要', '关键词', '引言', '参考文献', 'abstract',
            'keywords', 'introduction', 'references', 'citation'
        ]
        if any(indicator in content_lower for indicator in academic_indicators):
            return DocumentType.ACADEMIC_PAPER

        # 检查合同特征
        contract_indicators = [
            '甲方', '乙方', '条款', '违约责任', 'party', 'clause', 'breach'
        ]
        if any(indicator in content_lower for indicator in contract_indicators):
            return DocumentType.CONTRACT

        return DocumentType.SIMPLE_DOC

    @classmethod
    def is_financial_report(cls, file_path: str) -> bool:
        """快速判断是否为财务报告"""
        return cls.detect(file_path) == DocumentType.FINANCIAL_REPORT

    @classmethod
    def get_confidence(cls, file_path: str, detected_type: DocumentType) -> float:
        """获取检测置信度"""
        filename = Path(file_path).name.lower()

        # 高置信度：文件名包含明确类型关键词
        for keyword in cls.TYPE_KEYWORDS.get(detected_type, []):
            if keyword in filename:
                return 0.9

        # 中置信度：扩展名匹配
        file_ext = Path(file_path).suffix.lower()
        if file_ext in cls.EXTENSION_MAP:
            if detected_type in cls.EXTENSION_MAP[file_ext]:
                return 0.7

        # 低置信度：默认类型
        return 0.5
