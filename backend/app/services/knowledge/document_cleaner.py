"""
文档清洗模块
去除页眉页脚、无意义文本，规范化内容

功能：
- 自动识别和去除页眉页脚
- 去除页码、版权信息等噪音
- 文本规范化（空白符、标点等）
- 去除重复内容
- 保留有意义的段落
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import jieba

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """清洗统计信息"""
    original_length: int
    cleaned_length: int
    removed_headers: int
    removed_footers: int
    removed_page_numbers: int
    removed_noise: int
    removed_duplicates: int
    final_length: int
    reduction_ratio: float


class DocumentCleaner:
    """
    文档清洗器

    用于清洗PDF解析后的文本内容，去除噪音和冗余信息
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 清洗配置
        self.remove_headers = self.config.get('remove_headers', True)
        self.remove_footers = self.config.get('remove_footers', True)
        self.remove_page_numbers = self.config.get('remove_page_numbers', True)
        self.remove_noise = self.config.get('remove_noise', True)
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)

        # 阈值配置
        self.min_line_length = self.config.get('min_line_length', 10)
        self.min_paragraph_length = self.config.get('min_paragraph_length', 20)
        self.duplicate_threshold = self.config.get('duplicate_threshold', 0.9)
        self.max_repeated_chars = self.config.get('max_repeated_chars', 3)

        # 页眉页脚模式（正则表达式）
        self._compile_patterns()

    def _compile_patterns(self):
        """编译常用的页眉页脚模式"""

        # 页码模式
        self.page_number_patterns = [
            r'^\s*第\s*\d+\s*页\s*$',  # 第X页
            r'^\s*Page\s*\d+\s*(of|/)\s*\d+\s*$',  # Page X of Y
            r'^\s*-?\s*\d+\s*-\s*$',  # - 12 -
            r'^\s*\d+\s*/\s*\d+\s*$',  # 12/100
            r'^第\s*\d+\s*页.*共\s*\d+\s*页',  # 第X页共Y页
        ]

        # 版权和噪音模式
        self.noise_patterns = [
            r'^\s*保密\s*$',
            r'^\s*机密\s*$',
            r'^\s*Confidential\s*$',
            r'^\s*内部资料\s*$',
            r'^\s*版权所有.*$',
            r'^\s*Copyright\s*©.*$',
            r'^\s*All rights reserved.*$',
            r'^\s*\*+\s*$',
            r'^\s*-+\s*$',
        ]

        # 短行模式（可能是页眉页脚）
        self.short_line_patterns = [
            r'^[^a-zA-Z\u4e00-\u9fa5]{0,3}$',  # 纯符号或数字
            r'^[•·]\s*$',  # 单个点
        ]

    def clean_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, CleaningStats]:
        """
        清洗整个文档

        参数:
            content: 原始文档内容
            metadata: 文档元数据（可选）

        返回:
            (清洗后的内容, 统计信息)
        """
        stats = CleaningStats(
            original_length=len(content),
            cleaned_length=0,
            removed_headers=0,
            removed_footers=0,
            removed_page_numbers=0,
            removed_noise=0,
            removed_duplicates=0,
            final_length=0,
            reduction_ratio=0.0
        )

        try:
            lines = content.split('\n')
            cleaned_lines = []

            # 第一步：按段落分组
            paragraphs = self._group_into_paragraphs(lines)

            # 第二步：分析并去除页眉页脚
            if self.remove_headers or self.remove_footers:
                header_patterns, footer_patterns = self._detect_header_footer_patterns(paragraphs)

            # 第三步：逐段落清洗
            for para in paragraphs:
                # 跳过空行
                if not para.strip():
                    continue

                # 检测页眉
                if self.remove_headers and self._is_header(para, header_patterns):
                    stats.removed_headers += 1
                    continue

                # 检测页脚
                if self.remove_footers and self._is_footer(para, footer_patterns):
                    stats.removed_footers += 1
                    continue

                # 检测页码
                if self.remove_page_numbers and self._is_page_number(para):
                    stats.removed_page_numbers += 1
                    continue

                # 检测噪音
                if self.remove_noise and self._is_noise(para):
                    stats.removed_noise += 1
                    continue

                # 规范化文本
                cleaned_para = self._normalize_paragraph(para)

                # 检查最小长度
                if len(cleaned_para) < self.min_paragraph_length:
                    continue

                cleaned_lines.append(cleaned_para)

            # 第四步：去重
            if self.remove_duplicates:
                cleaned_lines = self._remove_duplicates(cleaned_lines)
                stats.removed_duplicates = len(lines) - len(cleaned_lines)

            # 第五步：最终规范化
            cleaned_content = '\n\n'.join(cleaned_lines)

            if self.normalize_whitespace:
                cleaned_content = self._normalize_whitespace(cleaned_content)

            # 更新统计
            stats.final_length = len(cleaned_content)
            stats.cleaned_length = stats.final_length
            stats.reduction_ratio = 1.0 - (stats.final_length / stats.original_length)

            logger.info(f"文档清洗完成: 原始 {stats.original_length} -> 清洗后 {stats.final_length} "
                       f"(减少 {stats.reduction_ratio*100:.1f}%)")

            return cleaned_content, stats

        except Exception as e:
            logger.error(f"文档清洗失败: {e}", exc_info=True)
            return content, stats

    def _group_into_paragraphs(self, lines: List[str]) -> List[str]:
        """将行分组成段落"""
        paragraphs = []
        current_paragraph = []

        for line in lines:
            stripped = line.strip()

            # 空行表示段落结束
            if not stripped:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue

            # 短行可能是标题或独立行
            if len(stripped) < self.min_line_length and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                paragraphs.append(stripped)
                current_paragraph = []
            else:
                current_paragraph.append(stripped)

        # 最后一个段落
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))

        return paragraphs

    def _detect_header_footer_patterns(
        self,
        paragraphs: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        检测页眉和页脚的模式

        通过分析文档开头和结尾的段落来识别常见的页眉页脚模式
        """
        # 分析前10个段落作为候选页眉
        header_candidates = {}
        for i, para in enumerate(paragraphs[:min(10, len(paragraphs))]):
            if len(para) < 100:  # 页眉通常较短
                if para not in header_candidates:
                    header_candidates[para] = []
                header_candidates[para].append(i)

        # 分析最后10个段落作为候选页脚
        footer_candidates = {}
        for i, para in enumerate(paragraphs[-min(10, len(paragraphs)):]):
            actual_index = len(paragraphs) - min(10, len(paragraphs)) + i
            if len(para) < 100:  # 页脚通常较短
                if para not in footer_candidates:
                    footer_candidates[para] = []
                footer_candidates[para].append(actual_index)

        # 提取高频出现的模式（重复出现3次以上）
        header_patterns = [
            para for para, indices in header_candidates.items()
            if len(indices) >= 3
        ]

        footer_patterns = [
            para for para, indices in footer_candidates.items()
            if len(indices) >= 3
        ]

        return header_patterns, footer_patterns

    def _is_header(self, text: str, header_patterns: List[str]) -> bool:
        """判断是否是页眉"""
        # 匹配已知模式
        for pattern in header_patterns:
            if text == pattern:
                return True

        # 匹配短行模式
        for pattern in self.short_line_patterns:
            if re.match(pattern, text):
                return True

        return False

    def _is_footer(self, text: str, footer_patterns: List[str]) -> bool:
        """判断是否是页脚"""
        # 匹配已知模式
        for pattern in footer_patterns:
            if text == pattern:
                return True

        # 匹配页码模式
        for pattern in self.page_number_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # 匹配噪音模式
        for pattern in self.noise_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False

    def _is_page_number(self, text: str) -> bool:
        """判断是否是页码"""
        stripped = text.strip()

        for pattern in self.page_number_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True

        return False

    def _is_noise(self, text: str) -> bool:
        """判断是否是噪音"""
        stripped = text.strip()

        # 过短
        if len(stripped) < self.min_line_length:
            return True

        # 纯符号
        if re.match(r'^[^\w\u4e00-\u9fa5]+$', stripped):
            return True

        # 过多重复字符
        if self._has_excessive_repetition(stripped):
            return True

        # 匹配噪音模式
        for pattern in self.noise_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True

        return False

    def _has_excessive_repetition(self, text: str) -> bool:
        """检测是否有过多重复字符"""
        for char in text:
            if text.count(char) > self.max_repeated_chars and len(char) > 0:
                # 检查是否连续重复
                if char * self.max_repeated_chars in text:
                    return True
        return False

    def _normalize_paragraph(self, text: str) -> str:
        """规范化段落"""
        # 去除首尾空白
        text = text.strip()

        # 规范化引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', ''').replace(''', '')

        # 规范化省略号
        text = re.sub(r'\.{2,}', '…', text)

        # 规范化破折号
        text = re.sub(r'-{2,}', '——', text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白符"""
        # 去除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]

        # 合并多个空行
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False

        # 重新组合
        text = '\n'.join(cleaned_lines)

        # 去除段落内多余空白
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _remove_duplicates(self, paragraphs: List[str]) -> List[str]:
        """去除重复段落"""
        seen = set()
        unique_paragraphs = []

        try:
            # 尝试使用Levenshtein距离进行模糊去重
            from Levenshtein import ratio as levenshtein_ratio

            for para in paragraphs:
                is_duplicate = False

                for seen_para in seen:
                    # 计算相似度
                    similarity = levenshtein_ratio(para, seen_para)

                    if similarity >= self.duplicate_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_paragraphs.append(para)
                    seen.add(para)

        except ImportError:
            # 如果没有Levenshtein库，使用精确匹配
            logger.warning("Levenshtein库未安装，使用精确去重")
            unique_paragraphs = list(dict.fromkeys(paragraphs))

        return unique_paragraphs


# 便捷函数
def clean_document(
    content: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[str, CleaningStats]:
    """
    清洗文档内容的便捷函数

    参数:
        content: 原始文档内容
        config: 配置字典

    返回:
        (清洗后的内容, 统计信息)

    示例:
        cleaned, stats = clean_document(original_text)
        print(f"清洗完成: {stats.reduction_ratio*100:.1f}% 的内容被移除")
    """
    cleaner = DocumentCleaner(config)
    return cleaner.clean_document(content)


def quick_clean(content: str) -> str:
    """
    快速清洗文档（使用默认配置）

    参数:
        content: 原始文档内容

    返回:
        清洗后的内容

    示例:
        cleaned = quick_clean(raw_text)
    """
    cleaned, _ = clean_document(content)
    return cleaned
