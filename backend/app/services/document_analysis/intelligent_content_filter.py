"""
智能内容过滤器 - Intelligent Content Filter
过滤文档中的无意义内容，提升数据质量
"""

import re
import logging
from typing import List, Set, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterStatistics:
    """过滤统计信息"""
    total_elements: int = 0
    filtered_elements: int = 0
    kept_elements: int = 0
    noise_types_removed: Dict[str, int] = None

    def __post_init__(self):
        if self.noise_types_removed is None:
            self.noise_types_removed = {}


class IntelligentContentFilter:
    """智能内容过滤器"""

    # 无意义内容模式库
    NOISE_PATTERNS = {
        'page_numbers': r'^\s*\d+\s*$',  # 纯页码
        'file_paths': r'^[A-Za-z]:\\.+|^.+/\.\.+$|^/.*$',  # 文件路径
        'print_markers': r'^\s*(第\s*\d+\s*页|Page\s*\d+|共\s*\d+\s*页|P\s*\d+|pp\s*\d+).*$',  # 打印标记
        'placeholders': r'^(在此处|点击此处|请在此|Click here|Type here).+(输入|填写|插入|编辑|添加|输入|edit|insert|add|type)',  # 模板文字
        'empty_lines': r'^\s*$',  # 空行
        'urls_only': r'^https?://[^\s]*$',  # 纯URL（没有其他内容）
        'email_only': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # 纯邮箱
        'numeric_only': r'^[\d\s\.,%\-\+\$€¥£]+$',  # 纯数字/货币（无上下文）
        'blank_page_marker': r'^\s*(空白页|Blank\s*Page|第\s*\d+\s*页\s*\n\s*$)',  # 空白页标记
        'watermark_like': r'^(保密|机密|Confidential|Draft|草稿|样本|Sample)$',  # 水印式文字
        'pagebreak_marker': r'^\s*-+\s*$',  # 分页标记（如 ----）
    }

    # 扩展的占位符模式（更全面的模板文字检测）
    PLACEHOLDER_PATTERNS = [
        r'在此处.*(输入|填写|插入|添加|编辑)',
        r'点击.*(输入|选择|上传)',
        r'请.*(输入|填写|选择|上传)',
        r'(Click|Type|Select|Upload|Insert|Add).*here',
        r'\[.*(此处|这里|待定|TBD|TO\s*BE\s*DETERMINED).*\]',
        r'（.*(待补充|待定|预留).*）',
        r'\(.*(待补充|待定|预留).*\)',
        r'^\s*(XXX|\?{3}|\.{3,})\s*$',  # XXX或???或...占位符
    ]

    # 页眉页脚识别模式
    HEADER_FOOTER_PATTERNS = {
        'confidential': r'^(机密|保密|内部资料|Confidential|Internal)',
        'company': r'.*(公司|集团|有限公司|股份有限公司|Corp|Ltd|Inc|LLC).*$',
        'date': r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}|January|February|March|April|May|June|July|August|September|October|November|December',
        'draft': r'^(草稿|Draft|DRAFT)$',
        'version': r'^(版本|Version|Ver)\s*\d+.*$',
    }

    # 重复内容阈值（相同内容出现超过此次数视为噪音）
    REPETITION_THRESHOLD = 3

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化过滤器

        Args:
            config: 配置字典
                - min_length: 最小字符长度 (默认10)
                - min_density: 最小信息密度 (默认0.3)
                - check_repetition: 是否检查重复 (默认True)
                - keep_first_header_footer: 是否保留首次出现的页眉页脚 (默认True)
        """
        self.config = config or {}
        self.min_length = self.config.get('min_length', 10)
        self.min_density = self.config.get('min_density', 0.3)
        self.check_repetition = self.config.get('check_repetition', True)
        self.keep_first_header_footer = self.config.get('keep_first_header_footer', True)

        # 编译正则表达式
        self.compiled_noise_patterns = {
            noise_type: re.compile(pattern)
            for noise_type, pattern in self.NOISE_PATTERNS.items()
        }

        self.compiled_header_footer_patterns = {
            hf_type: re.compile(pattern)
            for hf_type, pattern in self.HEADER_FOOTER_PATTERNS.items()
        }

        # 编译扩展的占位符模式
        self.compiled_placeholder_patterns = [
            re.compile(pattern) for pattern in self.PLACEHOLDER_PATTERNS
        ]

        # 重复内容计数器
        self.content_counter: Dict[str, int] = {}
        self.seen_headers_footers: Set[str] = set()
        self.header_footer_positions: Dict[str, List[int]] = {}  # 记录页眉页脚出现位置

        # 页面位置跟踪（用于识别页眉页脚）
        self.current_page = 0

    def reset(self):
        """重置过滤器状态"""
        self.content_counter.clear()
        self.seen_headers_footers.clear()
        self.header_footer_positions.clear()
        self.current_page = 0

    def is_meaningful_content(self, text: str) -> tuple:
        """
        判断文本是否有意义

        Args:
            text: 文本内容

        Returns:
            (is_meaningful, reason): (是否有意义, 原因)
        """
        if not text or not isinstance(text, str):
            return False, "empty_or_invalid"

        text_stripped = text.strip()

        # 1. 检查是否为噪音模式
        for noise_type, pattern in self.compiled_noise_patterns.items():
            if pattern.match(text_stripped):
                return False, f"noise_pattern_{noise_type}"

        # 1.5 检查扩展的占位符模式
        for pattern in self.compiled_placeholder_patterns:
            if pattern.search(text_stripped):
                return False, "placeholder_pattern"

        # 2. 检查长度阈值
        if len(text_stripped) < self.min_length:
            return False, "too_short"

        # 3. 检查信息密度
        char_count = len(text_stripped)
        non_space = len(text_stripped.replace(' ', '').replace('\t', '').replace('\n', ''))
        density = non_space / char_count if char_count > 0 else 0

        if density < self.min_density:
            return False, "low_density"

        # 4. 检查重复内容
        if self.check_repetition:
            normalized = text_stripped.lower()
            self.content_counter[normalized] = self.content_counter.get(normalized, 0) + 1

            if self.content_counter[normalized] > self.REPETITION_THRESHOLD:
                return False, "repetitive"

        return True, "valid"

    def is_header_footer(self, text: str) -> bool:
        """
        判断是否为页眉页脚

        Args:
            text: 文本内容

        Returns:
            bool: 是否为页眉页脚
        """
        text_stripped = text.strip().lower()

        # 检查是否匹配页眉页脚模式
        for pattern in self.compiled_header_footer_patterns.values():
            if pattern.match(text_stripped):
                return True

        # 检查是否为已见过的页眉页脚
        if text_stripped in self.seen_headers_footers:
            return True

        return False

    def register_header_footer(self, text: str, position: int = None):
        """
        记录页眉页脚（带位置信息）

        Args:
            text: 页眉页脚文本
            position: 在文档中的位置索引
        """
        normalized = text.strip().lower()
        if len(normalized) < 100:  # 只记录较短的页眉页脚
            self.seen_headers_footers.add(normalized)
            if position is not None:
                if normalized not in self.header_footer_positions:
                    self.header_footer_positions[normalized] = []
                self.header_footer_positions[normalized].append(position)

    def is_decorative_image(self, element: Any) -> bool:
        """
        判断图片是否为纯装饰性图案

        Args:
            element: 文档元素（包含metadata）

        Returns:
            bool: 是否为装饰性图片
        """
        try:
            elem_type = getattr(element, 'element_type', None)
            if elem_type not in ['image', 'chart']:
                return False

            metadata = getattr(element, 'metadata', {})
            if not metadata:
                # 没有metadata的图片，可能是装饰性的
                return True

            # 检查是否有OCR识别的文字
            ocr_text = metadata.get('ocr_text', '')
            image_type = metadata.get('image_type', '')
            description = metadata.get('description', '')

            # 如果没有OCR文字，没有描述，也没有类型信息，可能是装饰性
            if not ocr_text and not description and not image_type:
                return True

            # 如果明确标注为图表，则不是装饰性
            if image_type in ['chart', 'table', 'formula', 'diagram']:
                return False

            # 如果有OCR文字或描述，则不是装饰性
            if ocr_text or description:
                return False

            # 默认情况下，认为是有内容的
            return False

        except Exception:
            # 出错时保守处理，保留图片
            return False

    def is_blank_page(self, elements: List[Any], start_idx: int) -> bool:
        """
        判断是否为空白页（连续多个元素都是无意义的）

        Args:
            elements: 文档元素列表
            start_idx: 开始检查的索引位置

        Returns:
            bool: 是否为空白页
        """
        # 检查接下来的几个元素
        window_size = min(10, len(elements) - start_idx)
        if window_size < 3:
            return False

        meaningful_count = 0
        for i in range(start_idx, min(start_idx + window_size, len(elements))):
            element = elements[i]
            elem_type = getattr(element, 'element_type', None)

            # 跳过非内容类型
            if elem_type in ['heading', 'title']:
                meaningful_count += 2  # 标题更有意义
                continue

            if elem_type == 'paragraph':
                content = getattr(element, 'content', '')
                if content and len(content.strip()) > self.min_length:
                    is_meaningful, _ = self.is_meaningful_content(content)
                    if is_meaningful:
                        meaningful_count += 1

            # 图片、表格等非段落元素也视为有意义
            elif elem_type in ['table', 'image', 'chart']:
                meaningful_count += 1

        # 如果有意义的元素少于总数的30%，认为是空白页
        return meaningful_count < window_size * 0.3

    def filter_document_elements(
        self,
        elements: List[Any],
        element_type_attr: str = 'element_type',
        content_attr: str = 'content'
    ) -> tuple:
        """
        过滤文档元素

        Args:
            elements: 文档元素列表
            element_type_attr: 元素类型属性名
            content_attr: 内容属性名

        Returns:
            (filtered_elements, statistics): (过滤后的元素列表, 统计信息)
        """
        self.reset()
        filtered = []
        stats = FilterStatistics()
        stats.total_elements = len(elements)

        for idx, element in enumerate(elements):
            try:
                # 获取元素类型和内容
                elem_type = getattr(element, element_type_attr, None)
                content = getattr(element, content_attr, None)

                # 检查装饰性图片
                if elem_type in ['image', 'chart']:
                    if self.is_decorative_image(element):
                        stats.filtered_elements += 1
                        noise_type = "decorative_image"
                        stats.noise_types_removed[noise_type] = stats.noise_types_removed.get(noise_type, 0) + 1
                        logger.debug(f"Filtered decorative image at index {idx}")
                        continue
                    else:
                        # 保留有内容的图片
                        filtered.append(element)
                        stats.kept_elements += 1
                        continue

                # 检查空白页
                if elem_type == 'paragraph' and not content:
                    # 可能是空白页的开始，检查后续元素
                    if self.is_blank_page(elements, idx):
                        stats.filtered_elements += 1
                        noise_type = "blank_page"
                        stats.noise_types_removed[noise_type] = stats.noise_types_removed.get(noise_type, 0) + 1
                        logger.debug(f"Filtered blank page starting at index {idx}")
                        continue

                # 保留非段落类型元素（标题、表格等）
                if elem_type != 'paragraph':
                    filtered.append(element)
                    stats.kept_elements += 1
                    continue

                # 处理段落类型元素
                if not content or not isinstance(content, str):
                    filtered.append(element)
                    stats.kept_elements += 1
                    continue

                # 检查是否为页眉页脚（改进版：保留首次出现）
                if self.is_header_footer(content):
                    normalized = content.strip().lower()
                    positions = self.header_footer_positions.get(normalized, [])

                    if self.keep_first_header_footer and len(positions) == 0:
                        # 首次出现，保留
                        filtered.append(element)
                        stats.kept_elements += 1
                        self.register_header_footer(content, idx)
                    else:
                        # 重复出现，过滤
                        self.register_header_footer(content, idx)
                        stats.filtered_elements += 1
                        noise_type = "repetitive_header_footer"
                        stats.noise_types_removed[noise_type] = stats.noise_types_removed.get(noise_type, 0) + 1
                        logger.debug(f"Filtered repetitive header/footer: {content[:50]}")
                    continue

                # 检查内容是否有意义
                is_meaningful, reason = self.is_meaningful_content(content)

                if is_meaningful:
                    filtered.append(element)
                    stats.kept_elements += 1
                else:
                    stats.filtered_elements += 1
                    stats.noise_types_removed[reason] = stats.noise_types_removed.get(reason, 0) + 1
                    logger.debug(f"Filtered element: {reason} - {content[:50]}")

            except Exception as e:
                logger.error(f"Error filtering element: {e}")
                # 出错时保留元素，避免丢失数据
                filtered.append(element)
                stats.kept_elements += 1

        return filtered, stats

    def filter_text_list(self, texts: List[str]) -> tuple:
        """
        过滤文本列表

        Args:
            texts: 文本列表

        Returns:
            (filtered_texts, statistics): (过滤后的文本列表, 统计信息)
        """
        self.reset()
        filtered = []
        stats = FilterStatistics()
        stats.total_elements = len(texts)

        for text in texts:
            is_meaningful, reason = self.is_meaningful_content(text)

            if is_meaningful:
                filtered.append(text)
                stats.kept_elements += 1
            else:
                stats.filtered_elements += 1
                stats.noise_types_removed[reason] = stats.noise_types_removed.get(reason, 0) + 1

        return filtered, stats

    def get_filter_summary(self, statistics: FilterStatistics) -> str:
        """
        获取过滤摘要

        Args:
            statistics: 统计信息

        Returns:
            str: 摘要文本
        """
        if statistics.total_elements == 0:
            return "无元素需要过滤"

        filtered_ratio = statistics.filtered_elements / statistics.total_elements * 100
        kept_ratio = statistics.kept_elements / statistics.total_elements * 100

        summary = f"""
过滤统计摘要:
  总元素数: {statistics.total_elements}
  过滤元素: {statistics.filtered_elements} ({filtered_ratio:.1f}%)
  保留元素: {statistics.kept_elements} ({kept_ratio:.1f}%)

过滤类型分布:
"""
        for noise_type, count in statistics.noise_types_removed.items():
            ratio = count / statistics.total_elements * 100
            summary += f"  - {noise_type}: {count} ({ratio:.1f}%)\n"

        return summary
