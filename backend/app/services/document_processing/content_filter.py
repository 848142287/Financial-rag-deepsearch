"""Content Filter - 内容过滤器

为文档处理提供内容过滤功能
"""

from typing import Dict, Any, Optional
from app.core.structured_logging import get_structured_logger
import re

logger = get_structured_logger(__name__)


class DocumentContentFilter:
    """文档内容过滤器"""

    def __init__(self):
        self.name = "DocumentContentFilter"
        logger.info("DocumentContentFilter initialized")

    def filter_content(
        self,
        content: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        过滤文档内容

        Args:
            content: 文档内容
            filters: 过滤规则

        Returns:
            过滤结果
        """
        try:
            if filters is None:
                filters = {}

            # 基础过滤逻辑
            filtered_content = content
            removed_count = 0

            # 移除多余空白
            if filters.get("remove_extra_whitespace", True):
                filtered_content = re.sub(r'\s+', ' ', filtered_content).strip()

            # 移除特定标签
            if filters.get("remove_html_tags", False):
                filtered_content = re.sub(r'<[^>]+>', '', filtered_content)

            result = {
                "filtered": True,
                "original_length": len(content),
                "filtered_length": len(filtered_content),
                "removed_chars": len(content) - len(filtered_content),
                "content": filtered_content
            }

            logger.info(f"Content filtered: {result['removed_chars']} chars removed")
            return result

        except Exception as e:
            logger.error(f"Content filter failed: {e}")
            return {
                "filtered": False,
                "error": str(e),
                "content": content
            }

    def is_valid_content(self, content: str) -> bool:
        """
        检查内容是否有效

        Args:
            content: 文档内容

        Returns:
            是否有效
        """
        if not content or not isinstance(content, str):
            return False

        # 检查最小长度
        if len(content.strip()) < 10:
            return False

        return True

    def sanitize_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 输入文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # 标准化空白
        text = re.sub(r'\s+', ' ', text).strip()

        return text
