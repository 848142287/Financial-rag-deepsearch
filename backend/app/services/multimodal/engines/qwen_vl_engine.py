"""
Qwen-VL引擎占位符
临时实现，避免导入错误
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class QwenVLEngine:
    """Qwen-VL引擎占位符"""

    def __init__(self):
        logger.warning("QwenVLEngine is using placeholder implementation")

    async def parse_with_ocr(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """占位符OCR解析"""
        logger.warning(f"Placeholder OCR parsing for {file_path}")
        return {
            'text_blocks': [],
            'metadata': {'engine': 'placeholder_qwen_ocr'}
        }

    async def parse_with_vl_max(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """占位符VL-Max解析"""
        logger.warning(f"Placeholder VL-Max parsing for {file_path}")
        return {
            'analysis_results': [],
            'metadata': {'engine': 'placeholder_qwen_vl_max'}
        }