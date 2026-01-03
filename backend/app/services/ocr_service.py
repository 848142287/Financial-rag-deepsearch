"""
OCR服务统一入口点
提供对所有OCR服务的统一访问
使用GLM-4.6V云端API进行OCR识别
"""

from app.core.structured_logging import get_structured_logger
from typing import Dict, Any, Optional, List

logger = get_structured_logger(__name__)

# 导入OCR服务组件

# ============================================================================
# OCR服务获取函数
# ============================================================================

def get_ocr_service(**kwargs) -> OCRService:
    """
    获取OCR服务实例（推荐）

    OCR服务特性:
    - 使用GLM-4.6V云端API进行高精度OCR识别
    - 内置优化的OCR提示词
    - 支持图片和PDF文档
    - 自动格式保持和文本清理

    Args:
        **kwargs: OCR服务配置参数
            - enable_ocr: 是否启用OCR (默认: True)
            - model_kwargs: GLM-4.6V模型参数

    Returns:
        OCRService 实例
    """
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(**kwargs)
        logger.info("✅ 初始化OCR服务 (GLM-4.6V 云端API)")
    return _ocr_service

# 向后兼容别名
def get_hybrid_ocr_service(**kwargs) -> OCRService:
    """获取OCR服务实例（向后兼容函数）"""
    return get_ocr_service(**kwargs)

# 全局OCR服务实例
_ocr_service = None

# ============================================================================
# 便捷函数
# ============================================================================

async def ocr_extract_text_from_image(
    image_bytes: bytes,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数: 从图片中提取文本

    Args:
        image_bytes: 图片字节内容
        prompt: 可选的提示词（默认使用GLM-4.6V内置OCR提示词）

    Returns:
        包含识别结果的字典
    """
    ocr_service = get_ocr_service()
    return await ocr_service.extract_text_from_image(
        image_bytes,
        prompt=prompt
    )

async def ocr_extract_from_pdf(
    pdf_bytes: bytes,
    pages: Optional[list] = None,
    max_concurrent: int = 12
) -> List[Dict[str, Any]]:
    """
    便捷函数: 从 PDF 批量提取文本

    Args:
        pdf_bytes: PDF 文件字节
        pages: 要处理的页码列表（None 表示全部）
        max_concurrent: 最大并发数

    Returns:
        每页的提取结果列表
    """
    ocr_service = get_ocr_service()
    return await ocr_service.batch_extract_from_pdf(
        pdf_bytes,
        pages=pages,
        max_concurrent=max_concurrent
    )

# ============================================================================
# 导出所有公共接口
# ============================================================================

__all__ = [
    # 类
    'OCRService',
    'HybridOCRService',  # 向后兼容别名
    'OCRModelType',

    # 函数
    'get_ocr_service',
    'get_hybrid_ocr_service',  # 向后兼容别名
    'ocr_extract_text_from_image',
    'ocr_extract_from_pdf',
]
