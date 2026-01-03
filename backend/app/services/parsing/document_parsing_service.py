"""
独立的文档解析服务 - 使用新的统一解析系统
负责所有文档解析逻辑

支持的文档类型：
- PDF: 使用UnifiedPDFParser
- Word: 使用WordDocumentParser
- PowerPoint: 使用UnifiedPPTParser
- Excel: 使用UnifiedExcelParser
- Markdown/Text: 基础文本解析
"""

import tempfile
import os
from pathlib import Path
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# 导入新的统一解析系统

class DocumentParsingService:
    """
    文档解析服务

    功能：
    - 文件类型自动检测和路由
    - 使用统一解析流水线
    - 支持所有文档类型（PDF/Word/PPT/Excel等）
    - 返回标准化的解析结果
    """

    def __init__(self, services: Dict[str, Any] = None):
        self.services = services or {}
        self._initialized = False

        # 初始化统一解析服务
        self.unified_service = get_unified_document_service()

    async def parse_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        解析文档

        Args:
            file_content: 文件内容（字节）
            filename: 文件名
            document_id: 文档ID

        Returns:
            (text_content, markdown_content, parse_result)
        """
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            logger.info(f"📄 开始解析文档: {filename}")

            # 使用统一解析服务
            result = await parse_document(tmp_path)

            if result.success:
                logger.info(f"✅ 文档解析成功: {filename}")

                # 提取内容
                text_content = result.content or ""
                markdown_content = result.markdown or ""

                # 构建返回结果
                parse_result = {
                    'method': 'UnifiedDocumentPipeline',
                    'success': True,
                    'metadata': result.metadata,
                    'processing_stats': result.processing_stats,
                    'images_count': len(result.images),
                    'has_multimodal_analysis': result.multimodal_analysis is not None,
                    'has_deepseek_summary': bool(result.deepseek_summary),
                    'vectors_count': len(result.vector_ids),
                    'graph_entities_count': len(result.graph_entities),
                    'local_storage_path': result.local_storage_path,
                }

                # 添加多模态分析结果
                if result.multimodal_analysis:
                    parse_result['multimodal_analysis'] = {
                        'images_analyzed': result.multimodal_analysis.images_analyzed,
                        'charts_found': result.multimodal_analysis.charts_found,
                        'formulas_found': result.multimodal_analysis.formulas_found,
                        'tables_found': result.multimodal_analysis.tables_found,
                    }

                return text_content, markdown_content, parse_result
            else:
                logger.error(f"❌ 文档解析失败: {result.error}")
                return "", "", {
                    'method': 'UnifiedDocumentPipeline',
                    'success': False,
                    'error': result.error
                }

        except Exception as e:
            logger.error(f"❌ 文档解析异常: {e}", exc_info=True)
            return "", "", {
                'method': 'UnifiedDocumentPipeline',
                'success': False,
                'error': str(e)
            }

        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# 便捷函数
async def parse_document_simple(
    file_content: bytes,
    filename: str,
    document_id: str = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    便捷函数：解析文档

    Args:
        file_content: 文件内容（字节）
        filename: 文件名
        document_id: 文档ID（可选）

    Returns:
        (text_content, markdown_content, parse_result)
    """
    service = DocumentParsingService()
    return await service.parse_document(file_content, filename, document_id)

__all__ = [
    'DocumentParsingService',
    'parse_document_simple'
]
