"""
分批文档解析服务
对于大文档（>20页）进行分批解析，每批15页
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class BatchDocumentParser:
    """
    分批文档解析器

    功能：
    - 自动检测文档页数
    - 大文档（>20页）分批解析
    - 每批15页
    - 合并批次结果
    - 进度跟踪
    """

    # 配置
    PAGE_THRESHOLD = 20  # 超过20页启用分批
    BATCH_SIZE = 15      # 每批15页

    def __init__(self):
        """初始化分批解析器"""
        self.parser = None
        self._initialized = False

    async def initialize(self):
        """初始化解析器"""
        if self._initialized:
            return

        # 使用refactored版本的parser
        from app.services.parsers.refactored.pdf_parser_refactored import parse_pdf as parse_pdf_refactored
        from app.services.parsers.refactored.word_parser_refactored import parse_word as parse_word_refactored
        from app.services.parsers.refactored.excel_parser_refactored import parse_excel as parse_excel_refactored
        from app.services.parsers.refactored.ppt_parser_refactored import parse_ppt as parse_ppt_refactored

        self.parsers = {
            'pdf': parse_pdf_refactored,
            'word': parse_word_refactored,
            'excel': parse_excel_refactored,
            'ppt': parse_ppt_refactored
        }

        self._initialized = True
        logger.info("✅ 分批文档解析器初始化完成")

    async def parse_document(
        self,
        file_path: str,
        document_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        解析文档（自动判断是否需要分批）

        Args:
            file_path: 文件路径
            document_id: 文档ID
            progress_callback: 进度回调函数

        Returns:
            解析结果
        """
        if not self._initialized:
            await self.initialize()

        # 1. 检测文档类型和页数
        file_type = self._detect_file_type(file_path)
        total_pages = await self._get_page_count(file_path, file_type)

        logger.info(f"📄 文档 {document_id}: 类型={file_type}, 页数={total_pages}")

        # 2. 判断是否需要分批
        if total_pages <= self.PAGE_THRESHOLD:
            # 小文档，直接解析
            logger.info(f"📄 文档页数<=20，直接解析")
            return await self._parse_directly(file_path, file_type, document_id, progress_callback)
        else:
            # 大文档，分批解析
            logger.info(f"📄 文档页数>20，启用分批解析（每批{self.BATCH_SIZE}页）")
            return await self._parse_in_batches(
                file_path, file_type, document_id, total_pages, progress_callback
            )

    def _detect_file_type(self, file_path: str) -> str:
        """检测文档类型"""
        suffix = Path(file_path).suffix.lower()

        type_map = {
            '.pdf': 'pdf',
            '.doc': 'word',
            '.docx': 'word',
            '.xls': 'excel',
            '.xlsx': 'excel',
            '.ppt': 'ppt',
            '.pptx': 'ppt'
        }

        return type_map.get(suffix, 'unknown')

    async def _get_page_count(self, file_path: str, file_type: str) -> int:
        """获取文档页数"""
        try:
            if file_type == 'pdf':
                # 使用PyMuPDF获取页数
                import fitz
                doc = fitz.open(file_path)
                page_count = len(doc)
                doc.close()
                return page_count

            elif file_type == 'word':
                # 使用python-docx获取页数（估算）
                from docx import Document
                doc = Document(file_path)
                # Word文档页数难以准确获取，这里估算
                # 假设每页约30个段落
                page_count = max(1, len(doc.paragraphs) // 30)
                return page_count

            elif file_type == 'excel':
                # Excel按工作表数计算
                import openpyxl
                wb = openpyxl.load_workbook(file_path, read_only=True)
                sheet_count = len(wb.sheetnames)
                wb.close()
                return sheet_count

            elif file_type == 'ppt':
                # PowerPoint按幻灯片数计算
                from pptx import Presentation
                prs = Presentation(file_path)
                slide_count = len(prs.slides)
                return slide_count

            else:
                logger.warning(f"⚠️ 未知文件类型: {file_type}")
                return 1

        except Exception as e:
            logger.error(f"❌ 获取页数失败: {e}")
            return 1

    async def _parse_directly(
        self,
        file_path: str,
        file_type: str,
        document_id: str,
        progress_callback: Optional[callable]
    ) -> Dict[str, Any]:
        """直接解析文档（小文档）"""
        if progress_callback:
            await progress_callback(document_id, 0, 100, "开始解析...")

        parser_func = self.parsers.get(file_type)
        if not parser_func:
            raise ValueError(f"不支持的文件类型: {file_type}")

        # 执行解析
        result = await parser_func(file_path)

        if progress_callback:
            await progress_callback(document_id, 100, 100, "解析完成")

        return {
            'document_id': document_id,
            'success': True,
            'parsed': True,
            'total_pages': 1,
            'batches': 1,
            'result': result
        }

    async def _parse_in_batches(
        self,
        file_path: str,
        file_type: str,
        document_id: str,
        total_pages: int,
        progress_callback: Optional[callable]
    ) -> Dict[str, Any]:
        """分批解析大文档"""
        # 计算批次数
        num_batches = (total_pages + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        logger.info(f"📊 分批信息: 总页数={total_pages}, 每批={self.BATCH_SIZE}, 总批次数={num_batches}")

        if progress_callback:
            await progress_callback(document_id, 0, 100, f"开始分批解析（共{num_batches}批）...")

        # 解析每一批
        all_results = []
        current_page = 0

        for batch_idx in range(num_batches):
            start_page = batch_idx * self.BATCH_SIZE
            end_page = min(start_page + self.BATCH_SIZE, total_pages)

            logger.info(f"🔄 解析第{batch_idx + 1}/{num_batches}批: 页{start_page + 1}-{end_page}")

            if progress_callback:
                progress = int((batch_idx / num_batches) * 100)
                await progress_callback(
                    document_id,
                    progress,
                    100,
                    f"解析第{batch_idx + 1}/{num_batches}批（页{start_page + 1}-{end_page}）..."
                )

            # 解析当前批次
            batch_result = await self._parse_batch(
                file_path, file_type, start_page, end_page
            )

            all_results.append({
                'batch_index': batch_idx,
                'start_page': start_page + 1,  # 1-based
                'end_page': end_page,
                'result': batch_result
            })

            # 小延迟，避免资源占用过高
            await asyncio.sleep(0.5)

        # 合并批次结果
        merged_result = self._merge_batch_results(all_results, file_type)

        if progress_callback:
            await progress_callback(document_id, 100, 100, "分批解析完成")

        return {
            'document_id': document_id,
            'success': True,
            'parsed': True,
            'total_pages': total_pages,
            'batches': num_batches,
            'result': merged_result,
            'batch_results': all_results
        }

    async def _parse_batch(
        self,
        file_path: str,
        file_type: str,
        start_page: int,
        end_page: int
    ) -> Dict[str, Any]:
        """解析单个批次"""
        parser_func = self.parsers.get(file_type)
        if not parser_func:
            raise ValueError(f"不支持的文件类型: {file_type}")

        # 调用refactored parser，传入页码范围
        # 注意：这里需要refactored parser支持页码范围参数
        # 如果不支持，需要修改refactored parser
        try:
            # 尝试传入页码范围
            result = await parser_func(
                file_path,
                page_range=(start_page, end_page)
            )
        except TypeError:
            # 如果parser不支持page_range参数，解析全部
            logger.warning(f"⚠️ Parser不支持页码范围，解析全部")
            result = await parser_func(file_path)

        return result

    def _merge_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        file_type: str
    ) -> Dict[str, Any]:
        """合并批次结果"""
        merged = {
            'text': [],
            'markdown': [],
            'metadata': {
                'file_type': file_type,
                'batch_count': len(batch_results)
            },
            'chunks': [],
            'images': [],
            'tables': []
        }

        for batch in batch_results:
            result = batch['result']

            # 合并文本
            if 'text' in result:
                merged['text'].append(result['text'])

            # 合并markdown
            if 'markdown' in result:
                merged['markdown'].append(result['markdown'])

            # 合并chunks
            if 'chunks' in result:
                merged['chunks'].extend(result['chunks'])

            # 合并images
            if 'images' in result:
                merged['images'].extend(result['images'])

            # 合并tables
            if 'tables' in result:
                merged['tables'].extend(result['tables'])

        # 连接文本
        if merged['text']:
            merged['text'] = '\n\n'.join(merged['text'])

        if merged['markdown']:
            merged['markdown'] = '\n\n'.join(merged['markdown'])

        return merged


# 全局实例
_batch_parser_instance: Optional[BatchDocumentParser] = None


def get_batch_document_parser() -> BatchDocumentParser:
    """获取分批文档解析器实例"""
    global _batch_parser_instance

    if _batch_parser_instance is None:
        _batch_parser_instance = BatchDocumentParser()
        logger.info("✅ 初始化分批文档解析器")

    return _batch_parser_instance


__all__ = [
    'BatchDocumentParser',
    'get_batch_document_parser'
]
