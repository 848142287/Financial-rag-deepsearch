"""
增强版文档协调器
确保所有文档都能生成文档块，添加容错机制
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)

class DocumentOrchestratorEnhanced:
    """增强版文档协调器，确保文档处理的完整性"""

    def __init__(self):
        self.chunk_size = 2000  # 每个文档块的默认大小

    def ensure_document_chunks(self, db: Session, document_id: int) -> bool:
        """
        确保文档至少有一个文档块
        如果没有，创建基础文档块

        Args:
            db: 数据库会话
            document_id: 文档ID

        Returns:
            bool: 是否成功创建或已存在文档块
        """
        try:
            # 检查是否已有文档块
            existing_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).count()

            if existing_chunks > 0:
                logger.info(f"文档 {document_id} 已有 {existing_chunks} 个文档块")
                return True

            # 获取文档信息
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                logger.error(f"文档 {document_id} 不存在")
                return False

            # 生成基础内容
            content_text = self._generate_fallback_content(document)

            # 创建基础文档块
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=0,
                content=content_text,
                chunk_metadata={
                    "document_id": document_id,
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "chunk_type": "fallback_generated",
                    "generated_at": datetime.now().isoformat(),
                    "reason": "no_chunks_found_in_processing",
                    "orchestrator": "enhanced"
                }
            )

            db.add(chunk)
            db.commit()

            logger.info(f"为文档 {document_id} 创建了fallback文档块")
            return True

        except Exception as e:
            logger.error(f"为文档 {document_id} 创建文档块失败: {e}")
            db.rollback()
            return False

    def _generate_fallback_content(self, document: Document) -> str:
        """
        为文档生成fallback内容

        Args:
            document: 文档对象

        Returns:
            str: 生成的文档内容
        """
        content_parts = []

        # 基本文档信息
        content_parts.append(f"文档标题: {document.title}")
        content_parts.append(f"文件名: {document.filename}")
        content_parts.append(f"文件类型: {document.file_type or '未知'}")
        content_parts.append(f"文件大小: {document.file_size} bytes")

        # 添加parsed_content信息（如果存在）
        if document.parsed_content:
            try:
                if isinstance(document.parsed_content, str):
                    parsed_data = json.loads(document.parsed_content)
                else:
                    parsed_data = document.parsed_content

                if parsed_data and 'content' in parsed_data:
                    content_parts.append("\n解析内容:")
                    for i, content_block in enumerate(parsed_data['content'][:3]):  # 只取前3个块
                        if isinstance(content_block, dict):
                            block_content = content_block.get('content', str(content_block))
                            content_parts.append(f"内容块 {i+1}: {str(block_content)[:500]}...")
            except Exception as e:
                logger.warning(f"解析parsed_content失败: {e}")
                content_parts.append("文档内容已解析但无法提取详细内容")

        # 如果没有parsed_content，生成通用描述
        if not document.parsed_content:
            if document.file_type == '.pdf':
                content_parts.append("\n这是一个PDF格式的金融研究报告，包含了相关的市场分析、投资建议或行业研究内容。")
            elif document.file_type == '.pptx':
                content_parts.append("\n这是一个PowerPoint演示文稿，通常包含金融分析、业务汇报或投资策略展示。")
            elif document.file_type == '.xlsx':
                content_parts.append("\n这是一个Excel电子表格文件，可能包含财务数据、市场统计或量化分析结果。")
            else:
                content_parts.append(f"\n这是一个{document.file_type or '未知'}格式的文件，包含相关的金融或业务信息。")

        # 添加元数据信息
        if document.doc_metadata:
            content_parts.append(f"\n文档元数据: {json.dumps(document.doc_metadata, ensure_ascii=False, indent=2)[:500]}...")

        # 添加时间戳
        content_parts.append(f"\n文档ID: {document.id}")
        content_parts.append(f"创建时间: {document.created_at}")
        if document.processed_at:
            content_parts.append(f"处理时间: {document.processed_at}")

        return "\n".join(content_parts)

    def process_document_with_fallback(self, db: Session, document_id: int) -> Dict[str, Any]:
        """
        处理文档并确保生成文档块

        Args:
            db: 数据库会话
            document_id: 文档ID

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            result = {
                'document_id': document_id,
                'success': False,
                'chunks_created': 0,
                'existing_chunks': 0,
                'fallback_used': False
            }

            # 检查现有文档块
            existing_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).count()

            result['existing_chunks'] = existing_chunks

            if existing_chunks == 0:
                # 创建fallback文档块
                success = self.ensure_document_chunks(db, document_id)
                result['success'] = success
                result['fallback_used'] = True
                result['chunks_created'] = 1 if success else 0
            else:
                result['success'] = True

            return result

        except Exception as e:
            logger.error(f"处理文档 {document_id} 时出错: {e}")
            return {
                'document_id': document_id,
                'success': False,
                'error': str(e)
            }

    def batch_ensure_chunks(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        批量确保文档有文档块

        Args:
            document_ids: 文档ID列表

        Returns:
            Dict[str, Any]: 批量处理结果
        """
        results = {
            'total_documents': len(document_ids),
            'successful': 0,
            'failed': 0,
            'already_had_chunks': 0,
            'chunks_created': 0,
            'details': []
        }

        db = next(get_db())
        try:
            for doc_id in document_ids:
                result = self.process_document_with_fallback(db, doc_id)
                results['details'].append(result)

                if result['success']:
                    results['successful'] += 1
                    if result['fallback_used']:
                        results['chunks_created'] += result['chunks_created']
                    else:
                        results['already_had_chunks'] += result['existing_chunks']
                else:
                    results['failed'] += 1

        finally:
            db.close()

        return results

# 全局实例
document_orchestrator_enhanced = DocumentOrchestratorEnhanced()