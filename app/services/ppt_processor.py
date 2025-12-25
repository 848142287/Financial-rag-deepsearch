"""
PPT文档处理器
集成PPT解析功能到现有的文档处理管道
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from app.services.ppt_parser import ppt_parser
from app.core.database import get_db
from app.models.document import Document, DocumentChunk
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class PPTProcessor:
    """PPT文档处理器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunk_size = 1500  # 每个chunk的目标字符数
        self.overlap_size = 200  # chunk之间的重叠大小

    async def process_ppt_document(self, document_id: int, file_path: str) -> Dict[str, Any]:
        """
        处理PPT文档

        Args:
            document_id: 文档ID
            file_path: PPT文件路径

        Returns:
            处理结果
        """
        try:
            self.logger.info(f"开始处理PPT文档 {document_id}: {file_path}")

            # 1. 检查文件是否存在
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f"文件不存在: {file_path}",
                    'document_id': document_id
                }

            # 2. 解析PPT内容
            ppt_result = await ppt_parser.parse_ppt(file_path, extract_images=True)

            if not ppt_result.get('success', False):
                return {
                    'success': False,
                    'error': f"PPT解析失败: {ppt_result.get('error', 'Unknown error')}",
                    'document_id': document_id
                }

            # 3. 提取和处理文本内容
            text_content = ppt_result['text_content']
            if not text_content.strip():
                return {
                    'success': False,
                    'error': "PPT文档中没有提取到文本内容",
                    'document_id': document_id
                }

            # 4. 文本分块处理
            chunks_data = self._create_chunks(text_content, ppt_result)

            # 5. 保存chunks到数据库
            db = next(get_db())
            try:
                saved_chunks = await self._save_chunks_to_db(db, document_id, chunks_data)

                # 6. 更新文档元数据
                await self._update_document_metadata(db, document_id, ppt_result)

                result = {
                    'success': True,
                    'document_id': document_id,
                    'total_slides': ppt_result['total_slides'],
                    'total_chunks': len(saved_chunks),
                    'extracted_tables': len(ppt_result['tables']),
                    'extracted_images': len(ppt_result['images']),
                    'extracted_charts': len(ppt_result['charts']),
                    'text_length': len(text_content),
                    'processing_time': ppt_result['parsing_time']
                }

                self.logger.info(f"PPT文档处理完成: {result}")
                return result

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"处理PPT文档失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'document_id': document_id
            }

    def _create_chunks(self, text_content: str, ppt_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        创建文档chunks

        Args:
            text_content: 完整文本内容
            ppt_result: PPT解析结果

        Returns:
            chunks数据列表
        """
        chunks_data = []

        try:
            # 按幻灯片分组内容
            slide_contents = {}
            for slide in ppt_result['slides']:
                slide_num = slide['slide_number']
                slide_text = slide['text'].strip()

                if slide_text:
                    # 添加幻灯片标题作为上下文
                    if slide['title']:
                        slide_text = f"第{slide_num}页: {slide['title']}\n\n{slide_text}"
                    else:
                        slide_text = f"第{slide_num}页:\n\n{slide_text}"

                    slide_contents[slide_num] = slide_text

            # 按幻灯片顺序创建chunks
            chunk_index = 0
            for slide_num in sorted(slide_contents.keys()):
                slide_text = slide_contents[slide_num]

                # 如果单个幻灯片内容过长，进行分块
                if len(slide_text) <= self.chunk_size:
                    chunks_data.append({
                        'content': slide_text,
                        'chunk_index': chunk_index,
                        'slide_number': slide_num,
                        'chunk_type': 'slide',
                        'metadata': {
                            'original_slide': slide_num,
                            'has_title': any(slide['slide_number'] == slide_num for slide in ppt_result['slides'] if slide['title']),
                            'content_type': 'single_slide'
                        }
                    })
                    chunk_index += 1
                else:
                    # 长内容分块处理
                    sub_chunks = self._split_long_text(slide_text, chunk_index, slide_num)
                    chunks_data.extend(sub_chunks)
                    chunk_index += len(sub_chunks)

            # 添加表格内容作为特殊chunks
            table_index = 0
            for table in ppt_result['tables']:
                if table.get('data'):
                    table_text = self._format_table_content(table)
                    if table_text:
                        chunks_data.append({
                            'content': table_text,
                            'chunk_index': chunk_index,
                            'chunk_type': 'table',
                            'metadata': {
                                'table_index': table_index,
                                'rows': table.get('rows', 0),
                                'columns': table.get('columns', 0),
                                'content_type': 'table_data'
                            }
                        })
                        chunk_index += 1
                        table_index += 1

            # 添加图像描述作为chunks（如果有相关文本）
            for i, image in enumerate(ppt_result['images'][:10]):  # 限制前10个图像
                image_info = self._create_image_description(image, i)
                if image_info:
                    chunks_data.append({
                        'content': image_info,
                        'chunk_index': chunk_index,
                        'chunk_type': 'image',
                        'metadata': {
                            'image_index': i,
                            'slide_number': image.get('slide_number'),
                            'image_width': image.get('width'),
                            'image_height': image.get('height'),
                            'content_type': 'image_description'
                        }
                    })
                    chunk_index += 1

            self.logger.info(f"创建了 {len(chunks_data)} 个PPT chunks")
            return chunks_data

        except Exception as e:
            self.logger.error(f"创建PPT chunks失败: {str(e)}")
            return []

    def _split_long_text(self, text: str, start_index: int, slide_num: int) -> List[Dict[str, Any]]:
        """分割长文本"""
        chunks = []

        try:
            # 按段落分割
            paragraphs = text.split('\n\n')
            current_chunk = ""

            for paragraph in paragraphs:
                # 如果添加这个段落会超过chunk大小，保存当前chunk
                if len(current_chunk + paragraph) > self.chunk_size and current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'chunk_index': start_index + len(chunks),
                        'slide_number': slide_num,
                        'chunk_type': 'slide_part',
                        'metadata': {
                            'original_slide': slide_num,
                            'content_type': 'slide_split',
                            'part_number': len(chunks) + 1
                        }
                    })
                    current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph

            # 保存最后一个chunk
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'chunk_index': start_index + len(chunks),
                    'slide_number': slide_num,
                    'chunk_type': 'slide_part',
                    'metadata': {
                        'original_slide': slide_num,
                        'content_type': 'slide_split',
                        'part_number': len(chunks) + 1
                    }
                })

            return chunks

        except Exception as e:
            self.logger.error(f"分割长文本失败: {str(e)}")
            return []

    def _format_table_content(self, table: Dict[str, Any]) -> str:
        """格式化表格内容为文本"""
        try:
            if not table.get('data'):
                return ""

            data = table['data']
            if not data:
                return ""

            # 限制表格大小以避免过大的chunks
            max_rows = 20
            max_cols = 10

            # 获取实际行列数
            rows = min(len(data), max_rows)
            cols = min(len(data[0]) if data else 0, max_cols)

            if rows == 0 or cols == 0:
                return ""

            # 构建表格文本
            table_lines = []
            table_lines.append(f"表格 ({cols}列 x {rows}行):")

            # 表头
            header = data[0][:cols]
            table_lines.append(" | ".join(str(cell) for cell in header))
            table_lines.append("-" * (len(" | ".join(str(cell) for cell in header))))

            # 表格数据
            for i in range(1, min(rows, len(data))):
                row = data[i][:cols]
                table_lines.append(" | ".join(str(cell) for cell in row))

            return "\n".join(table_lines)

        except Exception as e:
            self.logger.warning(f"格式化表格内容失败: {str(e)}")
            return ""

    def _create_image_description(self, image: Dict[str, Any], index: int) -> Optional[str]:
        """创建图像描述"""
        try:
            if not image:
                return None

            description_parts = []

            # 基本信息
            description_parts.append(f"图片{index + 1}:")

            if image.get('slide_number'):
                description_parts.append(f"位于第{image['slide_number']}页")

            if image.get('width') and image.get('height'):
                description_parts.append(f"尺寸: {image['width']}x{image['height']}像素")

            if image.get('format'):
                description_parts.append(f"格式: {image['format']}")

            # 生成描述
            description = " ".join(description_parts)
            description += f"\n这是一个PPT中的图片，建议查看原文档以获取完整的视觉信息。"

            return description

        except Exception as e:
            self.logger.warning(f"创建图像描述失败: {str(e)}")
            return None

    async def _save_chunks_to_db(self, db: Session, document_id: int, chunks_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """保存chunks到数据库"""
        saved_chunks = []

        try:
            for chunk_data in chunks_data:
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_data['chunk_index'],
                    content=chunk_data['content'],
                    embedding_id=f"ppt_chunk_{chunk_data['chunk_index']}",
                    metadata=chunk_data.get('metadata', {})
                )

                db.add(chunk)
                saved_chunks.append(chunk)

            db.commit()

            self.logger.info(f"成功保存 {len(saved_chunks)} 个PPT chunks到数据库")
            return saved_chunks

        except Exception as e:
            db.rollback()
            self.logger.error(f"保存PPT chunks到数据库失败: {str(e)}")
            raise e

    async def _update_document_metadata(self, db: Session, document_id: int, ppt_result: Dict[str, Any]):
        """更新文档元数据"""
        try:
            # 查找文档
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return

            # 准备元数据
            metadata = {
                'ppt_info': {
                    'total_slides': ppt_result['total_slides'],
                    'total_tables': len(ppt_result['tables']),
                    'total_images': len(ppt_result['images']),
                    'total_charts': len(ppt_result['charts']),
                    'has_images': len(ppt_result['images']) > 0,
                    'has_tables': len(ppt_result['tables']) > 0,
                    'has_charts': len(ppt_result['charts']) > 0
                },
                'ppt_metadata': ppt_result.get('metadata', {}),
                'parsing_info': {
                    'parser_version': '1.0',
                    'parsing_time': ppt_result['parsing_time'],
                    'file_type': 'pptx'
                }
            }

            # 修复：存储解析内容到parsed_content字段
            parsed_content = {
                'content': [
                    {
                        'type': 'slides',
                        'total_slides': ppt_result['total_slides'],
                        'slides_data': ppt_result.get('slides', []),
                        'tables': ppt_result['tables'],
                        'images': ppt_result['images'],
                        'charts': ppt_result['charts'],
                        'text_content': ppt_result.get('text_content', ''),
                        'parsing_metadata': {
                            'parser_type': 'ppt_processor',
                            'parsing_time': ppt_result['parsing_time'],
                            'extracted_elements': {
                                'tables_count': len(ppt_result['tables']),
                                'images_count': len(ppt_result['images']),
                                'charts_count': len(ppt_result['charts'])
                            }
                        }
                    }
                ],
                'metadata': {
                    'document_type': 'presentation',
                    'total_elements': len(ppt_result['tables']) + len(ppt_result['images']) + len(ppt_result['charts']),
                    'file_format': 'pptx',
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            document.parsed_content = parsed_content

            # 更新文档元数据
            if document.doc_metadata:
                document.doc_metadata.update(metadata)
            else:
                document.doc_metadata = metadata

            # 更新文件类型
            if not document.file_type:
                document.file_type = 'pptx'

            # 更新状态为已完成
            document.status = 'completed'
            from datetime import datetime
            document.processed_at = datetime.now()

            db.commit()
            self.logger.info(f"成功更新文档 {document_id} 的PPT元数据")

        except Exception as e:
            db.rollback()
            self.logger.error(f"更新PPT文档元数据失败: {str(e)}")
            raise e

    def get_ppt_preview(self, document_id: int, file_path: str, max_slides: int = 3) -> Dict[str, Any]:
        """
        获取PPT预览信息

        Args:
            document_id: 文档ID
            file_path: PPT文件路径
            max_slides: 最大预览幻灯片数

        Returns:
            预览信息
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f"文件不存在: {file_path}",
                    'document_id': document_id
                }

            # 只解析前几页用于预览
            preview_result = {}

            # 快速解析基本信息
            import asyncio
            ppt_result = asyncio.run(ppt_parser.parse_ppt(file_path, extract_images=False))

            if not ppt_result.get('success', False):
                return {
                    'success': False,
                    'error': f"PPT解析失败: {ppt_result.get('error', 'Unknown error')}",
                    'document_id': document_id
                }

            # 提取预览信息
            preview_slides = ppt_result['slides'][:max_slides]

            preview_result = {
                'success': True,
                'document_id': document_id,
                'total_slides': ppt_result['total_slides'],
                'preview_slides_count': len(preview_slides),
                'file_name': os.path.basename(file_path),
                'title': ppt_result.get('metadata', {}).get('title', ''),
                'author': ppt_result.get('metadata', {}).get('author', ''),
                'slides': [],
                'summary': ppt_result.get('summary', ''),
                'created': ppt_result.get('metadata', {}).get('created'),
                'tables_count': len(ppt_result.get('tables', [])),
                'images_count': len(ppt_result.get('images', [])),
                'charts_count': len(ppt_result.get('charts', []))
            }

            # 只返回预览幻灯片的基本信息（不包含图像）
            for slide in preview_slides:
                slide_preview = {
                    'slide_number': slide['slide_number'],
                    'title': slide['title'],
                    'text_preview': slide['text'][:200] + ('...' if len(slide['text']) > 200 else ''),
                    'has_table': len(slide.get('tables', [])) > 0,
                    'has_chart': len(slide.get('charts', [])) > 0,
                    'image_count': len(slide.get('images', [])),
                    'layout': slide.get('layout', '')
                }
                preview_result['slides'].append(slide_preview)

            return preview_result

        except Exception as e:
            self.logger.error(f"获取PPT预览失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'document_id': document_id
            }

# 创建全局PPT处理器实例
ppt_processor = PPTProcessor()