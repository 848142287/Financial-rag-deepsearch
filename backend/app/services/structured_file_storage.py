"""
结构化文档文件存储服务
确保解析后的文件完全保持原始文档的结构和顺序
"""

import json
from app.core.structured_logging import get_structured_logger
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import aiofiles

logger = get_structured_logger(__name__)

@dataclass
class DocumentPage:
    """文档页面"""
    page_number: int
    content: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "content": self.content,
            "chunks": self.chunks,
            "images": self.images,
            "tables": self.tables,
            "metadata": self.metadata
        }

@dataclass
class StructuredDocument:
    """结构化文档"""
    document_id: str
    filename: str
    total_pages: int

    # 按页面顺序组织
    pages: List[DocumentPage] = field(default_factory=list)

    # 全局信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 处理信息
    processing_time: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "pages": [page.to_dict() for page in self.pages],
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "created_at": self.created_at
        }

class StructuredFileStorageService:
    """
    结构化文件存储服务

    确保解析后的文档完全保持：
    - 页面顺序
    - 段落顺序
    - 表格和图片的位置
    - 元数据的完整性
    """

    def __init__(self, base_dir: str = "./storage/structured_documents"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_document(
        self,
        document_id: str,
        filename: str,
        chunks: List[Any],
        metadata: Dict[str, Any]
    ) -> str:
        """
        保存结构化文档（保持页面顺序）

        Args:
            document_id: 文档ID
            filename: 文件名
            chunks: 分块列表（必须按顺序）
            metadata: 元数据

        Returns:
            保存的文件路径
        """
        try:
            # 按页面和chunk_index排序
            sorted_chunks = self._sort_chunks_by_page(chunks)

            # 按页面分组
            pages_map = self._group_chunks_by_page(sorted_chunks)

            # 创建结构化文档
            structured_doc = StructuredDocument(
                document_id=document_id,
                filename=filename,
                total_pages=len(pages_map),
                pages=self._create_document_pages(pages_map),
                metadata={
                    **metadata,
                    "total_chunks": len(chunks),
                    "chunk_order_preserved": True
                }
            )

            # 保存为JSON
            return await self._save_to_json(structured_doc)

        except Exception as e:
            logger.error(f"保存结构化文档失败: {e}")
            raise

    async def save_document_with_sections(
        self,
        document_id: str,
        filename: str,
        sections: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        """
        保存带章节的文档（保持章节顺序）

        Args:
            document_id: 文档ID
            filename: 文件名
            sections: 章节列表（必须按顺序）
            metadata: 元数据

        Returns:
            保存的文件路径
        """
        try:
            # 按页面排序章节
            sorted_sections = sorted(sections, key=lambda x: x.get('page_number', 0))

            structured_doc = StructuredDocument(
                document_id=document_id,
                filename=filename,
                total_pages=max(s.get('page_number', 0) for s in sections),
                pages=[
                    DocumentPage(
                        page_number=section['page_number'],
                        content=section.get('content', ''),
                        chunks=section.get('chunks', []),
                        images=section.get('images', []),
                        tables=section.get('tables', []),
                        metadata=section
                    )
                    for section in sorted_sections
                ],
                metadata={
                    **metadata,
                    "structure_type": "sections",
                    "section_order_preserved": True
                }
            )

            return await self._save_to_json(structured_doc)

        except Exception as e:
            logger.error(f"保存章节文档失败: {e}")
            raise

    def _sort_chunks_by_page(self, chunks: List[Any]) -> List[Any]:
        """
        按页面和chunk_index排序

        Args:
            chunks: 原始chunk列表

        Returns:
            排序后的chunk列表
        """
        def get_sort_key(chunk):
            # 获取排序键
            page_numbers = getattr(chunk, 'page_numbers', [])
            page_number = page_numbers[0] if page_numbers else 0

            chunk_index = getattr(chunk, 'chunk_index', 0)

            return (page_number, chunk_index)

        # 排序
        sorted_chunks = sorted(chunks, key=get_sort_key)

        # 验证顺序
        for i, chunk in enumerate(sorted_chunks):
            if hasattr(chunk, 'chunk_index'):
                # 可选：添加顺序验证
                if hasattr(chunk, 'metadata'):
                    chunk.metadata['_sort_index'] = i

        return sorted_chunks

    def _group_chunks_by_page(self, chunks: List[Any]) -> Dict[int, List[Any]]:
        """按页面分组chunks"""
        pages_map = {}

        for chunk in chunks:
            # 获取页面号
            if hasattr(chunk, 'page_numbers'):
                page_nums = chunk.page_numbers
                page_num = page_nums[0] if page_nums else 0
            elif hasattr(chunk, 'metadata'):
                page_num = chunk.metadata.get('page', 0)
            else:
                page_num = 0

            if page_num not in pages_map:
                pages_map[page_num] = []

            pages_map[page_num].append(chunk)

        return pages_map

    def _create_document_pages(
        self,
        pages_map: Dict[int, List[Any]]
    ) -> List[DocumentPage]:
        """创建文档页面"""
        pages = []

        for page_num in sorted(pages_map.keys()):
            page_chunks = pages_map[page_num]

            # 合并内容
            content = "\n\n".join([
                getattr(chunk, 'content', '')
                for chunk in page_chunks
            ])

            # 提取图片和表格
            images = []
            tables = []

            for chunk in page_chunks:
                if hasattr(chunk, 'metadata'):
                    metadata = chunk.metadata

                    # 检查是否有图片
                    if metadata.get('has_images'):
                        images.append({
                            "chunk_id": getattr(chunk, 'chunk_id', ''),
                            "content": getattr(chunk, 'content', '')[:200],
                            "metadata": metadata
                        })

                    # 检查是否有表格
                    if metadata.get('has_tables'):
                        tables.append({
                            "chunk_id": getattr(chunk, 'chunk_id', ''),
                            "content": getattr(chunk, 'content', ''),
                            "metadata": metadata
                        })

            page = DocumentPage(
                page_number=page_num,
                content=content,
                chunks=[self._chunk_to_dict(c) for c in page_chunks],
                images=images,
                tables=tables,
                metadata={
                    "chunk_count": len(page_chunks),
                    "image_count": len(images),
                    "table_count": len(tables)
                }
            )

            pages.append(page)

        return pages

    def _chunk_to_dict(self, chunk: Any) -> Dict[str, Any]:
        """转换chunk为字典"""
        if hasattr(chunk, 'to_dict'):
            return chunk.to_dict()
        elif hasattr(chunk, 'content'):
            return {
                "content": chunk.content,
                "chunk_id": getattr(chunk, 'chunk_id', ''),
                "chunk_index": getattr(chunk, 'chunk_index', 0),
                "page_numbers": getattr(chunk, 'page_numbers', []),
                "metadata": getattr(chunk, 'metadata', {})
            }
        else:
            return str(chunk)

    async def _save_to_json(self, doc: StructuredDocument) -> str:
        """保存为JSON文件"""
        # 创建文档目录
        doc_dir = self.base_dir / doc.document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整文档JSON
        json_path = doc_dir / "document.json"
        async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
            await f.write(
                json.dumps(doc.to_dict(), ensure_ascii=False, indent=2)
            )

        # 保存Markdown（保持顺序）
        md_path = doc_dir / "document.md"
        await self._save_to_markdown(doc, md_path)

        # 保存按页面分割的文件
        for page in doc.pages:
            page_path = doc_dir / f"page_{page.page_number:04d}.json"
            async with aiofiles.open(page_path, 'w', encoding='utf-8') as f:
                await f.write(
                    json.dumps(page.to_dict(), ensure_ascii=False, indent=2)
                )

        logger.info(f"结构化文档已保存: {json_path}")
        return str(json_path)

    async def _save_to_markdown(self, doc: StructuredDocument, md_path: Path):
        """保存为Markdown（保持页面顺序）"""
        lines = []

        # 添加标题
        lines.append(f"# {doc.filename}\n")
        lines.append(f"**文档ID:** {doc.document_id}\n")
        lines.append(f"**总页数:** {doc.total_pages}\n")
        lines.append(f"**处理时间:** {doc.processing_time:.2f}秒\n")
        lines.append("\n---\n\n")

        # 按页面添加内容
        for page in doc.pages:
            lines.append(f"## 第 {page.page_number} 页\n")
            lines.append(f"**内容长度:** {len(page.content)} 字符\n")
            lines.append(f"**分块数:** {len(page.chunks)}\n")
            lines.append(f"**图片数:** {len(page.images)}\n")
            lines.append(f"**表格数:** {len(page.tables)}\n")

            # 如果有表格，先显示
            if page.tables:
                lines.append("\n### 表格\n")
                for i, table in enumerate(page.tables):
                    lines.append(f"#### 表格 {i+1}\n")
                    lines.append(f"{table['content']}\n")
                    lines.append(f"*Chunk ID: {table['chunk_id']}*\n")

            # 如果有图片
            if page.images:
                lines.append("\n### 图片\n")
                for i, image in enumerate(page.images):
                    lines.append(f"#### 图片 {i+1}\n")
                    lines.append(f"{image['content']}\n")
                    lines.append(f"*Chunk ID: {image['chunk_id']}*\n")

            # 页面内容
            lines.append("\n### 内容\n")
            lines.append(page.content)
            lines.append("\n\n---\n\n")

        # 写入文件
        async with aiofiles.open(md_path, 'w', encoding='utf-8') as f:
            await f.write('\n'.join(lines))

    async def load_document(self, document_id: str) -> Optional[StructuredDocument]:
        """加载结构化文档"""
        try:
            doc_dir = self.base_dir / document_id
            json_path = doc_dir / "document.json"

            if not json_path.exists():
                return None

            async with aiofiles.open(json_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)

            # 重建StructuredDocument对象
            pages = [
                DocumentPage(**page_data)
                for page_data in data.get('pages', [])
            ]

            doc = StructuredDocument(
                document_id=data['document_id'],
                filename=data['filename'],
                total_pages=data['total_pages'],
                pages=pages,
                metadata=data.get('metadata', {}),
                processing_time=data.get('processing_time', 0.0)
            )

            return doc

        except Exception as e:
            logger.error(f"加载结构化文档失败: {e}")
            return None

# 全局实例
structured_file_storage = StructuredFileStorageService()
