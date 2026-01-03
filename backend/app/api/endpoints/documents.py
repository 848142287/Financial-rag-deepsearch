"""
文档管理API端点
提供文档的CRUD操作
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.core.database import get_async_db
from app.core.structured_logging import get_structured_logger
from app.models.document import Document, DocumentStatus

logger = get_structured_logger(__name__)
router = APIRouter(tags=["文档管理"])

# ============================================================================
# 响应模型
# ============================================================================

class DocumentListItem(BaseModel):
    """文档列表项"""
    id: int
    title: str
    filename: str
    status: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processing_result: Optional[dict] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentListItem]
    total: int
    page: int
    page_size: int
    total_pages: int

class DocumentDetail(BaseModel):
    """文档详情"""
    id: int
    title: str
    filename: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    content_type: Optional[str] = None
    status: str
    task_id: Optional[str] = None
    processing_mode: Optional[str] = None
    error_message: Optional[str] = None
    processing_result: Optional[dict] = None
    retry_count: int = 0
    doc_metadata: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processed_at: Optional[str] = None

    class Config:
        from_attributes = True

# ============================================================================
# API端点
# ============================================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="按状态筛选"),
    file_type: Optional[str] = Query(None, description="按文件类型筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    db: AsyncSession = Depends(get_async_db)
):
    """
    获取文档列表

    支持分页、状态筛选、文件类型筛选和关键词搜索
    """
    try:
        # 构建查询
        query = select(Document)

        # 状态筛选
        if status:
            query = query.where(Document.status == status)

        # 文件类型筛选
        if file_type:
            query = query.where(Document.file_type == file_type)

        # 搜索筛选（标题或文件名）
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                (Document.title.ilike(search_pattern)) |
                (Document.filename.ilike(search_pattern))
            )

        # 获取总数
        count_query = select(func.count()).select_from(query.alias())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # 分页和排序
        query = query.order_by(desc(Document.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)

        # 执行查询
        result = await db.execute(query)
        documents = result.scalars().all()

        # 计算总页数
        total_pages = (total + page_size - 1) // page_size

        # 转换为响应模型
        document_items = []
        for doc in documents:
            document_items.append(DocumentListItem(
                id=doc.id,
                title=doc.title,
                filename=doc.filename,
                status=doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                file_type=doc.file_type,
                file_size=doc.file_size,
                created_at=doc.created_at.isoformat() if doc.created_at else None,
                updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
                processing_result=doc.processing_result,
                error_message=doc.error_message
            ))

        return DocumentListResponse(
            documents=document_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """
    获取文档详情
    """
    try:
        # 查询文档
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")

        return DocumentDetail(
            id=document.id,
            title=document.title,
            filename=document.filename,
            file_path=document.file_path,
            file_size=document.file_size,
            file_type=document.file_type,
            content_type=document.content_type,
            status=document.status.value if isinstance(document.status, DocumentStatus) else document.status,
            task_id=document.task_id,
            processing_mode=document.processing_mode,
            error_message=document.error_message,
            processing_result=document.processing_result,
            retry_count=document.retry_count or 0,
            doc_metadata=document.doc_metadata,
            created_at=document.created_at.isoformat() if document.created_at else None,
            updated_at=document.updated_at.isoformat() if document.updated_at else None,
            processed_at=document.processed_at.isoformat() if document.processed_at else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档详情失败: {str(e)}")

__all__ = ["router"]
