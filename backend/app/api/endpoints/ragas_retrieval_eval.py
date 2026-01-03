"""
RAGAS检索评估API端点
提供检索质量评估功能
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_db
from app.core.structured_logging import get_structured_logger
from app.models.document import Document

logger = get_structured_logger(__name__)
router = APIRouter(tags=["RAGAS检索评估"])

# ============================================================================
# 响应模型
# ============================================================================

class EvalDocumentItem(BaseModel):
    """评估文档项"""
    id: int
    title: str
    filename: str
    status: str
    created_at: Optional[str] = None

    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[EvalDocumentItem]
    total: int

# ============================================================================
# API端点
# ============================================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_evaluation_documents(
    limit: int = Query(50, ge=1, le=100, description="返回数量限制"),
    status: Optional[str] = Query(None, description="按状态筛选"),
    db: AsyncSession = Depends(get_async_db)
):
    """
    获取可用于评估的文档列表
    """
    try:
        # 构建查询
        query = select(Document)

        # 只返回已完成的文档
        if status:
            query = query.where(Document.status == status)
        else:
            # 默认只返回已完成的文档
            query = query.where(Document.status == "completed")

        # 排序和限制
        query = query.order_by(desc(Document.created_at)).limit(limit)

        # 执行查询
        result = await db.execute(query)
        documents = result.scalars().all()

        # 简单计数 - 使用len而不是单独的count查询
        total = len(documents)

        # 转换为响应模型
        document_items = []
        for doc in documents:
            document_items.append(EvalDocumentItem(
                id=doc.id,
                title=doc.title,
                filename=doc.filename,
                status=doc.status.value if hasattr(doc.status, 'value') else doc.status,
                created_at=doc.created_at.isoformat() if doc.created_at else None
            ))

        return DocumentListResponse(
            documents=document_items,
            total=total
        )

    except Exception as e:
        logger.error(f"获取评估文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

__all__ = ["router"]
