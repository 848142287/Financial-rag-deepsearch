"""
统一文档处理API
整合文档管理和文档处理流水线的功能

特性：
- 统一的文档上传接口
- 完整的文档处理流水线
- 文档CRUD操作
- 处理进度查询
- 批量操作支持
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.core.database import get_async_db
from app.core.structured_logging import get_structured_logger
from app.models.document import Document, DocumentStatus
from app.services.orchestration.unified_orchestrator import get_orchestrator

logger = get_structured_logger(__name__)

router = APIRouter(prefix="/documents", tags=["统一文档处理"])

# ============================================================================
# 请求/响应模型
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    document_id: str
    filename: str
    status: str
    message: str
    processing_url: Optional[str] = None

class DocumentListItem(BaseModel):
    """文档列表项"""
    id: int
    document_id: str
    title: str
    filename: str
    status: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processing_progress: Optional[float] = 0.0
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
    document_id: str
    title: str
    filename: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    status: str
    processing_progress: Optional[float] = 0.0
    current_step: Optional[str] = None
    processing_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processed_at: Optional[str] = None

    class Config:
        from_attributes = True

class ProcessingOptions(BaseModel):
    """处理选项"""
    enable_parsing: bool = True
    enable_chunking: bool = True
    enable_embedding: bool = True
    enable_vector_storage: bool = True
    enable_entity_extraction: bool = False
    enable_knowledge_graph: bool = False

    # 高级选项
    parsing_mode: str = "auto"  # auto, fast, accurate
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

# ============================================================================
# 统一文档处理API
# ============================================================================

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    title: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
    processing_options: Optional[str] = None,  # JSON字符串
    db: AsyncSession = Depends(get_async_db)
):
    """
    统一文档上传接口

    功能：
    1. 上传文档文件
    2. 创建文档记录
    3. 启动处理流水线（可选同步/异步）

    支持的文件类型：.pdf, .xlsx, .xls, .docx, .pptx, .ppt, .md, .txt

    参数：
    - file: 文档文件
    - document_id: 可选的文档ID（自动生成如果不提供）
    - title: 可选的标题（默认使用文件名）
    - background_tasks: 是否后台处理（默认True）
    - processing_options: 处理选项（JSON字符串）
    """
    try:
        # 生成文档ID
        if not document_id:
            document_id = str(uuid.uuid4())

        # 生成标题
        if not title:
            title = Path(file.filename).stem if file.filename else document_id

        # 读取文件内容
        file_content = await file.read()
        file_size = len(file_content)
        file_type = Path(file.filename).suffix.lower().lstrip('.')

        # 创建文档记录
        document = Document(
            document_id=document_id,
            title=title,
            filename=file.filename or f"{document_id}.{file_type}",
            file_size=file_size,
            file_type=file_type,
            status=DocumentStatus.PENDING,
            processing_progress=0.0,
            current_step="等待处理..."
        )

        db.add(document)
        await db.commit()
        await db.refresh(document)

        logger.info(f"文档记录已创建: {document_id} - {title}")

        # 解析处理选项
        options = {
            "enable_parsing": True,
            "enable_chunking": True,
            "enable_embedding": True,
            "enable_vector_storage": True,
            "enable_entity_extraction": False,
            "enable_knowledge_graph": False
        }

        if processing_options:
            import json
            try:
                options.update(json.loads(processing_options))
            except json.JSONDecodeError:
                logger.warning(f"处理选项解析失败，使用默认配置")

        # 是否后台处理
        if background_tasks is None:
            # 默认后台处理
            background_tasks = BackgroundTasks()

        # 添加后台处理任务
        background_tasks.add_task(
            _process_document_background,
            document_id,
            file_content,
            file.filename or f"{document_id}.{file_type}",
            options
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename or f"{document_id}.{file_type}",
            status="processing",
            message="文档已上传，正在后台处理",
            processing_url=f"/api/v1/documents/{document_id}"
        )

    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")

@router.post("/upload-sync", response_model=DocumentDetail)
async def upload_document_sync(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    title: Optional[str] = None,
    enable_parsing: bool = True,
    enable_chunking: bool = True,
    enable_embedding: bool = True,
    enable_vector_storage: bool = True,
    db: AsyncSession = Depends(get_async_db)
):
    """
    同步文档上传和处理

    与upload端点相同，但同步等待处理完成
    适用于小文件或需要立即获得结果的场景
    """
    try:
        # 生成文档ID
        if not document_id:
            document_id = str(uuid.uuid4())

        # 生成标题
        if not title:
            title = Path(file.filename).stem if file.filename else document_id

        # 读取文件内容
        file_content = await file.read()
        file_size = len(file_content)
        file_type = Path(file.filename).suffix.lower().lstrip('.')

        # 创建文档记录
        document = Document(
            document_id=document_id,
            title=title,
            filename=file.filename or f"{document_id}.{file_type}",
            file_size=file_size,
            file_type=file_type,
            status=DocumentStatus.PROCESSING,
            processing_progress=0.0,
            current_step="初始化..."
        )

        db.add(document)
        await db.commit()
        await db.refresh(document)

        logger.info(f"文档记录已创建: {document_id} - {title}")

        # 同步处理
        orchestrator = get_orchestrator()
        await orchestrator.initialize()

        # 保存文件到临时位置
        import tempfile
        import os

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}")
        temp_file.write(file_content)
        temp_file.close()
        temp_file_path = temp_file.name

        try:
            # 处理文档
            result = await orchestrator.process_document(
                file_path=temp_file_path,
                document_id=document_id,
                filename=file.filename or f"{document_id}.{file_type}"
            )

            # 更新文档记录
            if result.success:
                document.status = DocumentStatus.COMPLETED
                document.processing_progress = 100.0
                document.current_step = "完成"
                document.processed_at = datetime.utcnow()
                document.processing_result = {
                    "stages": [stage.to_dict() for stage in result.stages],
                    "metrics": result.metrics
                }
            else:
                document.status = DocumentStatus.FAILED
                document.error_message = result.error

            await db.commit()
            await db.refresh(document)

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass

        return DocumentDetail(
            id=document.id,
            document_id=document.document_id,
            title=document.title,
            filename=document.filename,
            file_path=document.file_path,
            file_size=document.file_size,
            file_type=document.file_type,
            status=document.status.value,
            processing_progress=float(document.processing_progress or 0),
            current_step=document.current_step,
            processing_result=document.processing_result,
            error_message=document.error_message,
            created_at=document.created_at.isoformat() if document.created_at else None,
            updated_at=document.updated_at.isoformat() if document.updated_at else None,
            processed_at=document.processed_at.isoformat() if document.processed_at else None
        )

    except Exception as e:
        logger.error(f"同步文档处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

@router.get("", response_model=DocumentListResponse)
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
                document_id=doc.document_id,
                title=doc.title,
                filename=doc.filename,
                status=doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                file_type=doc.file_type,
                file_size=doc.file_size,
                created_at=doc.created_at.isoformat() if doc.created_at else None,
                updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
                processing_progress=float(doc.processing_progress or 0),
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

@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_async_db)
):
    """
    获取文档详情

    参数：
    - document_id: 文档ID
    """
    try:
        # 查询文档
        result = await db.execute(
            select(Document).where(Document.document_id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")

        return DocumentDetail(
            id=document.id,
            document_id=document.document_id,
            title=document.title,
            filename=document.filename,
            file_path=document.file_path,
            file_size=document.file_size,
            file_type=document.file_type,
            status=document.status.value if isinstance(document.status, DocumentStatus) else document.status,
            processing_progress=float(document.processing_progress or 0),
            current_step=document.current_step,
            processing_result=document.processing_result,
            error_message=document.error_message,
            created_at=document.created_at.isoformat() if document.created_at else None,
            updated_at=document.updated_at.isoformat() if document.updated_at else None,
            processed_at=document.processed_at.isoformat() if document.processed_at else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档详情失败: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    delete_vector: bool = Query(True, description="是否删除向量数据"),
    db: AsyncSession = Depends(get_async_db)
):
    """
    删除文档

    参数：
    - document_id: 文档ID
    - delete_vector: 是否同时删除向量数据
    """
    try:
        # 查询文档
        result = await db.execute(
            select(Document).where(Document.document_id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")

        # 删除向量数据
        if delete_vector:
            try:
                from app.services.vectorstore.milvus_vector_store import get_milvus_store
                store = get_milvus_store()
                await store.delete_document(document_id)
                logger.info(f"已删除文档 {document_id} 的向量数据")
            except Exception as e:
                logger.warning(f"删除向量数据失败: {e}")

        # 删除数据库记录
        await db.delete(document)
        await db.commit()

        logger.info(f"文档已删除: {document_id}")

        return {
            "success": True,
            "message": f"文档 {document_id} 已删除",
            "document_id": document_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

# ============================================================================
# 后台处理函数
# ============================================================================

async def _process_document_background(
    document_id: str,
    file_content: bytes,
    filename: str,
    options: Dict[str, Any]
):
    """后台处理文档"""
    import tempfile
    import os
    from app.core.database import get_async_db_context

    logger.info(f"开始后台处理文档: {document_id}")

    temp_file_path = None
    try:
        # 获取数据库会话
        async with get_async_db_context() as db:
            # 更新文档状态
            result = await db.execute(
                select(Document).where(Document.document_id == document_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                logger.error(f"文档不存在: {document_id}")
                return

            # 更新状态为处理中
            document.status = DocumentStatus.PROCESSING
            document.processing_progress = 0.0
            document.current_step = "初始化..."
            await db.commit()

            # 保存文件到临时位置
            file_type = Path(filename).suffix.lower().lstrip('.')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}")
            temp_file.write(file_content)
            temp_file.close()
            temp_file_path = temp_file.name

            # 获取编排器
            orchestrator = get_orchestrator()
            await orchestrator.initialize()

            # 处理文档
            logger.info(f"开始处理文档: {document_id}")
            process_result = await orchestrator.process_document(
                file_path=temp_file_path,
                document_id=document_id,
                filename=filename
            )

            # 更新处理结果
            if process_result.success:
                document.status = DocumentStatus.COMPLETED
                document.processing_progress = 100.0
                document.current_step = "完成"
                document.processed_at = datetime.utcnow()
                document.processing_result = {
                    "stages": [stage.to_dict() for stage in process_result.stages],
                    "metrics": process_result.metrics
                }
                logger.info(f"文档处理成功: {document_id}")
            else:
                document.status = DocumentStatus.FAILED
                document.error_message = process_result.error
                logger.error(f"文档处理失败: {document_id} - {process_result.error}")

            await db.commit()

    except Exception as e:
        logger.error(f"后台处理文档失败: {document_id} - {e}")

        # 更新错误状态
        try:
            async with get_async_db_context() as db:
                result = await db.execute(
                    select(Document).where(Document.document_id == document_id)
                )
                document = result.scalar_one_or_none()
                if document:
                    document.status = DocumentStatus.FAILED
                    document.error_message = str(e)
                    await db.commit()
        except Exception as db_error:
            logger.error(f"更新错误状态失败: {db_error}")

    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

# 导出
__all__ = ['router']
