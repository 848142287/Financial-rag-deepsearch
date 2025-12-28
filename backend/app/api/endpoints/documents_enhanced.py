"""
增强的文档管理API端点
集成文档去重功能
"""

import logging
import os
import tempfile
import aiofiles
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse

from app.core.database import get_db
from app.models.document import Document
from app.services.upload_service import UploadService
from app.services.document_deduplication import document_deduplication_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload-with-deduplication")
async def upload_with_deduplication(
    file: UploadFile = File(..., description="上传的文件"),
    title: Optional[str] = Form(None, description="文档标题"),
    description: Optional[str] = Form(None, description="文档描述"),
    category: Optional[str] = Form(None, description="文档分类"),
    tags: Optional[str] = Form(None, description="文档标签，逗号分隔"),
    skip_deduplication: bool = Form(False, description="跳过去重检查"),
    force_upload: bool = Form(False, description="强制上传"),
    db: Session = Depends(get_db)
):
    """
    增强的文档上传API，集成去重检查功能
    """
    try:
        # 如果没有提供title，从文件名提取
        if not title and file.filename:
            # 移除文件扩展名作为默认标题
            title = os.path.splitext(file.filename)[0]

        # 设置默认值
        if not description:
            description = f"文件: {file.filename}"
        if not category:
            category = "券商研报"
        if not tags:
            tags = "研报,文档"

        logger.info(f"收到文档上传请求: {file.filename}, 标题: {title}")

        # 读取文件内容用于哈希计算
        binary_content = b""  # 保存原始二进制内容
        text_content_for_hash = ""  # 用于哈希计算的文本内容
        file_path = ""

        # 创建临时文件
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name

            # 读取原始二进制内容
            binary_content = await file.read()

            # 写入临时文件
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(binary_content)

            # 尝试读取文本内容用于哈希计算
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    text_content_for_hash = await f.read()

                # 如果是图片等二进制文件，使用二进制内容哈希
                if not text_content_for_hash.strip():
                    logger.info(f"文件 {file.filename} 是二进制文件，使用二进制内容计算哈希")
                    text_content_for_hash = binary_content.decode('latin-1')  # 用于哈希计算

            except UnicodeDecodeError:
                # 无法解码为文本，使用二进制内容
                logger.info(f"文件 {file.filename} 无法解码为UTF-8，使用二进制内容计算哈希")
                text_content_for_hash = binary_content.decode('latin-1')

        # 确保使用实际的二进制文件大小
        actual_file_size = len(binary_content)
        logger.info(f"文件 {file.filename} 实际大小: {actual_file_size} bytes")

        # 计算文件哈希
        file_hash = await document_deduplication_service.calculate_file_hash(file_path)
        content_hash = document_deduplication_service.calculate_content_hash(text_content_for_hash)

        if not file_hash or not content_hash:
            raise HTTPException(status_code=400, detail="无法计算文件哈希")

        # 简化的去重检查 - 直接基于MD5哈希
        is_duplicate = False
        existing_document_id = None
        similarity_score = 0.0

        if not skip_deduplication and not force_upload:
            # 直接检查MySQL数据库中的重复
            db = next(get_db())
            try:
                existing_doc = db.query(Document).filter(
                    Document.file_hash == file_hash
                ).first()

                if existing_doc:
                    is_duplicate = True
                    existing_document_id = existing_doc.id
                    similarity_score = 1.0  # MD5完全匹配
                    logger.info(f"发现重复文档: file_hash={file_hash[:8]}, existing_id={existing_doc.id}")
            finally:
                db.close()

        # 执行文档处理和存储
        upload_service = UploadService()
        await upload_service.initialize()  # 确保服务已初始化

        # 保存文件到MinIO
        minio_path = await upload_service.save_file(file, file_path)

        # 保存元数据到数据库
        document = Document(
            title=title,
            filename=file.filename,
            file_path=minio_path,
            file_size=actual_file_size,  # 使用实际文件大小
            content_type=file.content_type or 'text/plain',
            file_hash=file_hash,  # 添加文件MD5哈希
            content_hash=content_hash,
            status="DUPLICATE" if is_duplicate else "PROCESSING",  # 根据重复检查结果设置状态
            doc_metadata={
                "description": description,
                "category": category,
                "tags": tags.split(',') if tags else [],
                "original_filename": file.filename,
                "upload_source": "api_upload",
                "duplicate_check": {
                    "is_duplicate": is_duplicate,
                    "similarity_score": similarity_score,
                    "existing_document_id": existing_document_id
                }
            }
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        # 修改：所有文档（包括重复文档）都需要触发完整的解析流水线
        # 重复文档信息仅作为元数据标记，不影响处理流程
        from app.tasks.unified_task_manager import process_document_unified

        try:
            # 所有文档都启动统一任务管理器处理（包括重复文档）
            task = process_document_unified.delay(str(document.id), file.filename)
            task_id = task.id

            # 更新文档为统一处理模式
            document.processing_mode = "unified_pipeline"
            document.task_id = task_id  # 保存任务ID到数据库

            if is_duplicate:
                # 重复文档也进行完整处理，但在元数据中标记
                logger.info(f"重复文档也将进行完整解析: ID={document.id}, 任务ID={task_id}, 相似文档ID={existing_document_id}")
                status_message = f"检测到重复文档(相似文档ID:{existing_document_id})，但仍将进行完整解析处理"
                response_status = "processing_duplicate"
                # 状态改为PROCESSING，确保会触发解析流水线
                document.status = "PROCESSING"
            else:
                # 非重复文档，正常处理
                logger.info(f"文档上传成功，启动解析流水线: ID={document.id}, 任务ID={task_id}, 文件={file.filename}")
                status_message = "文档上传成功，正在解析处理中"
                response_status = "processing"

            db.commit()

        except Exception as e:
            logger.error(f"统一文档处理任务启动失败: {e}")
            task_id = f"unified_{document.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # 任务启动失败时的处理
            if is_duplicate:
                status_message = "重复文档标记成功，但处理任务启动失败"
                response_status = "duplicate_failed"
            else:
                status_message = "文档上传成功，但处理任务启动失败"
                response_status = "processing_failed"

        # 删除临时文件
        try:
            os.unlink(file_path)
        except:
            pass

        return JSONResponse(
            content={
                "success": True,
                "message": status_message,
                "document_id": document.id,
                "task_id": task_id,
                "status": response_status,
                "file_path": minio_path,
                "size": actual_file_size,  # 使用实际文件大小
                "deduplication_check": {
                    "performed": not skip_deduplication,
                    "is_duplicate": is_duplicate if not skip_deduplication else False,
                    "similarity_score": similarity_score if not skip_deduplication else 0.0,
                    "existing_document_id": existing_document_id
                }
            }
        )

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}")

        # 清理临时文件
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")

@router.post("/batch-upload-with-deduplication")
async def batch_upload_with_deduplication(
    files: List[UploadFile] = File(..., description="要上传的文件列表"),
    skip_deduplication: bool = Form(False, description="跳过去重检查"),
    force_upload: bool = Form(False, description="强制上传"),
    db: Session = Depends(get_db)
):
    """
    批量文档上传，集成去重检查
    """
    try:
        results = []
        skipped_files = []
        duplicate_files = []

        for file in files:
            try:
                # 读取文件内容
                content = ""
                async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    file_path = temp_file.name

                    async with aiofiles.open(file_path, 'wb') as f:
                        file_content = await file.read()
                        await f.write(file_content)

                    # 尝试读取文本内容
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            text_content = await f.read()
                            content = text_content
                    except UnicodeDecodeError:
                        content = f"[二进制文件] {file.filename}, 大小: {len(file_content)} bytes"

                # 计算哈希
                file_hash = await document_deduplication_service.calculate_file_hash(file_path)
                content_hash = document_deduplication_service.calculate_content_hash(content)

                if not skip_deduplication and not force_upload:
                    # 检查重复
                    duplicate_check = await document_deduplication_service.check_document_duplication(
                        file_path=file_path,
                        content=content,
                        file_metadata={'filename': file.filename}
                    )

                    if duplicate_check.is_duplicate and not force_upload:
                        skipped_files.append({
                            'filename': file.filename,
                            'reason': duplicate_check.recommendations[0] if duplicate_check.recommendations else "文档已存在",
                            'similarity': duplicate_check.similarity_score
                        })
                        continue

                # 执行上传逻辑（这里简化，实际应该调用上传服务）
                result = {
                    'filename': file.filename,
                    'status': 'uploaded',
                    'size': len(file_content)
                }
                results.append(result)

                # 清理临时文件
                os.unlink(file_path)

            except Exception as e:
                logger.error(f"批量上传文件 {file.filename} 失败: {e}")
                skipped_files.append({
                    'filename': file.filename,
                    'reason': f"处理失败: {str(e)}",
                    'similarity': 0.0
                })

    except Exception as e:
        logger.error(f"批量上传处理失败: {e}")

    return JSONResponse(
            content={
                "success": True,
                "message": f"批量上传完成",
                "results": results,
                "skipped": skipped_files,
                "duplicates": duplicate_files,
                "total_processed": len(files),
                "success_count": len(results),
                "skipped_count": len(skipped_files)
            }
        )

@router.get("/upload-status/{task_id}")
async def get_upload_status(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    获取上传任务状态
    """
    try:
        # 这里应该从任务队列获取状态
        # 简化实现，实际应该查询Celery任务状态
        document = db.query(Document).filter(
            Document.task_id == task_id
        ).first()

        if not document:
            raise HTTPException(status_code=404, detail="任务不存在")

        return {
            "task_id": task_id,
            "document_id": document.id,
            "status": document.status,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None
        }

    except Exception as e:
        logger.error(f"获取上传状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取上传状态失败: {str(e)}")


@router.get("/{document_id}/preview")
async def preview_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    预览文档内容
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 尝试从MinIO获取文件内容
        content = ""
        upload_service = UploadService()
        await upload_service.initialize()

        try:
            # 从MinIO获取文件内容
            file_content = await upload_service.get_file_content(document.file_path)

            # 从metadata中提取description
            metadata = document.doc_metadata or {}
            description = metadata.get('description', '')

            # 根据文件类型处理内容
            if document.file_type.lower() in ['.txt', '.md']:
                content = file_content.decode('utf-8')
            elif document.file_type.lower() == '.pdf':
                # 对于PDF文件，返回基本信息
                content = f"PDF文档: {document.title}\n描述: {description or '无描述'}\n状态: {document.status}\n文件大小: {document.file_size} bytes"
            elif document.file_type.lower() in ['.docx', '.doc']:
                # 对于Word文档，返回基本信息
                content = f"Word文档: {document.title}\n描述: {description or '无描述'}\n状态: {document.status}\n文件大小: {document.file_size} bytes"
            else:
                # 其他格式，返回基本信息或文本内容
                try:
                    content = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    content = f"文档: {document.title}\n类型: {document.file_type}\n状态: {document.status}\n文件大小: {document.file_size} bytes\n文件路径: {document.file_path}"

        except Exception as e:
            logger.error(f"从MinIO读取文档内容失败: {e}")
            # 如果无法从MinIO获取，返回基本信息
            metadata = document.doc_metadata or {}
            description = metadata.get('description', '')
            content = f"文档: {document.title}\n类型: {document.file_type}\n状态: {document.status}\n文件大小: {document.file_size} bytes\n描述: {description or '无描述'}"

        # 从metadata中提取description
        metadata = document.doc_metadata or {}
        description = metadata.get('description', '')

        return {
            "document_id": document.id,
            "title": document.title,
            "description": description,
            "content": content,
            "file_path": document.file_path,
            "status": document.status,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预览文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"预览文档失败: {str(e)}")


@router.get("/")
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    title: Optional[str] = None,
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    获取文档列表，支持筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size

        # 构建查询
        query = db.query(Document)

        # 应用筛选条件
        if title:
            query = query.filter(Document.title.contains(title))
        if status:
            query = query.filter(Document.status == status.upper())
        if file_type:
            query = query.filter(Document.file_type == file_type)

        # 查询文档列表
        documents = query.offset(offset).limit(page_size).all()
        total = query.count()

        # 格式化返回结果
        document_list = []
        for doc in documents:
            # 从metadata中提取description
            metadata = doc.doc_metadata or {}
            description = metadata.get('description', '')

            # Handle null/empty filename
            file_name = doc.filename
            if not file_name or file_name == 'null':
                # Extract filename from file_path or use default
                if doc.file_path:
                    file_name = doc.file_path.split('/')[-1] or f"document_{doc.id}.dat"
                else:
                    file_name = f"document_{doc.id}.dat"

            document_list.append({
                "id": doc.id,
                "title": doc.title,
                "description": description,
                "file_name": file_name,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "category": metadata.get('category', ''),
                "tags": metadata.get('tags', []),
                "content_type": doc.content_type
            })

        return {
            "documents": document_list,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size
        }

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    完整删除文档及其所有相关数据（向量、图谱、缓存、文件等）
    """
    try:
        # 导入完整删除服务
        from app.services.document_deletion_service import document_deletion_service

        logger.info(f"开始完整删除文档: {document_id}")

        # 执行完整删除
        deletion_result = await document_deletion_service.delete_document_complete(document_id, db)

        if deletion_result["success"]:
            logger.info(f"文档 {document_id} 完整删除成功")
            return deletion_result
        else:
            logger.error(f"文档 {document_id} 删除失败: {deletion_result['message']}")
            raise HTTPException(status_code=500, detail=deletion_result["message"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档删除失败: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"文档删除失败: {str(e)}")


@router.delete("/batch")
async def batch_delete_documents(
    document_ids: List[int],
    db: Session = Depends(get_db)
):
    """
    批量完整删除文档及其所有相关数据
    """
    try:
        # 导入完整删除服务
        from app.services.document_deletion_service import document_deletion_service

        logger.info(f"开始批量删除文档: {document_ids}")

        if not document_ids:
            raise HTTPException(status_code=400, detail="文档ID列表不能为空")

        if len(document_ids) > 50:  # 限制批量删除数量
            raise HTTPException(status_code=400, detail="批量删除文档数量不能超过50个")

        deletion_results = {
            "success": True,
            "message": "批量删除操作完成",
            "total_documents": len(document_ids),
            "successful_deletions": [],
            "failed_deletions": [],
            "deletion_summary": {
                "total_deleted_items": 0,
                "details": {}
            },
            "errors": []
        }

        # 逐个删除文档
        for document_id in document_ids:
            try:
                logger.info(f"删除文档: {document_id}")

                # 执行完整删除
                deletion_result = await document_deletion_service.delete_document_complete(document_id, db)

                if deletion_result["success"]:
                    deletion_results["successful_deletions"].append({
                        "document_id": document_id,
                        "title": deletion_result.get("document_title", "Unknown"),
                        "deleted_items": deletion_result.get("total_deleted_items", 0)
                    })

                    # 累计删除项目数量
                    deletion_results["deletion_summary"]["total_deleted_items"] += deletion_result.get("total_deleted_items", 0)

                    # 保存详细删除信息
                    deletion_results["deletion_summary"]["details"][str(document_id)] = deletion_result.get("deletion_summary", {})

                    logger.info(f"文档 {document_id} 删除成功")
                else:
                    deletion_results["failed_deletions"].append({
                        "document_id": document_id,
                        "error": deletion_result.get("message", "删除失败")
                    })
                    deletion_results["errors"].append(f"文档 {document_id}: {deletion_result.get('message', '删除失败')}")

            except Exception as e:
                error_msg = f"文档 {document_id}: {str(e)}"
                deletion_results["failed_deletions"].append({
                    "document_id": document_id,
                    "error": error_msg
                })
                deletion_results["errors"].append(error_msg)
                logger.error(f"删除文档 {document_id} 失败: {e}")

        success_count = len(deletion_results["successful_deletions"])
        failed_count = len(deletion_results["failed_deletions"])

        # 设置最终状态和消息
        if failed_count == 0:
            deletion_results["success"] = True
            deletion_results["message"] = f"所有 {success_count} 个文档删除成功"
        elif success_count == 0:
            deletion_results["success"] = False
            deletion_results["message"] = f"所有 {failed_count} 个文档删除失败"
        else:
            deletion_results["success"] = True  # 部分成功
            deletion_results["message"] = f"成功删除 {success_count} 个文档，失败 {failed_count} 个文档"

        logger.info(f"批量删除完成: 成功 {success_count}, 失败 {failed_count}, 总计删除项目 {deletion_results['deletion_summary']['total_deleted_items']}")

        return JSONResponse(
            status_code=200 if success_count > 0 else 500,
            content=deletion_results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}")