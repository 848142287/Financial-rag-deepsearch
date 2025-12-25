"""
后台任务
"""

import os
import glob
import logging
import asyncio
from datetime import datetime, timedelta
from celery import current_task
from app.core.celery import celery_app
# DEPRECATED: Use ConsolidatedDocumentService instead - from app.services.consolidated_document_service import ConsolidatedDocumentService document_parser
from app.services.embedding_service import embedding_service
from app.services.milvus_service import milvus_service

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="process_document")
def process_document(document_id: int, file_path: str, file_type: str):
    """处理文档的异步任务"""
    try:
        logger.info(f"开始处理文档: {document_id}, 文件类型: {file_type}")

        # 更新任务状态
        task_id = current_task.request.id

        # 创建新的事件循环来处理异步调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 解析文档
        result = loop.run_until_complete(document_parser.parse_document(file_path))

        # 文本分块
        chunks = document_parser.chunk_text(
            result['text_content'],
            chunk_size=512,
            chunk_overlap=50
        )

        # 生成嵌入向量
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = loop.run_until_complete(embedding_service.get_embeddings(chunk_texts))

        # 将嵌入添加到分块数据
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk['embedding'] = embedding

        # 存储到Milvus
        embedding_ids = loop.run_until_complete(milvus_service.insert_embeddings(
            document_id=document_id,
            chunks=chunks
        ))

        logger.info(f"文档处理完成: {document_id}, 生成 {len(chunks)} 个分块")

        return {
            "document_id": document_id,
            "chunks_count": len(chunks),
            "embedding_ids": embedding_ids,
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"文档处理失败: {document_id}, 错误: {e}")
        raise


@celery_app.task(bind=True, name="bulk_process_documents")
def bulk_process_documents(document_list):
    """批量处理文档"""
    results = []

    for doc_info in document_list:
        try:
            # 异步处理单个文档
            result = process_document.delay(
                doc_info['document_id'],
                doc_info['file_path'],
                doc_info['file_type']
            )
            results.append({
                "document_id": doc_info['document_id'],
                "task_id": result.id,
                "status": "queued"
            })
        except Exception as e:
            logger.error(f"批量处理文档失败: {doc_info['document_id']}, 错误: {e}")
            results.append({
                "document_id": doc_info['document_id'],
                "status": "failed",
                "error": str(e)
            })

    return results


@celery_app.task(name="cleanup_temp_files")
def cleanup_temp_files():
    """清理临时文件"""
    try:
        temp_dirs = [
            "/tmp/mineru_output",
            "/app/uploads/temp",
            "/tmp/magic_pdf_output"
        ]

        cleaned_files = 0
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                # 清理超过24小时的文件
                cutoff_time = datetime.now() - timedelta(hours=24)

                for file_path in glob.glob(os.path.join(temp_dir, "*")):
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            cleaned_files += 1

        logger.info(f"临时文件清理完成，删除了 {cleaned_files} 个文件")
        return {"cleaned_files": cleaned_files}

    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")
        raise


@celery_app.task(name="health_check")
def health_check():
    """系统健康检查"""
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "services": {}
        }

        # 检查各个服务的健康状态
        # 这里可以添加对各种服务的健康检查
        # 例如：数据库连接、向量数据库、图数据库等

        return health_status

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "unhealthy",
            "error": str(e)
        }


@celery_app.task(name="index_document")
def index_document(document_id: int):
    """索引文档到向量数据库"""
    try:
        # 实现文档索引逻辑
        # 这里会调用向量化和索引服务

        logger.info(f"开始索引文档: {document_id}")

        # 模拟索引过程
        import time
        time.sleep(2)

        logger.info(f"文档索引完成: {document_id}")

        return {
            "document_id": document_id,
            "status": "indexed",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"文档索引失败: {document_id}, 错误: {e}")
        raise


@celery_app.task(name="update_embeddings")
def update_embeddings(document_id: int):
    """更新文档的嵌入向量"""
    try:
        logger.info(f"开始更新文档嵌入: {document_id}")

        # 实现嵌入向量更新逻辑
        # 这里会重新计算并更新向量数据库中的嵌入

        return {
            "document_id": document_id,
            "status": "updated",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"更新文档嵌入失败: {document_id}, 错误: {e}")
        raise