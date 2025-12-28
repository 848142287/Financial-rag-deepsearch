"""
统一任务管理器 - 整合所有分散的任务功能
将文档处理、搜索、分析等功能集中管理
"""

from celery import current_task
from app.core.celery import celery_app  # 使用主celery配置
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

# 导入统一服务整合器
from app.services.core_service_integrator import get_service_integrator

# 导入数据库相关 - 移到顶部避免作用域问题
from app.core.database import SessionLocal
from app.models.document import Document

logger = logging.getLogger(__name__)


def run_async(coro):
    """运行异步协程的同步包装器"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(bind=True)
def process_document_unified(self, document_id: str, original_filename: str, user_id: str = None):
    """
    统一的文档处理任务
    整合所有分散的文档处理功能，使用统一的服务整合器
    """
    task_id = self.request.id

    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': '初始化服务', 'progress': 5}
        )

        # 获取统一服务整合器
        integrator = get_service_integrator()
        run_async(integrator.initialize())

        self.update_state(
            state='PROGRESS',
            meta={'status': '获取文档内容', 'progress': 15}
        )

        # 获取文档内容
        from app.services.minio_service import MinIOService

        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise Exception(f"文档不存在: {document_id}")

            # 下载文件内容
            minio_service = MinIOService()

            async def get_file_content():
                # 修复：移除 file_path 中的 'documents/' 前缀（如果存在）
                # 因为 MinIO 中实际存储路径是 '2025/12/27/...' 而不是 'documents/2025/12/27/...'
                actual_path = document.file_path
                if actual_path.startswith('documents/'):
                    actual_path = actual_path[len('documents/'):]

                content = await minio_service.download_file(actual_path)
                if content is None:
                    raise Exception(f"文件下载失败: {document.file_path}")
                return content

            file_content = run_async(get_file_content())

        finally:
            db.close()

        self.update_state(
            state='PROGRESS',
            meta={'status': '执行完整处理流水线', 'progress': 25}
        )

        # 使用统一服务整合器处理文档
        result = run_async(integrator.process_document_complete(
            file_content, original_filename, document_id
        ))

        # 更新数据库
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                # 更新处理状态和结果
                document.processing_mode = "unified_pipeline"
                # 正确存储：processing_result 存储元数据，parsed_content 存储文本内容
                import json
                document.processing_result = json.dumps(result, ensure_ascii=False)

                # 从结果中提取文本内容
                stages = result.get('stages', {})
                if 'analysis' in stages:
                    # 尝试获取解析的文本
                    parsed_text = result.get('parsed_text', '')
                    markdown = result.get('markdown', '')
                    document.parsed_content = markdown or parsed_text or str(result)
                else:
                    document.parsed_content = str(result)

                if result.get('success'):
                    document.status = 'completed'
                    document.processed_at = datetime.now()
                else:
                    document.status = 'PROCESSING_FAILED'
                    document.error_message = result.get('error', 'Unknown error')

                db.commit()

        finally:
            db.close()

        self.update_state(
            state='SUCCESS',
            meta={'status': '处理完成', 'progress': 100, 'result': result}
        )

        logger.info(f"✅ 统一文档处理完成: {document_id}")
        return result

    except Exception as e:
        logger.error(f"❌ 统一文档处理失败 {document_id}: {e}")

        # 更新错误状态
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.status = 'PROCESSING_FAILED'
                document.error_message = str(e)
                db.commit()
        finally:
            db.close()

        # 重试逻辑
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True)
def search_documents_unified(self, query: str, top_k: int = 10, filters: Optional[Dict] = None):
    """
    统一的文档搜索任务
    整合向量搜索和知识图谱搜索
    """
    task_id = self.request.id

    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': '初始化搜索服务', 'progress': 20}
        )

        # 获取统一服务整合器
        integrator = get_service_integrator()
        run_async(integrator.initialize())

        self.update_state(
            state='PROGRESS',
            meta={'status': '执行搜索', 'progress': 60}
        )

        # 执行搜索
        results = run_async(integrator.search_documents(query, top_k, filters))

        self.update_state(
            state='SUCCESS',
            meta={'status': '搜索完成', 'progress': 100, 'result_count': len(results)}
        )

        return {
            'query': query,
            'results': results,
            'total_count': len(results),
            'task_id': task_id
        }

    except Exception as e:
        logger.error(f"❌ 统一搜索失败: {e}")
        raise self.retry(exc=e, countdown=30, max_retries=2)


@celery_app.task(bind=True)
def system_health_check(self):
    """
    系统健康检查任务
    检查所有服务的状态
    """
    try:
        integrator = get_service_integrator()

        # 检查服务状态
        status = run_async(integrator.get_service_status())
        config_summary = integrator.get_config_summary()

        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': status['services'],
            'config': config_summary
        }

    except Exception as e:
        logger.error(f"❌ 系统健康检查失败: {e}")
        return {
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }


# 向后兼容的别名
process_document_complete = process_document_unified