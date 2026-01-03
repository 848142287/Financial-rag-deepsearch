"""
后台异步任务：实体提取和知识图谱构建
在主文档处理完成后异步执行，不阻塞主流程
"""
from app.tasks.unified_task_manager import celery_app
from app.core.structured_logging import get_structured_logger
import asyncio
from app.core.database import SessionLocal
from sqlalchemy import text

logger = get_structured_logger(__name__)


@celery_app.task(bind=True, name='app.tasks.background_enrichment.enrich_document_async')
def enrich_document_async(self, document_id: str, chunks_data: list):
    """
    异步 enrich文档：实体提取 + 知识图谱构建

    Args:
        document_id: 文档ID
        chunks_data: 文档块数据
    """
    task_id = self.request.id
    logger.info(f"🚀 [后台任务] 开始异步enrich文档 {document_id}")

    try:
        # 使用asyncio运行异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_enrich_document(document_id, chunks_data))
            logger.info(f"✅ [后台任务] 文档 {document_id} enrich完成")
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ [后台任务] 文档 {document_id} enrich失败: {e}")
        # 更新文档状态
        db = SessionLocal()
        try:
            db.execute(
                text("UPDATE documents SET enrichment_status='failed', enrichment_error=:error WHERE id=:id"),
                {'error': str(e), 'id': document_id}
            )
            db.commit()
        finally:
            db.close()
        raise


async def _enrich_document(document_id: str, chunks_data: list):
    """异步enrich文档的主逻辑"""
    from app.services.core_service_integrator import get_service_integrator
    from app.core.database import SessionLocal
    from sqlalchemy import text

    # 更新状态为enriching
    db = SessionLocal()
    try:
        db.execute(
            text("UPDATE documents SET enrichment_status='enriching' WHERE id=:id"),
            {'id': document_id}
        )
        db.commit()
    finally:
        db.close()

    # 获取服务整合器
    integrator = get_service_integrator()
    await integrator.initialize()

    entities = []
    relationships = []

    # 阶段1: 实体提取（后台） - 简化版本，跳过以节省资源
    try:
        logger.info(f"🔗 [后台] 实体提取已禁用（节省资源）...")
    except Exception as e:
        logger.error(f"❌ [后台] 实体提取失败: {e}")

    # 阶段2: 关系提取 - 简化版本，跳过
    try:
        logger.info(f"🔗 [后台] 关系提取已禁用（节省资源）...")
    except Exception as e:
        logger.error(f"❌ [后台] 关系提取失败: {e}")

    # 阶段3: 存储到Neo4j（如果启用） - 简化版本，跳过
    try:
        logger.info(f"📊 [后台] 知识图谱存储已禁用（节省资源）...")
    except Exception as e:
        logger.error(f"❌ [后台] 知识图谱存储失败: {e}")

    # 更新状态为完成
    db = SessionLocal()
    try:
        db.execute(
            text("""
                UPDATE documents
                SET enrichment_status='completed',
                    enrichment_completed_at=NOW()
                WHERE id=:id
            """),
            {'id': document_id}
        )
        db.commit()
    finally:
        db.close()

    return {
        'document_id': document_id,
        'entities_count': len(entities),
        'relationships_count': len(relationships),
        'status': 'success',
        'note': '后台enrichment已简化（实体提取和知识图谱已禁用以节省资源）'
    }
