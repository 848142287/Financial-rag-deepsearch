"""
异步后台任务系统
实现向量embedding和Neo4j知识图谱抽取的异步后台处理
不阻塞前端，支持多线程并发
"""

from app.tasks.unified_task_manager import celery_app
from app.core.structured_logging import get_structured_logger
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = get_structured_logger(__name__)

# 线程池执行器用于CPU密集型任务
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vector_kg_worker")

@celery_app.task(
    bind=True,
    name='app.tasks.async_vector_kg_tasks.vectorize_document_async',
    soft_time_limit=600,  # 10分钟
    max_retries=2,
    default_retry_delay=60
)
def vectorize_document_async(
    self,
    document_id: str,
    chunks_data: List[Dict[str, Any]],
    collection_name: str = "financial_documents"
):
    """
    异步向量化文档并存储到Milvus

    Args:
        document_id: 文档ID
        chunks_data: 文档块列表
        collection_name: Milvus集合名称

    Returns:
        向量化结果统计
    """
    task_id = self.request.id
    logger.info(f"🚀 [异步任务] 开始向量化文档 {document_id}, 共{len(chunks_data)}个块")

    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': '初始化embedding服务', 'progress': 10}
        )

        # 在线程池中运行向量化任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _vectorize_document(document_id, chunks_data, collection_name, self.update_state)
            )
            logger.info(f"✅ [异步任务] 文档 {document_id} 向量化完成: {result['vectors_count']}个向量")
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ [异步任务] 文档 {document_id} 向量化失败: {e}")
        raise

async def _vectorize_document(
    document_id: str,
    chunks_data: List[Dict[str, Any]],
    collection_name: str,
    progress_callback=None
) -> Dict[str, Any]:
    """异步向量化文档主逻辑"""

    from app.services.embeddings.unified_embedding_service import get_embedding_service
    from app.services.vectorstore.milvus_vector_store import MilvusVectorStore
    from app.core.database import SessionLocal
    from sqlalchemy import text

    # 1. 初始化embedding服务
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '初始化embedding服务', 'progress': 20})

    embedding_service = get_embedding_service()
    await embedding_service.initialize()

    # 2. 批量生成向量（支持并发）
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '生成向量embeddings', 'progress': 40})

    all_texts = [chunk.get('text', '') for chunk in chunks_data if chunk.get('text')]
    all_texts = [text for text in all_texts if text.strip()]  # 过滤空文本

    logger.info(f"📝 准备向量化 {len(all_texts)} 个文本块")

    # 分批处理，避免内存溢出
    batch_size = 32
    all_vectors = []

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        logger.info(f"🔄 处理批次 {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size}")

        # 使用线程池并发生成向量
        vectors = await embedding_service.embed_batch(batch_texts)
        all_vectors.extend(vectors)

        if progress_callback:
            progress_pct = 40 + int((i + len(batch_texts)) / len(all_texts) * 30)
            progress_callback(state='PROGRESS', meta={'status': f'生成向量embeddings ({progress_pct}%)', 'progress': progress_pct})

    # 3. 存储到Milvus
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '存储向量到Milvus', 'progress': 70})

    vector_store = MilvusVectorStore()

    # 准备插入数据
    insert_data = []
    for i, (chunk, vector) in enumerate(zip(chunks_data, all_vectors)):
        if not chunk.get('text') or not chunk.get('text').strip():
            continue

        insert_data.append({
            'document_id': document_id,
            'chunk_index': i,
            'text': chunk.get('text', ''),
            'vector': vector.tolist() if hasattr(vector, 'tolist') else vector,
            'metadata': {
                'page': chunk.get('page', 0),
                'section': chunk.get('section', ''),
                'chunk_type': chunk.get('type', 'text')
            }
        })

    # 批量插入Milvus
    result = await vector_store.insert_documents(
        collection_name=collection_name,
        documents=insert_data
    )

    # 4. 更新数据库状态
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '更新数据库', 'progress': 90})

    db = SessionLocal()
    try:
        db.execute(
            text("""
                UPDATE documents
                SET vectorization_status='completed',
                    vectorization_completed_at=NOW(),
                    vectors_count=:count
                WHERE id=:id
            """),
            {'count': len(insert_data), 'id': document_id}
        )
        db.commit()
    finally:
        db.close()

    logger.info(f"✅ 向量化完成: {len(insert_data)}个向量已存储到Milvus")

    return {
        'document_id': document_id,
        'vectors_count': len(insert_data),
        'collection_name': collection_name,
        'status': 'success'
    }

@celery_app.task(
    bind=True,
    name='app.tasks.async_vector_kg_tasks.extract_knowledge_graph_async',
    soft_time_limit=900,  # 15分钟
    max_retries=2,
    default_retry_delay=60
)
def extract_knowledge_graph_async(
    self,
    document_id: str,
    parsed_content: str,
    graph_name: str = "financial_kg"
):
    """
    异步抽取Neo4j知识图谱

    Args:
        document_id: 文档ID
        parsed_content: 解析后的文档内容
        graph_name: 图谱名称

    Returns:
        知识图谱抽取结果统计
    """
    task_id = self.request.id
    logger.info(f"🚀 [异步任务] 开始抽取知识图谱 {document_id}")

    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': '初始化知识图谱服务', 'progress': 10}
        )

        # 在线程池中运行知识图谱抽取
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _extract_knowledge_graph(document_id, parsed_content, graph_name, self.update_state)
            )
            logger.info(f"✅ [异步任务] 文档 {document_id} 知识图谱抽取完成: {result['entities_count']}个实体, {result['relationships_count']}个关系")
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ [异步任务] 文档 {document_id} 知识图谱抽取失败: {e}")
        raise

async def _extract_knowledge_graph(
    document_id: str,
    parsed_content: str,
    graph_name: str,
    progress_callback=None
) -> Dict[str, Any]:
    """异步抽取知识图谱主逻辑"""

    from app.services.financial_entity_extractor import FinancialEntityExtractor
    from app.services.financial_relationship_extractor import FinancialRelationshipExtractor
    from app.services.financial_metrics_extractor import FinancialMetricsExtractor
    from app.core.database import SessionLocal
    from sqlalchemy import text
    from neo4j import GraphDatabase
    from app.core.config import settings

    # 1. 初始化Neo4j连接
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '连接Neo4j', 'progress': 20})

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )

    # 2. 抽取实体
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '抽取实体', 'progress': 40})

    entity_extractor = FinancialEntityExtractor()
    entities = await entity_extractor.extract_entities(parsed_content)

    logger.info(f"📊 抽取到 {len(entities)} 个实体")

    # 3. 抽取关系
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '抽取关系', 'progress': 60})

    relationship_extractor = FinancialRelationshipExtractor()
    relationships = await relationship_extractor.extract_relationships(parsed_content, entities)

    logger.info(f"🔗 抽取到 {len(relationships)} 个关系")

    # 4. 抽取财务指标
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '抽取财务指标', 'progress': 70})

    metrics_extractor = FinancialMetricsExtractor()
    metrics = await metrics_extractor.extract_metrics(parsed_content)

    logger.info(f"📈 抽取到 {len(metrics)} 个财务指标")

    # 5. 存储到Neo4j
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '存储到Neo4j', 'progress': 80})

    async def store_to_neo4j():
        """异步存储到Neo4j"""
        with driver.session() as session:
            # 存储实体
            for entity in entities:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.document_id = $document_id,
                        e.properties = $properties,
                        e.updated_at = datetime()
                    """,
                    id=entity.get('id', f"{document_id}_{entity.get('name', '')}_{entity.get('type', '')}"),
                    name=entity.get('name', ''),
                    type=entity.get('type', ''),
                    document_id=document_id,
                    properties=entity.get('properties', {})
                )

            # 存储关系
            for rel in relationships:
                await session.run(
                    """
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATIONSHIP {type: $rel_type}]->(target)
                    SET r.document_id = $document_id,
                        r.properties = $properties,
                        r.updated_at = datetime()
                    """,
                    source_id=rel.get('source_id'),
                    target_id=rel.get('target_id'),
                    rel_type=rel.get('type', 'RELATED_TO'),
                    document_id=document_id,
                    properties=rel.get('properties', {})
                )

            # 存储财务指标
            for metric in metrics:
                await session.run(
                    """
                    MERGE (m:Metric {name: $name, document_id: $document_id})
                    SET m.value = $value,
                        m.unit = $unit,
                        m.period = $period,
                        m.properties = $properties,
                        m.updated_at = datetime()
                    """,
                    name=metric.get('name', ''),
                    document_id=document_id,
                    value=metric.get('value', ''),
                    unit=metric.get('unit', ''),
                    period=metric.get('period', ''),
                    properties=metric.get('properties', {})
                )

    await store_to_neo4j()

    # 6. 更新数据库状态
    if progress_callback:
        progress_callback(state='PROGRESS', meta={'status': '更新数据库', 'progress': 90})

    db = SessionLocal()
    try:
        db.execute(
            text("""
                UPDATE documents
                SET kg_extraction_status='completed',
                    kg_extraction_completed_at=NOW(),
                    entities_count=:entities,
                    relationships_count=:relationships,
                    metrics_count=:metrics
                WHERE id=:id
            """),
            {
                'entities': len(entities),
                'relationships': len(relationships),
                'metrics': len(metrics),
                'id': document_id
            }
        )
        db.commit()
    finally:
        db.close()

    driver.close()

    logger.info(f"✅ 知识图谱抽取完成: {len(entities)}个实体, {len(relationships)}个关系, {len(metrics)}个指标")

    return {
        'document_id': document_id,
        'entities_count': len(entities),
        'relationships_count': len(relationships),
        'metrics_count': len(metrics),
        'graph_name': graph_name,
        'status': 'success'
    }

@celery_app.task(
    bind=True,
    name='app.tasks.async_vector_kg_tasks.pipeline_document_async',
    soft_time_limit=1800,  # 30分钟
    max_retries=1
)
def pipeline_document_async(
    self,
    document_id: str,
    parsed_content: str,
    chunks_data: List[Dict[str, Any]]
):
    """
    完整的异步处理流水线：向量化 + 知识图谱抽取

    Args:
        document_id: 文档ID
        parsed_content: 解析后的内容
        chunks_data: 文档块数据

    Returns:
        完整处理结果
    """
    task_id = self.request.id
    logger.info(f"🚀 [异步流水线] 开始处理文档 {document_id}")

    try:
        # 1. 触发向量化任务
        logger.info(f"📝 触发向量化任务...")
        vector_task = vectorize_document_async.apply_async(
            args=[document_id, chunks_data],
            link=self.request.id
        )

        # 2. 触发知识图谱抽取任务
        logger.info(f"📊 触发知识图谱抽取任务...")
        kg_task = extract_knowledge_graph_async.apply_async(
            args=[document_id, parsed_content],
            link=self.request.id
        )

        # 等待两个任务完成
        vector_result = vector_task.get(timeout=600)
        kg_result = kg_task.get(timeout=900)

        logger.info(f"✅ [异步流水线] 文档 {document_id} 处理完成")

        return {
            'document_id': document_id,
            'vector_result': vector_result,
            'kg_result': kg_result,
            'status': 'completed'
        }

    except Exception as e:
        logger.error(f"❌ [异步流水线] 文档 {document_id} 处理失败: {e}")
        raise

# 导出任务
__all__ = [
    'vectorize_document_async',
    'extract_knowledge_graph_async',
    'pipeline_document_async'
]
