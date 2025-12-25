"""
增强多模态文档处理任务 - 强制使用qwen-vl-plus
包含图片分析、图表分析、公式提取、实体关系抽取
"""

from celery import current_task
from app.core.async_tasks.celery_app import celery_app
import logging
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.models.document import Document, DocumentChunk
from app.core.database import SessionLocal
from app.services.real_qwen_service import RealQwenService, RealQwenConfig
from app.services.minio_service import MinIOService
from app.services.milvus_service import MilvusService
from app.services.neo4j_service import Neo4jService
from app.services.qwen_embedding_service import QwenEmbeddingService
from sqlalchemy.orm import Session

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
def process_enhanced_multimodal_document(self, document_id: str, original_filename: str, user_id: str = None):
    """
    增强多模态文档处理任务
    强制使用qwen-vl-plus进行高级分析
    """
    task_id = self.request.id

    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': '开始增强多模态分析', 'progress': 5}
        )

        # 初始化增强服务
        config = RealQwenConfig(
            enable_image_analysis=True,
            enable_chart_analysis=True,
            enable_formula_extraction=True,
            enable_entity_extraction=True
        )

        qwen_service = RealQwenService(config)
        embedding_service = QwenEmbeddingService()

        # 获取文档信息
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise Exception(f"文档不存在: {document_id}")

            # 更新文档状态
            document.processing_mode = "enhanced_multimodal_pipeline"
            document.task_id = task_id
            db.commit()

            # 从MinIO获取文件
            minio_service = MinIOService()

            async def get_file_content():
                return await minio_service.download_file(document.file_path)

            file_content = run_async(get_file_content())

        finally:
            db.close()

        self.update_state(
            state='PROGRESS',
            meta={'status': '使用qwen-vl-plus分析文档', 'progress': 20}
        )

        # 使用RealQwenService进行多模态分析
        sections = [{'page': 1, 'content': ''}]  # 占位符，会被服务重新分析

        async def analyze_document():
            return await qwen_service.analyze_document_multimodal(
                file_content, original_filename, sections
            )

        analysis_results = run_async(analyze_document())

        self.update_state(
            state='PROGRESS',
            meta={'status': '提取实体关系', 'progress': 50}
        )

        # 实体关系抽取
        async def extract_entities():
            return await qwen_service.extract_entity_relationships(
                analysis_results.get('summary', '')
            )

        entity_results = run_async(extract_entities())

        self.update_state(
            state='PROGRESS',
            meta={'status': '生成向量嵌入', 'progress': 70}
        )

        # 生成向量嵌入
        all_text_parts = []

        # 添加章节内容
        for section in analysis_results.get('sections_analysis', []):
            all_text_parts.append(section.get('summary', ''))
            all_text_parts.extend(section.get('key_points', []))

        # 添加图片描述
        for img in analysis_results.get('images_found', []):
            all_text_parts.append(img.get('description', ''))

        # 添加图表分析
        if analysis_results.get('charts_found'):
            all_text_parts.append("图表数据分析结果")

        # 添加公式解释
        for formula in analysis_results.get('formulas_found', []):
            all_text_parts.append(formula.get('explanation', ''))

        full_text = ' '.join(filter(None, all_text_parts))

        # 生成嵌入向量
        async def generate_embeddings():
            return await embedding_service.generate_embeddings([full_text])

        embedding_results = run_async(generate_embeddings())

        self.update_state(
            state='PROGRESS',
            meta={'status': '保存到存储系统', 'progress': 85}
        )

        # 保存到数据库
        db = SessionLocal()
        try:
            # 更新文档信息
            document.parsed_content = json.dumps({
                'analysis_results': analysis_results,
                'entity_results': entity_results,
                'models_used': ['qwen-vl-plus', 'text-embedding-v4'],
                'processing_timestamp': datetime.now().isoformat(),
                'enhanced_features': {
                    'image_analysis': analysis_results.get('images_found', []),
                    'chart_analysis': analysis_results.get('charts_found', []),
                    'formula_extraction': analysis_results.get('formulas_found', []),
                    'entity_extraction': entity_results
                }
            }, ensure_ascii=False)

            document.status = 'COMPLETED'
            document.processed_at = datetime.now()

            # 创建文档块
            if embedding_results and len(embedding_results) > 0:
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=1,
                    content=full_text[:1000],  # 限制长度
                    chunk_metadata={
                        'processing_mode': 'enhanced_multimodal',
                        'has_images': len(analysis_results.get('images_found', [])) > 0,
                        'has_charts': len(analysis_results.get('charts_found', [])) > 0,
                        'has_formulas': len(analysis_results.get('formulas_found', [])) > 0,
                        'entities_count': len(entity_results) if entity_results else 0
                    }
                )
                db.add(chunk)

            db.commit()

        finally:
            db.close()

        self.update_state(
            state='PROGRESS',
            meta={'status': '处理完成', 'progress': 100}
        )

        # 返回处理结果
        result = {
            'status': 'completed',
            'document_id': document_id,
            'models_used': ['qwen-vl-plus', 'text-embedding-v4'],
            'enhanced_features': {
                'images_found': len(analysis_results.get('images_found', [])),
                'charts_found': len(analysis_results.get('charts_found', [])),
                'formulas_found': len(analysis_results.get('formulas_found', [])),
                'entities_extracted': len(entity_results) if entity_results else 0
            },
            'processing_time': datetime.now().isoformat()
        }

        logger.info(f"增强多模态处理完成: {document_id}")
        return result

    except Exception as e:
        logger.error(f"增强多模态处理失败 {document_id}: {e}")

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


# 向后兼容的别名
process_document_complete = process_enhanced_multimodal_document