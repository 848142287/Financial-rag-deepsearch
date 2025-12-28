"""
多存储后端处理器
借鉴新代码的多存储集成模式，整合所有存储后端
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime

from app.services.enhanced_qwen_processor import EnhancedQwenProcessor
from app.services.enhanced_storage_service import EnhancedStorageService
from app.services.enhanced_vector_service import enhanced_vector_service
from app.services.neo4j_service import neo4j_service
from app.services.minio_service import MinIOService
from app.models.document import Document, DocumentChunk, DocumentStatus
from sqlalchemy.orm import Session
from app.core.database import get_db

logger = logging.getLogger(__name__)


class MultiBackendProcessor:
    """多存储后端处理器"""

    def __init__(self):
        """初始化处理器"""
        self.qwen_processor = EnhancedQwenProcessor()
        self.storage_service = EnhancedStorageService()
        self.vector_service = enhanced_vector_service
        self.graph_service = neo4j_service
        self.minio_service = MinIOService()

    async def process_document_complete(
        self,
        file_path: str,
        db: Session,
        document_id: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """完整的文档处理流程"""
        try:
            file_path = Path(file_path)
            options = options or {}

            processing_result = {
                'file_path': str(file_path),
                'document_id': document_id,
                'timestamp': datetime.utcnow().isoformat(),
                'stages': {}
            }

            # 阶段1: 文件存储到MinIO
            logger.info(f"阶段1: 存储文件到MinIO - {file_path.name}")
            try:
                # 使用增强的存储服务（去重上传）
                upload_result = await self.storage_service.upload_with_deduplication(
                    file_path=file_path,
                    metadata={
                        'processing_options': options,
                        'original_path': str(file_path),
                        'file_type': file_path.suffix.lower()
                    }
                )

                processing_result['stages']['minio'] = {
                    'status': 'success',
                    'object_name': upload_result['object_name'],
                    'file_hash': upload_result['file_hash'],
                    'deduplication': upload_result['status'] == 'exists'
                }
            except Exception as e:
                logger.error(f"MinIO存储失败: {str(e)}")
                processing_result['stages']['minio'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                return processing_result

            # 阶段2: 文档解析（使用增强的Qwen处理器）
            logger.info(f"阶段2: 文档解析")
            try:
                # 读取文件内容进行分析
                with open(file_path, 'rb') as f:
                    image_data = f.read()

                # 如果是图像文件，直接使用视觉处理器
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    analysis_result = self.qwen_processor.process_document_page(image_data)

                    # 构建chunk数据
                    chunks = [{
                        'id': f"{document_id}_0",
                        'content': analysis_result.get('full_text', ''),
                        'type': 'text',
                        'page_num': 1,
                        'chunk_index': 0,
                        'metadata': {
                            'formulas': analysis_result.get('formulas', []),
                            'tables': analysis_result.get('tables', []),
                            'processing_metadata': analysis_result.get('processing_metadata', {})
                        }
                    }]
                else:
                    # 对于其他格式，使用现有的解析器
                    from app.services.consolidated_document_service import ConsolidatedDocumentService as DocumentProcessor
                    doc_processor = DocumentProcessor(db)

                    # 获取文件扩展名
                    file_ext = file_path.suffix.lower()
                    if file_ext in doc_processor.supported_formats:
                        content_extractor = doc_processor.supported_formats[file_ext]
                        content, metadata = await content_extractor(str(file_path), options)

                        # 简单分块（可以优化）
                        chunks = self._create_chunks_from_content(content, document_id)
                    else:
                        # 使用Qwen进行通用解析
                        analysis_result = self.qwen_processor.process_document_page(image_data)
                        chunks = [{
                            'id': f"{document_id}_0",
                            'content': analysis_result.get('full_text', ''),
                            'type': 'text',
                            'page_num': 1,
                            'chunk_index': 0,
                            'metadata': analysis_result.get('processing_metadata', {})
                        }]

                processing_result['stages']['parsing'] = {
                    'status': 'success',
                    'chunk_count': len(chunks),
                    'total_content_length': sum(len(c['content']) for c in chunks)
                }
            except Exception as e:
                logger.error(f"文档解析失败: {str(e)}")
                processing_result['stages']['parsing'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                chunks = []

            # 阶段3: 存储到数据库
            logger.info(f"阶段3: 存储到数据库")
            try:
                if document_id:
                    # 更新现有文档
                    document = db.query(Document).filter(Document.id == document_id).first()
                    if document:
                        # 删除旧chunks
                        db.query(DocumentChunk).filter(
                            DocumentChunk.document_id == document_id
                        ).delete()

                        # 更新文档信息
                        document.status = DocumentStatus.PROCESSING
                        document.doc_metadata = document.doc_metadata or {}
                        document.doc_metadata.update(processing_result['stages'])
                else:
                    # 创建新文档记录
                    document = Document(
                        filename=file_path.name,
                        file_path=upload_result['object_name'],
                        file_size=file_path.stat().st_size,
                        status=DocumentStatus.PROCESSING,
                        doc_metadata={
                            'processing_stages': processing_result['stages'],
                            'file_hash': upload_result['file_hash']
                        }
                    )
                    db.add(document)
                    db.flush()
                    document_id = document.id

                # 创建chunks
                chunk_objects = []
                for chunk in chunks:
                    chunk_obj = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk['chunk_index'],
                        content=chunk['content'],
                        chunk_type=chunk.get('type', 'text'),
                        page_num=chunk.get('page_num', 0),
                        chunk_metadata=chunk.get('metadata', {})
                    )
                    chunk_objects.append(chunk_obj)

                db.add_all(chunk_objects)
                db.commit()

                processing_result['stages']['database'] = {
                    'status': 'success',
                    'document_id': document_id,
                    'chunk_count': len(chunk_objects)
                }
            except Exception as e:
                logger.error(f"数据库存储失败: {str(e)}")
                processing_result['stages']['database'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                if document_id:
                    db.rollback()

            # 阶段4: 向量化存储（异步执行）
            logger.info(f"阶段4: 向量化存储")
            if chunks and document_id:
                try:
                    # 使用增强的向量服务
                    vector_result = await self.vector_service.index_document_chunks(
                        document_id=str(document_id),
                        chunks=chunks
                    )

                    processing_result['stages']['vectors'] = {
                        'status': 'success',
                        **vector_result
                    }
                except Exception as e:
                    logger.error(f"向量化失败: {str(e)}")
                    processing_result['stages']['vectors'] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            # 阶段5: 知识图谱构建（异步执行）
            logger.info(f"阶段5: 知识图谱构建")
            if document_id:
                try:
                    # 创建文档节点
                    doc_node_id = await self.graph_service.create_document_node(
                        document_id=document_id,
                        title=file_path.stem,
                        metadata=processing_result.get('stages', {})
                    )

                    # 创建块节点
                    if chunks:
                        chunk_ids = await self.graph_service.create_chunk_nodes(
                            document_id=str(document_id),
                            chunks=chunks
                        )

                        # 为每个chunk提取实体
                        total_entities = 0
                        for i, chunk in enumerate(chunks):
                            chunk_id = f"{document_id}_{i}"
                            entities = await self.graph_service.enhanced_entity_extraction(
                                chunk_id=chunk_id,
                                content=chunk['content']
                            )
                            total_entities += len(entities)

                            # 创建实体间关系
                            await self.graph_service.create_entity_relationships(chunk_id)

                        processing_result['stages']['knowledge_graph'] = {
                            'status': 'success',
                            'document_node': doc_node_id,
                            'chunk_count': len(chunk_ids),
                            'entity_count': total_entities
                        }
                    else:
                        processing_result['stages']['knowledge_graph'] = {
                            'status': 'success',
                            'document_node': doc_node_id,
                            'chunk_count': 0,
                            'entity_count': 0
                        }
                except Exception as e:
                    logger.error(f"知识图谱构建失败: {str(e)}")
                    processing_result['stages']['knowledge_graph'] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            # 更新文档状态为完成
            if document_id:
                try:
                    document = db.query(Document).filter(Document.id == document_id).first()
                    if document:
                        document.status = DocumentStatus.COMPLETED
                        document.processed_at = datetime.utcnow()
                        document.doc_metadata = document.doc_metadata or {}
                        document.doc_metadata['complete_processing_result'] = processing_result
                        db.commit()
                except Exception as e:
                    logger.error(f"更新文档状态失败: {str(e)}")

            # 计算总体状态
            success_stages = sum(
                1 for stage in processing_result['stages'].values()
                if stage['status'] == 'success'
            )
            processing_result['overall_status'] = (
                'success' if success_stages == 5 else
                'partial' if success_stages > 0 else 'failed'
            )

            logger.info(f"文档处理完成: {file_path.name} - {processing_result['overall_status']}")
            return processing_result

        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}")
            raise

    def _create_chunks_from_content(self, content: str, document_id: int) -> List[Dict[str, Any]]:
        """从内容创建chunks"""
        chunks = []
        chunk_size = 1000  # 可配置
        chunk_overlap = 200

        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                chunk_text = content[start:]
            else:
                # 尝试在句子边界分割
                chunk_text = content[start:end]
                sentence_end = max(
                    chunk_text.rfind('。'),
                    chunk_text.rfind('！'),
                    chunk_text.rfind('？'),
                    chunk_text.rfind('. '),
                    chunk_text.rfind('\n\n')
                )

                if sentence_end > chunk_size * 0.8:
                    chunk_text = chunk_text[:sentence_end + 1]
                    end = start + len(chunk_text)

            if chunk_text.strip():
                chunks.append({
                    'id': f"{document_id}_{chunk_index}",
                    'content': chunk_text.strip(),
                    'type': 'text',
                    'page_num': 0,
                    'chunk_index': chunk_index,
                    'metadata': {}
                })

            start = end - chunk_overlap if end < len(content) else end
            chunk_index += 1

        return chunks

    async def search_multi_backend(
        self,
        query: str,
        search_types: List[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """多后端搜索"""
        if search_types is None:
            search_types = ['vector', 'graph', 'hybrid']

        results = {
            'query': query,
            'timestamp': datetime.utcnow().isoformat(),
            'results': {}
        }

        # 向量搜索
        if 'vector' in search_types:
            try:
                vector_results = await self.vector_service.search_documents(
                    query=query,
                    top_k=limit,
                    filters=filters
                )
                results['results']['vector'] = {
                    'status': 'success',
                    'count': len(vector_results),
                    'items': vector_results
                }
            except Exception as e:
                results['results']['vector'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # 知识图谱搜索
        if 'graph' in search_types:
            try:
                graph_results = await self.graph_service.semantic_search(
                    query=query,
                    limit=limit,
                    filters=filters
                )
                results['results']['graph'] = {
                    'status': 'success',
                    'count': len(graph_results),
                    'items': graph_results
                }
            except Exception as e:
                results['results']['graph'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # 混合搜索
        if 'hybrid' in search_types:
            try:
                # 使用统一存储管理器的混合搜索
                from app.services.unified_storage_manager import unified_storage_manager
                hybrid_results = await unified_storage_manager.hybrid_retrieve(
                    query=query,
                    top_k=limit,
                    filters=filters,
                    search_strategy='hybrid'
                )
                results['results']['hybrid'] = {
                    'status': 'success',
                    'count': len(hybrid_results),
                    'items': [self._convert_search_result(r) for r in hybrid_results]
                }
            except Exception as e:
                results['results']['hybrid'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        return results

    def _convert_search_result(self, result) -> Dict[str, Any]:
        """转换搜索结果格式"""
        return {
            'id': result.id,
            'content': result.content,
            'document_id': result.document_id,
            'document_title': result.document_title,
            'score': result.score,
            'source': result.source,
            'chunk_type': result.chunk_type,
            'page_num': result.page_num,
            'metadata': result.metadata
        }

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        try:
            from app.services.unified_storage_manager import unified_storage_manager

            # 获取统一存储统计
            storage_stats = await unified_storage_manager.get_comprehensive_stats()

            # 添加MinIO统计
            minio_stats = await self.storage_service.get_storage_statistics()

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'storage': storage_stats._asdict(),
                'minio': minio_stats,
                'services': {
                    'vector_service': 'active',
                    'graph_service': 'active',
                    'storage_service': 'active',
                    'minio_service': 'active'
                }
            }

        except Exception as e:
            logger.error(f"获取综合统计失败: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}


# 全局处理器实例
multi_backend_processor = MultiBackendProcessor()


# 便利函数
async def process_document_with_all_backends(
    file_path: str,
    db: Session,
    document_id: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """使用所有后端处理文档的便利函数"""
    return await multi_backend_processor.process_document_complete(
        file_path=file_path,
        db=db,
        document_id=document_id,
        options=options
    )


async def search_all_backends(
    query: str,
    search_types: List[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """搜索所有后端的便利函数"""
    return await multi_backend_processor.search_multi_backend(
        query=query,
        search_types=search_types,
        limit=limit
    )