"""
存储适配器：连接增强本地存储与现有系统
提供向后兼容的接口
"""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from app.services.storage.enhanced_local_storage import enhanced_local_storage, DocumentMetadata

logger = logging.getLogger(__name__)


class StorageAdapter:
    """
    存储适配器

    提供与现有core_service_integrator兼容的接口
    同时使用增强的本地存储功能
    """

    def __init__(self):
        self.enhanced_storage = enhanced_local_storage

    async def save_parsed_document_to_local(
        self,
        document_id: str,
        filename: str,
        text_content: str,
        markdown_content: Optional[str] = None,
        structured_content: Optional[Dict] = None,
        analysis_result: Optional[Dict] = None,
        chunks: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        semantic_enhancement: Optional[Dict] = None,
        knowledge_graph: Optional[Dict] = None,
        extraction_method: str = "unknown",
        extraction_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        保存解析后的文档到本地文件（兼容接口）

        Args:
            document_id: 文档ID
            filename: 文件名
            text_content: 文本内容
            markdown_content: Markdown内容
            structured_content: 结构化内容
            analysis_result: 分析结果
            chunks: 文档分块
            embeddings: 向量嵌入
            semantic_enhancement: 语义增强
            knowledge_graph: 知识图谱
            extraction_method: 提取方法
            extraction_time: 提取耗时

        Returns:
            保存结果
        """
        try:
            # 如果没有提供Markdown内容，使用文本内容
            if not markdown_content:
                markdown_content = text_content

            # 使用增强存储服务保存
            result = await self.enhanced_storage.save_document(
                document_id=document_id,
                filename=filename,
                file_path=f"/app/storage/parsed_docs/{document_id}",
                content=text_content,
                markdown_content=markdown_content,
                chunks=chunks,
                embeddings=embeddings,
                semantic_enhancement=semantic_enhancement,
                knowledge_graph=knowledge_graph,
                analysis_result=analysis_result or {},
                extraction_method=extraction_method,
                extraction_time=extraction_time
            )

            if result['success']:
                logger.info(f"✅ 文档已保存: {document_id}")
                return {
                    'status': 'success',
                    'document_id': document_id,
                    'path': result['path'],
                    'files': [str(f) for f in result['files']],
                    'metadata': result['metadata'],
                    'quality': result['quality']
                }
            else:
                logger.error(f"❌ 保存失败: {result.get('error')}")
                return {
                    'status': 'error',
                    'error': result.get('error')
                }

        except Exception as e:
            logger.error(f"保存文档失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def save_from_enhanced_pipeline(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        analysis_result: Any,
        semantic_enhancements: Optional[List] = None,
        enriched_entities: Optional[List] = None,
        enriched_relations: Optional[List] = None,
        enhanced_embeddings: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        保存增强流水线的结果

        Args:
            document_id: 文档ID
            filename: 文件名
            file_path: 文件路径
            analysis_result: DocumentAnalysisResult
            semantic_enhancements: 语义增强结果
            enriched_entities: 增强实体
            enriched_relations: 增强关系
            enhanced_embeddings: 增强嵌入

        Returns:
            保存结果
        """
        try:
            # 准备数据
            chunks = None
            embeddings = None
            semantic_data = None
            kg_data = None

            # 从analysis_result提取数据
            if hasattr(analysis_result, 'chunks'):
                chunks = [
                    {
                        'index': i,
                        'content': chunk.page_content,
                        'metadata': chunk.metadata
                    }
                    for i, chunk in enumerate(analysis_result.chunks)
                ]

            if hasattr(analysis_result, 'embeddings'):
                embeddings = analysis_result.embeddings

            # 准备语义增强数据
            if semantic_enhancements:
                semantic_data = {
                    'count': len(semantic_enhancements),
                    'enhancements': [
                        {
                            'chunk_id': se.chunk_id,
                            'semantic_category': se.semantic_category,
                            'enhanced_summary': se.enhanced_summary,
                            'key_insights': se.key_insights,
                            'sentiment': se.sentiment,
                            'risk_level': se.risk_level,
                            'opportunity_level': se.opportunity_level
                        }
                        for se in semantic_enhancements
                    ]
                }

            # 准备知识图谱数据
            if enriched_entities or enriched_relations:
                kg_data = {
                    'entities': [
                        {
                            'canonical_id': ee.canonical_id,
                            'name': ee.name,
                            'type': ee.type,
                            'confidence': ee.confidence,
                            'importance_score': ee.importance_score,
                            'sources': ee.sources
                        }
                        for ee in (enriched_entities or [])
                    ],
                    'relations': [
                        {
                            'relation_id': er.relation_id,
                            'subject': er.subject_canonical_id,
                            'object': er.object_canonical_id,
                            'relation_type': er.relation_type,
                            'direction': er.direction,
                            'confidence': er.confidence
                        }
                        for er in (enriched_relations or [])
                    ]
                }

            # 准备增强嵌入数据
            embedding_vectors = None
            if enhanced_embeddings:
                embedding_vectors = []
                for ee in enhanced_embeddings:
                    if hasattr(ee, 'embeddings'):
                        # 获取最佳嵌入
                        best_emb = ee.get_best_embedding()
                        if best_emb is not None:
                            embedding_vectors.append(best_emb.tolist())

            # 保存文档
            result = await self.enhanced_storage.save_document(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                content=analysis_result.raw_text if hasattr(analysis_result, 'raw_text') else "",
                markdown_content=analysis_result.raw_text if hasattr(analysis_result, 'raw_text') else None,
                chunks=chunks,
                embeddings=embedding_vectors,
                semantic_enhancement=semantic_data,
                knowledge_graph=kg_data,
                extraction_method=analysis_result.extraction_method if hasattr(analysis_result, 'extraction_method') else "enhanced",
                extraction_time=analysis_result.extraction_time if hasattr(analysis_result, 'extraction_time') else 0.0
            )

            if result['success']:
                logger.info(f"✅ 增强流水线结果已保存: {document_id}")
                return result
            else:
                logger.error(f"❌ 保存失败: {result.get('error')}")
                return result

        except Exception as e:
            logger.error(f"保存增强流水线结果失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """获取文档元数据"""
        try:
            doc_dir = self.enhanced_storage.base_path / document_id
            metadata_file = doc_dir / 'metadata.json'

            if not metadata_file.exists():
                logger.warning(f"元数据文件不存在: {document_id}")
                return None

            data = json.loads(metadata_file.read_text(encoding='utf-8'))
            return DocumentMetadata(**data)

        except Exception as e:
            logger.error(f"获取元数据失败: {e}")
            return None

    async def list_documents(
        self,
        quality_level: Optional[str] = None,
        min_quality_score: Optional[float] = None
    ) -> List[Dict]:
        """列出文档"""
        try:
            filters = {}
            if quality_level:
                filters['quality_level'] = quality_level
            if min_quality_score:
                filters['min_quality_score'] = min_quality_score

            return await self.enhanced_storage.search_documents("", filters)

        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return []

    async def get_storage_statistics(self) -> Dict:
        """获取存储统计"""
        return await self.enhanced_storage.get_document_stats()

    async def check_document_quality(self, document_id: str) -> Dict:
        """检查文档质量"""
        try:
            doc_dir = self.enhanced_storage.base_path / document_id
            quality_file = doc_dir / 'quality_report.json'

            if not quality_file.exists():
                return {'quality': 'unknown', 'reason': 'No quality report'}

            return json.loads(quality_file.read_text(encoding='utf-8'))

        except Exception as e:
            logger.error(f"检查质量失败: {e}")
            return {'quality': 'error', 'reason': str(e)}


# 全局实例
storage_adapter = StorageAdapter()


# 兼容性函数：向后兼容旧接口
async def save_parsed_document_to_local(*args, **kwargs):
    """兼容性包装函数"""
    return await storage_adapter.save_parsed_document_to_local(*args, **kwargs)
