"""
轻量级文档处理编排器 - 替代CoreServiceIntegrator
将2758行拆分为多个小服务，通过编排器协调
"""

from datetime import datetime
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class DocumentProcessingOrchestrator:
    """
    文档处理编排器（轻量级）

    职责：
    - 协调各个服务
    - 管理处理流程
    - 收集处理结果

    不包含具体业务逻辑，仅做协调
    """

    def __init__(self, services: Dict[str, Any]):
        self.services = services
        self._services = {}

    async def initialize(self):
        """初始化各个服务"""
        # 导入并初始化各个独立服务
        from app.services.parsing.document_parsing_service import DocumentParsingService
        from app.services.vector.vector_generation_service import VectorGenerationService
        from app.services.storage.storage_coordinator_service import StorageCoordinatorService
        from app.services.optimized_entity_extractor import OptimizedEntityExtractor

        # 初始化解析服务
        self._services['parser'] = DocumentParsingService(self.services)
        logger.info("✅ 文档解析服务已初始化")

        # 初始化向量生成服务
        self._services['vector_gen'] = VectorGenerationService(
            embedding_service=self.services['embedding'],
            batch_size=50,
            max_concurrent=10
        )
        logger.info("✅ 向量生成服务已初始化")

        # 初始化存储协调服务
        self._services['storage'] = StorageCoordinatorService(self.services)
        logger.info("✅ 存储协调服务已初始化")

        # 初始化实体提取器
        self._services['entity_extractor'] = OptimizedEntityExtractor(
            config={'enable_llm_fallback': True}
        )
        logger.info("✅ 实体提取器已初始化")

        self._initialized = True
        logger.info("✅ 文档处理编排器初始化完成")

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        处理文档（编排器）

        Returns:
            处理结果
        """
        logger.info(f"🚀 开始处理文档: {filename}")

        result = {
            'document_id': document_id,
            'filename': filename,
            'processing_start': datetime.now().isoformat(),
            'stages': {}
        }

        try:
            # 阶段1: 解析文档
            logger.info("📄 阶段1: 文档解析...")
            text_content, markdown_content, parse_result = await self._services['parser'].parse_document(
                file_content, filename, document_id
            )

            if not text_content and not markdown_content:
                error_msg = "文档解析失败"
                logger.error(f"❌ {error_msg}")
                return {**result, 'success': False, 'error': error_msg}

            result['stages']['parsing'] = {
                'status': 'completed',
                'method': parse_result.get('method'),
                'text_length': len(text_content),
                'markdown_length': len(markdown_content)
            }

            # 阶段2: 文档分割
            logger.info("✂️ 阶段2: 文档分割...")
            chunks_data = await self._chunk_document(text_content, {})
            logger.info(f"✅ 文档已分割为 {len(chunks_data)} 个chunks")

            result['stages']['chunking'] = {
                'status': 'completed',
                'chunk_count': len(chunks_data)
            }

            # 阶段3: 实体提取
            logger.info("🔗 阶段3: 实体提取...")
            entities, relationships = await self._extract_entities(chunks_data, document_id)

            result['stages']['entities'] = {
                'status': 'completed',
                'entity_count': len(entities),
                'relationship_count': len(relationships)
            }

            # 阶段4: 向量生成
            logger.info("🔢 阶段4: 向量生成...")
            chunks_with_embeddings = await self._services['vector_gen'].generate_vectors_batch(
                chunks_data, document_id
            )

            # 验证向量质量
            vector_stats = self._services['vector_gen'].validate_embeddings(chunks_with_embeddings)

            result['stages']['embeddings'] = {
                'status': 'completed',
                'chunks_processed': len(chunks_with_embeddings),
                'vector_dimension': vector_stats['dimension'],
                'valid_rate': vector_stats['valid_rate']
            }

            # 阶段5: 并行存储
            logger.info("💾 阶段5: 数据存储...")
            storage_result = await self._services['storage'].store_all(
                document_id,
                chunks_with_embeddings,
                entities,
                relationships,
                enable_kg=True
            )

            result['stages']['storage'] = storage_result

            result['processing_end'] = datetime.now().isoformat()
            result['status'] = 'completed'
            result['success'] = True

            logger.info(f"✅ 文档处理完成: {filename}")
            return result

        except Exception as e:
            logger.error(f"❌ 文档处理失败: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            result['success'] = False
            return result

    async def _chunk_document(
        self,
        text_content: str,
        analysis_result: Dict
    ) -> list:
        """分割文档（简化版）"""
        # 这里应该调用实际的chunker
        # 为了示例，返回简单的分块
        chunk_size = 1000
        chunks = []

        for i in range(0, len(text_content), chunk_size):
            chunks.append({
                'chunk_id': f"chunk_{i // chunk_size}",
                'document_id': '',
                'chunk_index': i // chunk_size,
                'content': text_content[i:i + chunk_size],
                'metadata': {}
            })

        return chunks

    async def _extract_entities(
        self,
        chunks: list,
        document_id: str
    ) -> tuple:
        """提取实体和关系"""
        from app.services.unified_document_service import UnifiedChunk

        # 转换为UnifiedChunk格式
        unified_chunks = [
            UnifiedChunk(
                chunk_id=chunk.get('chunk_id'),
                document_id=document_id,
                chunk_index=chunk.get('chunk_index'),
                content=chunk.get('content'),
                metadata=chunk.get('metadata', {})
            )
            for chunk in chunks
        ]

        # 提取实体
        extracted_entities = await self._services['entity_extractor'].extract_entities_batch(
            unified_chunks,
            config={'min_confidence': 0.6}
        )

        # 转换格式
        entities = [
            {
                'name': ent.text,
                'type': ent.entity_type,
                'confidence': ent.confidence,
                'source': ent.source,
                'properties': ent.metadata or {}
            }
            for ent in extracted_entities
        ]

        # 提取关系（简化版）
        relationships = self._extract_relationships_simple(entities, chunks)

        return entities, relationships

    def _extract_relationships_simple(
        self,
        entities: list,
        chunks: list
    ) -> list:
        """简化版关系提取"""
        relationships = []

        # 基于共现提取关系
        for chunk in chunks:
            chunk_entities = [
                e for e in entities
                if e.get('name') and e.get('name') in chunk.get('content', '')
            ]

            # 为每对实体建立关系
            for i, ent1 in enumerate(chunk_entities):
                for ent2 in chunk_entities[i+1:]:
                    relationships.append({
                        'from_entity': ent1['name'],
                        'to_entity': ent2['name'],
                        'type': 'RELATED_TO',
                        'confidence': 0.7
                    })

        return relationships[:500]  # 限制数量
