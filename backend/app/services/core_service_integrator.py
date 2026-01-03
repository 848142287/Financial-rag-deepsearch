"""
统一服务整合器 - 整合所有分散的服务
重新实现以修复缺失的文件
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class ServiceConfig:
    """服务配置"""
    enable_multimodal: bool = True
    enable_entity_extraction: bool = True
    enable_knowledge_graph: bool = True
    async_entity_extraction: bool = True
    async_knowledge_graph: bool = True
    enable_vector_storage: bool = True
    enable_mysql_storage: bool = True
    enable_chunking: bool = True

class CoreServiceIntegrator:
    """统一服务整合器"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        self._services = {}
        self._initialized = False

    async def initialize(self):
        """初始化所有服务"""
        if self._initialized:
            return

        logger.info("🚀 初始化统一服务整合器...")

        # 导入服务
        from app.services.minio_service import MinIOService
        from app.services.enhanced_milvus_service import EnhancedMilvusService  # 使用增强版Milvus服务
        # from app.services.neo4j_service import Neo4jService  # Neo4j服务暂时禁用（模块不存在）
        from app.services.embeddings.unified_embedding_service import get_embedding_service
        # from app.services.unified_document_service import UnifiedDocumentService  # 模块不存在，已禁用
        # from app.services.knowledge.entity_extractor import FinancialEntityExtractor  # 实体提取暂时禁用

        # 初始化MinIO
        self._services['minio'] = MinIOService()
        # MinIOService没有initialize方法，不需要调用
        logger.info("✅ MinIO服务已初始化")

        # 初始化Milvus (使用增强版)
        self._services['milvus'] = EnhancedMilvusService()
        # EnhancedMilvusService没有initialize方法，不需要调用
        logger.info("✅ Milvus服务已初始化")

        # 初始化Neo4j (暂时禁用)
        # if self.config.enable_knowledge_graph:
        #     self._services['neo4j'] = Neo4jService()
        #     await self._services['neo4j'].initialize()
        #     logger.info("✅ Neo4j服务已初始化")

        # 初始化Embedding服务
        self._services['embedding'] = get_embedding_service()
        logger.info("✅ Embedding服务已初始化")

        # 初始化文档服务 (暂时禁用 - 模块不存在)
        # self._services['document'] = UnifiedDocumentService()
        # logger.info("✅ 文档服务已初始化")

        # 初始化实体提取器 (暂时禁用)
        # if self.config.enable_entity_extraction:
        #     self._services['entity_extractor'] = FinancialEntityExtractor()
        #     logger.info("✅ 实体提取器已初始化")

        self._initialized = True
        logger.info("✅ 统一服务整合器初始化完成")

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        处理文档（完整流水线）

        Returns:
            处理结果
        """
        logger.info(f"🚀 开始处理文档: {filename}")

        result = {
            'document_id': document_id,
            'filename': filename,
            'processing_start': datetime.now().isoformat(),
            'stages': {},
            'success': False
        }

        try:
            # 阶段1: 解析文档
            logger.info("📄 阶段1: 文档解析...")
            parse_result = await self._parse_document(file_content, filename, document_id)
            result['stages']['parsing'] = parse_result

            if not parse_result.get('success'):
                return result

            # 阶段2: 文本分块
            logger.info("🔪 阶段2: 文本分块...")
            chunks = await self._create_chunks(parse_result, document_id)
            result['stages']['chunking'] = {'status': 'completed', 'chunk_count': len(chunks)}

            # 阶段3: 向量生成
            logger.info("🔢 阶段3: 向量生成...")
            chunks_with_embeddings = await self._generate_embeddings(chunks, document_id)
            result['stages']['embeddings'] = {'status': 'completed', 'embedding_count': len(chunks_with_embeddings)}

            # 阶段4: 存储到MySQL和Milvus
            logger.info("💾 阶段4: 存储数据...")
            await self._store_document_data(document_id, parse_result, chunks_with_embeddings)
            result['stages']['storage'] = {'status': 'completed'}

            # 保存chunks用于后台enrichment
            result['chunks'] = chunks_with_embeddings

            # 阶段5: 实体提取（异步后台）
            result['stages']['entities'] = {
                'status': 'async_pending',
                'reason': '实体提取已转为后台异步任务'
            }

            # 阶段6: 知识图谱（异步后台）
            result['stages']['knowledge_graph'] = {
                'status': 'async_pending',
                'reason': '知识图谱已转为后台异步任务'
            }

            result['success'] = True
            result['processing_end'] = datetime.now().isoformat()

            logger.info(f"✅ 文档处理完成: {document_id}")
            return result

        except Exception as e:
            logger.error(f"❌ 文档处理失败: {e}")
            result['error'] = str(e)
            result['success'] = False
            return result

    async def _parse_document(self, file_content: bytes, filename: str, document_id: str) -> Dict[str, Any]:
        """解析文档"""
        from app.services.advanced_pdf_parser import AdvancedPDFParser

        parser = AdvancedPDFParser()

        # 保存临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
            f.write(file_content)
            temp_path = f.name

        try:
            # 解析PDF
            parse_result = await parser.parse_pdf_async(temp_path)

            if not parse_result.success:
                return {
                    'success': False,
                    'error': parse_result.error or 'PDF解析失败',
                    'text': '',
                    'markdown': ''
                }

            return {
                'success': True,
                'text': parse_result.text,
                'markdown': parse_result.markdown,
                'tables': parse_result.tables,
                'images': parse_result.images,
                'formulas': parse_result.formulas,
                'charts': parse_result.charts,
                'metadata': parse_result.metadata,
                'parsing_stats': parse_result.parsing_stats,
                'method': parse_result.method
            }
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def _create_chunks(self, parse_result: Dict, document_id: str) -> List[Dict]:
        """创建文档块"""
        text = parse_result.get('text', '')
        markdown = parse_result.get('markdown', '')
        content = markdown or text

        # 简单分块策略
        chunk_size = 1000
        chunks = []

        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i + chunk_size]
            chunks.append({
                'chunk_id': f"{document_id}_{i // chunk_size}",
                'document_id': document_id,
                'chunk_index': i // chunk_size,
                'content': chunk_text,
                'metadata': {
                    'source': 'pdf_parser',
                    'chunk_size': chunk_size
                }
            })

        return chunks

    async def _generate_embeddings(self, chunks: List[Dict], document_id: str) -> List[Dict]:
        """生成向量"""
        chunks_with_embeddings = []
        semaphore = asyncio.Semaphore(1)  # 限制并发

        async def generate_embedding(chunk: Dict):
            async with semaphore:
                embedding = await self._services['embedding'].embed(chunk['content'])
                chunk['embedding'] = embedding
                chunk['embedding_id'] = None  # Will be set after storage
                return chunk

        tasks = [generate_embedding(chunk) for chunk in chunks]
        chunks_with_embeddings = await asyncio.gather(*tasks)

        return chunks_with_embeddings

    async def _store_document_data(self, document_id: str, parse_result: Dict, chunks: List[Dict]):
        """存储文档数据到MySQL和Milvus"""
        from app.core.database import SessionLocal
        from sqlalchemy import text as sql_text

        db = SessionLocal()
        try:
            # 更新文档的parsed_content - 存储为JSON对象
            markdown = parse_result.get('markdown', '')
            parsed_text = parse_result.get('text', '')

            # 构建JSON格式的parsed_content
            import json
            content_json = json.dumps({
                'text': parsed_text,
                'markdown': markdown
            }, ensure_ascii=False)

            db.execute(
                sql_text("UPDATE documents SET parsed_content=:content WHERE id=:id"),
                {'content': content_json, 'id': document_id}
            )
            db.commit()

            # 存储chunks到MySQL和Milvus
            embedding_ids = []
            milvus_chunks = []

            for chunk in chunks:
                # 保存到MySQL
                chunk_record = DocumentChunk(
                    document_id=int(document_id),
                    chunk_index=chunk['chunk_index'],
                    content=chunk['content'],
                    metadata=chunk.get('metadata', {})
                )
                db.add(chunk_record)
                db.flush()

                # 准备Milvus数据
                if chunk.get('embedding') is not None:
                    milvus_chunks.append({
                        'chunk_id': chunk['chunk_id'],
                        'content': chunk['content'],
                        'embedding': chunk['embedding'].tolist() if hasattr(chunk['embedding'], 'tolist') else chunk['embedding'],
                        'chunk_index': chunk['chunk_index'],
                        'page_number': chunk.get('metadata', {}).get('page', 0),
                        'chunk_type': 'text'
                    })

            # 批量插入到Milvus
            if milvus_chunks:
                try:
                    milvus_ids = await self._services['milvus'].insert_chunks_with_full_metadata(
                        document_id=str(document_id),
                        chunks_data=milvus_chunks
                    )
                    embedding_ids = [str(mid) for mid in milvus_ids]
                    logger.info(f"✅ 插入了 {len(milvus_ids)} 个向量到Milvus")
                except Exception as e:
                    logger.warning(f"Milvus存储失败（不影响主流程）: {e}")

            db.commit()
            logger.info(f"✅ 存储了 {len(chunks)} 个文档块")

        finally:
            db.close()

    async def search_documents(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """搜索文档"""
        # 生成查询向量
        query_embedding = await self._services['embedding'].get_embedding(query)

        # 向量搜索
        results = await self._services['milvus'].search_vectors(
            collection_name='document_chunks',
            query_vector=query_embedding,
            limit=top_k
        )

        return results

    async def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'services': {
                'minio': 'healthy' if 'minio' in self._services else 'disabled',
                'milvus': 'healthy' if 'milvus' in self._services else 'disabled',
                'neo4j': 'healthy' if 'neo4j' in self._services else 'disabled',
                'embedding': 'healthy' if 'embedding' in self._services else 'disabled',
            },
            'initialized': self._initialized
        }

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'enable_entity_extraction': self.config.enable_entity_extraction,
            'enable_knowledge_graph': self.config.enable_knowledge_graph,
            'async_entity_extraction': self.config.async_entity_extraction,
            'async_knowledge_graph': self.config.async_knowledge_graph,
        }

# 全局实例
_integrator: Optional[CoreServiceIntegrator] = None

def get_service_integrator() -> CoreServiceIntegrator:
    """获取服务整合器实例"""
    global _integrator
    if _integrator is None:
        _integrator = CoreServiceIntegrator()
    return _integrator
