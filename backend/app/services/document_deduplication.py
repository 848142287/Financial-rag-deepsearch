"""
文档去重服务
检查上传文档是否已经存在于系统中，避免重复处理
"""

import hashlib
from app.core.structured_logging import get_structured_logger
from typing import Dict, List, Optional

from app.core.database import get_db
from app.models.document import Document
from app.services.vectorstore.unified_milvus_service import UnifiedMilvusService
from app.services.neo4j_service import Neo4jService
from app.services.minio_service import MinIOService
from app.core.cache_manager import cache_manager

logger = get_structured_logger(__name__)

class DocumentDeduplicationResult:
    """文档去重检查结果"""

    def __init__(self):
        self.is_duplicate = False
        self.similarity_score = 0.0
        self.existing_document_id = None
        self.existing_document_info = {}
        self.duplicate_sources = []
        self.matching_details = {}
        self.recommendations = []

    def add_duplicate_source(self, source: str, similarity: float, details: Dict):
        """添加重复来源"""
        self.duplicate_sources.append({
            'source': source,
            'similarity': similarity,
            'details': details
        })
        if similarity > self.similarity_score:
            self.similarity_score = similarity
            self.is_duplicate = True

    def set_duplicate_document(self, document_id: int, document_info: Dict):
        """设置重复的文档信息"""
        self.existing_document_id = document_id
        self.existing_document_info = document_info
        self.is_duplicate = True

class DocumentDeduplicationService:
    """文档去重服务"""

    def __init__(self):
        self.milvus_service = MilvusService()
        self.neo4j_service = Neo4jService()
        self.minio_service = MinIOService()
        self.similarity_threshold = 0.95  # MD5完全匹配阈值
        self.content_similarity_threshold = 0.85  # 内容相似度阈值

    async def calculate_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            return ""

    def calculate_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        try:
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"计算内容哈希失败: {e}")
            return ""

    async def check_mysql_duplicate_by_hash(self, file_hash: str, content_hash: str) -> Optional[Dict]:
        """检查MySQL中是否有相同哈希的文档"""
        try:
            db = next(get_db())

            # 检查文件哈希
            doc = db.query(Document).filter(
                Document.file_hash == file_hash
            ).first()

            if not doc:
                # 检查内容哈希
                doc = db.query(Document).filter(
                    Document.content_hash == content_hash
                ).first()

            if doc:
                return {
                    'document_id': doc.id,
                    'title': doc.title,
                    'description': doc.description,
                    'file_path': doc.file_path,
                    'upload_time': doc.created_at,
                    'status': doc.status
                }

            return None

        except Exception as e:
            logger.error(f"检查MySQL重复失败: {e}")
            return None

    async def check_milvus_duplicate_by_content(self, content: str, limit: int = 5) -> List[Dict]:
        """在Milvus中搜索相似内容"""
        try:
            # 生成内容向量
            from app.services.embeddings.unified_embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            embedding = await embedding_service.embed(content)

            if not embedding:
                return []

            # 在Milvus中搜索
            results = await self.milvus_service.search(
                query_embedding=embedding,
                limit=limit,
                score_threshold=self.content_similarity_threshold
            )

            return results

        except Exception as e:
            logger.error(f"检查Milvus重复失败: {e}")
            return []

    async def check_neo4j_duplicate_by_content(self, content: str, limit: int = 3) -> List[Dict]:
        """在Neo4j中搜索相似内容"""
        try:
            # 提取关键实体
            from app.services.entity_extraction.financial_entity_extractor import FinancialEntityExtractor
            extractor = FinancialEntityExtractor()
            entities = await extractor.extract_entities(content)

            # 在Neo4j中搜索包含相似实体的文档
            similar_docs = []
            for entity in entities[:5]:  # 限制实体数量
                query = f"""
                MATCH (d:Document)-[:HAS_ENTITY]->(e:Entity {{name: '{entity['text']}'}})
                RETURN d.id, d.title, d.content, d.created_at
                LIMIT {limit}
                """

                neo4j_results = await self.neo4j_service.execute_query(query)
                if neo4j_results:
                    similar_docs.extend(neo4j_results)

            return similar_docs[:limit]

        except Exception as e:
            logger.error(f"检查Neo4j重复失败: {e}")
            return []

    async def check_minio_duplicate_by_hash(self, file_hash: str) -> Optional[Dict]:
        """检查MinIO中是否有相同文件"""
        try:
            # 构建可能的存储路径
            possible_paths = [
                f"documents/{file_hash[:2]}/{file_hash}/document",
                f"uploads/{file_hash[:2]}/{file_hash}/content",
                f"parsed/{file_hash[:2]}/{file_hash}/parsed",
            ]

            for path in possible_paths:
                try:
                    objects = self.minio_service.list_objects(path)
                    if objects:
                        return {
                            'path': path,
                            'objects': objects,
                            'storage_type': 'minio'
                        }
                except:
                    continue

            return None

        except Exception as e:
            logger.error(f"检查MinIO重复失败: {e}")
            return None

    async def check_cache_duplicate(self, file_hash: str, content_hash: str) -> Optional[Dict]:
        """检查缓存中的重复信息"""
        try:
            cache_key = f"document_duplicate:{file_hash}:{content_hash}"
            cached_result = await cache_manager.get(cache_key)

            if cached_result:
                return cached_result

            return None

        except Exception as e:
            logger.error(f"检查缓存失败: {e}")
            return None

    async def cache_duplicate_result(self, file_hash: str, content_hash: str, result: Dict):
        """缓存重复检查结果"""
        try:
            cache_key = f"document_duplicate:{file_hash}:{content_hash}"
            # 缓存24小时
            await cache_manager.set(cache_key, result, ttl=86400)

        except Exception as e:
            logger.error(f"缓存重复结果失败: {e}")

    async def check_document_duplication(
        self,
        file_path: str,
        content: str,
        file_metadata: Dict = None
    ) -> DocumentDeduplicationResult:
        """
        检查文档重复
        """
        result = DocumentDeduplicationResult()

        # 计算哈希
        file_hash = await self.calculate_file_hash(file_path)
        content_hash = self.calculate_content_hash(content)

        if not file_hash or not content_hash:
            logger.warning("无法计算文档哈希，跳过重复检查")
            return result

        logger.info(f"检查文档重复: file_hash={file_hash[:8]}, content_hash={content_hash[:8]}")

        # 1. 检查缓存
        cached_result = await self.check_cache_duplicate(file_hash, content_hash)
        if cached_result:
            logger.info(f"从缓存中发现重复文档: {cached_result.get('document_id')}")
            result.is_duplicate = True
            result.existing_document_id = cached_result.get('document_id')
            result.existing_document_info = cached_result.get('document_info', {})
            return result

        # 2. 检查MySQL数据库（完全匹配）
        mysql_result = await self.check_mysql_duplicate_by_hash(file_hash, content_hash)
        if mysql_result:
            logger.info(f"MySQL中发现重复文档: {mysql_result['document_id']}")
            result.is_duplicate = True
            result.set_duplicate_document(mysql_result['document_id'], mysql_result)
            result.add_duplicate_source('mysql', 1.0, {'match_type': 'exact_hash'})
            await self.cache_duplicate_result(file_hash, content_hash, {
                'document_id': mysql_result['document_id'],
                'document_info': mysql_result
            })
            return result

        # 3. 检查MinIO存储（完全匹配）
        minio_result = await self.check_minio_duplicate_by_hash(file_hash)
        if minio_result:
            logger.info(f"MinIO中发现重复文件: {minio_result['path']}")
            result.is_duplicate = True
            result.add_duplicate_source('minio', 1.0, {
                'storage_path': minio_result['path'],
                'objects_count': len(minio_result.get('objects', []))
            })

        # 4. 检查向量数据库（内容相似度）
        milvus_results = await self.check_milvus_duplicate_by_content(content, limit=3)
        if milvus_results:
            logger.info(f"Milvus中发现 {len(milvus_results)}个相似文档")
            for milvus_result in milvus_results:
                similarity = milvus_result.get('score', 0)
                if similarity >= self.content_similarity_threshold:
                    result.add_duplicate_source('milvus', similarity, {
                        'document_id': milvus_result.get('document_id'),
                        'chunk_id': milvus_result.get('chunk_id'),
                        'content_preview': milvus_result.get('content', '')[:100] + '...'
                    })

        # 5. 检查知识图谱（实体匹配）
        neo4j_results = await self.check_neo4j_duplicate_by_content(content, limit=3)
        if neo4j_results:
            logger.info(f"Neo4j中发现 {len(neo4j_results)}个相似文档")
            for neo4j_result in neo4j_results:
                # 简单的相似度计算（基于共享实体数量）
                similarity = min(len(neo4j_results) / 10.0, 1.0)  # 假设最多10个实体
                if similarity >= self.content_similarity_threshold:
                    result.add_duplicate_source('neo4j', similarity, {
                        'document_id': neo4j_result.get('id'),
                        'title': neo4j_result.get('title'),
                        'entities_count': len(neo4j_results)
                    })

        # 6. 缓存结果
        await self.cache_duplicate_result(file_hash, content_hash, {
            'is_duplicate': result.is_duplicate,
            'similarity_score': result.similarity_score,
            'duplicate_sources': result.duplicate_sources,
            'existing_document_id': result.existing_document_id
        })

        # 7. 生成推荐
        result.recommendations = self._generate_recommendations(result)

        return result

    def _generate_recommendations(self, result: DocumentDeduplicationResult) -> List[str]:
        """生成处理建议"""
        recommendations = []

        if result.is_duplicate:
            if result.similarity_score >= 0.99:
                recommendations.append("此文档与现有文档几乎完全相同，建议直接使用现有文档")
            elif result.similarity_score >= 0.95:
                recommendations.append("此文档与现有文档高度相似，建议查看现有文档后再决定是否上传")
            elif result.similarity_score >= 0.85:
                recommendations.append("此文档与现有文档较为相似，建议确认是否需要上传")
            else:
                recommendations.append("此文档与现有文档相似度较低，可以正常上传")
        else:
            recommendations.append("此文档与系统中现有文档不重复，可以正常上传")

        # 根据数据源给出具体建议
        if result.duplicate_sources:
            sources = [source['source'] for source in result.duplicate_sources]
            if len(sources) > 1:
                recommendations.append(f"在{', '.join(sources)}中都发现了相关数据")
            else:
                recommendations.append(f"在{sources[0]}中发现了相关数据")

        return recommendations

    async def get_duplicate_summary(self, file_hash: str, content_hash: str) -> Dict:
        """获取重复检查摘要"""
        result = await self.check_document_duplication("", "", {})

        return {
            'is_duplicate': result.is_duplicate,
            'similarity_score': result.similarity_score,
            'duplicate_sources_count': len(result.duplicate_sources),
            'duplicate_sources': [
                {
                    'source': source['source'],
                    'similarity': source['similarity'],
                    'details': source['details']
                }
                for source in result.duplicate_sources
            ],
            'existing_document_id': result.existing_document_id,
            'recommendations': result.recommendations
        }

# 全局去重服务实例
document_deduplication_service = DocumentDeduplicationService()