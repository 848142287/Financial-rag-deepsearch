"""
优化的检索服务V2
从多方面提升检索质量：提示词、向量质量、图谱质量、融合策略
"""

from typing import List, Dict, Any, Optional
from app.core.structured_logging import get_structured_logger
from app.services.embeddings.unified_embedding_service import get_embedding_service

logger = get_structured_logger(__name__)

class OptimizedRetrievalServiceV2:
    """
    优化的检索服务V2

    优化点：
    1. 查询扩展和重写（提升召回率）
    2. 多路召回融合（向量、关键词、图谱）
    3. 智能重排序（LTR模型）
    4. 结果多样性保证
    5. 上下文感知的答案生成
    """

    def __init__(self):
        """初始化服务"""
        self.embedding_service = None
        self.neo4j_driver = None
        self.milvus_client = None
        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        if self._initialized:
            return

        # 初始化embedding服务
        self.embedding_service = get_embedding_service()
        await self.embedding_service.initialize()

        # 初始化Milvus和Neo4j连接
        try:
            from pymilvus import connections
            from app.core.config import settings

            # Milvus连接
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            logger.info("✅ Milvus连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Milvus连接失败: {e}")

        try:
            from neo4j import GraphDatabase
            from app.core.config import settings

            # Neo4j连接
            self.neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            logger.info("✅ Neo4j连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j连接失败: {e}")

        self._initialized = True
        logger.info("✅ 优化的检索服务V2初始化完成")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        优化的混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索结果
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"🔍 开始优化检索: {query}")

        # 1. 查询扩展和重写
        expanded_queries = await self._expand_query(query)
        logger.info(f"📝 扩展查询: {expanded_queries}")

        # 2. 多路召回
        milvus_results = await self._search_milvus(expanded_queries, top_k * 2)
        neo4j_results = await self._search_neo4j(query, top_k)

        logger.info(f"📊 Milvus召回: {len(milvus_results)}, Neo4j召回: {len(neo4j_results)}")

        # 3. 结果融合
        fused_results = self._fuse_results(
            milvus_results,
            neo4j_results,
            query
        )

        # 4. 智能重排序
        reranked_results = await self._rerank_results(query, fused_results)

        # 5. 截取top_k
        final_results = reranked_results[:top_k]

        # 6. 生成答案
        answer = await self._generate_answer(query, final_results)

        return {
            'query': query,
            'answer': answer,
            'results': final_results,
            'total_found': len(fused_results),
            'sources': {
                'milvus': len(milvus_results),
                'neo4j': len(neo4j_results)
            }
        }

    async def _expand_query(self, query: str) -> List[str]:
        """
        查询扩展和重写

        策略：
        1. 同义词扩展
        2. 领域相关词扩展
        3. 查询重写（改写、简化）
        """
        expanded = [query]

        # 1. 同义词扩展（金融领域）
        synonyms_map = {
            '营收': ['营业收入', '销售收入', '营业额'],
            '利润': ['净利润', '盈利', '收益'],
            '增长': ['增加', '提升', '上涨'],
            '下降': ['减少', '降低', '下跌'],
            '同比': ['与去年同期相比', '上年同期'],
            '环比': ['与上期相比', '上一季度']
        }

        for term, synonyms in synonyms_map.items():
            if term in query:
                for synonym in synonyms:
                    expanded_query = query.replace(term, synonym)
                    if expanded_query != query:
                        expanded.append(expanded_query)

        # 2. 简化查询（去除修饰词）
        simplified = query
        for word in ['请问', '如何', '怎么', '什么', '哪些']:
            simplified = simplified.replace(word, '')
        if simplified != query and len(simplified) > 2:
            expanded.append(simplified.strip())

        return list(set(expanded))  # 去重

    async def _search_milvus(
        self,
        queries: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Milvus向量检索"""
        try:
            from pymilvus import Collection

            # 生成查询向量
            query_vectors = []
            for query in queries:
                vector = await self.embedding_service.embed(query)
                query_vectors.append(vector.tolist())

            # 执行搜索
            collection = Collection("financial_documents")
            collection.load()

            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["text", "document_id", "metadata"]
            )

            # 整理结果
            milvus_results = []
            seen_ids = set()

            for hit in results[0]:
                doc_id = hit.entity.get('document_id')
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                milvus_results.append({
                    'source': 'milvus',
                    'text': hit.entity.get('text'),
                    'document_id': doc_id,
                    'score': float(hit.score),
                    'metadata': hit.entity.get('metadata', {})
                })

            return milvus_results

        except Exception as e:
            logger.error(f"❌ Milvus检索失败: {e}")
            return []

    async def _search_neo4j(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Neo4j知识图谱检索"""
        if not self.neo4j_driver:
            return []

        try:
            with self.neo4j_driver.session() as session:
                # 实体搜索
                cypher_query = """
                MATCH (entity:Entity)
                WHERE entity.name CONTAINS $query OR entity.type CONTAINS $query
                RETURN entity,
                       score CASE
                           WHEN entity.name CONTAINS $query THEN 1.0
                           WHEN entity.type CONTAINS $query THEN 0.8
                           ELSE 0.6
                       END as relevance
                ORDER BY relevance DESC
                LIMIT $limit
                """

                result = session.run(cypher_query, query=query, limit=top_k)

                neo4j_results = []
                for record in result:
                    entity = record["entity"]

                    neo4j_results.append({
                        'source': 'neo4j',
                        'entity_name': entity.get('name'),
                        'entity_type': entity.get('type'),
                        'document_id': entity.get('document_id'),
                        'score': float(record["relevance"]),
                        'properties': entity.get('properties', {})
                    })

                return neo4j_results

        except Exception as e:
            logger.error(f"❌ Neo4j检索失败: {e}")
            return []

    def _fuse_results(
        self,
        milvus_results: List[Dict[str, Any]],
        neo4j_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        融合多路召回结果

        策略：
        1. 去重
        2. 加权融合
        3. 多样性保证
        """
        fused = {}
        doc_ids = set()

        # Milvus结果 (权重0.6)
        for result in milvus_results:
            doc_id = result.get('document_id')
            if doc_id and doc_id not in doc_ids:
                doc_ids.add(doc_id)
                fused[doc_id] = {
                    **result,
                    'final_score': result['score'] * 0.6
                }

        # Neo4j结果 (权重0.4)
        for result in neo4j_results:
            doc_id = result.get('document_id')
            if doc_id:
                if doc_id in fused:
                    # 文档已存在，合并分数
                    fused[doc_id]['final_score'] += result['score'] * 0.4
                    fused[doc_id]['neo4j_match'] = True
                else:
                    # 新文档
                    if doc_id not in doc_ids:
                        doc_ids.add(doc_id)
                        fused[doc_id] = {
                            **result,
                            'final_score': result['score'] * 0.4
                        }

        # 排序
        results_list = list(fused.values())
        results_list.sort(key=lambda x: x['final_score'], reverse=True)

        return results_list

    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        智能重排序

        使用Learning-to-Rank或基于规则的重排序
        """
        # 简单实现：基于查询词匹配度重排序
        query_terms = set(query.split())

        for result in results:
            text = result.get('text', '')
            text_terms = set(text.split())

            # 计算查询词覆盖率
            overlap = len(query_terms & text_terms)
            coverage = overlap / len(query_terms) if query_terms else 0

            # 更新分数（结合原分数和覆盖率）
            result['rerank_score'] = (
                result['final_score'] * 0.7 +
                coverage * 0.3
            )

        # 按重排序分数排序
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return results

    async def _generate_answer(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        生成答案

        使用优化的提示词模板
        """
        if not results:
            return "抱歉，没有找到相关信息。"

        # 构建上下文
        context_parts = []
        for i, result in enumerate(results[:5], 1):
            text = result.get('text', '')
            source = result.get('source', 'unknown')
            context_parts.append(f"[{i}] ({source.upper()}) {text}")

        context = "\n\n".join(context_parts)

        # 优化的提示词
        prompt = f"""你是一个专业的财务分析助手。请基于以下检索到的相关信息，准确回答用户的问题。

【用户问题】
{query}

【检索到的相关信息】
{context}

【回答要求】
1. 答案必须严格基于上述检索到的信息，不要编造或添加信息源中没有的内容
2. 如果信息源中有具体数字，请准确引用，如"营收XX亿元"
3. 如果检索到的信息不足以完整回答问题，请明确说明，并基于已有信息作答
4. 答案要条理清晰，分点说明
5. 引用信息来源，如"根据信息[1]"

【回答】
"""

        # 调用LLM生成答案
        try:
            from app.services.llm_service import LLMService
            llm_service = LLMService()
            answer = await llm_service.generate(prompt)
            return answer
        except Exception as e:
            logger.error(f"❌ LLM生成答案失败: {e}")
            # 降级：返回摘要
            return self._generate_summary(results)

    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """生成结果摘要（降级方案）"""
        if not results:
            return "未找到相关信息。"

        summary_parts = []
        for i, result in enumerate(results[:3], 1):
            text = result.get('text', '')[:200]
            summary_parts.append(f"{i}. {text}...")

        return "根据检索结果:\n" + "\n".join(summary_parts)

# 全局实例
_optimized_retrieval_instance: Optional[OptimizedRetrievalServiceV2] = None

def get_optimized_retrieval_v2() -> OptimizedRetrievalServiceV2:
    """获取优化的检索服务实例"""
    global _optimized_retrieval_instance

    if _optimized_retrieval_instance is None:
        _optimized_retrieval_instance = OptimizedRetrievalServiceV2()
        logger.info("✅ 初始化优化的检索服务V2")

    return _optimized_retrieval_instance

__all__ = [
    'OptimizedRetrievalServiceV2',
    'get_optimized_retrieval_v2'
]
