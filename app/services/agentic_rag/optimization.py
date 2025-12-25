"""
Agentic RAG检索优化策略
缓存、并行处理、金融适配等优化
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import time

from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    hot_query_ttl: int = 300      # 热门查询5分钟
    vector_result_ttl: int = 600   # 向量结果10分钟
    metadata_ttl: int = 3600       # 元数据1小时
    max_cache_size: int = 10000    # 最大缓存条目
    cache_key_prefix: str = "agentic_rag"


@dataclass
class ParallelConfig:
    """并行处理配置"""
    max_parallel_retrievals: int = 3
    document_parse_workers: int = 4
    vector_batch_size: int = 32
    enable_async_processing: bool = True


class RetrievalOptimizer:
    """检索优化器"""

    def __init__(self):
        self.redis = get_redis_client()
        self.cache_config = CacheConfig()
        self.parallel_config = ParallelConfig()

        # 金融术语增强词典
        self.financial_enhanced_terms = {
            '市场情绪': ['投资者情绪', '市场信心', '市场恐慌指数', 'VIX指数'],
            '宏观经济': ['GDP', 'CPI', 'PMI', '货币政策', '财政政策'],
            '行业分析': ['产业链', '竞争格局', '市场集中度', '行业周期'],
            '公司分析': ['商业模式', '核心竞争力', '护城河', 'ROE', 'ROA'],
            '技术分析': ['K线', '均线', 'MACD', 'RSI', '布林带'],
            '基本面分析': ['市盈率', '市净率', '现金流', '负债率', '成长性'],
            '风险管理': ['VaR', '最大回撤', '夏普比率', '波动率']
        }

        # 时间敏感查询模式
        self.temporal_patterns = [
            r'最近.*?年',
            r'过去.*?月',
            r'\d{4}年.*?至今',
            r'今年.*?表现',
            r'近期.*?趋势',
            r'最新.*?数据'
        ]

    async def cache_hot_query(self, query_hash: str, result: Dict[str, Any]):
        """缓存热门查询结果"""
        try:
            key = f"{self.cache_config.cache_key_prefix}:hot:{query_hash}"
            await self.redis.setex(
                key,
                self.cache_config.hot_query_ttl,
                json.dumps(result, ensure_ascii=False)
            )
            logger.debug(f"Cached hot query result: {query_hash}")
        except Exception as e:
            logger.error(f"Failed to cache hot query: {str(e)}")

    async def get_cached_hot_query(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """获取缓存的热门查询"""
        try:
            key = f"{self.cache_config.cache_key_prefix}:hot:{query_hash}"
            cached = await self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Failed to get cached hot query: {str(e)}")
        return None

    async def cache_vector_results(
        self,
        query_vector: List[float],
        results: List[Dict[str, Any]]
    ):
        """缓存向量检索结果"""
        try:
            # 将向量转换为字符串键
            vector_hash = hashlib.md5(str(query_vector).encode()).hexdigest()
            key = f"{self.cache_config.cache_key_prefix}:vector:{vector_hash}"

            await self.redis.setex(
                key,
                self.cache_config.vector_result_ttl,
                json.dumps(results, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"Failed to cache vector results: {str(e)}")

    async def get_cached_vector_results(
        self,
        query_vector: List[float]
    ) -> Optional[List[Dict[str, Any]]]:
        """获取缓存的向量结果"""
        try:
            vector_hash = hashlib.md5(str(query_vector).encode()).hexdigest()
            key = f"{self.cache_config.cache_key_prefix}:vector:{vector_hash}"
            cached = await self.redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Failed to get cached vector results: {str(e)}")
        return None

    async def parallel_document_parsing(
        self,
        documents: List[Any],
        parse_func: Callable
    ) -> List[Any]:
        """并行解析文档"""
        if not self.parallel_config.enable_async_processing:
            # 串行处理
            return [await parse_func(doc) for doc in documents]

        # 分批并行处理
        batch_size = self.parallel_config.document_parse_workers
        tasks = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            task = asyncio.create_task(self._parse_document_batch(batch, parse_func))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        # 展平结果
        return [item for batch_result in results for item in batch_result]

    async def _parse_document_batch(
        self,
        batch: List[Any],
        parse_func: Callable
    ) -> List[Any]:
        """解析文档批次"""
        tasks = [parse_func(doc) for doc in batch]
        return await asyncio.gather(*tasks)

    async def batch_vector_generation(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """批量生成向量"""
        from app.services.embedding_service import embedding_service

        batch_size = self.parallel_config.vector_batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await embedding_service.get_embeddings(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def enhance_financial_query(self, query: str) -> str:
        """增强金融查询"""
        enhanced_query = query

        # 添加金融术语
        for term, alternatives in self.financial_enhanced_terms.items():
            if term in query:
                # 添加相关术语到查询中
                for alt in alternatives[:2]:  # 最多添加2个相关术语
                    if alt not in query:
                        enhanced_query += f" {alt}"

        # 处理时间敏感查询
        import re
        for pattern in self.temporal_patterns:
            if re.search(pattern, query):
                # 添加时间关键词
                enhanced_query += " 时间序列 数据"

        return enhanced_query

    def temporal_query_processing(self, query: str) -> Dict[str, Any]:
        """时间敏感查询处理"""
        import re
        from datetime import datetime, timedelta

        result = {
            'is_temporal': False,
            'time_range': None,
            'time_keywords': [],
            'enhanced_query': query
        }

        # 检测时间模式
        current_year = datetime.now().year
        time_mappings = {
            '今年': str(current_year),
            '去年': str(current_year - 1),
            '最近一年': f"{current_year - 1}年至今",
            '最近三年': f"{current_year - 3}年至今"
        }

        for chinese_term, time_expr in time_mappings.items():
            if chinese_term in query:
                result['is_temporal'] = True
                result['time_keywords'].append(chinese_term)
                result['enhanced_query'] = query.replace(chinese_term, time_expr)

        # 提取具体年份
        years = re.findall(r'20\d{2}', query)
        if years:
            result['is_temporal'] = True
            result['time_keywords'].extend(years)

        return result

    async def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取热门查询"""
        try:
            # 从Redis获取查询计数
            pattern = f"{self.cache_config.cache_key_prefix}:query_count:*"
            keys = await self.redis.keys(pattern)

            query_counts = []
            for key in keys:
                count = await self.redis.get(key)
                if count:
                    query = key.decode().split(':')[-1]
                    query_counts.append({
                        'query': query,
                        'count': int(count)
                    })

            # 排序并返回前N个
            query_counts.sort(key=lambda x: x['count'], reverse=True)
            return query_counts[:limit]

        except Exception as e:
            logger.error(f"Failed to get popular queries: {str(e)}")
            return []

    async def record_query_frequency(self, query: str):
        """记录查询频率"""
        try:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            key = f"{self.cache_config.cache_key_prefix}:query_count:{query_hash}"

            # 原子性递增
            await self.redis.incr(key)
            await self.redis.expire(key, 86400 * 7)  # 7天过期

        except Exception as e:
            logger.error(f"Failed to record query frequency: {str(e)}")

    async def optimize_query_cache(self):
        """优化查询缓存"""
        try:
            # 获取所有缓存键
            pattern = f"{self.cache_config.cache_key_prefix}:*"
            keys = await self.redis.keys(pattern)

            if len(keys) > self.cache_config.max_cache_size:
                # 计算需要清理的键数量
                cleanup_count = len(keys) - self.cache_config.max_cache_size + 100

                # 随机选择要删除的键
                import random
                keys_to_delete = random.sample(keys, cleanup_count)

                # 删除旧的缓存
                for key in keys_to_delete:
                    await self.redis.delete(key)

                logger.info(f"Cleaned up {cleanup_count} cache entries")

        except Exception as e:
            logger.error(f"Failed to optimize query cache: {str(e)}")


class MultiModalIntegrator:
    """多模态结果整合器"""

    def __init__(self):
        pass

    async def integrate_table_summaries(
        self,
        text_results: List[Dict[str, Any]],
        table_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """整合表格摘要"""
        integrated_results = []

        # 合并文本和表格结果
        for text_result in text_results:
            # 查找相关的表格
            related_tables = self._find_related_tables(
                text_result, table_results
            )

            result = text_result.copy()
            if related_tables:
                result['related_tables'] = [
                    {
                        'summary': table.get('summary', ''),
                        'key_data': table.get('key_data', {}),
                        'relevance': table.get('relevance', 0.5)
                    }
                    for table in related_tables[:3]  # 最多3个相关表格
                ]
                result['score'] += len(related_tables) * 0.1  # 提升相关性分数

            integrated_results.append(result)

        # 添加独立的表格结果
        for table in table_results:
            if not self._is_table_covered(table, text_results):
                integrated_results.append({
                    'content': f"表格数据摘要：{table.get('summary', '')}",
                    'metadata': table.get('metadata', {}),
                    'source': 'table',
                    'score': table.get('relevance', 0.5),
                    'table_data': table
                })

        return integrated_results

    async def integrate_image_descriptions(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """整合图片描述"""
        integrated_results = []

        for text_result in text_results:
            # 查找相关的图片
            related_images = self._find_related_images(
                text_result, image_results
            )

            result = text_result.copy()
            if related_images:
                result['related_images'] = [
                    {
                        'description': img.get('description', ''),
                        'chart_type': img.get('chart_type', ''),
                        'relevance': img.get('relevance', 0.5)
                    }
                    for img in related_images[:2]  # 最多2个相关图片
                ]
                result['score'] += len(related_images) * 0.05  # 提升相关性分数

            integrated_results.append(result)

        return integrated_results

    def _find_related_tables(
        self,
        text_result: Dict[str, Any],
        table_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """查找相关表格"""
        text_content = text_result.get('content', '').lower()
        related = []

        for table in table_results:
            table_keywords = table.get('keywords', [])
            keyword_match = sum(1 for kw in table_keywords if kw in text_content)

            if keyword_match > 0:
                table['relevance'] = keyword_match / len(table_keywords)
                related.append(table)

        # 按相关性排序
        related.sort(key=lambda x: x['relevance'], reverse=True)
        return related

    def _find_related_images(
        self,
        text_result: Dict[str, Any],
        image_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """查找相关图片"""
        text_content = text_result.get('content', '').lower()
        related = []

        for image in image_results:
            image_description = image.get('description', '').lower()
            # 简单的文本匹配
            if any(word in text_content for word in image_description.split()[:5]):
                image['relevance'] = 0.7
                related.append(image)

        return related

    def _is_table_covered(
        self,
        table: Dict[str, Any],
        text_results: List[Dict[str, Any]]
    ) -> bool:
        """检查表格是否已被文本结果覆盖"""
        table_keywords = set(table.get('keywords', []))

        for text_result in text_results:
            text_content = text_result.get('content', '').lower()
            if len(table_keywords & set(text_content.split())) > 2:
                return True

        return False

    async def integrate_all(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """整合所有模态结果"""
        # 分离不同类型的结果
        text_results = [r for r in results if r.get('source') in ['vector', 'graph', 'keyword']]
        table_results = [r for r in results if r.get('source') == 'table']
        image_results = [r for r in results if r.get('source') == 'image']

        # 整合表格
        if table_results:
            results = await self.integrate_table_summaries(text_results, table_results)

        # 整合图片
        if image_results:
            results = await self.integrate_image_descriptions(results, image_results)

        return results


# 数据一致性管理器
class ConsistencyManager:
    """数据一致性管理器"""

    def __init__(self):
        self.redis = get_redis_client()

    async def ensure_transaction_consistency(
        self,
        operations: List[Callable]
    ) -> bool:
        """确保事务一致性"""
        try:
            # 使用Redis的WATCH/MULTI/EXEC实现事务
            async with self.redis.pipeline() as pipe:
                # 开始事务监控
                await pipe.watch("global_lock")

                # 尝试获取全局锁
                lock_acquired = await pipe.setnx("global_lock", "1", ex=30)
                if not lock_acquired:
                    return False

                # 执行操作
                await pipe.multi()
                for operation in operations:
                    await operation(pipe)

                # 提交事务
                results = await pipe.execute()

                # 释放锁
                await self.redis.delete("global_lock")

                return all(results)

        except Exception as e:
            logger.error(f"Transaction failed: {str(e)}")
            # 释放锁
            await self.redis.delete("global_lock")
            return False

    async def check_data_consistency(self) -> Dict[str, Any]:
        """检查数据一致性"""
        try:
            consistency_report = {
                'vector_store_status': 'unknown',
                'graph_db_status': 'unknown',
                'cache_status': 'unknown',
                'inconsistencies': []
            }

            # 检查向量存储
            try:
                from app.services.milvus_service import MilvusService
                milvus = MilvusService()
                collections = await milvus.list_collections()
                consistency_report['vector_store_status'] = 'healthy' if collections else 'empty'
            except Exception as e:
                consistency_report['vector_store_status'] = 'error'
                consistency_report['inconsistencies'].append(f"Vector store error: {str(e)}")

            # 检查图数据库
            try:
                from app.services.neo4j_service import Neo4jService
                neo4j = Neo4jService()
                count = await neo4j.count_entities()
                consistency_report['graph_db_status'] = 'healthy'
            except Exception as e:
                consistency_report['graph_db_status'] = 'error'
                consistency_report['inconsistencies'].append(f"Graph DB error: {str(e)}")

            # 检查缓存
            try:
                keys = await self.redis.keys("*")
                consistency_report['cache_status'] = 'healthy'
            except Exception as e:
                consistency_report['cache_status'] = 'error'
                consistency_report['inconsistencies'].append(f"Cache error: {str(e)}")

            return consistency_report

        except Exception as e:
            logger.error(f"Consistency check failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局实例
retrieval_optimizer = RetrievalOptimizer()
multi_modal_integrator = MultiModalIntegrator()
consistency_manager = ConsistencyManager()