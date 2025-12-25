"""
优化的检索服务
优先从MongoDB查询解析后的文件信息，然后回退到其他存储系统
"""

import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from pymilvus import Collection, connections
from neo4j import GraphDatabase
from pymongo import MongoClient

logger = logging.getLogger(__name__)

class OptimizedRetrievalService:
    """优化的检索服务"""

    def __init__(self):
        self.mongo_client = None
        self.neo4j_driver = None
        self.milvus_connected = False
        self._initialize_connections()

    def _initialize_connections(self):
        """初始化各种存储连接"""
        try:
            # MongoDB连接
            self.mongo_client = MongoClient(
                'mongodb://admin:password@localhost:27017/',
                serverSelectionTimeoutMS=5000
            )
            self.mongo_db = self.mongo_client['financial_rag']
            # 测试连接
            self.mongo_db.command('ping')
            logger.info("✅ MongoDB连接成功")
        except Exception as e:
            logger.warning(f"⚠️ MongoDB连接失败: {e}")
            self.mongo_client = None

        try:
            # Neo4j连接
            self.neo4j_driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "password")
            )
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Neo4j连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j连接失败: {e}")
            self.neo4j_driver = None

        try:
            # Milvus连接
            connections.connect(alias="default", host='milvus', port='19530')
            self.milvus_connected = True
            logger.info("✅ Milvus连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Milvus连接失败: {e}")
            self.milvus_connected = False

    def search_mongodb_parsed_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """从MongoDB搜索解析后的文档内容"""
        if not self.mongo_client:
            return []

        try:
            # 生成查询哈希用于缓存
            query_hash = hashlib.md5(query.encode()).hexdigest()

            # 先检查缓存
            cache_collection = self.mongo_db['search_cache']
            cached_result = cache_collection.find_one({
                "query_hash": query_hash,
                "created_at": {"$gte": datetime.utcnow() - timedelta(minutes=30)}
            })

            if cached_result:
                logger.info(f"✅ 从MongoDB缓存获取结果")
                return cached_result.get("results", [])

            # 搜索解析内容
            parsed_content_collection = self.mongo_db['document_parsed_content']

            # 文本搜索
            text_results = list(parsed_content_collection.find({
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"content": {"$regex": query, "$options": "i"}},
                    {"sections.title": {"$regex": query, "$options": "i"}},
                    {"sections.content": {"$regex": query, "$options": "i"}}
                ]
            }).limit(limit))

            # 转换为统一格式
            results = []
            for doc in text_results:
                results.append({
                    "source": "mongodb",
                    "document_id": doc.get("document_id"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "sections": doc.get("sections", []),
                    "metadata": doc.get("metadata", {}),
                    "score": 1.0,  # MongoDB搜索暂时给固定分数
                    "relevance": "text_match"
                })

            # 缓存结果
            if results:
                cache_collection.insert_one({
                    "query_hash": query_hash,
                    "query": query,
                    "results": results,
                    "created_at": datetime.utcnow()
                })

            logger.info(f"✅ 从MongoDB找到 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"❌ MongoDB搜索失败: {e}")
            return []

    def search_milvus_vectors(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """从Milvus搜索向量相似度"""
        if not self.milvus_connected:
            return []

        try:
            # 这里需要实际的向量嵌入服务
            # 暂时返回空结果
            logger.info("⚠️ Milvus搜索需要向量嵌入服务")
            return []

        except Exception as e:
            logger.error(f"❌ Milvus搜索失败: {e}")
            return []

    def search_neo4j_knowledge_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """从Neo4j搜索知识图谱"""
        if not self.neo4j_driver:
            return []

        try:
            with self.neo4j_driver.session() as session:
                # 实体识别和关系搜索
                cypher_query = """
                MATCH (entity)-[rel]->(related)
                WHERE entity.name CONTAINS $query OR entity.type CONTAINS $query
                RETURN entity, rel, related,
                       score CASE
                           WHEN entity.name CONTAINS $query THEN 1.0
                           WHEN entity.type CONTAINS $query THEN 0.8
                           ELSE 0.6
                       END as relevance
                ORDER BY relevance DESC
                LIMIT $limit
                """

                result = session.run(cypher_query, query=query, limit=limit)

                knowledge_results = []
                for record in result:
                    entity = record["entity"]
                    rel = record["rel"]
                    related = record["related"]

                    knowledge_results.append({
                        "source": "neo4j",
                        "entity": dict(entity),
                        "relationship": dict(rel),
                        "related_entity": dict(related),
                        "relevance": record["relevance"],
                        "type": "knowledge_graph"
                    })

                logger.info(f"✅ 从Neo4j找到 {len(knowledge_results)} 个知识图谱结果")
                return knowledge_results

        except Exception as e:
            logger.error(f"❌ Neo4j搜索失败: {e}")
            return []

    def hybrid_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """混合搜索：结合MongoDB、Milvus和Neo4j"""
        logger.info(f"🔍 开始混合搜索: {query}")

        # 1. 优先从MongoDB搜索解析内容
        mongodb_results = self.search_mongodb_parsed_content(query, limit)

        # 2. 并行搜索其他存储
        milvus_results = self.search_milvus_vectors(query, limit)
        neo4j_results = self.search_neo4j_knowledge_graph(query, limit)

        # 3. 合并和排序结果
        all_results = []

        # MongoDB结果（高优先级）
        for result in mongodb_results:
            all_results.append({
                **result,
                "priority": 1,
                "source_weight": 0.6
            })

        # Neo4j结果（中优先级）
        for result in neo4j_results:
            all_results.append({
                **result,
                "priority": 2,
                "source_weight": 0.3
            })

        # Milvus结果（低优先级，因为需要向量嵌入）
        for result in milvus_results:
            all_results.append({
                **result,
                "priority": 3,
                "source_weight": 0.1
            })

        # 按优先级和相关性排序
        all_results.sort(key=lambda x: (x["priority"], -x.get("relevance", 0)))

        # 截取到指定数量
        final_results = all_results[:limit]

        # 生成综合答案
        answer = self.generate_answer(query, final_results)

        search_summary = {
            "query": query,
            "total_results": len(all_results),
            "returned_results": len(final_results),
            "sources_used": {
                "mongodb": len(mongodb_results),
                "milvus": len(milvus_results),
                "neo4j": len(neo4j_results)
            },
            "mongodb_priority": True,
            "cached": len([r for r in mongodb_results if r.get("from_cache", False)])
        }

        return {
            "query": query,
            "answer": answer,
            "results": final_results,
            "summary": search_summary,
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """基于搜索结果生成答案"""
        if not results:
            return f"抱歉，没有找到与'{query}'相关的信息。"

        # 主要从MongoDB结果生成答案
        mongodb_results = [r for r in results if r.get("source") == "mongodb"]

        if mongodb_results:
            # 使用第一个最相关的结果
            best_result = mongodb_results[0]

            if best_result.get("content"):
                content = best_result["content"][:500]
                return f"根据相关文档信息：\n\n{content}...\n\n这个信息来自文档：{best_result.get('title', '未知文档')}"

        # 如果没有MongoDB结果，使用其他结果
        if results:
            neo4j_results = [r for r in results if r.get("source") == "neo4j"]
            if neo4j_results:
                entity_info = neo4j_results[0].get("entity", {})
                return f"在知识图谱中找到相关信息：{entity_info.get('name', '未知实体')} ({entity_info.get('type', '未知类型')})"

        return f"找到 {len(results)} 个相关结果，但需要进一步处理才能生成详细答案。"

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "mongodb": {
                "connected": self.mongo_client is not None,
                "collections": 0
            },
            "neo4j": {
                "connected": self.neo4j_driver is not None
            },
            "milvus": {
                "connected": self.milvus_connected
            }
        }

        # 获取MongoDB集合统计
        if self.mongo_client:
            try:
                collections = self.mongo_db.list_collection_names()
                status["mongodb"]["collections"] = len(collections)

                # 获取文档数量
                total_docs = 0
                for collection_name in collections:
                    collection = self.mongo_db[collection_name]
                    total_docs += collection.count_documents({})
                status["mongodb"]["total_documents"] = total_docs

            except Exception as e:
                logger.error(f"获取MongoDB统计失败: {e}")

        return status

# 创建全局实例
optimized_retrieval_service = OptimizedRetrievalService()