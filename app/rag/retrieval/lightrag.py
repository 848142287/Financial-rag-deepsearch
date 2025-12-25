"""
LightRAG - 轻量级快速检索
使用向量检索 + 全文搜索实现快速响应
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

try:
    from pymilvus import connections, Collection
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

from sqlalchemy import create_engine, text
import pymysql


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_vector_search: bool = True
    use_fulltext_search: bool = True
    rerank: bool = True
    rerank_top_k: int = 10


class LightRAG:
    """轻量级快速检索"""

    def __init__(
        self,
        mysql_config: Dict[str, Any],
        milvus_config: Dict[str, Any],
        config: RetrievalConfig = None
    ):
        """
        初始化LightRAG

        Args:
            mysql_config: MySQL配置
            milvus_config: Milvus配置
            config: 检索配置
        """
        self.mysql_config = mysql_config
        self.milvus_config = milvus_config
        self.config = config or RetrievalConfig()

        # MySQL连接
        self.mysql_engine = create_engine(
            f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@"
            f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
        )

        # Milvus连接
        if MILVUS_AVAILABLE:
            connections.connect(
                alias="default",
                host=milvus_config.get("host", "milvus"),
                port=milvus_config.get("port", "19530")
            )
            self.collection = Collection(milvus_config.get("collection", "document_embeddings"))

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        if not MILVUS_AVAILABLE:
            return []

        try:
            self.collection.load()

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["document_id", "chunk_index", "content"]
            )

            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "id": str(hit.entity.get("document_id")),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "content": hit.entity.get("content", ""),
                    "score": float(1 / (1 + hit.distance)),  # 转换为相似度
                    "retrieval_method": "vector"
                })

            return formatted_results

        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

    def fulltext_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        全文检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        try:
            with self.mysql_engine.connect() as conn:
                # 使用MySQL全文搜索
                sql_query = text("""
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.chunk_index,
                        dc.content,
                        MATCH(dc.content) AGAINST(:query IN NATURAL LANGUAGE MODE) as score
                    FROM document_chunks dc
                    WHERE MATCH(dc.content) AGAINST(:query IN NATURAL LANGUAGE MODE)
                    ORDER BY score DESC
                    LIMIT :limit
                """)

                result = conn.execute(
                    sql_query,
                    {"query": query, "limit": top_k}
                )

                formatted_results = []
                for row in result:
                    formatted_results.append({
                        "id": str(row.document_id),
                        "chunk_index": row.chunk_index,
                        "content": row.content,
                        "score": float(row.score) if row.score else 0.0,
                        "retrieval_method": "fulltext"
                    })

                return formatted_results

        except Exception as e:
            print(f"Fulltext search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        混合检索（向量 + 全文）

        Args:
            query: 查询文本
            query_embedding: 查询向量（可选）

        Returns:
            检索结果列表
        """
        all_results = []
        result_ids = set()

        # 向量检索
        if self.config.use_vector_search and query_embedding:
            vector_results = self.vector_search(
                query_embedding,
                top_k=self.config.rerank_top_k if self.config.rerank else self.config.top_k
            )

            for result in vector_results:
                result_id = f"{result['id']}_{result['chunk_index']}"
                if result_id not in result_ids and result["score"] >= self.config.similarity_threshold:
                    result_ids.add(result_id)
                    all_results.append(result)

        # 全文检索
        if self.config.use_fulltext_search:
            ft_results = self.fulltext_search(
                query,
                top_k=self.config.rerank_top_k if self.config.rerank else self.config.top_k
            )

            for result in ft_results:
                result_id = f"{result['id']}_{result['chunk_index']}"
                if result_id not in result_ids:
                    result_ids.add(result_id)
                    all_results.append(result)

        # 重排序
        if self.config.rerank and len(all_results) > self.config.top_k:
            # 简单的重排序：结合向量分数和全文分数
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            all_results = all_results[:self.config.top_k]

        return all_results

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        执行检索

        Args:
            query: 查询文本
            query_embedding: 查询向量（可选）

        Returns:
            检索结果
        """
        start_time = time.time()

        results = self.hybrid_search(query, query_embedding)

        # 获取完整的文档信息
        document_ids = list(set([r["id"] for r in results]))

        doc_details = self._get_document_details(document_ids)

        # 组装最终结果
        final_results = []
        for result in results:
            doc_info = doc_details.get(result["id"], {})
            final_results.append({
                **result,
                "title": doc_info.get("title", ""),
                "filename": doc_info.get("filename", ""),
                "metadata": doc_info.get("metadata", {})
            })

        retrieval_time = time.time() - start_time

        return {
            "query": query,
            "results": final_results,
            "total_retrieved": len(final_results),
            "retrieval_time": retrieval_time,
            "method": "lightrag"
        }

    def _get_document_details(
        self,
        document_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """获取文档详情"""
        if not document_ids:
            return {}

        try:
            with self.mysql_engine.connect() as conn:
                placeholders = ",".join([":id" + str(i) for i in range(len(document_ids))])
                sql_query = text(f"""
                    SELECT
                        id,
                        title,
                        filename,
                        metadata
                    FROM documents
                    WHERE id IN ({placeholders})
                """)

                params = {f"id{i}": doc_id for i, doc_id in enumerate(document_ids)}
                result = conn.execute(sql_query, params)

                doc_details = {}
                for row in result:
                    doc_details[str(row.id)] = {
                        "title": row.title,
                        "filename": row.filename,
                        "metadata": row.metadata if row.metadata else {}
                    }

                return doc_details

        except Exception as e:
            print(f"Failed to get document details: {e}")
            return {}

    def close(self):
        """关闭连接"""
        if MILVUS_AVAILABLE:
            try:
                connections.disconnect("default")
            except:
                pass
