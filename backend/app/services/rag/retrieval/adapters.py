"""
向量存储和BM25存储适配器
将现有的Milvus服务和新的BM25服务适配到混合检索框架
"""

from abc import ABC, abstractmethod
from app.core.structured_logging import get_structured_logger
import pickle
from pathlib import Path

logger = get_structured_logger(__name__)

# ============================================================================
# Milvus 向量存储适配器
# ============================================================================

class VectorStoreAdapter(ABC):
    """向量存储适配器接口"""

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 元数据过滤条件

        Returns:
            [{"doc_id", "content", "score", "metadata"}, ...]
        """
        pass

class MilvusVectorStoreAdapter(VectorStoreAdapter):
    """
    Milvus向量存储适配器

    将现有的MilvusService适配到混合检索框架
    """

    def __init__(self, milvus_service, embedding_service=None):
        """
        初始化适配器

        Args:
            milvus_service: MilvusService实例
            embedding_service: 向量编码服务（用于将查询文本转换为向量）
        """
        self.milvus = milvus_service
        self.embedding_service = embedding_service

        logger.info("MilvusVectorStoreAdapter初始化完成")

    async def search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any] = None,
        source: str = "milvus"
    ) -> List[Dict[str, Any]]:
        """
        向量搜索（文本查询）

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 元数据过滤条件
            source: 来源标识

        Returns:
            搜索结果列表
        """
        try:
            # 1. 将查询文本转换为向量
            if self.embedding_service:
                query_embedding = await self._encode_query(query)
            else:
                # 如果没有embedding服务，返回空结果
                logger.warning("未配置embedding服务，无法进行向量搜索")
                return []

            # 2. 构建document_ids过滤条件
            document_ids = None
            if filters and "document_ids" in filters:
                document_ids = filters["document_ids"]

            # 3. 调用Milvus搜索
            search_results = await self.milvus.search(
                query_embedding=query_embedding,
                limit=top_k,
                document_ids=document_ids,
                score_threshold=0.0  # 不过滤，由Reranker处理
            )

            # 4. 转换结果格式
            results = []
            for r in search_results:
                results.append({
                    "doc_id": str(r.get("chunk_id", r.get("document_id"))),
                    "content": r["content"],
                    "score": float(r["score"]),
                    "metadata": r.get("metadata", {}),
                    "source": source
                })

            logger.debug(f"Milvus搜索返回{len(results)}个结果")
            return results

        except Exception as e:
            logger.error(f"Milvus向量搜索失败: {e}")
            return []

    async def _encode_query(self, query: str) -> List[float]:
        """
        将查询文本编码为向量

        Args:
            query: 查询文本

        Returns:
            向量（list of float）
        """
        try:
            # 尝试使用统一的embedding服务
            from app.services.embeddings.unified_embedding_service import get_unified_llm_service_initialized

            embedding_service = await get_unified_llm_service_initialized()
            result = await embedding_service.embed(query)

            # 转换为list
            if hasattr(result, 'tolist'):
                return result.tolist()
            elif isinstance(result, list):
                return result
            else:
                # 如果是numpy array
                import numpy as np
                return np.array(result).tolist()

        except Exception as e:
            logger.error(f"查询编码失败: {e}")
            raise

# ============================================================================
# BM25 存储适配器
# ============================================================================

class BM25StoreAdapter(ABC):
    """BM25存储适配器接口"""

    @abstractmethod
    async def search_multi(
        self,
        queries: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        多查询BM25搜索

        Args:
            queries: 查询列表
            top_k: 返回数量

        Returns:
            [{"doc_id", "content", "score", "metadata"}, ...]
        """
        pass

    @abstractmethod
    async def add_documents(self, documents: List[str]):
        """添加文档到索引"""
        pass

class SimpleBM25Store(BM25StoreAdapter):
    """
    简单的BM25存储实现

    使用rank-bm25库实现中文BM25检索
    """

    def __init__(
        self,
        index_path: str = "data/bm25_index.pkl",
        use_cache: bool = True
    ):
        """
        初始化BM25存储

        Args:
            index_path: 索引文件路径
            use_cache: 是否使用缓存
        """
        self.index_path = Path(index_path)
        self.use_cache = use_cache

        self.bm25 = None
        self.documents = []
        self.doc_metadata = []

        # 尝试加载已有索引
        if self.use_cache and self.index_path.exists():
            self._load_index()
        else:
            logger.info("BM25索引不存在，需要先创建索引")

        logger.info(f"SimpleBM25Store初始化: docs={len(self.documents)}, path={index_path}")

    async def search_multi(
        self,
        queries: List[str],
        top_k: int,
        source: str = "bm25"
    ) -> List[Dict[str, Any]]:
        """
        多查询BM25搜索

        Args:
            queries: 查询列表
            top_k: 返回数量
            source: 来源标识

        Returns:
            搜索结果列表
        """
        if self.bm25 is None:
            logger.warning("BM25索引未初始化")
            return []

        try:
            from rank_bm25 import BM25Okapi
            import jieba

            # 多查询合并（取最高分）
            merged_scores = {}

            for query in queries:
                # 分词
                tokens = list(jieba.cut(query))

                # 计算分数
                scores = self.bm25.get_scores(tokens)

                # 合并分数
                for doc_id, score in enumerate(scores):
                    merged_scores[doc_id] = max(
                        merged_scores.get(doc_id, -float('inf')),
                        score
                    )

            # 排序并转换为结果格式
            sorted_docs = sorted(
                merged_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            results = []
            for doc_id, score in sorted_docs:
                if doc_id < len(self.documents):
                    results.append({
                        "doc_id": str(doc_id),
                        "content": self.documents[doc_id],
                        "score": float(score),
                        "metadata": self.doc_metadata[doc_id] if doc_id < len(self.doc_metadata) else {},
                        "source": source
                    })

            logger.debug(f"BM25搜索返回{len(results)}个结果")
            return results

        except Exception as e:
            logger.error(f"BM25搜索失败: {e}")
            return []

    async def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """
        添加文档到索引

        Args:
            documents: 文档列表
            metadata: 元数据列表
        """
        try:
            from rank_bm25 import BM25Okapi
            import jieba

            # 分词
            tokenized_corpus = [
                list(jieba.cut(doc))
                for doc in documents
            ]

            # 创建BM25索引
            self.bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25)
            self.documents = documents
            self.doc_metadata = metadata or [{} for _ in documents]

            # 保存索引
            self._save_index()

            logger.info(f"BM25索引创建完成: docs={len(documents)}")

        except Exception as e:
            logger.error(f"BM25索引创建失败: {e}")
            raise

    def _save_index(self):
        """保存索引到文件"""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents,
                    'metadata': self.doc_metadata
                }, f)

            logger.info(f"BM25索引已保存: {self.index_path}")

        except Exception as e:
            logger.error(f"BM25索引保存失败: {e}")

    def _load_index(self):
        """从文件加载索引"""
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)

            self.bm25 = data['bm25']
            self.documents = data['documents']
            self.doc_metadata = data.get('metadata', [])

            logger.info(f"BM25索引已加载: docs={len(self.documents)}")

        except Exception as e:
            logger.error(f"BM25索引加载失败: {e}")
            self.bm25 = None
            self.documents = []
            self.doc_metadata = []

    def clear(self):
        """清空索引"""
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []

        if self.index_path.exists():
            self.index_path.unlink()

        logger.info("BM25索引已清空")

# ============================================================================
# 便捷函数
# ============================================================================

def get_milvus_adapter(milvus_service, embedding_service=None) -> MilvusVectorStoreAdapter:
    """
    获取Milvus适配器

    Args:
        milvus_service: MilvusService实例
        embedding_service: 向量编码服务（可选）

    Returns:
        MilvusVectorStoreAdapter
    """
    return MilvusVectorStoreAdapter(milvus_service, embedding_service)

def get_bm25_adapter(
    index_path: str = "data/bm25_index.pkl",
    use_cache: bool = True
) -> SimpleBM25Store:
    """
    获取BM25适配器

    Args:
        index_path: 索引文件路径
        use_cache: 是否使用缓存

    Returns:
        SimpleBM25Store
    """
    return SimpleBM25Store(index_path, use_cache)
