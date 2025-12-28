"""
增强的混合检索模块
支持稠密向量 + 稀疏向量混合检索，使用RRF融合算法

不需要新的大模型，仅增强检索算法
"""

import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json

logger = logging.getLogger(__name__)

try:
    from pymilvus import (
        connections, Collection, utility, AnnSearchRequest, RRFRanker
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("PyMilvus未安装，部分功能将不可用")


class SearchMode(Enum):
    """检索模式"""
    HYBRID = "hybrid"      # 混合检索（稠密+稀疏）
    DENSE = "dense"        # 仅稠密向量检索
    SPARSE = "sparse"      # 仅稀疏向量检索（BM25）


class HybridSearchConfig:
    """混合检索配置"""

    def __init__(
        self,
        mode: str = "hybrid",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        top_k: int = 10,
        enable_rerank: bool = True
    ):
        """
        参数:
            mode: 检索模式 (hybrid/dense/sparse)
            dense_weight: 稠密向量权重
            sparse_weight: 稀疏向量权重
            rrf_k: RRF融合参数（倒数排名融合）
            top_k: 返回结果数量
            enable_rerank: 是否启用重排序
        """
        self.mode = mode
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.enable_rerank = enable_rerank


class EnhancedHybridSearch:
    """增强的混合检索引擎"""

    def __init__(self, collection: Collection):
        """
        参数:
            collection: Milvus集合对象
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("需要安装PyMilvus: pip install pymilvus")

        self.collection = collection
        self.logger = logging.getLogger(self.__class__.__name__)

    async def search(
        self,
        query_vector: List[float],
        query_text: str,
        embed_function,
        config: Optional[HybridSearchConfig] = None,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索

        参数:
            query_vector: 查询的稠密向量
            query_text: 查询文本（用于稀疏检索/BM25）
            embed_function: 向量嵌入函数
            config: 检索配置
            filter_expr: 过滤表达式
            output_fields: 输出字段列表

        返回:
            检索结果列表，每个结果包含id、score、content等字段
        """
        if config is None:
            config = HybridSearchConfig()

        if output_fields is None:
            output_fields = ["id", "text", "content", "metadata", "file_code", "file_name"]

        try:
            # 确保集合已加载
            self._ensure_collection_loaded()

            # 准备搜索请求列表
            search_requests = []

            # 根据模式准备搜索请求
            if config.mode in [SearchMode.HYBRID.value, SearchMode.DENSE.value]:
                # 准备稠密向量搜索请求
                dense_search_params = {
                    "metric_type": "IP",  # 内积相似度
                    "params": {"ef": 10}
                }

                dense_req = AnnSearchRequest(
                    data=[query_vector],
                    anns_field="vector",
                    param=dense_search_params,
                    limit=config.top_k,
                    expr=filter_expr
                )
                search_requests.append(dense_req)
                self.logger.debug("已添加稠密向量搜索请求")

            if config.mode in [SearchMode.HYBRID.value, SearchMode.SPARSE.value]:
                # 检查集合是否有sparse_vector字段
                if self._has_sparse_vector_field():
                    # 准备稀疏向量搜索请求（BM25）
                    sparse_search_params = {
                        "metric_type": "BM25",
                        "params": {
                            "drop_ratio_search": 0.1,
                            "bm25_k1": 1.2,
                            "bm25_b": 0.75
                        }
                    }

                    sparse_req = AnnSearchRequest(
                        data=[query_text],  # 使用原始查询文本
                        anns_field="sparse_vector",
                        param=sparse_search_params,
                        limit=config.top_k,
                        expr=filter_expr
                    )
                    search_requests.append(sparse_req)
                    self.logger.debug("已添加稀疏向量搜索请求")

            # 执行搜索
            if not search_requests:
                self.logger.error("没有有效的搜索请求")
                return []

            results = await self._execute_search(
                search_requests=search_requests,
                config=config,
                output_fields=output_fields
            )

            self.logger.info(f"检索完成，找到 {len(results)} 个结果")
            return results

        except Exception as e:
            self.logger.error(f"混合检索失败: {e}", exc_info=True)
            return []

    async def _execute_search(
        self,
        search_requests: List[AnnSearchRequest],
        config: HybridSearchConfig,
        output_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """执行搜索并处理结果"""

        try:
            # 判断使用哪种搜索方式
            if config.mode == SearchMode.HYBRID.value and len(search_requests) > 1:
                # 混合检索：使用hybrid_search和RRF融合器
                ranker = RRFRanker(k=config.rrf_k)

                search_results = self.collection.hybrid_search(
                    reqs=search_requests,
                    rerank=ranker,
                    limit=config.top_k,
                    output_fields=output_fields
                )

                # hybrid_search返回列表的列表，取第一个
                if search_results and len(search_results) > 0:
                    search_results = search_results[0]
                else:
                    return []

            else:
                # 单一类型检索：使用普通的search方法
                # 取第一个（也是唯一的）搜索请求
                req = search_requests[0]

                # 根据请求类型判断字段名
                if req.anns_field == "vector":
                    data = req.data  # 已经是向量列表
                else:
                    # sparse_vector使用原始文本
                    data = req.data

                search_results = self.collection.search(
                    data=data,
                    anns_field=req.anns_field,
                    param=req.param,
                    limit=config.top_k,
                    expr=req.expr,
                    output_fields=output_fields
                )

                # search返回列表的列表，取第一个
                if search_results and len(search_results) > 0:
                    search_results = search_results[0]
                else:
                    return []

            # 转换结果格式
            formatted_results = []
            for hit in search_results:
                result_dict = {
                    "id": hit.id,
                    "score": hit.score,
                    "distance": getattr(hit, 'distance', None)
                }

                # 提取所有请求的字段
                if hit.entity:
                    for field_name in output_fields:
                        result_dict[field_name] = hit.entity.get(field_name)

                    # 解析metadata字段
                    if "metadata" in result_dict:
                        metadata = result_dict["metadata"]
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                        result_dict["metadata"] = metadata
                else:
                    # 如果没有entity，填充默认值
                    for field_name in output_fields:
                        result_dict[field_name] = None

                formatted_results.append(result_dict)

            # 按分数排序（降序）
            formatted_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            return formatted_results

        except Exception as e:
            self.logger.error(f"执行搜索失败: {e}", exc_info=True)
            return []

    def _ensure_collection_loaded(self):
        """确保集合已加载到内存"""
        try:
            if not self.collection.is_loaded:
                self.collection.load()
                self.logger.debug(f"集合 {self.collection.name} 已加载")
        except Exception as e:
            self.logger.warning(f"检查集合加载状态失败: {e}")

    def _has_sparse_vector_field(self) -> bool:
        """检查集合是否有sparse_vector字段"""
        try:
            schema = self.collection.schema
            for field in schema.fields:
                if field.name == "sparse_vector":
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"检查sparse_vector字段失败: {e}")
            return False

    async def batch_search(
        self,
        queries: List[Dict[str, Any]],
        embed_function,
        config: Optional[HybridSearchConfig] = None,
        filter_expr: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        批量检索

        参数:
            queries: 查询列表，每个查询包含query_vector和query_text
            embed_function: 向量嵌入函数
            config: 检索配置
            filter_expr: 过滤表达式

        返回:
            检索结果列表的列表
        """
        if config is None:
            config = HybridSearchConfig()

        results = []

        for query in queries:
            query_vector = query.get("query_vector")
            query_text = query.get("query_text", "")

            if not query_vector:
                results.append([])
                continue

            result = await self.search(
                query_vector=query_vector,
                query_text=query_text,
                embed_function=embed_function,
                config=config,
                filter_expr=filter_expr
            )
            results.append(result)

        return results


def create_hybrid_search_config(
    mode: str = "hybrid",
    **kwargs
) -> HybridSearchConfig:
    """
    创建混合检索配置的辅助函数

    参数:
        mode: 检索模式 (hybrid/dense/sparse)
        **kwargs: 其他配置参数

    返回:
        HybridSearchConfig对象
    """
    return HybridSearchConfig(mode=mode, **kwargs)


# 向下兼容的工具函数
def calculate_rrf_score(
    rank1: int,
    rank2: int,
    k: int = 60
) -> float:
    """
    计算RRF（Reciprocal Rank Fusion）分数

    参数:
        rank1: 在第一个结果列表中的排名
        rank2: 在第二个结果列表中的排名
        k: RRF参数

    返回:
        融合后的分数
    """
    score = 1.0 / (k + rank1) + 1.0 / (k + rank2)
    return score
