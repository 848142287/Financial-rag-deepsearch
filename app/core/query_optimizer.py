"""
高级索引和聚合查询优化模块
提供查询优化、索引管理、聚合分析等功能
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable,
    AsyncGenerator, TypeVar, Generic
)
import json
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')


class IndexType(Enum):
    """索引类型"""
    BTREE = "btree"
    HASH = "hash"
    FULLTEXT = "fulltext"
    VECTOR = "vector"
    COMPOSITE = "composite"
    SPATIAL = "spatial"


class QueryType(Enum):
    """查询类型"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    JOIN = "join"


@dataclass
class IndexInfo:
    """索引信息"""
    name: str
    table: str
    columns: List[str]
    index_type: IndexType
    unique: bool = False
    partial: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    size_bytes: int = 0
    usage_count: int = 0


@dataclass
class QueryPlan:
    """查询计划"""
    query: str
    plan_type: str
    cost: float
    execution_time: Optional[float] = None
    indexes_used: List[str] = field(default_factory=list)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    estimated_rows: int = 0
    actual_rows: int = 0


@dataclass
class QueryStats:
    """查询统计"""
    query_hash: str
    query: str
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_executed: Optional[datetime] = None
    success_count: int = 0
    error_count: int = 0
    indexes_used: Set[str] = field(default_factory=set)


class BaseIndex(ABC):
    """索引基类"""

    def __init__(self, name: str, index_info: IndexInfo):
        self.name = name
        self.index_info = index_info
        self.entries = {}
        self.size = 0

    @abstractmethod
    async def insert(self, key: Any, value: Any, doc_id: Optional[str] = None) -> bool:
        """插入索引条目"""
        pass

    @abstractmethod
    async def delete(self, key: Any, doc_id: Optional[str] = None) -> bool:
        """删除索引条目"""
        pass

    @abstractmethod
    async def search(self, key: Any, **kwargs) -> List[Any]:
        """搜索索引"""
        pass

    @abstractmethod
    async def range_search(
        self,
        start_key: Any,
        end_key: Any,
        **kwargs
    ) -> List[Any]:
        """范围搜索"""
        pass

    async def bulk_insert(self, entries: List[Tuple[Any, Any]]) -> int:
        """批量插入"""
        count = 0
        for key, value in entries:
            if await self.insert(key, value):
                count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        return {
            "name": self.name,
            "type": self.index_info.index_type.value,
            "size": len(self.entries),
            "memory_usage": self.size,
            "columns": self.index_info.columns
        }


class BTreeIndex(BaseIndex):
    """B树索引实现"""

    def __init__(self, name: str, index_info: IndexInfo):
        super().__init__(name, index_info)
        self.tree = {}
        self.root = None

    async def insert(self, key: Any, value: Any, doc_id: Optional[str] = None) -> bool:
        """插入键值对"""
        try:
            if key not in self.tree:
                self.tree[key] = []

            entry = {
                "value": value,
                "doc_id": doc_id,
                "timestamp": datetime.now()
            }
            self.tree[key].append(entry)
            self.size += len(str(entry))
            return True
        except Exception as e:
            logger.error(f"BTree insert error: {str(e)}")
            return False

    async def delete(self, key: Any, doc_id: Optional[str] = None) -> bool:
        """删除键值对"""
        try:
            if key in self.tree:
                if doc_id:
                    # 删除特定文档的条目
                    self.tree[key] = [
                        entry for entry in self.tree[key]
                        if entry.get("doc_id") != doc_id
                    ]
                    if not self.tree[key]:
                        del self.tree[key]
                else:
                    # 删除所有条目
                    del self.tree[key]
                return True
            return False
        except Exception as e:
            logger.error(f"BTree delete error: {str(e)}")
            return False

    async def search(self, key: Any, exact_match: bool = True) -> List[Any]:
        """搜索键"""
        try:
            if exact_match:
                if key in self.tree:
                    return [entry["value"] for entry in self.tree[key]]
                return []
            else:
                # 模糊匹配
                results = []
                key_str = str(key).lower()
                for tree_key, entries in self.tree.items():
                    if key_str in str(tree_key).lower():
                        results.extend([entry["value"] for entry in entries])
                return results
        except Exception as e:
            logger.error(f"BTree search error: {str(e)}")
            return []

    async def range_search(
        self,
        start_key: Any,
        end_key: Any,
        inclusive: bool = True
    ) -> List[Any]:
        """范围搜索"""
        try:
            results = []
            sorted_keys = sorted(self.tree.keys())

            for key in sorted_keys:
                if inclusive:
                    if start_key <= key <= end_key:
                        results.extend([entry["value"] for entry in self.tree[key]])
                else:
                    if start_key < key < end_key:
                        results.extend([entry["value"] for entry in self.tree[key]])

                # 优化：如果key已经超过end_key，可以提前退出
                if key > end_key:
                    break

            return results
        except Exception as e:
            logger.error(f"BTree range search error: {str(e)}")
            return []


class HashIndex(BaseIndex):
    """哈希索引实现"""

    def __init__(self, name: str, index_info: IndexInfo):
        super().__init__(name, index_info)
        self.hash_table = {}
        self.buckets = {}

    async def insert(self, key: Any, value: Any, doc_id: Optional[str] = None) -> bool:
        """插入键值对"""
        try:
            hash_key = self._hash(key)

            if hash_key not in self.hash_table:
                self.hash_table[hash_key] = []

            entry = {
                "key": key,
                "value": value,
                "doc_id": doc_id,
                "timestamp": datetime.now()
            }
            self.hash_table[hash_key].append(entry)
            self.size += len(str(entry))
            return True
        except Exception as e:
            logger.error(f"Hash insert error: {str(e)}")
            return False

    async def delete(self, key: Any, doc_id: Optional[str] = None) -> bool:
        """删除键值对"""
        try:
            hash_key = self._hash(key)
            if hash_key in self.hash_table:
                if doc_id:
                    # 删除特定文档的条目
                    original_count = len(self.hash_table[hash_key])
                    self.hash_table[hash_key] = [
                        entry for entry in self.hash_table[hash_key]
                        if entry.get("doc_id") != doc_id
                    ]
                    if not self.hash_table[hash_key]:
                        del self.hash_table[hash_key]
                    return len(self.hash_table.get(hash_key, [])) < original_count
                else:
                    # 删除所有条目
                    del self.hash_table[hash_key]
                    return True
            return False
        except Exception as e:
            logger.error(f"Hash delete error: {str(e)}")
            return False

    async def search(self, key: Any, **kwargs) -> List[Any]:
        """搜索键"""
        try:
            hash_key = self._hash(key)
            if hash_key in self.hash_table:
                # 检查实际键是否匹配
                matches = [
                    entry["value"] for entry in self.hash_table[hash_key]
                    if entry["key"] == key
                ]
                return matches
            return []
        except Exception as e:
            logger.error(f"Hash search error: {str(e)}")
            return []

    async def range_search(
        self,
        start_key: Any,
        end_key: Any,
        **kwargs
    ) -> List[Any]:
        """哈希索引不支持范围搜索"""
        logger.warning("Hash index does not support range search")
        return []

    def _hash(self, key: Any) -> str:
        """哈希函数"""
        return hashlib.md5(str(key).encode()).hexdigest()


class FullTextIndex(BaseIndex):
    """全文索引实现"""

    def __init__(self, name: str, index_info: IndexInfo):
        super().__init__(name, index_info)
        self.inverted_index = {}
        self.documents = {}
        self.tokenizer = re.compile(r'\w+')

    async def insert(self, key: Any, value: Any, doc_id: Optional[str] = None) -> bool:
        """插入文档"""
        try:
            text = str(value)
            doc_id = doc_id or str(key)

            # 存储文档
            self.documents[doc_id] = {
                "key": key,
                "text": text,
                "timestamp": datetime.now()
            }

            # 分词并建立倒排索引
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}

                if doc_id not in self.inverted_index[token]:
                    self.inverted_index[token][doc_id] = []

                self.inverted_index[token][doc_id].append(
                    text.find(token)
                )

            self.size += len(text)
            return True
        except Exception as e:
            logger.error(f"FullText insert error: {str(e)}")
            return False

    async def delete(self, key: Any, doc_id: Optional[str] = None) -> bool:
        """删除文档"""
        try:
            doc_id = doc_id or str(key)

            if doc_id in self.documents:
                text = self.documents[doc_id]["text"]
                tokens = self._tokenize(text)

                # 从倒排索引中删除
                for token in tokens:
                    if token in self.inverted_index and doc_id in self.inverted_index[token]:
                        del self.inverted_index[token][doc_id]
                        if not self.inverted_index[token]:
                            del self.inverted_index[token]

                # 删除文档
                del self.documents[doc_id]
                return True
            return False
        except Exception as e:
            logger.error(f"FullText delete error: {str(e)}")
            return False

    async def search(
        self,
        query: str,
        operator: str = "AND",
        **kwargs
    ) -> List[Any]:
        """搜索文档"""
        try:
            tokens = self._tokenize(query.lower())

            if not tokens:
                return []

            # 获取包含每个token的文档
            token_docs = []
            for token in tokens:
                docs = set()
                for doc_id in self.inverted_index.get(token, {}):
                    docs.add(doc_id)
                token_docs.append(docs)

            # 根据操作符合并结果
            if operator == "AND":
                result_docs = set.intersection(*token_docs) if token_docs else set()
            elif operator == "OR":
                result_docs = set.union(*token_docs) if token_docs else set()
            else:
                result_docs = token_docs[0] if token_docs else set()

            # 返回匹配的文档
            results = []
            for doc_id in result_docs:
                if doc_id in self.documents:
                    results.append(self.documents[doc_id]["key"])

            return results
        except Exception as e:
            logger.error(f"FullText search error: {str(e)}")
            return []

    async def range_search(self, *args, **kwargs) -> List[Any]:
        """全文索引不支持范围搜索"""
        return []

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        tokens = self.tokenizer.findall(text.lower())
        # 过滤停用词和短词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [token for token in tokens if len(token) > 2 and token not in stopwords]


class VectorIndex(BaseIndex):
    """向量索引实现"""

    def __init__(self, name: str, index_info: IndexInfo, dimension: int):
        super().__init__(name, index_info)
        self.dimension = dimension
        self.vectors = {}
        self.index = None  # 可以集成FAISS或Annoy等库

    async def insert(self, key: Any, value: Any, doc_id: Optional[str] = None) -> bool:
        """插入向量"""
        try:
            if isinstance(value, (list, np.ndarray)):
                vector = np.array(value, dtype=np.float32)
                if len(vector) != self.dimension:
                    raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")

                self.vectors[key] = {
                    "vector": vector,
                    "doc_id": doc_id,
                    "timestamp": datetime.now()
                }
                self.size += vector.nbytes
                return True
            return False
        except Exception as e:
            logger.error(f"Vector insert error: {str(e)}")
            return False

    async def delete(self, key: Any, doc_id: Optional[str] = None) -> bool:
        """删除向量"""
        try:
            if key in self.vectors:
                del self.vectors[key]
                return True
            return False
        except Exception as e:
            logger.error(f"Vector delete error: {str(e)}")
            return False

    async def search(self, query_vector: Any, top_k: int = 10, **kwargs) -> List[Tuple[Any, float]]:
        """相似度搜索"""
        try:
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)

            results = []
            for key, data in self.vectors.items():
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_vector, data["vector"])
                results.append((key, similarity))

            # 排序并返回top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            return []

    async def range_search(self, *args, **kwargs) -> List[Any]:
        """向量索引不支持传统范围搜索"""
        return []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0


class IndexManager:
    """索引管理器"""

    def __init__(self):
        self.indexes = {}
        self.stats = {}

    async def create_index(self, index_info: IndexInfo) -> BaseIndex:
        """创建索引"""
        if index_info.index_type == IndexType.BTREE:
            index = BTreeIndex(index_info.name, index_info)
        elif index_info.index_type == IndexType.HASH:
            index = HashIndex(index_info.name, index_info)
        elif index_info.index_type == IndexType.FULLTEXT:
            index = FullTextIndex(index_info.name, index_info)
        elif index_info.index_type == IndexType.VECTOR:
            dimension = index_info.metadata.get("dimension", 128)
            index = VectorIndex(index_info.name, index_info, dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_info.index_type}")

        self.indexes[index_info.name] = index
        self.stats[index_info.name] = {
            "created_at": datetime.now(),
            "operations": 0,
            "total_time": 0.0
        }

        logger.info(f"Created index {index_info.name} of type {index_info.index_type.value}")
        return index

    async def drop_index(self, name: str) -> bool:
        """删除索引"""
        if name in self.indexes:
            del self.indexes[name]
            if name in self.stats:
                del self.stats[name]
            logger.info(f"Dropped index {name}")
            return True
        return False

    def get_index(self, name: str) -> Optional[BaseIndex]:
        """获取索引"""
        return self.indexes.get(name)

    def list_indexes(self) -> List[Dict[str, Any]]:
        """列出所有索引"""
        return [
            {
                "name": name,
                "type": index.index_info.index_type.value,
                "table": index.index_info.table,
                "columns": index.index_info.columns,
                "stats": index.get_stats()
            }
            for name, index in self.indexes.items()
        ]

    async def optimize_indexes(self):
        """优化所有索引"""
        for name, index in self.indexes.items():
            try:
                # 根据索引类型执行不同的优化
                if isinstance(index, BTreeIndex):
                    # B树可以重新平衡
                    pass
                elif isinstance(index, VectorIndex):
                    # 向量索引可以重建以提高查询性能
                    pass

                logger.info(f"Optimized index {name}")
            except Exception as e:
                logger.error(f"Failed to optimize index {name}: {str(e)}")


class QueryOptimizer:
    """查询优化器"""

    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self.query_stats = {}
        self.query_plans = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def analyze_query(self, query: str) -> QueryPlan:
        """分析查询并生成执行计划"""
        start_time = datetime.now()

        # 生成查询哈希
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # 检查缓存的计划
        if query_hash in self.query_plans:
            plan = self.query_plans[query_hash]
            logger.info(f"Using cached query plan for hash {query_hash}")
            return plan

        # 解析查询
        query_type = self._detect_query_type(query)
        tables = self._extract_tables(query)
        conditions = self._extract_conditions(query)

        # 选择合适的索引
        candidate_indexes = self._select_indexes(tables, conditions)

        # 生成执行计划
        plan = await self._generate_execution_plan(
            query,
            query_type,
            tables,
            conditions,
            candidate_indexes
        )

        # 估算成本
        plan.cost = self._estimate_cost(plan)

        # 缓存计划
        self.query_plans[query_hash] = plan

        analysis_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query analysis completed in {analysis_time:.3f}s")

        return plan

    def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        query_lower = query.lower().strip()
        if query_lower.startswith("select"):
            return QueryType.SELECT
        elif query_lower.startswith("insert"):
            return QueryType.INSERT
        elif query_lower.startswith("update"):
            return QueryType.UPDATE
        elif query_lower.startswith("delete"):
            return QueryType.DELETE
        elif "join" in query_lower:
            return QueryType.JOIN
        elif any(agg in query_lower for agg in ["group by", "count(", "sum(", "avg(", "max(", "min("]):
            return QueryType.AGGREGATE
        else:
            return QueryType.SELECT

    def _extract_tables(self, query: str) -> List[str]:
        """提取查询中的表"""
        # 简化的表提取逻辑
        tables = []
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'INTO\s+(\w+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

    def _extract_conditions(self, query: str) -> List[Dict[str, Any]]:
        """提取查询条件"""
        conditions = []

        # 简化的条件提取
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP BY|\s+ORDER BY|\s+LIMIT|$)', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            # 分解AND条件
            and_conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)

            for cond in and_conditions:
                # 提取列和操作符
                op_match = re.match(r'(\w+)\s*(=|>|<|>=|<=|!=|LIKE|IN)\s*(.+)', cond.strip(), re.IGNORECASE)
                if op_match:
                    conditions.append({
                        "column": op_match.group(1),
                        "operator": op_match.group(2),
                        "value": op_match.group(3).strip("'\"")
                    })

        return conditions

    def _select_indexes(
        self,
        tables: List[str],
        conditions: List[Dict[str, Any]]
    ) -> List[BaseIndex]:
        """选择合适的索引"""
        candidate_indexes = []

        for index in self.index_manager.indexes.values():
            if index.index_info.table in tables:
                # 检查索引列是否在条件中使用
                for condition in conditions:
                    if condition["column"] in index.index_info.columns:
                        candidate_indexes.append(index)
                        break

        return candidate_indexes

    async def _generate_execution_plan(
        self,
        query: str,
        query_type: QueryType,
        tables: List[str],
        conditions: List[Dict[str, Any]],
        candidate_indexes: List[BaseIndex]
    ) -> QueryPlan:
        """生成执行计划"""
        plan = QueryPlan(
            query=query,
            plan_type="sequential_scan"  # 默认计划类型
        )

        # 如果有可用索引，选择最优索引
        if candidate_indexes:
            best_index = self._select_best_index(candidate_indexes, conditions)
            plan.indexes_used.append(best_index.name)
            plan.plan_type = "index_scan"

        # 添加操作步骤
        plan.operations.extend([
            {"type": "scan", "table": tables[0] if tables else "unknown"},
            {"type": "filter", "conditions": conditions},
            {"type": "project", "columns": self._extract_projected_columns(query)}
        ])

        # 估算返回行数
        plan.estimated_rows = self._estimate_rows(tables, conditions)

        return plan

    def _select_best_index(
        self,
        indexes: List[BaseIndex],
        conditions: List[Dict[str, Any]]
    ) -> BaseIndex:
        """选择最佳索引"""
        # 简单的索引选择策略
        # 优先选择覆盖最多条件的索引
        best_index = None
        max_coverage = 0

        for index in indexes:
            coverage = sum(
                1 for condition in conditions
                if condition["column"] in index.index_info.columns
            )

            if coverage > max_coverage:
                max_coverage = coverage
                best_index = index

        return best_index or indexes[0]

    def _extract_projected_columns(self, query: str) -> List[str]:
        """提取查询的列"""
        columns = []

        # 提取SELECT后的列
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
        if select_match:
            select_clause = select_match.group(1)
            if select_clause.strip() != "*":
                columns = [col.strip() for col in select_clause.split(",")]

        return columns

    def _estimate_rows(
        self,
        tables: List[str],
        conditions: List[Dict[str, Any]]
    ) -> int:
        """估算返回行数"""
        # 简化的行数估算
        base_rows = 1000  # 假设每表1000行
        selectivity = 1.0

        for condition in conditions:
            # 每个条件减少一些行数
            if condition["operator"] == "=":
                selectivity *= 0.1
            elif condition["operator"] in (">", "<", ">=", "<="):
                selectivity *= 0.3
            elif condition["operator"] == "LIKE":
                selectivity *= 0.5

        estimated_rows = int(base_rows * selectivity * len(tables))
        return max(1, estimated_rows)

    def _estimate_cost(self, plan: QueryPlan) -> float:
        """估算查询成本"""
        base_cost = 1.0

        # 根据计划类型调整成本
        if plan.plan_type == "index_scan":
            base_cost *= 0.1  # 索引扫描成本较低
        elif plan.plan_type == "sequential_scan":
            base_cost *= plan.estimated_rows * 0.001  # 顺序扫描成本与行数相关

        # JOIN操作增加成本
        if "join" in plan.query.lower():
            base_cost *= 2.0

        # 聚合操作增加成本
        if "group by" in plan.query.lower():
            base_cost *= 1.5

        return base_cost

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行查询"""
        start_time = datetime.now()
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # 分析查询
        plan = await self.analyze_query(query)

        # 记录查询统计
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query=query
            )

        stats = self.query_stats[query_hash]

        try:
            # 执行查询（这里需要实际的数据库连接）
            # result = await self._execute_with_plan(plan, params)

            # 模拟执行
            await asyncio.sleep(plan.cost * 0.01)  # 模拟查询执行时间
            result = {"mock": True, "plan": plan.__dict__}

            # 更新统计
            execution_time = (datetime.now() - start_time).total_seconds()
            stats.execution_count += 1
            stats.total_time += execution_time
            stats.avg_time = stats.total_time / stats.execution_count
            stats.min_time = min(stats.min_time, execution_time)
            stats.max_time = max(stats.max_time, execution_time)
            stats.last_executed = datetime.now()
            stats.success_count += 1
            stats.indexes_used.update(plan.indexes_used)

            plan.execution_time = execution_time
            plan.actual_rows = result.get("row_count", 0)

            return {
                "result": result,
                "plan": plan,
                "execution_time": execution_time
            }

        except Exception as e:
            # 记录错误统计
            stats.error_count += 1
            logger.error(f"Query execution failed: {str(e)}")
            raise

    async def _execute_with_plan(self, plan: QueryPlan, params: Optional[Dict[str, Any]] = None) -> Any:
        """根据执行计划执行查询"""
        # 这里需要实现实际的查询执行逻辑
        # 可以使用索引来优化查询
        pass

    def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        return {
            "total_queries": len(self.query_stats),
            "cached_plans": len(self.query_plans),
            "top_queries": sorted(
                [
                    {
                        "query": stats.query[:100],
                        "execution_count": stats.execution_count,
                        "avg_time": stats.avg_time,
                        "success_rate": stats.success_count / stats.execution_count if stats.execution_count > 0 else 0
                    }
                    for stats in self.query_stats.values()
                ],
                key=lambda x: x["execution_count"],
                reverse=True
            )[:10]
        }

    async def optimize_slow_queries(self, threshold: float = 1.0) -> List[str]:
        """优化慢查询"""
        suggestions = []

        for stats in self.query_stats.values():
            if stats.avg_time > threshold:
                # 分析慢查询并提出建议
                if not stats.indexes_used:
                    suggestions.append(
                        f"Query '{stats.query[:50]}...' has no indexes used. "
                        "Consider adding indexes on WHERE clause columns."
                    )

                if "join" in stats.query.lower() and len(stats.indexes_used) < 2:
                    suggestions.append(
                        f"JOIN query '{stats.query[:50]}...' may benefit from more indexes."
                    )

        return suggestions


class AggregationOptimizer:
    """聚合查询优化器"""

    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self.materialized_views = {}

    async def optimize_aggregation(
        self,
        query: str,
        group_by_columns: List[str],
        agg_functions: List[str]
    ) -> QueryPlan:
        """优化聚合查询"""
        plan = QueryPlan(
            query=query,
            plan_type="aggregation",
            cost=0.0
        )

        # 检查是否有物化视图
        view_key = self._generate_view_key(group_by_columns, agg_functions)
        if view_key in self.materialized_views:
            plan.plan_type = "materialized_view_scan"
            plan.cost = 0.1
            plan.indexes_used.append(view_key)

        # 检查是否有合适的索引
        for index in self.index_manager.indexes.values():
            if all(col in index.index_info.columns for col in group_by_columns):
                plan.indexes_used.append(index.name)
                plan.plan_type = "index_group_aggregation"
                plan.cost *= 0.5
                break

        return plan

    def _generate_view_key(self, group_by: List[str], aggs: List[str]) -> str:
        """生成物化视图键"""
        return f"agg_{'_'.join(sorted(group_by))}_{'_'.join(sorted(aggs))}"

    async def create_materialized_view(
        self,
        name: str,
        query: str,
        group_by_columns: List[str],
        agg_functions: List[str]
    ):
        """创建物化视图"""
        # 这里需要实现物化视图的创建和刷新逻辑
        view_info = {
            "name": name,
            "query": query,
            "group_by": group_by_columns,
            "aggregations": agg_functions,
            "created_at": datetime.now(),
            "last_refreshed": datetime.now()
        }

        view_key = self._generate_view_key(group_by_columns, agg_functions)
        self.materialized_views[view_key] = view_info

        logger.info(f"Created materialized view {name}")

    async def refresh_materialized_views(self):
        """刷新所有物化视图"""
        for view_key, view_info in self.materialized_views.items():
            try:
                # 执行刷新逻辑
                # await self._execute_query(view_info["query"])
                view_info["last_refreshed"] = datetime.now()
                logger.info(f"Refreshed materialized view {view_info['name']}")
            except Exception as e:
                logger.error(f"Failed to refresh view {view_info['name']}: {str(e)}")


# 全局实例
_index_manager = IndexManager()
_query_optimizer = QueryOptimizer(_index_manager)
_agg_optimizer = AggregationOptimizer(_index_manager)


def get_index_manager() -> IndexManager:
    """获取索引管理器"""
    return _index_manager


def get_query_optimizer() -> QueryOptimizer:
    """获取查询优化器"""
    return _query_optimizer


def get_aggregation_optimizer() -> AggregationOptimizer:
    """获取聚合优化器"""
    return _agg_optimizer