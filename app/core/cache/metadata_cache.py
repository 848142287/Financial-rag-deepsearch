"""
元数据缓存
缓存文档元数据、实体关系、知识图谱等信息
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class MetadataType(Enum):
    """元数据类型"""
    DOCUMENT = "document"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SCHEMA = "schema"
    INDEX = "index"
    TAXONOMY = "taxonomy"
    ONTOLOGY = "ontology"


@dataclass
class DocumentMetadata:
    """文档元数据"""
    id: str
    title: str
    content_hash: str
    file_type: str
    size: int
    created_at: datetime
    updated_at: datetime
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    language: str = "zh"
    page_count: int = 0
    word_count: int = 0
    key_entities: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityMetadata:
    """实体元数据"""
    id: str
    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    confidence: float = 0.0
    source_documents: Set[str] = field(default_factory=set)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    validation_status: str = "unvalidated"


@dataclass
class RelationshipMetadata:
    """关系元数据"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float = 0.0
    source_documents: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    bidirectional: bool = False
    weight: float = 1.0


@dataclass
class KnowledgeGraphMetadata:
    """知识图谱元数据"""
    graph_id: str
    name: str
    description: Optional[str] = None
    entity_count: int = 0
    relationship_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    schema_version: str = "1.0"
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data_type: MetadataType
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    version: int = 1
    dependencies: Set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl

    def update_access(self) -> None:
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class MetadataCache:
    """元数据缓存管理器"""

    def __init__(self, max_entries: int = 50000, default_ttl: int = 7200):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._type_index: Dict[MetadataType, Set[str]] = {
            meta_type: set() for meta_type in MetadataType
        }
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._reverse_dependency_graph: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'type_distribution': {meta_type.value: 0 for meta_type in MetadataType}
        }

    def _generate_key(self, data_type: MetadataType, identifier: str,
                     params: Dict[str, Any] = None) -> str:
        """生成缓存键"""
        key_parts = [data_type.value, identifier]

        if params:
            # 排序参数确保一致的键
            sorted_params = json.dumps(params, sort_keys=True)
            key_parts.append(sorted_params)

        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """获取文档元数据"""
        return await self._get(MetadataType.DOCUMENT, doc_id)

    async def set_document(self, doc_id: str, metadata: DocumentMetadata,
                          ttl: Optional[int] = None) -> bool:
        """设置文档元数据"""
        return await self._set(MetadataType.DOCUMENT, doc_id, metadata, ttl)

    async def get_entity(self, entity_id: str) -> Optional[EntityMetadata]:
        """获取实体元数据"""
        return await self._get(MetadataType.ENTITY, entity_id)

    async def set_entity(self, entity_id: str, metadata: EntityMetadata,
                        ttl: Optional[int] = None) -> bool:
        """设置实体元数据"""
        return await self._set(MetadataType.ENTITY, entity_id, metadata, ttl)

    async def get_relationship(self, relationship_id: str) -> Optional[RelationshipMetadata]:
        """获取关系元数据"""
        return await self._get(MetadataType.RELATIONSHIP, relationship_id)

    async def set_relationship(self, relationship_id: str, metadata: RelationshipMetadata,
                              ttl: Optional[int] = None) -> bool:
        """设置关系元数据"""
        return await self._set(MetadataType.RELATIONSHIP, relationship_id, metadata, ttl)

    async def get_knowledge_graph(self, graph_id: str) -> Optional[KnowledgeGraphMetadata]:
        """获取知识图谱元数据"""
        return await self._get(MetadataType.KNOWLEDGE_GRAPH, graph_id)

    async def set_knowledge_graph(self, graph_id: str, metadata: KnowledgeGraphMetadata,
                                 ttl: Optional[int] = None) -> bool:
        """设置知识图谱元数据"""
        return await self._set(MetadataType.KNOWLEDGE_GRAPH, graph_id, metadata, ttl)

    async def get_custom(self, data_type: MetadataType, key: str,
                        params: Dict[str, Any] = None) -> Optional[Any]:
        """获取自定义元数据"""
        cache_key = self._generate_key(data_type, key, params)
        entry = await self._get_entry(cache_key)
        return entry.data if entry else None

    async def set_custom(self, data_type: MetadataType, key: str, data: Any,
                        ttl: Optional[int] = None, dependencies: List[str] = None,
                        params: Dict[str, Any] = None) -> bool:
        """设置自定义元数据"""
        cache_key = self._generate_key(data_type, key, params)
        return await self._set_entry(cache_key, data_type, data, ttl, dependencies or [])

    async def _get(self, data_type: MetadataType, identifier: str) -> Optional[Any]:
        """通用获取方法"""
        cache_key = self._generate_key(data_type, identifier)
        entry = await self._get_entry(cache_key)
        return entry.data if entry else None

    async def _set(self, data_type: MetadataType, identifier: str, data: Any,
                  ttl: Optional[int] = None) -> bool:
        """通用设置方法"""
        cache_key = self._generate_key(data_type, identifier)
        return await self._set_entry(cache_key, data_type, data, ttl)

    async def _get_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        async with self._lock:
            self._stats['total_requests'] += 1

            entry = self._cache.get(cache_key)
            if entry is None:
                self._stats['cache_misses'] += 1
                return None

            if entry.is_expired():
                await self._remove_entry(cache_key)
                self._stats['cache_misses'] += 1
                return None

            entry.update_access()
            self._stats['cache_hits'] += 1

            logger.debug(f"元数据缓存命中: {cache_key}")
            return entry

    async def _set_entry(self, cache_key: str, data_type: MetadataType, data: Any,
                        ttl: Optional[int] = None, dependencies: List[str] = None) -> bool:
        """设置缓存条目"""
        try:
            async with self._lock:
                # 检查是否需要清理空间
                if len(self._cache) >= self.max_entries and cache_key not in self._cache:
                    await self._evict_lru()

                entry = CacheEntry(
                    key=cache_key,
                    data_type=data_type,
                    data=data,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl=ttl or self.default_ttl,
                    dependencies=set(dependencies or [])
                )

                self._cache[cache_key] = entry
                self._type_index[data_type].add(cache_key)

                # 更新依赖关系
                for dep in dependencies:
                    if dep not in self._dependency_graph:
                        self._dependency_graph[dep] = set()
                    self._dependency_graph[dep].add(cache_key)

                    if cache_key not in self._reverse_dependency_graph:
                        self._reverse_dependency_graph[cache_key] = set()
                    self._reverse_dependency_graph[cache_key].add(dep)

                # 更新统计
                self._stats['type_distribution'][data_type.value] = len(self._type_index[data_type])

                logger.debug(f"元数据缓存设置成功: {cache_key}")
                return True

        except Exception as e:
            logger.error(f"元数据缓存设置失败: {e}")
            return False

    async def invalidate(self, data_type: Optional[MetadataType] = None,
                        identifier: Optional[str] = None,
                        pattern: Optional[str] = None) -> int:
        """使缓存失效"""
        try:
            async with self._lock:
                invalidated_count = 0

                if data_type and identifier:
                    # 精确匹配
                    cache_key = self._generate_key(data_type, identifier)
                    if await self._remove_entry(cache_key):
                        invalidated_count += 1

                elif pattern:
                    # 模式匹配
                    keys_to_remove = []
                    for cache_key in self._cache.keys():
                        if pattern in cache_key:
                            keys_to_remove.append(cache_key)

                    for key in keys_to_remove:
                        if await self._remove_entry(key):
                            invalidated_count += 1

                elif data_type:
                    # 按类型清理
                    keys_to_remove = list(self._type_index[data_type])
                    for cache_key in keys_to_remove:
                        if await self._remove_entry(cache_key):
                            invalidated_count += 1

                logger.info(f"元数据缓存失效完成: {invalidated_count} 个条目")
                return invalidated_count

        except Exception as e:
            logger.error(f"元数据缓存失效失败: {e}")
            return 0

    async def invalidate_dependencies(self, dependency_key: str) -> int:
        """使依赖失效"""
        try:
            async with self._lock:
                invalidated_count = 0

                if dependency_key in self._dependency_graph:
                    dependent_keys = self._dependency_graph[dependency_key].copy()
                    for cache_key in dependent_keys:
                        if await self._remove_entry(cache_key):
                            invalidated_count += 1

                logger.info(f"依赖失效完成: {dependency_key}, 影响条目: {invalidated_count}")
                return invalidated_count

        except Exception as e:
            logger.error(f"依赖失效失败: {e}")
            return 0

    async def _remove_entry(self, cache_key: str) -> bool:
        """移除缓存条目"""
        if cache_key not in self._cache:
            return False

        entry = self._cache[cache_key]

        # 从类型索引中移除
        self._type_index[entry.data_type].discard(cache_key)

        # 清理依赖关系
        if cache_key in self._reverse_dependency_graph:
            for dep in self._reverse_dependency_graph[cache_key]:
                self._dependency_graph[dep].discard(cache_key)

        # 删除条目
        del self._cache[cache_key]
        if cache_key in self._reverse_dependency_graph:
            del self._reverse_dependency_graph[cache_key]

        self._stats['evictions'] += 1
        return True

    async def _evict_lru(self) -> None:
        """LRU淘汰"""
        if not self._cache:
            return

        # 找到最少使用的条目
        lru_key = min(self._cache.items(), key=lambda x: x[1].last_accessed)[0]
        await self._remove_entry(lru_key)
        self._stats['evictions'] += 1
        logger.debug(f"LRU淘汰元数据缓存: {lru_key}")

    async def clear(self, data_type: Optional[MetadataType] = None) -> bool:
        """清空缓存"""
        try:
            async with self._lock:
                if data_type:
                    # 清空指定类型
                    keys_to_remove = list(self._type_index[data_type])
                    for cache_key in keys_to_remove:
                        await self._remove_entry(cache_key)
                else:
                    # 清空所有
                    self._cache.clear()
                    self._type_index = {meta_type: set() for meta_type in MetadataType}
                    self._dependency_graph.clear()
                    self._reverse_dependency_graph.clear()

                logger.info(f"元数据缓存清空完成: {data_type.value if data_type else '全部'}")
                return True

        except Exception as e:
            logger.error(f"元数据缓存清空失败: {e}")
            return False

    async def cleanup_expired(self) -> int:
        """清理过期缓存"""
        try:
            async with self._lock:
                expired_keys = []
                for cache_key, entry in self._cache.items():
                    if entry.is_expired():
                        expired_keys.append(cache_key)

                for cache_key in expired_keys:
                    await self._remove_entry(cache_key)

                if expired_keys:
                    logger.info(f"清理过期元数据缓存: {len(expired_keys)} 个条目")

                return len(expired_keys)

        except Exception as e:
            logger.error(f"清理过期元数据缓存失败: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self._stats['total_requests']
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = self._stats['cache_hits'] / total_requests

        return {
            'total_requests': total_requests,
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'hit_rate': hit_rate,
            'cached_entries': len(self._cache),
            'max_entries': self.max_entries,
            'evictions': self._stats['evictions'],
            'type_distribution': self._stats['type_distribution'].copy(),
            'dependency_count': len(self._dependency_graph)
        }

    async def search_metadata(self, data_type: MetadataType,
                            filters: Dict[str, Any] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """搜索元数据"""
        try:
            async with self._lock:
                results = []

                for cache_key in self._type_index[data_type]:
                    entry = self._cache.get(cache_key)
                    if entry and not entry.is_expired():
                        # 应用过滤条件
                        if self._matches_filters(entry.data, filters):
                            results.append({
                                'key': cache_key,
                                'data_type': entry.data_type.value,
                                'data': entry.data,
                                'created_at': entry.created_at.isoformat(),
                                'access_count': entry.access_count
                            })

                        if len(results) >= limit:
                            break

                return results

        except Exception as e:
            logger.error(f"搜索元数据失败: {e}")
            return []

    def _matches_filters(self, data: Any, filters: Dict[str, Any]) -> bool:
        """检查数据是否匹配过滤条件"""
        if not filters:
            return True

        for field, condition in filters.items():
            if not hasattr(data, field):
                continue

            value = getattr(data, field)

            if isinstance(condition, dict):
                # 复杂条件
                if 'equals' in condition and value != condition['equals']:
                    return False
                if 'contains' in condition and condition['contains'] not in str(value):
                    return False
                if 'in' in condition and value not in condition['in']:
                    return False
                if 'gt' in condition and value <= condition['gt']:
                    return False
                if 'lt' in condition and value >= condition['lt']:
                    return False
            else:
                # 简单条件
                if value != condition:
                    return False

        return True


# 全局元数据缓存实例
metadata_cache: Optional[MetadataCache] = None


def get_metadata_cache(max_entries: int = 50000, default_ttl: int = 7200) -> MetadataCache:
    """获取元数据缓存实例"""
    global metadata_cache

    if metadata_cache is None:
        metadata_cache = MetadataCache(max_entries=max_entries, default_ttl=default_ttl)

    return metadata_cache