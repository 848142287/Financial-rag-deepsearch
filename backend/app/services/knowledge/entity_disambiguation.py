"""
实体消歧和对齐服务
实现跨文档的实体消歧和合并
"""
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from app.core.graph_config import (
    GraphEntityType,
    generate_entity_id,
    normalize_entity_name,
    graph_entity_config
)

logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """实体提及"""
    id: str
    name: str
    normalized_name: str
    type: GraphEntityType
    confidence: float
    document_id: str
    chunk_id: str
    properties: Dict
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class EntityCluster:
    """实体簇（消歧后的实体）"""
    canonical_id: str
    canonical_name: str
    type: GraphEntityType
    mentions: List[EntityMention]
    aliases: Set[str]
    confidence: float
    properties: Dict

    @property
    def mention_count(self) -> int:
        return len(self.mentions)

    @property
    def document_count(self) -> int:
        return len(set(m.document_id for m in self.mentions))


class EntityDisambiguationService:
    """实体消歧服务"""

    def __init__(self):
        self.config = graph_entity_config

        # 金融领域公司别名词典
        self.company_aliases = self._load_company_aliases()

        # 实体相似度缓存
        self.similarity_cache = {}

    def _load_company_aliases(self) -> Dict[str, List[str]]:
        """加载公司别名词典"""
        return {
            "腾讯控股有限公司": ["腾讯", "Tencent", "腾讯科技", "Tencent Holdings"],
            "阿里巴巴集团控股有限公司": ["阿里巴巴", "阿里", "Alibaba", "Alibaba Group"],
            "百度在线网络技术公司": ["百度", "Baidu", "百度公司"],
            "京东集团": ["京东", "JD.com", "JD"],
            "美团点评": ["美团", "Meituan", "美团点评"],
            "滴滴出行": ["滴滴", "Didi", "滴滴科技"],
            "字节跳动": ["ByteDance", "字节"],
            "华为技术有限公司": ["华为", "Huawei"],
            "中国平安保险（集团）股份有限公司": ["中国平安", "平安保险", "Ping An"],
            "中国建设银行股份有限公司": ["建设银行", "建行", "CCB"],
            "中国工商银行股份有限公司": ["工商银行", "工行", "ICBC"],
            "中国银行股份有限公司": ["中国银行", "中行", "BOC"],
            "中国农业银行股份有限公司": ["农业银行", "农行", "ABC"],
            "招商银行股份有限公司": ["招商银行", "招行", "CMB"],
        }

    async def disambiguate_entities(
        self,
        entities: List[EntityMention],
        existing_clusters: Optional[List[EntityCluster]] = None
    ) -> List[EntityCluster]:
        """
        实体消歧主函数

        Args:
            entities: 实体提及列表
            existing_clusters: 已存在的实体簇（可选）

        Returns:
            消歧后的实体簇列表
        """
        try:
            logger.info(f"开始实体消歧，输入实体数: {len(entities)}")

            # 步骤1: 标准化实体名称
            for entity in entities:
                entity.normalized_name = normalize_entity_name(entity.name, entity.type)

            # 步骤2: 按类型分组
            by_type = self._group_by_type(entities)

            # 步骤3: 在每个类型内进行消歧
            all_clusters = []

            for entity_type, type_entities in by_type.items():
                type_clusters = await self._disambiguate_by_type(
                    type_entities,
                    existing_clusters or []
                )
                all_clusters.extend(type_clusters)

            # 步骤4: 与现有簇合并
            if existing_clusters:
                all_clusters = await self._merge_with_existing(
                    all_clusters,
                    existing_clusters
                )

            logger.info(
                f"实体消歧完成: {len(entities)} 个提及 -> "
                f"{len(all_clusters)} 个实体簇"
            )

            return all_clusters

        except Exception as e:
            logger.error(f"实体消歧失败: {e}")
            # 返回未消歧的实体（每个实体一个簇）
            return [
                EntityCluster(
                    canonical_id=entity.id,
                    canonical_name=entity.name,
                    type=entity.type,
                    mentions=[entity],
                    aliases={entity.name},
                    confidence=entity.confidence,
                    properties=entity.properties
                )
                for entity in entities
            ]

    def _group_by_type(
        self,
        entities: List[EntityMention]
    ) -> Dict[GraphEntityType, List[EntityMention]]:
        """按类型分组"""
        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.type].append(entity)
        return dict(by_type)

    async def _disambiguate_by_type(
        self,
        entities: List[EntityMention],
        existing_clusters: List[EntityCluster]
    ) -> List[EntityCluster]:
        """
        在同一类型内进行消歧

        Args:
            entities: 同类型的实体提及
            existing_clusters: 已存在的实体簇

        Returns:
            实体簇列表
        """
        if not entities:
            return []

        # 使用聚类算法进行实体对齐
        clusters = []
        unassigned = set(range(len(entities)))

        # 1. 首先使用别名词典进行精确匹配
        alias_groups = self._group_by_aliases(entities)
        for group in alias_groups:
            cluster = self._create_cluster_from_group(entities, group)
            clusters.append(cluster)
            unassigned -= group

        # 2. 使用相似度进行聚类
        while unassigned:
            i = unassigned.pop()
            entity_i = entities[i]

            # 创建新簇
            cluster = EntityCluster(
                canonical_id=generate_entity_id(entity_i.name, entity_i.type),
                canonical_name=entity_i.normalized_name,
                type=entity_i.type,
                mentions=[entity_i],
                aliases={entity_i.name, entity_i.normalized_name},
                confidence=entity_i.confidence,
                properties=entity_i.properties.copy()
            )

            # 查找相似实体
            similar_indices = []
            for j in list(unassigned):
                entity_j = entities[j]
                similarity = await self._compute_entity_similarity(
                    entity_i,
                    entity_j
                )

                if similarity >= self.config.entity_similarity_threshold:
                    similar_indices.append(j)

            # 添加相似实体到簇
            for j in similar_indices:
                entity_j = entities[j]
                cluster.mentions.append(entity_j)
                cluster.aliases.add(entity_j.name)
                cluster.aliases.add(entity_j.normalized_name)
                cluster.confidence = max(cluster.confidence, entity_j.confidence)
                cluster.properties.update(entity_j.properties)
                unassigned.remove(j)

            clusters.append(cluster)

        return clusters

    def _group_by_aliases(
        self,
        entities: List[EntityMention]
    ) -> List[Set[int]]:
        """
        使用别名词典进行分组

        Args:
            entities: 实体列表

        Returns:
            分组索引列表
        """
        groups = []
        assigned = set()

        # 构建别名到索引的映射
        alias_to_indices = defaultdict(set)
        for i, entity in enumerate(entities):
            aliases = self._get_all_aliases(entity)
            for alias in aliases:
                alias_to_indices[alias].add(i)

        # 查找共享别名的实体组
        for alias, indices in alias_to_indices.items():
            if len(indices) > 1 and indices - assigned:
                groups.append(indices)
                assigned.update(indices)

        return groups

    def _get_all_aliases(self, entity: EntityMention) -> Set[str]:
        """获取实体的所有别名"""
        aliases = {entity.name, entity.normalized_name}

        # 从实体属性中获取别名
        if entity.aliases:
            aliases.update(entity.aliases)

        # 从词典中查找别名
        for canonical, alias_list in self.company_aliases.items():
            if entity.name in alias_list or entity.normalized_name in alias_list:
                aliases.add(canonical)
                aliases.update(alias_list)

        return aliases

    async def _compute_entity_similarity(
        self,
        entity1: EntityMention,
        entity2: EntityMention
    ) -> float:
        """
        计算两个实体的相似度

        Args:
            entity1: 实体1
            entity2: 实体2

        Returns:
            相似度 (0-1)
        """
        # 缓存键
        cache_key = (entity1.id, entity2.id)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # 1. 名称相似度 (50%)
        name_sim = self._compute_name_similarity(
            entity1.normalized_name,
            entity2.normalized_name
        )

        # 2. 属性相似度 (30%)
        prop_sim = self._compute_property_similarity(
            entity1.properties,
            entity2.properties
        )

        # 3. 上下文相似度 (20%) - 基于共现文档
        context_sim = 0.0
        if entity1.document_id == entity2.document_id:
            context_sim = 1.0  # 同一文档，高置信度
        else:
            # 不同文档，需要更多信息（暂时设为0.5）
            context_sim = 0.5

        # 综合相似度
        similarity = (
            name_sim * 0.5 +
            prop_sim * 0.3 +
            context_sim * 0.2
        )

        # 缓存结果
        self.similarity_cache[cache_key] = similarity

        return similarity

    def _compute_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度（使用Levenshtein距离）"""
        if name1 == name2:
            return 1.0

        # Levenshtein距离
        distance = self._levenshtein_distance(name1, name2)
        max_len = max(len(name1), len(name2))

        if max_len == 0:
            return 0.0

        similarity = 1.0 - (distance / max_len)
        return similarity

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算Levenshtein编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _compute_property_similarity(
        self,
        props1: Dict,
        props2: Dict
    ) -> float:
        """计算属性相似度"""
        if not props1 or not props2:
            return 0.0

        # 共同属性
        common_keys = set(props1.keys()) & set(props2.keys())
        if not common_keys:
            return 0.0

        # 计算属性值相似度
        similarities = []
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值类型：计算相对差异
                if val1 == val2:
                    sim = 1.0
                else:
                    sim = 1.0 - min(abs(val1 - val2) / max(abs(val1), abs(val2)), 1.0)
                similarities.append(sim)
            else:
                # 字符串类型：精确匹配
                sim = 1.0 if str(val1) == str(val2) else 0.0
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _create_cluster_from_group(
        self,
        entities: List[EntityMention],
        group_indices: Set[int]
    ) -> EntityCluster:
        """从实体组创建簇"""
        mentions = [entities[i] for i in group_indices]

        # 选择置信度最高的作为规范名称
        canonical_mention = max(mentions, key=lambda m: m.confidence)

        # 合并别名
        aliases = set()
        for m in mentions:
            aliases.add(m.name)
            aliases.add(m.normalized_name)
            if m.aliases:
                aliases.update(m.aliases)

        # 合并属性
        properties = {}
        for m in mentions:
            properties.update(m.properties)

        # 计算综合置信度
        confidence = max(m.confidence for m in mentions)

        return EntityCluster(
            canonical_id=generate_entity_id(
                canonical_mention.name,
                canonical_mention.type
            ),
            canonical_name=canonical_mention.normalized_name,
            type=canonical_mention.type,
            mentions=mentions,
            aliases=aliases,
            confidence=confidence,
            properties=properties
        )

    async def _merge_with_existing(
        self,
        new_clusters: List[EntityCluster],
        existing_clusters: List[EntityCluster]
    ) -> List[EntityCluster]:
        """与现有簇合并"""
        merged = []

        for new_cluster in new_clusters:
            matched = False

            for existing_cluster in existing_clusters:
                if new_cluster.type != existing_cluster.type:
                    continue

                # 计算簇相似度
                similarity = await self._compute_cluster_similarity(
                    new_cluster,
                    existing_cluster
                )

                if similarity >= self.config.entity_similarity_threshold:
                    # 合并簇
                    merged_cluster = self._merge_clusters(
                        existing_cluster,
                        new_cluster
                    )
                    merged.append(merged_cluster)
                    matched = True
                    break

            if not matched:
                merged.append(new_cluster)

        # 添加未匹配的现有簇
        for existing_cluster in existing_clusters:
            if not any(
                ec.canonical_id == existing_cluster.canonical_id
                for ec in merged
            ):
                merged.append(existing_cluster)

        return merged

    async def _compute_cluster_similarity(
        self,
        cluster1: EntityCluster,
        cluster2: EntityCluster
    ) -> float:
        """计算簇相似度"""
        # 检查名称是否匹配
        if cluster1.canonical_name == cluster2.canonical_name:
            return 1.0

        # 检查别名是否重叠
        alias_overlap = cluster1.aliases & cluster2.aliases
        if alias_overlap:
            return 0.9

        # 使用Levenshtein距离计算名称相似度
        name_sim = self._compute_name_similarity(
            cluster1.canonical_name,
            cluster2.canonical_name
        )

        return name_sim

    def _merge_clusters(
        self,
        base_cluster: EntityCluster,
        new_cluster: EntityCluster
    ) -> EntityCluster:
        """合并两个簇"""
        # 合并提及
        all_mentions = base_cluster.mentions + new_cluster.mentions

        # 合并别名
        all_aliases = base_cluster.aliases | new_cluster.aliases

        # 合并属性
        all_properties = {**base_cluster.properties, **new_cluster.properties}

        # 更新置信度
        merged_confidence = max(
            base_cluster.confidence,
            new_cluster.confidence
        )

        return EntityCluster(
            canonical_id=base_cluster.canonical_id,
            canonical_name=base_cluster.canonical_name,
            type=base_cluster.type,
            mentions=all_mentions,
            aliases=all_aliases,
            confidence=merged_confidence,
            properties=all_properties
        )


# 全局实例
entity_disambiguation_service = EntityDisambiguationService()
