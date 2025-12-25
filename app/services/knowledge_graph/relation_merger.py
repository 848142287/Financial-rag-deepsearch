"""
关系融合器
实现知识图谱中关系的合并和去重
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)


class RelationMergeStrategy(Enum):
    """关系合并策略"""
    UNION = "union"  # 合并所有关系
    INTERSECTION = "intersection"  # 取交集
    HIGHEST_CONFIDENCE = "highest_confidence"  # 选择置信度最高的
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    SEMANTIC_SIMILARITY = "semantic_similarity"  # 基于语义相似度


@dataclass
class MergedRelation:
    """合并后的关系"""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    weight: float
    source_relations: List[str]  # 原始关系ID列表
    properties: Dict[str, Any]
    metadata: Dict[str, Any]
    merge_count: int
    merge_method: RelationMergeStrategy


@dataclass
class MergeResult:
    """合并结果"""
    merged_relations: List[MergedRelation]
    original_relations: int
    merged_count: int
    compression_rate: float
    strategy_used: str
    statistics: Dict[str, Any]


class RelationMerger:
    """关系融合器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 合并参数
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.default_strategy = RelationMergeStrategy(
            self.config.get('default_strategy', 'highest_confidence')
        )

        # 权重配置
        self.weights = {
            'semantic_similarity': self.config.get('semantic_weight', 0.4),
            'structural_similarity': self.config.get('structural_weight', 0.3),
            'confidence_score': self.config.get('confidence_weight', 0.2),
            'source_reliability': self.config.get('source_weight', 0.1)
        }

        # 初始化工具
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

        # 关系类型映射
        self.predicate_mappings = {
            '相等': ['is', 'equals', 'same_as', '等同'],
            '属于': 'belongs_to',
            '包含': 'contains',
            '位于': 'located_in',
            '创建': ['created_by', 'author'],
            '拥有': ['has', 'owns', 'possesses'],
            '相关': ['related_to', 'associated_with', 'correlated']
        }

    async def merge_relations(
        self,
        relations: List[Any],
        entity_id_map: Dict[str, Any],
        strategy: Optional[RelationMergeStrategy] = None
    ) -> MergeResult:
        """融合关系"""
        try:
            logger.info(f"开始融合 {len(relations)} 个关系")

            if not strategy:
                strategy = self.default_strategy

            # 预处理关系
            processed_relations = await self._preprocess_relations(relations, entity_id_map)

            # 根据策略执行融合
            if strategy == RelationMergeStrategy.UNION:
                merged_relations = await self._union_merge(processed_relations)
            elif strategy == RelationMergeStrategy.INTERSECTION:
                merged_relations = await self._intersection_merge(processed_relations)
            elif strategy == RelationMergeStrategy.HIGHEST_CONFIDENCE:
                merged_relations = await self._highest_confidence_merge(processed_relations)
            elif strategy == RelationMergeStrategy.WEIGHTED_AVERAGE:
                merged_relations = await self._weighted_average_merge(processed_relations)
            elif strategy == RelationMergeStrategy.SEMANTIC_SIMILARITY:
                merged_relations = await self._semantic_similarity_merge(processed_relations)
            else:
                merged_relations = await self._default_merge(processed_relations)

            # 计算统计信息
            original_count = len(relations)
            merged_count = len(merged_relations)
            compression_rate = 1 - (merged_count / original_count) if original_count > 0 else 0

            statistics = await self._calculate_merge_statistics(
                relations, merged_relations, strategy
            )

            logger.info(f"关系融合完成，原始: {original_count}, 合并后: {merged_count}, 压缩率: {compression_rate:.2%}")

            return MergeResult(
                merged_relations=merged_relations,
                original_relations=original_count,
                merged_count=merged_count,
                compression_rate=compression_rate,
                strategy_used=strategy.value,
                statistics=statistics
            )

        except Exception as e:
            logger.error(f"关系融合失败: {str(e)}")
            raise

    async def _preprocess_relations(self, relations: List[Any], entity_id_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预处理关系"""
        processed = []

        for relation in relations:
            # 映射实体ID
            subject_id = entity_id_map.get(relation.subject, relation.subject)
            object_id = entity_id_map.get(relation.target or relation.object, relation.target or relation.object)

            # 标准化谓词
            predicate = self._normalize_predicate(relation.predicate or relation.type)

            processed_relation = {
                'id': relation.id,
                'subject': subject_id,
                'predicate': predicate,
                'object': object_id,
                'confidence': getattr(relation, 'confidence', 1.0),
                'weight': getattr(relation, 'weight', 1.0),
                'properties': getattr(relation, 'properties', {}),
                'metadata': getattr(relation, 'metadata', {}),
                'source_documents': getattr(relation, 'source_documents', set()),
                'original_relation': relation
            }

            # 生成关系签名用于快速匹配
            processed_relation['signature'] = self._generate_signature(
                subject_id, predicate, object_id
            )

            processed.append(processed_relation)

        return processed

    def _normalize_predicate(self, predicate: str) -> str:
        """标准化谓词"""
        if not predicate:
            return "unknown"

        predicate = predicate.lower().strip()

        # 查找映射
        for standard, variants in self.predicate_mappings.items():
            if isinstance(variants, list):
                if predicate in variants:
                    return standard
            elif predicate == variants:
                return standard

        # 简单的词根处理
        if predicate.endswith('s') and len(predicate) > 3:
            predicate = predicate[:-1]

        return predicate

    def _generate_signature(self, subject: str, predicate: str, obj: str) -> str:
        """生成关系签名"""
        return f"{subject}_{predicate}_{obj}"

    async def _union_merge(self, relations: List[Dict[str, Any]]) -> List[MergedRelation]:
        """合并策略：联合所有关系"""
        signature_groups = defaultdict(list)

        # 按签名分组
        for relation in relations:
            signature_groups[relation['signature']].append(relation)

        merged_relations = []

        for signature, group in signature_groups.items():
            if len(group) > 1:
                # 合并多个相同的关系
                merged = await self._merge_relation_group(group, RelationMergeStrategy.UNION)
                merged_relations.append(merged)
            else:
                # 单个关系直接使用
                rel = group[0]
                merged = MergedRelation(
                    id=rel['id'],
                    subject=rel['subject'],
                    predicate=rel['predicate'],
                    object=rel['object'],
                    confidence=rel['confidence'],
                    weight=rel['weight'],
                    source_relations=[rel['id']],
                    properties=rel['properties'],
                    metadata=rel['metadata'],
                    merge_count=1,
                    merge_method=RelationMergeStrategy.UNION
                )
                merged_relations.append(merged)

        return merged_relations

    async def _intersection_merge(self, relations: List[Dict[str, Any]]) -> List[MergedRelation]:
        """合并策略：取交集"""
        signature_groups = defaultdict(list)

        # 按签名分组
        for relation in relations:
            signature_groups[relation['signature']].append(relation)

        merged_relations = []

        for signature, group in signature_groups.items():
            # 只保留在多个源中出现的关系
            if len(group) >= 2:
                # 检查是否在不同源中出现
                sources = set()
                for rel in group:
                    sources.update(rel['source_documents'])

                if len(sources) >= 2:
                    merged = await self._merge_relation_group(group, RelationMergeStrategy.INTERSECTION)
                    merged_relations.append(merged)

        return merged_relations

    async def _highest_confidence_merge(self, relations: List[Dict[str, Any]]) -> List[MergedRelation]:
        """合并策略：选择置信度最高的"""
        signature_groups = defaultdict(list)

        # 按签名分组
        for relation in relations:
            signature_groups[relation['signature']].append(relation)

        merged_relations = []

        for signature, group in signature_groups.items():
            # 选择置信度最高的关系
            best_relation = max(group, key=lambda r: r['confidence'])

            # 合并其他关系的元数据
            all_properties = {}
            all_metadata = {}
            all_sources = set()
            source_ids = []

            for rel in group:
                all_properties.update(rel['properties'])
                all_metadata.update(rel['metadata'])
                all_sources.update(rel['source_documents'])
                source_ids.append(rel['id'])

            merged = MergedRelation(
                id=best_relation['id'],
                subject=best_relation['subject'],
                predicate=best_relation['predicate'],
                object=best_relation['object'],
                confidence=best_relation['confidence'],
                weight=best_relation['weight'],
                source_relations=source_ids,
                properties=all_properties,
                metadata={**all_metadata, 'all_sources': list(all_sources)},
                merge_count=len(group),
                merge_method=RelationMergeStrategy.HIGHEST_CONFIDENCE
            )
            merged_relations.append(merged)

        return merged_relations

    async def _weighted_average_merge(self, relations: List[Dict[str, Any]]) -> List[MergedRelation]:
        """合并策略：加权平均"""
        signature_groups = defaultdict(list)

        # 按签名分组
        for relation in relations:
            signature_groups[relation['signature']].append(relation)

        merged_relations = []

        for signature, group in signature_groups.items():
            # 计算加权平均
            total_weight = sum(rel['weight'] for rel in group)

            if total_weight == 0:
                total_weight = len(group)

            # 加权平均置信度
            avg_confidence = sum(rel['confidence'] * rel['weight'] for rel in group) / total_weight

            # 合并属性和元数据
            all_properties = {}
            all_metadata = {}
            all_sources = set()
            source_ids = []

            for rel in group:
                all_properties.update(rel['properties'])
                all_metadata.update(rel['metadata'])
                all_sources.update(rel['source_documents'])
                source_ids.append(rel['id'])

            # 选择第一个关系作为基础
            base_relation = group[0]

            merged = MergedRelation(
                id=f"merged_{signature}",
                subject=base_relation['subject'],
                predicate=base_relation['predicate'],
                object=base_relation['object'],
                confidence=avg_confidence,
                weight=total_weight / len(group),  # 平均权重
                source_relations=source_ids,
                properties=all_properties,
                metadata={**all_metadata, 'all_sources': list(all_sources)},
                merge_count=len(group),
                merge_method=RelationMergeStrategy.WEIGHTED_AVERAGE
            )
            merged_relations.append(merged)

        return merged_relations

    async def _semantic_similarity_merge(self, relations: List[Dict[str, Any]]) -> List[MergedRelation]:
        """合并策略：基于语义相似度"""
        if len(relations) < 2:
            return await self._default_merge(relations)

        # 构建相似度矩阵
        relation_texts = []
        for rel in relations:
            text = f"{rel['subject']} {rel['predicate']} {rel['object']}"
            relation_texts.append(text)

        # 计算TF-IDF向量
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(relation_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            # 如果TF-IDF失败，使用默认策略
            return await self._default_merge(relations)

        # 聚类相似的关系
        clusters = self._cluster_by_similarity(relations, similarity_matrix)

        # 合并每个聚类中的关系
        merged_relations = []
        for cluster in clusters:
            if len(cluster) > 1:
                merged = await self._merge_relation_group(cluster, RelationMergeStrategy.SEMANTIC_SIMILARITY)
                merged_relations.append(merged)
            else:
                rel = cluster[0]
                merged = MergedRelation(
                    id=rel['id'],
                    subject=rel['subject'],
                    predicate=rel['predicate'],
                    object=rel['object'],
                    confidence=rel['confidence'],
                    weight=rel['weight'],
                    source_relations=[rel['id']],
                    properties=rel['properties'],
                    metadata=rel['metadata'],
                    merge_count=1,
                    merge_method=RelationMergeStrategy.SEMANTIC_SIMILARITY
                )
                merged_relations.append(merged)

        return merged_relations

    def _cluster_by_similarity(
        self,
        relations: List[Dict[str, Any]],
        similarity_matrix: np.ndarray
    ) -> List[List[Dict[str, Any]]]:
        """基于相似度聚类"""
        clusters = []
        used_indices = set()

        for i, rel in enumerate(relations):
            if i in used_indices:
                continue

            cluster = [rel]
            used_indices.add(i)

            # 查找相似的关系
            for j in range(i + 1, len(relations)):
                if j in used_indices:
                    continue

                if similarity_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(relations[j])
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters

    async def _default_merge(self, relations: List[Dict[str, Any]]) -> List[MergedRelation]:
        """默认合并策略"""
        return await self._highest_confidence_merge(relations)

    async def _merge_relation_group(
        self,
        relations: List[Dict[str, Any]],
        strategy: RelationMergeStrategy
    ) -> MergedRelation:
        """合并关系组"""
        if len(relations) == 1:
            rel = relations[0]
            return MergedRelation(
                id=rel['id'],
                subject=rel['subject'],
                predicate=rel['predicate'],
                object=rel['object'],
                confidence=rel['confidence'],
                weight=rel['weight'],
                source_relations=[rel['id']],
                properties=rel['properties'],
                metadata=rel['metadata'],
                merge_count=1,
                merge_method=strategy
            )

        # 选择基础关系
        base_relation = relations[0]

        # 合并属性
        merged_properties = {}
        for rel in relations:
            merged_properties.update(rel['properties'])

        # 合并元数据
        merged_metadata = {
            'merge_timestamp': asyncio.get_event_loop().time(),
            'merge_strategy': strategy.value,
            'source_count': len(relations)
        }

        # 合并源文档
        all_sources = set()
        source_ids = []
        for rel in relations:
            all_sources.update(rel['source_documents'])
            source_ids.append(rel['id'])
        merged_metadata['all_sources'] = list(all_sources)

        # 计算合并后的置信度和权重
        if strategy == RelationMergeStrategy.HIGHEST_CONFIDENCE:
            confidence = max(rel['confidence'] for rel in relations)
            weight = max(rel['weight'] for rel in relations)
        elif strategy == RelationMergeStrategy.WEIGHTED_AVERAGE:
            total_weight = sum(rel['weight'] for rel in relations)
            confidence = sum(rel['confidence'] * rel['weight'] for rel in relations) / total_weight
            weight = total_weight / len(relations)
        else:
            confidence = sum(rel['confidence'] for rel in relations) / len(relations)
            weight = sum(rel['weight'] for rel in relations) / len(relations)

        # 生成合并后的ID
        signature = base_relation['signature']
        merged_id = f"merged_{strategy.value}_{signature}"

        return MergedRelation(
            id=merged_id,
            subject=base_relation['subject'],
            predicate=base_relation['predicate'],
            object=base_relation['object'],
            confidence=confidence,
            weight=weight,
            source_relations=source_ids,
            properties=merged_properties,
            metadata=merged_metadata,
            merge_count=len(relations),
            merge_method=strategy
        )

    async def _calculate_merge_statistics(
        self,
        original_relations: List[Any],
        merged_relations: List[MergedRelation],
        strategy: RelationMergeStrategy
    ) -> Dict[str, Any]:
        """计算合并统计信息"""
        stats = {
            'strategy': strategy.value,
            'predicate_distribution': defaultdict(int),
            'confidence_distribution': {
                'mean': 0.0,
                'min': 1.0,
                'max': 0.0,
                'std': 0.0
            },
            'merge_count_distribution': defaultdict(int),
            'compression_details': {},
            'quality_metrics': {}
        }

        # 统计谓词分布
        for merged in merged_relations:
            stats['predicate_distribution'][merged.predicate] += 1

        # 统计置信度分布
        if merged_relations:
            confidences = [merged.confidence for merged in merged_relations]
            stats['confidence_distribution'] = {
                'mean': np.mean(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'std': np.std(confidences)
            }

        # 统计合并数量分布
        for merged in merged_relations:
            stats['merge_count_distribution'][merged.merge_count] += 1

        # 压缩详情
        original_count = len(original_relations)
        merged_count = len(merged_relations)
        stats['compression_details'] = {
            'original_count': original_count,
            'merged_count': merged_count,
            'reduced_count': original_count - merged_count,
            'compression_rate': (original_count - merged_count) / original_count if original_count > 0 else 0
        }

        # 质量指标
        stats['quality_metrics'] = {
            'avg_confidence_improvement': self._calculate_confidence_improvement(
                original_relations, merged_relations
            ),
            'duplicate_reduction_rate': self._calculate_duplicate_reduction(
                original_relations, merged_relations
            ),
            'consistency_score': self._calculate_consistency_score(merged_relations)
        }

        return stats

    def _calculate_confidence_improvement(
        self,
        original_relations: List[Any],
        merged_relations: List[MergedRelation]
    ) -> float:
        """计算置信度改进"""
        if not original_relations or not merged_relations:
            return 0.0

        original_avg = np.mean([getattr(r, 'confidence', 1.0) for r in original_relations])
        merged_avg = np.mean([m.confidence for m in merged_relations])

        return merged_avg - original_avg

    def _calculate_duplicate_reduction(
        self,
        original_relations: List[Any],
        merged_relations: List[MergedRelation]
    ) -> float:
        """计算重复减少率"""
        if not original_relations:
            return 0.0

        # 计算原始关系中的重复
        signatures = set()
        duplicates = 0
        for rel in original_relations:
            subject = rel.subject
            predicate = rel.predicate or rel.type
            obj = rel.target or rel.object
            signature = f"{subject}_{predicate}_{obj}"
            if signature in signatures:
                duplicates += 1
            else:
                signatures.add(signature)

        return duplicates / len(original_relations)

    def _calculate_consistency_score(self, merged_relations: List[MergedRelation]) -> float:
        """计算一致性分数"""
        if not merged_relations:
            return 0.0

        # 基于合并数量的分布计算一致性
        merge_counts = [m.merge_count for m in merged_relations]
        if len(merge_counts) <= 1:
            return 1.0

        # 标准差越小，一致性越高
        std_dev = np.std(merge_counts)
        max_std = np.sqrt(len(merge_counts) - 1)
        consistency = 1.0 - (std_dev / max_std) if max_std > 0 else 1.0

        return consistency