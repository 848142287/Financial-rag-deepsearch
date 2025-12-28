"""
实体对齐器
使用多种算法实现实体的精确对齐
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import networkx as nx
from collections import defaultdict
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    import difflib
    LEVENSHTEIN_AVAILABLE = False
import jieba
import re

logger = logging.getLogger(__name__)


class AlignmentMethod(Enum):
    """对齐方法"""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    STRUCTURE_SIMILARITY = "structure_similarity"
    HYBRID = "hybrid"


@dataclass
class EntityAlignment:
    """实体对齐结果"""
    cluster_id: str
    entities: List[Any]  # EntityNode实例
    alignment_confidence: float
    alignment_method: AlignmentMethod
    representative_entity: Any
    alignment_features: Dict[str, Any]


@dataclass
class AlignmentResult:
    """对齐结果"""
    clusters: Dict[str, EntityAlignment]
    total_entities: int
    aligned_entities: int
    alignment_rate: float
    method_used: str
    statistics: Dict[str, Any]


class EntityAligner:
    """实体对齐器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 对齐参数
        self.exact_match_threshold = self.config.get('exact_match_threshold', 1.0)
        self.fuzzy_match_threshold = self.config.get('fuzzy_match_threshold', 0.8)
        self.semantic_threshold = self.config.get('semantic_threshold', 0.85)
        self.structure_threshold = self.config.get('structure_threshold', 0.7)

        # 权重配置
        self.weights = {
            'name_similarity': self.config.get('name_weight', 0.4),
            'type_similarity': self.config.get('type_weight', 0.2),
            'attribute_similarity': self.config.get('attribute_weight', 0.3),
            'context_similarity': self.config.get('context_weight', 0.1)
        }

        # 初始化工具
        self._initialize_tools()

    def _initialize_tools(self):
        """初始化工具"""
        # TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            tokenizer=self._tokenize_chinese,
            lowercase=False
        )

    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词"""
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 使用jieba分词
        tokens = list(jieba.cut(text))
        # 过滤空字符串和单字符
        return [token for token in tokens if len(token) > 1]

    async def align_entities(
        self,
        entities: List[Any],
        threshold: float = 0.8,
        method: str = "hybrid"
    ) -> AlignmentResult:
        """对齐实体"""
        try:
            logger.info(f"开始对齐 {len(entities)} 个实体，方法: {method}")

            # 预处理实体
            processed_entities = await self._preprocess_entities(entities)

            # 选择对齐方法
            alignment_method = AlignmentMethod(method)

            if alignment_method == AlignmentMethod.EXACT_MATCH:
                clusters = await self._exact_match_alignment(processed_entities)
            elif alignment_method == AlignmentMethod.FUZZY_MATCH:
                clusters = await self._fuzzy_match_alignment(processed_entities, threshold)
            elif alignment_method == AlignmentMethod.SEMANTIC_SIMILARITY:
                clusters = await self._semantic_alignment(processed_entities, threshold)
            elif alignment_method == AlignmentMethod.STRUCTURE_SIMILARITY:
                clusters = await self._structure_alignment(processed_entities, threshold)
            else:  # HYBRID
                clusters = await self._hybrid_alignment(processed_entities, threshold)

            # 后处理聚类结果
            clusters = await self._postprocess_clusters(clusters)

            # 计算统计信息
            total_aligned = sum(len(cluster.entities) for cluster in clusters.values())
            alignment_rate = total_aligned / len(entities) if entities else 0

            statistics = await self._calculate_alignment_statistics(clusters, entities)

            logger.info(f"实体对齐完成，共 {len(clusters)} 个聚类，对齐率: {alignment_rate:.2%}")

            return AlignmentResult(
                clusters=clusters,
                total_entities=len(entities),
                aligned_entities=total_aligned,
                alignment_rate=alignment_rate,
                method_used=method,
                statistics=statistics
            )

        except Exception as e:
            logger.error(f"实体对齐失败: {str(e)}")
            raise

    async def _preprocess_entities(self, entities: List[Any]) -> List[Dict[str, Any]]:
        """预处理实体"""
        processed = []

        for entity in entities:
            # 提取实体特征
            features = {
                'id': entity.id,
                'label': entity.label,
                'type': entity.type,
                'properties': entity.properties or {},
                'normalized_label': self._normalize_text(entity.label),
                'tokens': self._tokenize_chinese(entity.label),
                'attributes': self._extract_attributes(entity.properties or {}),
                'context': self._extract_context(entity)
            }

            # 生成向量表示
            features['vector'] = await self._generate_entity_vector(entity)

            processed.append(features)

        return processed

    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        # 转换为小写
        text = text.lower()
        # 移除空格和标点
        text = re.sub(r'[\s,，。.！!?？]', '', text)
        return text

    def _extract_attributes(self, properties: Dict[str, Any]) -> List[str]:
        """提取属性特征"""
        attributes = []
        for key, value in properties.items():
            if isinstance(value, str):
                attributes.extend(self._tokenize_chinese(value))
            elif isinstance(value, (int, float)):
                attributes.append(f"{key}:{value}")
        return attributes

    def _extract_context(self, entity: Any) -> List[str]:
        """提取上下文信息"""
        context = []

        # 从源文档中提取
        if hasattr(entity, 'source_documents') and entity.source_documents:
            for doc in entity.source_documents:
                context.append(f"source:{doc}")

        # 从元数据中提取
        if hasattr(entity, 'metadata') and entity.metadata:
            for key, value in entity.metadata.items():
                if isinstance(value, str):
                    context.extend(self._tokenize_chinese(value))

        return context

    async def _generate_entity_vector(self, entity: Any) -> np.ndarray:
        """生成实体向量"""
        # 组合实体的所有文本信息
        texts = [entity.label]

        # 添加属性文本
        if entity.properties:
            for value in entity.properties.values():
                if isinstance(value, str):
                    texts.append(value)

        # 使用TF-IDF生成向量
        try:
            combined_text = ' '.join(texts)
            vector = self.tfidf_vectorizer.fit_transform([combined_text])
            return vector.toarray()[0]
        except:
            # 如果TF-IDF失败，使用简单的特征向量
            return np.random.rand(100)

    async def _exact_match_alignment(self, entities: List[Dict[str, Any]]) -> Dict[str, EntityAlignment]:
        """精确匹配对齐"""
        clusters = {}
        label_map = defaultdict(list)

        # 按标准化标签分组
        for entity in entities:
            label_map[entity['normalized_label']].append(entity)

        # 创建聚类
        cluster_id = 0
        for label, entity_list in label_map.items():
            if len(entity_list) > 1:  # 只保留有重复的
                # 选择代表性实体（置信度最高的）
                representative = max(entity_list, key=lambda e: getattr(e.get('original'), 'confidence', 1.0))

                cluster = EntityAlignment(
                    cluster_id=f"exact_{cluster_id}",
                    entities=[e['original'] for e in entity_list],
                    alignment_confidence=1.0,
                    alignment_method=AlignmentMethod.EXACT_MATCH,
                    representative_entity=representative['original'],
                    alignment_features={
                        'match_type': 'exact',
                        'normalized_label': label
                    }
                )
                clusters[cluster.cluster_id] = cluster
                cluster_id += 1

        return clusters

    async def _fuzzy_match_alignment(
        self,
        entities: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, EntityAlignment]:
        """模糊匹配对齐"""
        clusters = {}
        processed = set()
        cluster_id = 0

        for i, entity1 in enumerate(entities):
            if i in processed:
                continue

            cluster_entities = [entity1]
            processed.add(i)

            # 查找相似的实体
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if j in processed:
                    continue

                # 计算编辑距离相似度
                similarity = self._calculate_edit_similarity(
                    entity1['normalized_label'],
                    entity2['normalized_label']
                )

                if similarity >= threshold:
                    cluster_entities.append(entity2)
                    processed.add(j)

            # 创建聚类
            if len(cluster_entities) > 1:
                representative = max(cluster_entities, key=lambda e: getattr(e.get('original'), 'confidence', 1.0))

                cluster = EntityAlignment(
                    cluster_id=f"fuzzy_{cluster_id}",
                    entities=[e['original'] for e in cluster_entities],
                    alignment_confidence=sum(similarity for similarity in [
                        self._calculate_edit_similarity(cluster_entities[0]['normalized_label'], e['normalized_label'])
                        for e in cluster_entities[1:]
                    ]) / len(cluster_entities),
                    alignment_method=AlignmentMethod.FUZZY_MATCH,
                    representative_entity=representative['original'],
                    alignment_features={
                        'match_type': 'fuzzy',
                        'avg_similarity': np.mean([
                            self._calculate_edit_similarity(cluster_entities[0]['normalized_label'], e['normalized_label'])
                            for e in cluster_entities[1:]
                        ])
                    }
                )
                clusters[cluster.cluster_id] = cluster
                cluster_id += 1

        return clusters

    def _calculate_edit_similarity(self, str1: str, str2: str) -> float:
        """计算编辑距离相似度"""
        if not str1 or not str2:
            return 0.0

        if LEVENSHTEIN_AVAILABLE:
            distance = Levenshtein.distance(str1, str2)
        else:
            # 使用difflib作为后备方案
            import difflib
            distance = len(''.join(difflib.SequenceMatcher(None, str1, str2).get_opcodes()))

        max_len = max(len(str1), len(str2))
        return 1.0 - distance / max_len if max_len > 0 else 1.0

    async def _semantic_alignment(
        self,
        entities: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, EntityAlignment]:
        """语义相似度对齐"""
        if len(entities) < 2:
            return {}

        # 构建相似度矩阵
        vectors = np.array([entity['vector'] for entity in entities])
        similarity_matrix = cosine_similarity(vectors)

        # 使用DBSCAN聚类
        clustering = DBSCAN(
            eps=1 - threshold,
            min_samples=2,
            metric='precomputed'
        )

        # 转换为距离矩阵
        distance_matrix = 1 - similarity_matrix
        labels = clustering.fit_predict(distance_matrix)

        # 创建聚类
        clusters = {}
        cluster_id = 0

        for label in set(labels):
            if label == -1:  # 忽略噪声点
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_entities = [entities[i] for i in cluster_indices]

            if len(cluster_entities) > 1:
                # 计算聚类内平均相似度
                cluster_similarities = []
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i < j:
                            cluster_similarities.append(similarity_matrix[i][j])

                avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0.0

                representative = max(cluster_entities, key=lambda e: getattr(e.get('original'), 'confidence', 1.0))

                cluster = EntityAlignment(
                    cluster_id=f"semantic_{cluster_id}",
                    entities=[e['original'] for e in cluster_entities],
                    alignment_confidence=avg_similarity,
                    alignment_method=AlignmentMethod.SEMANTIC_SIMILARITY,
                    representative_entity=representative['original'],
                    alignment_features={
                        'match_type': 'semantic',
                        'cluster_size': len(cluster_entities),
                        'avg_similarity': avg_similarity
                    }
                )
                clusters[cluster.cluster_id] = cluster
                cluster_id += 1

        return clusters

    async def _structure_alignment(
        self,
        entities: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, EntityAlignment]:
        """结构相似度对齐"""
        clusters = {}
        processed = set()
        cluster_id = 0

        for i, entity1 in enumerate(entities):
            if i in processed:
                continue

            cluster_entities = [entity1]
            processed.add(i)

            # 查找结构相似的实体
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if j in processed:
                    continue

                # 计算结构相似度
                similarity = self._calculate_structure_similarity(entity1, entity2)

                if similarity >= threshold:
                    cluster_entities.append(entity2)
                    processed.add(j)

            # 创建聚类
            if len(cluster_entities) > 1:
                representative = max(cluster_entities, key=lambda e: getattr(e.get('original'), 'confidence', 1.0))

                cluster = EntityAlignment(
                    cluster_id=f"structure_{cluster_id}",
                    entities=[e['original'] for e in cluster_entities],
                    alignment_confidence=similarity,
                    alignment_method=AlignmentMethod.STRUCTURE_SIMILARITY,
                    representative_entity=representative['original'],
                    alignment_features={
                        'match_type': 'structure',
                        'type_match': entity1['type'] == cluster_entities[1]['type'] if len(cluster_entities) > 1 else True,
                        'attribute_overlap': self._calculate_attribute_overlap(entity1, entity2) if len(cluster_entities) > 1 else 0
                    }
                )
                clusters[cluster.cluster_id] = cluster
                cluster_id += 1

        return clusters

    def _calculate_structure_similarity(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """计算结构相似度"""
        similarity = 0.0

        # 类型相似度
        if entity1['type'] == entity2['type']:
            similarity += self.weights['type_similarity']

        # 属性相似度
        attr_overlap = self._calculate_attribute_overlap(entity1, entity2)
        similarity += self.weights['attribute_similarity'] * attr_overlap

        # 上下文相似度
        context_overlap = self._calculate_context_overlap(entity1, entity2)
        similarity += self.weights['context_similarity'] * context_overlap

        return similarity

    def _calculate_attribute_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """计算属性重叠度"""
        attrs1 = set(entity1['attributes'])
        attrs2 = set(entity2['attributes'])

        if not attrs1 and not attrs2:
            return 1.0

        intersection = attrs1.intersection(attrs2)
        union = attrs1.union(attrs2)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_context_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """计算上下文重叠度"""
        context1 = set(entity1['context'])
        context2 = set(entity2['context'])

        if not context1 and not context2:
            return 1.0

        intersection = context1.intersection(context2)
        union = context1.union(context2)

        return len(intersection) / len(union) if union else 0.0

    async def _hybrid_alignment(
        self,
        entities: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, EntityAlignment]:
        """混合对齐方法"""
        # 依次应用不同的对齐方法
        all_clusters = {}

        # 1. 精确匹配
        exact_clusters = await self._exact_match_alignment(entities)
        all_clusters.update(exact_clusters)

        # 2. 模糊匹配
        fuzzy_threshold = max(threshold, self.fuzzy_match_threshold)
        fuzzy_clusters = await self._fuzzy_match_alignment(entities, fuzzy_threshold)
        all_clusters.update(fuzzy_clusters)

        # 3. 语义相似度
        semantic_clusters = await self._semantic_alignment(entities, threshold)
        all_clusters.update(semantic_clusters)

        # 4. 结构相似度
        structure_clusters = await self._structure_alignment(entities, threshold)
        all_clusters.update(structure_clusters)

        # 合并重叠的聚类
        merged_clusters = await self._merge_overlapping_clusters(all_clusters)

        return merged_clusters

    async def _merge_overlapping_clusters(self, clusters: Dict[str, EntityAlignment]) -> Dict[str, EntityAlignment]:
        """合并重叠的聚类"""
        if not clusters:
            return clusters

        # 构建实体到聚类的映射
        entity_to_clusters = defaultdict(list)
        for cluster_id, cluster in clusters.items():
            for entity in cluster.entities:
                entity_to_clusters[entity.id].append(cluster)

        # 找到需要合并的聚类
        merged_clusters = {}
        processed_entities = set()
        merge_id = 0

        for entity_id, entity_clusters in entity_to_clusters.items():
            if entity_id in processed_entities:
                continue

            # 收集所有相关的聚类
            related_clusters = []
            entities_to_process = [entity_id]

            while entities_to_process:
                current_entity_id = entities_to_process.pop()
                if current_entity_id in processed_entities:
                    continue

                processed_entities.add(current_entity_id)

                for cluster in entity_to_clusters[current_entity_id]:
                    if cluster not in related_clusters:
                        related_clusters.append(cluster)
                        for entity in cluster.entities:
                            if entity.id not in processed_entities:
                                entities_to_process.append(entity.id)

            # 创建合并后的聚类
            if len(related_clusters) > 1:
                all_entities = []
                all_confidences = []
                all_methods = []

                for cluster in related_clusters:
                    all_entities.extend(cluster.entities)
                    all_confidences.append(cluster.alignment_confidence)
                    all_methods.append(cluster.alignment_method.value)

                # 选择代表性实体
                representative = max(all_entities, key=lambda e: getattr(e, 'confidence', 1.0))

                merged_cluster = EntityAlignment(
                    cluster_id=f"merged_{merge_id}",
                    entities=all_entities,
                    alignment_confidence=np.mean(all_confidences),
                    alignment_method=AlignmentMethod.HYBRID,
                    representative_entity=representative,
                    alignment_features={
                        'merged_from': [c.cluster_id for c in related_clusters],
                        'original_methods': all_methods,
                        'merge_count': len(related_clusters)
                    }
                )
                merged_clusters[merged_cluster.cluster_id] = merged_cluster
                merge_id += 1
            elif len(related_clusters) == 1:
                # 直接添加
                merged_clusters[related_clusters[0].cluster_id] = related_clusters[0]

        return merged_clusters

    async def _postprocess_clusters(self, clusters: Dict[str, EntityAlignment]) -> Dict[str, EntityAlignment]:
        """后处理聚类结果"""
        # 移除只包含单个实体的聚类
        filtered_clusters = {
            cluster_id: cluster
            for cluster_id, cluster in clusters.items()
            if len(cluster.entities) > 1
        }

        # 重新计算置信度
        for cluster in filtered_clusters.values():
            cluster.alignment_confidence = await self._recalculate_cluster_confidence(cluster)

        return filtered_clusters

    async def _recalculate_cluster_confidence(self, cluster: EntityAlignment) -> float:
        """重新计算聚类置信度"""
        if not cluster.entities:
            return 0.0

        # 基于实体的平均置信度
        entity_confidences = [
            getattr(entity, 'confidence', 1.0)
            for entity in cluster.entities
        ]
        avg_entity_confidence = np.mean(entity_confidences)

        # 基于聚类内一致性
        consistency_score = await self._calculate_cluster_consistency(cluster)

        # 综合置信度
        final_confidence = 0.7 * avg_entity_confidence + 0.3 * consistency_score

        return final_confidence

    async def _calculate_cluster_consistency(self, cluster: EntityAlignment) -> float:
        """计算聚类一致性"""
        if len(cluster.entities) < 2:
            return 1.0

        # 计算实体对之间的平均相似度
        similarities = []

        for i, entity1 in enumerate(cluster.entities):
            for entity2 in cluster.entities[i+1:]:
                # 类型一致性
                type_match = 1.0 if entity1.type == entity2.type else 0.0
                similarities.append(type_match)

        return np.mean(similarities) if similarities else 1.0

    async def _calculate_alignment_statistics(
        self,
        clusters: Dict[str, EntityAlignment],
        original_entities: List[Any]
    ) -> Dict[str, Any]:
        """计算对齐统计信息"""
        stats = {
            'total_clusters': len(clusters),
            'cluster_size_distribution': [],
            'alignment_method_distribution': defaultdict(int),
            'confidence_distribution': [],
            'type_distribution': defaultdict(int),
            'unaligned_entities': len(original_entities)
        }

        # 统计每个聚类
        for cluster in clusters.values():
            cluster_size = len(cluster.entities)
            stats['cluster_size_distribution'].append(cluster_size)
            stats['alignment_method_distribution'][cluster.alignment_method.value] += 1
            stats['confidence_distribution'].append(cluster.alignment_confidence)

            # 减少未对齐实体数
            stats['unaligned_entities'] -= cluster_size

            # 统计类型分布
            for entity in cluster.entities:
                stats['type_distribution'][entity.type] += 1

        # 计算平均统计
        if stats['cluster_size_distribution']:
            stats['avg_cluster_size'] = np.mean(stats['cluster_size_distribution'])
            stats['max_cluster_size'] = max(stats['cluster_size_distribution'])
            stats['min_cluster_size'] = min(stats['cluster_size_distribution'])
        else:
            stats['avg_cluster_size'] = 0
            stats['max_cluster_size'] = 0
            stats['min_cluster_size'] = 0

        if stats['confidence_distribution']:
            stats['avg_confidence'] = np.mean(stats['confidence_distribution'])
            stats['max_confidence'] = max(stats['confidence_distribution'])
            stats['min_confidence'] = min(stats['confidence_distribution'])
        else:
            stats['avg_confidence'] = 0
            stats['max_confidence'] = 0
            stats['min_confidence'] = 0

        return stats