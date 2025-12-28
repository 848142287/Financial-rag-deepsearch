"""
图谱质量验证服务
验证实体和关系的质量，检测冲突和异常
"""
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from app.core.graph_config import (
    GraphEntityType,
    GraphRelationType,
    graph_quality_config
)

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """质量等级"""
    HIGH = "high"           # 高质量
    MEDIUM = "medium"       # 中等质量
    LOW = "low"            # 低质量
    INVALID = "invalid"    # 无效


@dataclass
class ValidationIssue:
    """验证问题"""
    severity: str          # severity: error, warning, info
    issue_type: str        # 问题类型
    description: str       # 描述
    entity_id: Optional[str] = None
    relation_id: Optional[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class GraphQualityValidator:
    """图谱质量验证器"""

    def __init__(self):
        self.config = graph_quality_config

        # 冲突关系规则
        self.conflict_rules = self._load_conflict_rules()

    def _load_conflict_rules(self) -> Dict[Tuple[GraphRelationType, GraphRelationType], str]:
        """
        加载冲突关系规则

        Returns:
            {
                (rel_type1, rel_type2): "冲突描述"
            }
        """
        return {
            # 投资关系冲突
            (GraphRelationType.INVESTS_IN, GraphRelationType.SUBSIDIARY_OF):
                "不能既是投资关系又是子公司关系",

            # 竞争与合作冲突
            (GraphRelationType.PARTNER_OF, GraphRelationType.COMPETITOR_OF):
                "不能既是合作伙伴又是竞争对手",

            # 任职关系冲突（一个人不能在两个职位相同的公司任职）
            (GraphRelationType.CEO_OF, GraphRelationType.CEO_OF):
                "一个人不能同时是两家公司的CEO",

            # 所有权关系冲突
            (GraphRelationType.SUBSIDIARY_OF, GraphRelationType.PARENT_OF):
                "不能既是子公司又是母公司",
        }

    async def validate_entity(
        self,
        entity_data: Dict
    ) -> Tuple[QualityLevel, List[ValidationIssue]]:
        """
        验证实体质量

        Args:
            entity_data: 实体数据

        Returns:
            (质量等级, 问题列表)
        """
        issues = []

        # 1. 验证必填字段
        required_fields = ["name", "type", "canonical_id"]
        for field in required_fields:
            if field not in entity_data or not entity_data[field]:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="missing_required_field",
                    description=f"缺少必填字段: {field}",
                    entity_id=entity_data.get("canonical_id"),
                    suggestions=[f"添加 {field} 字段"]
                ))

        # 2. 验证实体类型
        if "type" in entity_data:
            try:
                entity_type = GraphEntityType(entity_data["type"])
            except ValueError:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="invalid_entity_type",
                    description=f"无效的实体类型: {entity_data['type']}",
                    entity_id=entity_data.get("canonical_id"),
                    suggestions=["使用有效的实体类型"]
                ))

        # 3. 验证置信度
        confidence = entity_data.get("confidence", 0.0)
        if confidence < self.config.min_entity_confidence:
            issues.append(ValidationIssue(
                severity="warning",
                issue_type="low_confidence",
                description=f"实体置信度过低: {confidence}",
                entity_id=entity_data.get("canonical_id"),
                suggestions=["提高实体识别准确度", "使用LLM增强抽取"]
            ))

        # 4. 验证实体名称
        if "name" in entity_data:
            name = entity_data["name"]
            if len(name) < 2:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="invalid_name",
                    description=f"实体名称过短: '{name}'",
                    entity_id=entity_data.get("canonical_id"),
                    suggestions=["使用完整的实体名称"]
                ))
            elif len(name) > 100:
                issues.append(ValidationIssue(
                    severity="warning",
                    issue_type="invalid_name",
                    description=f"实体名称过长: {len(name)} 字符",
                    entity_id=entity_data.get("canonical_id"),
                    suggestions=["缩短实体名称或使用简称"]
                ))

        # 5. 验证提及次数
        mention_count = entity_data.get("mention_count", 0)
        if mention_count < self.config.min_entity_mentions:
            issues.append(ValidationIssue(
                severity="info",
                issue_type="low_mention_count",
                description=f"实体提及次数较少: {mention_count}",
                entity_id=entity_data.get("canonical_id"),
                suggestions=["考虑合并相似实体", "检查是否为误识别"]
            ))

        # 确定质量等级
        quality_level = self._determine_quality_level(issues)

        return quality_level, issues

    async def validate_relation(
        self,
        relation_data: Dict
    ) -> Tuple[QualityLevel, List[ValidationIssue]]:
        """
        验证关系质量

        Args:
            relation_data: 关系数据

        Returns:
            (质量等级, 问题列表)
        """
        issues = []

        # 1. 验证必填字段
        required_fields = ["subject", "object", "relation_type"]
        for field in required_fields:
            if field not in relation_data or not relation_data[field]:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="missing_required_field",
                    description=f"缺少必填字段: {field}",
                    relation_id=relation_data.get("id"),
                    suggestions=[f"添加 {field} 字段"]
                ))

        # 2. 验证关系类型
        if "relation_type" in relation_data:
            try:
                rel_type = GraphRelationType(relation_data["relation_type"])
            except ValueError:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="invalid_relation_type",
                    description=f"无效的关系类型: {relation_data['relation_type']}",
                    relation_id=relation_data.get("id"),
                    suggestions=["使用有效的关系类型"]
                ))

        # 3. 验证置信度
        confidence = relation_data.get("confidence", 0.0)
        if confidence < self.config.min_relation_confidence:
            issues.append(ValidationIssue(
                severity="warning",
                issue_type="low_confidence",
                description=f"关系置信度过低: {confidence}",
                relation_id=relation_data.get("id"),
                suggestions=["使用LLM增强抽取", "增加证据支持"]
            ))

        # 4. 验证主体和客体不同
        if "subject" in relation_data and "object" in relation_data:
            if relation_data["subject"] == relation_data["object"]:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="self_relation",
                    description="关系的主体和客体相同",
                    relation_id=relation_data.get("id"),
                    suggestions=["检查实体抽取结果", "修正关系方向"]
                ))

        # 5. 验证关系方向性
        if "direction" in relation_data:
            direction = relation_data["direction"]
            if direction == "unknown":
                issues.append(ValidationIssue(
                    severity="info",
                    issue_type="unknown_direction",
                    description="关系方向未确定",
                    relation_id=relation_data.get("id"),
                    suggestions=["使用依存句法分析", "LLM判断方向"]
                ))

        # 6. 验证证据文本
        if "evidence" in relation_data:
            evidence = relation_data["evidence"]
            if len(evidence) < 5:
                issues.append(ValidationIssue(
                    severity="warning",
                    issue_type="weak_evidence",
                    description=f"证据文本过短: '{evidence}'",
                    relation_id=relation_data.get("id"),
                    suggestions=["提供更完整的证据上下文"]
                ))

        # 确定质量等级
        quality_level = self._determine_quality_level(issues)

        return quality_level, issues

    async def validate_graph_consistency(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> List[ValidationIssue]:
        """
        验证图谱一致性

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            问题列表
        """
        issues = []

        # 1. 检测孤立节点
        if self.config.remove_isolated_nodes:
            connected_entity_ids = set()
            for rel in relations:
                connected_entity_ids.add(rel.get("subject"))
                connected_entity_ids.add(rel.get("object"))

            for entity in entities:
                entity_id = entity.get("canonical_id")
                if entity_id and entity_id not in connected_entity_ids:
                    mention_count = entity.get("mention_count", 0)
                    if mention_count < self.config.min_connections:
                        issues.append(ValidationIssue(
                            severity="info",
                            issue_type="isolated_entity",
                            description=f"孤立实体: {entity.get('name')}",
                            entity_id=entity_id,
                            suggestions=["检查是否需要移除", "检查关系抽取是否遗漏"]
                        ))

        # 2. 检测冲突关系
        if self.config.detect_conflicting_relations:
            conflicts = await self._detect_conflicting_relations(relations)
            issues.extend(conflicts)

        # 3. 检测重复实体
        duplicates = await self._detect_duplicate_entities(entities)
        issues.extend(duplicates)

        # 4. 检测重复关系
        duplicate_rels = await self._detect_duplicate_relations(relations)
        issues.extend(duplicate_rels)

        return issues

    async def _detect_conflicting_relations(
        self,
        relations: List[Dict]
    ) -> List[ValidationIssue]:
        """检测冲突关系"""
        issues = []

        # 构建实体对的关系索引
        rel_index = {}
        for rel in relations:
            subject = rel.get("subject")
            obj = rel.get("object")
            rel_type = rel.get("relation_type")

            if not (subject and obj and rel_type):
                continue

            key = (subject, obj)
            if key not in rel_index:
                rel_index[key] = []
            rel_index[key].append(rel)

        # 检查冲突
        for (subject, obj), rels in rel_index.items():
            if len(rels) < 2:
                continue

            # 检查关系类型对是否冲突
            for i, rel1 in enumerate(rels):
                for rel2 in rels[i+1:]:
                    type1 = rel1.get("relation_type")
                    type2 = rel2.get("relation_type")

                    if not (type1 and type2):
                        continue

                    # 检查是否是冲突的关系对
                    conflict_key = (
                        GraphRelationType(type1),
                        GraphRelationType(type2)
                    )
                    reverse_conflict_key = (
                        GraphRelationType(type2),
                        GraphRelationType(type1)
                    )

                    if conflict_key in self.conflict_rules:
                        issues.append(ValidationIssue(
                            severity="warning",
                            issue_type="conflicting_relations",
                            description=self.conflict_rules[conflict_key],
                            relation_id=rel1.get("id"),
                            suggestions=[
                                f"选择置信度更高的关系",
                                f"检查关系类型: {type1} vs {type2}"
                            ]
                        ))
                    elif reverse_conflict_key in self.conflict_rules:
                        issues.append(ValidationIssue(
                            severity="warning",
                            issue_type="conflicting_relations",
                            description=self.conflict_rules[reverse_conflict_key],
                            relation_id=rel1.get("id"),
                            suggestions=[
                                f"选择置信度更高的关系",
                                f"检查关系类型: {type1} vs {type2}"
                            ]
                        ))

        return issues

    async def _detect_duplicate_entities(
        self,
        entities: List[Dict]
    ) -> List[ValidationIssue]:
        """检测重复实体"""
        issues = []

        # 按名称分组
        name_groups = {}
        for entity in entities:
            name = entity.get("name", "").strip().lower()
            if not name:
                continue

            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(entity)

        # 报告重复
        for name, group in name_groups.items():
            if len(group) > 1:
                # 检查是否是不同的canonical_id（真正的重复）
                canonical_ids = set(e.get("canonical_id") for e in group)
                if len(canonical_ids) > 1:
                    issues.append(ValidationIssue(
                        severity="warning",
                        issue_type="duplicate_entities",
                        description=f"发现重复实体: {name} ({len(group)}个)",
                        suggestions=["合并重复实体", "检查实体消歧逻辑"]
                    ))

        return issues

    async def _detect_duplicate_relations(
        self,
        relations: List[Dict]
    ) -> List[ValidationIssue]:
        """检测重复关系"""
        issues = []

        # 按主体-客体-类型分组
        rel_groups = {}
        for rel in relations:
            subject = rel.get("subject")
            obj = rel.get("object")
            rel_type = rel.get("relation_type")

            if not (subject and obj and rel_type):
                continue

            key = (subject, obj, rel_type)
            if key not in rel_groups:
                rel_groups[key] = []
            rel_groups[key].append(rel)

        # 报告重复
        for key, group in rel_groups.items():
            if len(group) > 1:
                issues.append(ValidationIssue(
                    severity="info",
                    issue_type="duplicate_relations",
                    description=f"发现重复关系: {key[0]} -> {key[1]} ({key[2]}) ({len(group)}个)",
                    suggestions=["合并重复关系", "保留置信度最高的"]
                ))

        return issues

    def _determine_quality_level(
        self,
        issues: List[ValidationIssue]
    ) -> QualityLevel:
        """根据问题确定质量等级"""
        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")

        if error_count > 0:
            return QualityLevel.INVALID
        elif warning_count >= 3:
            return QualityLevel.LOW
        elif warning_count >= 1:
            return QualityLevel.MEDIUM
        else:
            return QualityLevel.HIGH

    async def resolve_conflicts(
        self,
        relations: List[Dict],
        strategy: str = "highest_confidence"
    ) -> List[Dict]:
        """
        解决关系冲突

        Args:
            relations: 关系列表
            strategy: 解决策略 (highest_confidence, most_recent, keep_all)

        Returns:
            解决冲突后的关系列表
        """
        if strategy == "keep_all":
            return relations

        # 按主体-客体分组
        rel_groups = {}
        for rel in relations:
            subject = rel.get("subject")
            obj = rel.get("object")

            if not (subject and obj):
                continue

            key = (subject, obj)
            if key not in rel_groups:
                rel_groups[key] = []
            rel_groups[key].append(rel)

        resolved = []

        for key, group in rel_groups.items():
            if len(group) == 1:
                resolved.extend(group)
            else:
                # 解决冲突
                if strategy == "highest_confidence":
                    # 选择置信度最高的
                    best = max(group, key=lambda r: r.get("confidence", 0.0))
                    resolved.append(best)
                elif strategy == "most_recent":
                    # 选择最新的
                    best = max(
                        group,
                        key=lambda r: r.get("created_at", "")
                    )
                    resolved.append(best)

        return resolved

    async def generate_quality_report(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> Dict:
        """
        生成质量报告

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            质量报告字典
        """
        # 验证所有实体
        entity_quality_stats = {
            QualityLevel.HIGH: 0,
            QualityLevel.MEDIUM: 0,
            QualityLevel.LOW: 0,
            QualityLevel.INVALID: 0
        }

        all_entity_issues = []
        for entity in entities:
            quality_level, issues = await self.validate_entity(entity)
            entity_quality_stats[quality_level] += 1
            all_entity_issues.extend(issues)

        # 验证所有关系
        relation_quality_stats = {
            QualityLevel.HIGH: 0,
            QualityLevel.MEDIUM: 0,
            QualityLevel.LOW: 0,
            QualityLevel.INVALID: 0
        }

        all_relation_issues = []
        for relation in relations:
            quality_level, issues = await self.validate_relation(relation)
            relation_quality_stats[quality_level] += 1
            all_relation_issues.extend(issues)

        # 验证图谱一致性
        consistency_issues = await self.validate_graph_consistency(
            entities, relations
        )

        # 统计问题
        error_count = sum(
            1 for i in all_entity_issues + all_relation_issues + consistency_issues
            if i.severity == "error"
        )
        warning_count = sum(
            1 for i in all_entity_issues + all_relation_issues + consistency_issues
            if i.severity == "warning"
        )
        info_count = sum(
            1 for i in all_entity_issues + all_relation_issues + consistency_issues
            if i.severity == "info"
        )

        # 计算总体质量分数
        total_items = len(entities) + len(relations)
        high_quality_count = (
            entity_quality_stats[QualityLevel.HIGH] +
            relation_quality_stats[QualityLevel.HIGH]
        )
        quality_score = (
            high_quality_count / total_items * 100
            if total_items > 0 else 0
        )

        return {
            "summary": {
                "total_entities": len(entities),
                "total_relations": len(relations),
                "quality_score": round(quality_score, 2),
                "error_count": error_count,
                "warning_count": warning_count,
                "info_count": info_count
            },
            "entity_quality": {
                "high": entity_quality_stats[QualityLevel.HIGH],
                "medium": entity_quality_stats[QualityLevel.MEDIUM],
                "low": entity_quality_stats[QualityLevel.LOW],
                "invalid": entity_quality_stats[QualityLevel.INVALID]
            },
            "relation_quality": {
                "high": relation_quality_stats[QualityLevel.HIGH],
                "medium": relation_quality_stats[QualityLevel.MEDIUM],
                "low": relation_quality_stats[QualityLevel.LOW],
                "invalid": relation_quality_stats[QualityLevel.INVALID]
            },
            "consistency_issues": len(consistency_issues),
            "recommendations": self._generate_recommendations(
                all_entity_issues,
                all_relation_issues,
                consistency_issues
            )
        }

    def _generate_recommendations(
        self,
        entity_issues: List[ValidationIssue],
        relation_issues: List[ValidationIssue],
        consistency_issues: List[ValidationIssue]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        all_issues = entity_issues + relation_issues + consistency_issues

        # 统计问题类型
        issue_types = {}
        for issue in all_issues:
            issue_type = issue.issue_type
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1

        # 生成建议
        if "low_confidence" in issue_types:
            count = issue_types["low_confidence"]
            recommendations.append(
                f"发现 {count} 个低置信度实体/关系，建议使用LLM增强抽取"
            )

        if "duplicate_entities" in issue_types:
            count = issue_types["duplicate_entities"]
            recommendations.append(
                f"发现 {count} 组重复实体，建议启用实体消歧"
            )

        if "conflicting_relations" in issue_types:
            count = issue_types["conflicting_relations"]
            recommendations.append(
                f"发现 {count} 组冲突关系，建议使用置信度最高的"
            )

        if "isolated_entity" in issue_types:
            count = issue_types["isolated_entity"]
            recommendations.append(
                f"发现 {count} 个孤立实体，建议检查是否需要移除"
            )

        if "unknown_direction" in issue_types:
            count = issue_types["unknown_direction"]
            recommendations.append(
                f"发现 {count} 个方向未确定的关系，建议使用依存句法分析"
            )

        if not recommendations:
            recommendations.append("图谱质量良好，无需特殊改进")

        return recommendations


# 全局实例
graph_quality_validator = GraphQualityValidator()
