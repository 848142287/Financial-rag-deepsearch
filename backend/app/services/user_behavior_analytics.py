"""
用户行为分析和个性化推荐系统
通过分析用户行为模式，提供个性化的金融内容推荐
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict, Counter
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import pandas as pd

from .data_balancer import DocumentCategory
from .financial_llm_service import financial_llm_service

logger = logging.getLogger(__name__)


class UserActionType(Enum):
    """用户行为类型"""
    SEARCH = "search"  # 搜索
    CLICK = "click"  # 点击
    VIEW = "view"  # 查看
    LIKE = "like"  # 点赞
    SHARE = "share"  # 分享
    DOWNLOAD = "download"  # 下载
    BOOKMARK = "bookmark"  # 收藏
    FEEDBACK = "feedback"  # 反馈


class RecommendationStrategy(Enum):
    """推荐策略"""
    CONTENT_BASED = "content_based"  # 基于内容
    COLLABORATIVE = "collaborative"  # 协同过滤
    HYBRID = "hybrid"  # 混合推荐
    POPULARITY = "popularity"  # 热门推荐
    PERSONALIZED = "personalized"  # 个性化推荐


@dataclass
class UserAction:
    """用户行为记录"""
    user_id: str
    action_type: UserActionType
    document_id: str
    timestamp: datetime
    query: Optional[str] = None
    duration: Optional[int] = None  # 查看时长（秒）
    rating: Optional[int] = None  # 用户评分 (1-5)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preferences: Dict[str, float]  # 偏好分数
    interests: List[str]  # 兴趣标签
    expertise_level: str  # 专业程度: beginner, intermediate, expert
    search_history: List[str]  # 搜索历史
    behavior_patterns: Dict[str, Any]  # 行为模式
    created_at: datetime
    updated_at: datetime


@dataclass
class Recommendation:
    """推荐结果"""
    document_id: str
    score: float
    reason: str  # 推荐原因
    strategy: str  # 推荐策略
    metadata: Optional[Dict[str, Any]] = None


class UserBehaviorAnalytics:
    """用户行为分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化用户行为分析器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 存储用户行为数据
        self.user_actions: List[UserAction] = []
        self.user_profiles: Dict[str, UserProfile] = {}

        # 推荐模型
        self.content_vectorizer = None
        self.user_item_matrix = None
        self.topic_model = None

        # 配置参数
        self.min_actions_for_profile = self.config.get("min_actions_for_profile", 5)
        self.profile_update_threshold = self.config.get("profile_update_threshold", 10)
        self.recommendation_cache_ttl = self.config.get("recommendation_cache_ttl", 3600)  # 1小时

        # 个性化权重
        self.weights = {
            "recent_weight": 0.4,  # 最近行为权重
            "frequency_weight": 0.3,  # 频率权重
            "duration_weight": 0.2,  # 时长权重
            "rating_weight": 0.1  # 评分权重
        }

    async def initialize(self):
        """初始化分析器"""
        try:
            logger.info("初始化用户行为分析器")

            # 初始化TF-IDF向量化器
            self.content_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            # 初始化主题模型
            self.topic_model = NMF(n_components=10, random_state=42)

            logger.info("用户行为分析器初始化完成")

        except Exception as e:
            logger.error(f"用户行为分析器初始化失败: {str(e)}")
            raise

    async def record_action(self, action: UserAction) -> bool:
        """
        记录用户行为

        Args:
            action: 用户行为

        Returns:
            是否记录成功
        """
        try:
            self.user_actions.append(action)

            # 更新用户画像
            await self._update_user_profile(action.user_id)

            logger.debug(f"记录用户行为: {action.user_id}, {action.action_type.value}")
            return True

        except Exception as e:
            logger.error(f"用户行为记录失败: {str(e)}")
            return False

    async def _update_user_profile(self, user_id: str):
        """更新用户画像"""
        try:
            # 获取用户的所有行为
            user_actions = [
                action for action in self.user_actions
                if action.user_id == user_id
            ]

            if len(user_actions) < self.min_actions_for_profile:
                return  # 行为数量不足，不更新画像

            # 计算用户偏好
            preferences = await self._calculate_user_preferences(user_actions)

            # 提取兴趣标签
            interests = await self._extract_user_interests(user_actions)

            # 评估专业程度
            expertise_level = await self._assess_expertise_level(user_actions)

            # 获取搜索历史
            search_history = [
                action.query for action in user_actions
                if action.action_type == UserActionType.SEARCH and action.query
            ]

            # 分析行为模式
            behavior_patterns = await self._analyze_behavior_patterns(user_actions)

            # 创建或更新用户画像
            now = datetime.now()
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                profile.preferences.update(preferences)
                profile.interests = interests
                profile.expertise_level = expertise_level
                profile.search_history = search_history[-50:]  # 保留最近50条
                profile.behavior_patterns = behavior_patterns
                profile.updated_at = now
            else:
                profile = UserProfile(
                    user_id=user_id,
                    preferences=preferences,
                    interests=interests,
                    expertise_level=expertise_level,
                    search_history=search_history,
                    behavior_patterns=behavior_patterns,
                    created_at=now,
                    updated_at=now
                )
                self.user_profiles[user_id] = profile

            logger.info(f"用户画像更新完成: {user_id}")

        except Exception as e:
            logger.error(f"用户画像更新失败: {str(e)}")

    async def _calculate_user_preferences(self, user_actions: List[UserAction]) -> Dict[str, float]:
        """计算用户偏好"""
        try:
            preferences = defaultdict(float)

            # 按文档类别统计
            category_scores = defaultdict(float)
            category_counts = defaultdict(int)

            for action in user_actions:
                # 获取文档类别（简化实现）
                category = await self._get_document_category(action.document_id)
                if category:

                    # 根据行为类型加权
                    weight = self._get_action_weight(action)

                    # 时间衰减
                    time_weight = self._calculate_time_weight(action.timestamp)

                    # 综合权重
                    total_weight = weight * time_weight

                    category_scores[category] += total_weight
                    category_counts[category] += 1

            # 归一化偏好分数
            total_score = sum(category_scores.values())
            if total_score > 0:
                for category, score in category_scores.items():
                    preferences[category] = score / total_score

            return dict(preferences)

        except Exception as e:
            logger.error(f"用户偏好计算失败: {str(e)}")
            return {}

    async def _extract_user_interests(self, user_actions: List[UserAction]) -> List[str]:
        """提取用户兴趣标签"""
        try:
            interests = []

            # 从搜索查询中提取关键词
            search_queries = [
                action.query for action in user_actions
                if action.action_type == UserActionType.SEARCH and action.query
            ]

            if search_queries:
                # 使用金融模型提取关键词
                all_queries = " ".join(search_queries)
                keyword_result = await financial_llm_service.extract_keywords(all_queries, top_k=20)

                interests = [kw[0] for kw in keyword_result.result]

            return interests

        except Exception as e:
            logger.error(f"兴趣标签提取失败: {str(e)}")
            return []

    async def _assess_expertise_level(self, user_actions: List[UserAction]) -> str:
        """评估用户专业程度"""
        try:
            # 专业程度评估指标
            indicators = {
                "search_complexity": 0,  # 搜索复杂度
                "document_diversity": 0,  # 文档多样性
                "interaction_depth": 0,  # 交互深度
                "time_spent": 0  # 花费时间
            }

            # 搜索复杂度：基于查询长度和专业词汇
            search_actions = [
                action for action in user_actions
                if action.action_type == UserActionType.SEARCH and action.query
            ]

            if search_actions:
                avg_query_length = np.mean([len(action.query.split()) for action in search_actions])
                indicators["search_complexity"] = min(avg_query_length / 10, 1.0)

            # 文档多样性：基于访问的不同类别数量
            categories = set()
            for action in user_actions:
                category = await self._get_document_category(action.document_id)
                if category:
                    categories.add(category)

            indicators["document_diversity"] = min(len(categories) / 5, 1.0)

            # 交互深度：基于深度行为（收藏、分享、下载等）
            deep_actions = [
                action for action in user_actions
                if action.action_type in [UserActionType.BOOKMARK, UserActionType.SHARE, UserActionType.DOWNLOAD]
            ]
            indicators["interaction_depth"] = min(len(deep_actions) / len(user_actions), 1.0)

            # 时间花费：基于平均查看时长
            view_actions = [
                action for action in user_actions
                if action.action_type == UserActionType.VIEW and action.duration
            ]

            if view_actions:
                avg_duration = np.mean([action.duration for action in view_actions])
                indicators["time_spent"] = min(avg_duration / 300, 1.0)  # 5分钟为满分

            # 综合评分
            total_score = sum(indicators.values()) / len(indicators)

            if total_score >= 0.7:
                return "expert"
            elif total_score >= 0.4:
                return "intermediate"
            else:
                return "beginner"

        except Exception as e:
            logger.error(f"专业程度评估失败: {str(e)}")
            return "beginner"

    async def _analyze_behavior_patterns(self, user_actions: List[UserAction]) -> Dict[str, Any]:
        """分析用户行为模式"""
        try:
            patterns = {}

            # 活跃时间段
            hours = [action.timestamp.hour for action in user_actions]
            hour_counter = Counter(hours)
            most_active_hour = hour_counter.most_common(1)[0][0] if hour_counter else 9

            patterns["most_active_hour"] = most_active_hour

            # 访问频率
            if len(user_actions) > 1:
                time_diffs = []
                sorted_actions = sorted(user_actions, key=lambda x: x.timestamp)

                for i in range(1, len(sorted_actions)):
                    diff = (sorted_actions[i].timestamp - sorted_actions[i-1].timestamp).total_seconds() / 3600
                    time_diffs.append(diff)

                patterns["avg_visit_interval"] = np.mean(time_diffs) if time_diffs else 0

            # 偏好的行为类型
            action_types = [action.action_type.value for action in user_actions]
            action_counter = Counter(action_types)
            patterns["preferred_actions"] = action_counter.most_common(3)

            return patterns

        except Exception as e:
            logger.error(f"行为模式分析失败: {str(e)}")
            return {}

    def _get_action_weight(self, action: UserAction) -> float:
        """获取行为权重"""
        action_weights = {
            UserActionType.SEARCH: 0.1,
            UserActionType.VIEW: 0.2,
            UserActionType.CLICK: 0.3,
            UserActionType.LIKE: 0.4,
            UserActionType.SHARE: 0.5,
            UserActionType.DOWNLOAD: 0.6,
            UserActionType.BOOKMARK: 0.7,
            UserActionType.FEEDBACK: 0.8
        }

        base_weight = action_weights.get(action.action_type, 0.1)

        # 根据评分调整
        if action.rating:
            rating_weight = action.rating / 5.0
            base_weight *= (0.5 + 0.5 * rating_weight)

        # 根据查看时长调整
        if action.duration:
            duration_weight = min(action.duration / 300, 1.0)  # 5分钟为满分
            base_weight *= (0.5 + 0.5 * duration_weight)

        return base_weight

    def _calculate_time_weight(self, timestamp: datetime) -> float:
        """计算时间权重（时间衰减）"""
        now = datetime.now()
        days_diff = (now - timestamp).days

        # 指数衰减
        decay_rate = 0.1
        return np.exp(-decay_rate * days_diff)

    async def _get_document_category(self, document_id: str) -> Optional[str]:
        """获取文档类别（简化实现）"""
        # 这里应该从数据库或文档元数据中获取
        # 现在返回模拟数据
        categories = ["年报", "季报", "研究报告", "新闻分析", "政策文件"]
        import hashlib
        index = int(hashlib.md5(document_id.encode()).hexdigest(), 16) % len(categories)
        return categories[index]

    async def get_recommendations(
        self,
        user_id: str,
        strategy: RecommendationStrategy = RecommendationStrategy.HYBRID,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Recommendation]:
        """
        获取个性化推荐

        Args:
            user_id: 用户ID
            strategy: 推荐策略
            top_k: 推荐数量
            filters: 过滤条件

        Returns:
            推荐结果列表
        """
        try:
            logger.info(f"为用户{user_id}生成推荐，策略: {strategy.value}")

            if user_id not in self.user_profiles:
                # 新用户，返回热门推荐
                return await self._get_popular_recommendations(top_k)

            user_profile = self.user_profiles[user_id]

            if strategy == RecommendationStrategy.CONTENT_BASED:
                recommendations = await self._content_based_recommendations(user_profile, top_k)
            elif strategy == RecommendationStrategy.COLLABORATIVE:
                recommendations = await self._collaborative_recommendations(user_profile, top_k)
            elif strategy == RecommendationStrategy.HYBRID:
                recommendations = await self._hybrid_recommendations(user_profile, top_k)
            elif strategy == RecommendationStrategy.POPULARITY:
                recommendations = await self._get_popular_recommendations(top_k)
            else:  # PERSONALIZED
                recommendations = await self._personalized_recommendations(user_profile, top_k)

            # 应用过滤条件
            if filters:
                recommendations = self._apply_recommendation_filters(recommendations, filters)

            logger.info(f"生成推荐完成: {len(recommendations)}个结果")
            return recommendations

        except Exception as e:
            logger.error(f"推荐生成失败: {str(e)}")
            return []

    async def _content_based_recommendations(
        self,
        user_profile: UserProfile,
        top_k: int
    ) -> List[Recommendation]:
        """基于内容的推荐"""
        try:
            recommendations = []

            # 基于用户兴趣标签推荐
            for interest in user_profile.interests:
                # 模拟相关文档搜索
                related_docs = await self._search_related_documents(interest, limit=3)
                for doc_id, score in related_docs:
                    recommendation = Recommendation(
                        document_id=doc_id,
                        score=score,
                        reason=f"基于您的兴趣: {interest}",
                        strategy="content_based",
                        metadata={"interest": interest}
                    )
                    recommendations.append(recommendation)

            # 基于用户偏好类别推荐
            for category, preference_score in user_profile.preferences.items():
                if preference_score > 0.1:  # 偏好阈值
                    related_docs = await self._search_category_documents(category, limit=2)
                    for doc_id, score in related_docs:
                        recommendation = Recommendation(
                            document_id=doc_id,
                            score=score * preference_score,
                            reason=f"基于您的偏好: {category}",
                            strategy="content_based",
                            metadata={"category": category}
                        )
                        recommendations.append(recommendation)

            # 按分数排序并返回top_k
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"基于内容推荐失败: {str(e)}")
            return []

    async def _collaborative_recommendations(
        self,
        user_profile: UserProfile,
        top_k: int
    ) -> List[Recommendation]:
        """协同过滤推荐"""
        try:
            recommendations = []

            # 找到相似用户
            similar_users = await self._find_similar_users(user_profile.user_id)

            # 获取相似用户喜欢的文档
            candidate_docs = defaultdict(float)
            for similar_user_id, similarity_score in similar_users:
                similar_user_actions = [
                    action for action in self.user_actions
                    if action.user_id == similar_user_id and
                       action.action_type in [UserActionType.LIKE, UserActionType.BOOKMARK]
                ]

                for action in similar_user_actions:
                    candidate_docs[action.document_id] += similarity_score * self._get_action_weight(action)

            # 过滤用户已看过的文档
            user_viewed_docs = {
                action.document_id for action in self.user_actions
                if action.user_id == user_profile.user_id
            }

            # 生成推荐
            for doc_id, score in candidate_docs.items():
                if doc_id not in user_viewed_docs:
                    recommendation = Recommendation(
                        document_id=doc_id,
                        score=score,
                        reason="相似用户也喜欢",
                        strategy="collaborative",
                        metadata={"similar_users": len(similar_users)}
                    )
                    recommendations.append(recommendation)

            # 按分数排序并返回top_k
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"协同过滤推荐失败: {str(e)}")
            return []

    async def _hybrid_recommendations(
        self,
        user_profile: UserProfile,
        top_k: int
    ) -> List[Recommendation]:
        """混合推荐"""
        try:
            # 获取多种推荐策略的结果
            content_recs = await self._content_based_recommendations(user_profile, top_k * 2)
            collaborative_recs = await self._collaborative_recommendations(user_profile, top_k * 2)
            popularity_recs = await self._get_popular_recommendations(top_k)

            # 合并和重新评分
            all_recommendations = defaultdict(dict)

            # 添加基于内容的推荐（权重0.4）
            for rec in content_recs:
                if rec.document_id not in all_recommendations:
                    all_recommendations[rec.document_id] = {
                        "document_id": rec.document_id,
                        "score": 0,
                        "reasons": [],
                        "strategy": "hybrid"
                    }
                all_recommendations[rec.document_id]["score"] += rec.score * 0.4
                all_recommendations[rec.document_id]["reasons"].append(rec.reason)

            # 添加协同过滤推荐（权重0.4）
            for rec in collaborative_recs:
                if rec.document_id not in all_recommendations:
                    all_recommendations[rec.document_id] = {
                        "document_id": rec.document_id,
                        "score": 0,
                        "reasons": [],
                        "strategy": "hybrid"
                    }
                all_recommendations[rec.document_id]["score"] += rec.score * 0.4
                all_recommendations[rec.document_id]["reasons"].append(rec.reason)

            # 添加热门推荐（权重0.2）
            for rec in popularity_recs:
                if rec.document_id not in all_recommendations:
                    all_recommendations[rec.document_id] = {
                        "document_id": rec.document_id,
                        "score": 0,
                        "reasons": [],
                        "strategy": "hybrid"
                    }
                all_recommendations[rec.document_id]["score"] += rec.score * 0.2
                all_recommendations[rec.document_id]["reasons"].append(rec.reason)

            # 生成最终推荐列表
            final_recommendations = []
            for doc_id, rec_data in all_recommendations.items():
                recommendation = Recommendation(
                    document_id=rec_data["document_id"],
                    score=rec_data["score"],
                    reason="; ".join(rec_data["reasons"]),
                    strategy=rec_data["strategy"]
                )
                final_recommendations.append(recommendation)

            # 按分数排序并返回top_k
            final_recommendations.sort(key=lambda x: x.score, reverse=True)
            return final_recommendations[:top_k]

        except Exception as e:
            logger.error(f"混合推荐失败: {str(e)}")
            return []

    async def _personalized_recommendations(
        self,
        user_profile: UserProfile,
        top_k: int
    ) -> List[Recommendation]:
        """个性化推荐"""
        try:
            recommendations = []

            # 根据专业程度调整推荐策略
            if user_profile.expertise_level == "expert":
                # 专家用户：推荐深度分析和专业报告
                expert_docs = await self._search_expert_documents(top_k)
                for doc_id, score in expert_docs:
                    recommendation = Recommendation(
                        document_id=doc_id,
                        score=score,
                        reason="专业深度分析",
                        strategy="personalized",
                        metadata={"expertise_level": user_profile.expertise_level}
                    )
                    recommendations.append(recommendation)

            elif user_profile.expertise_level == "intermediate":
                # 中级用户：平衡基础和进阶内容
                intermediate_docs = await self._search_intermediate_documents(top_k)
                for doc_id, score in intermediate_docs:
                    recommendation = Recommendation(
                        document_id=doc_id,
                        score=score,
                        reason="进阶学习内容",
                        strategy="personalized",
                        metadata={"expertise_level": user_profile.expertise_level}
                    )
                    recommendations.append(recommendation)

            else:  # beginner
                # 初级用户：推荐基础知识和入门材料
                beginner_docs = await self._search_beginner_documents(top_k)
                for doc_id, score in beginner_docs:
                    recommendation = Recommendation(
                        document_id=doc_id,
                        score=score,
                        reason="基础知识入门",
                        strategy="personalized",
                        metadata={"expertise_level": user_profile.expertise_level}
                    )
                    recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"个性化推荐失败: {str(e)}")
            return []

    async def _get_popular_recommendations(self, top_k: int) -> List[Recommendation]:
        """获取热门推荐"""
        try:
            # 统计文档热度
            doc_popularity = defaultdict(float)

            for action in self.user_actions:
                weight = self._get_action_weight(action)
                doc_popularity[action.document_id] += weight

            # 按热度排序
            sorted_docs = sorted(doc_popularity.items(), key=lambda x: x[1], reverse=True)

            recommendations = []
            for doc_id, popularity_score in sorted_docs[:top_k]:
                recommendation = Recommendation(
                    document_id=doc_id,
                    score=popularity_score,
                    reason="热门内容",
                    strategy="popularity"
                )
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"热门推荐失败: {str(e)}")
            return []

    async def _find_similar_users(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """找到相似用户"""
        try:
            if user_id not in self.user_profiles:
                return []

            target_profile = self.user_profiles[user_id]
            similarities = []

            for other_id, other_profile in self.user_profiles.items():
                if other_id == user_id:
                    continue

                # 计算偏好相似度
                similarity = self._calculate_profile_similarity(target_profile, other_profile)
                if similarity > 0.1:  # 相似度阈值
                    similarities.append((other_id, similarity))

            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"相似用户查找失败: {str(e)}")
            return []

    def _calculate_profile_similarity(self, profile1: UserProfile, profile2: UserProfile) -> float:
        """计算用户画像相似度"""
        try:
            # 偏好相似度
            common_categories = set(profile1.preferences.keys()) & set(profile2.preferences.keys())
            if not common_categories:
                return 0.0

            preference_similarity = 0.0
            for category in common_categories:
                pref1 = profile1.preferences[category]
                pref2 = profile2.preferences[category]
                preference_similarity += 1 - abs(pref1 - pref2)

            preference_similarity /= len(common_categories)

            # 兴趣相似度
            interest1 = set(profile1.interests)
            interest2 = set(profile2.interests)
            interest_similarity = len(interest1 & interest2) / len(interest1 | interest2) if interest1 | interest2 else 0.0

            # 专业程度相似度
            expertise_similarity = 1.0 if profile1.expertise_level == profile2.expertise_level else 0.5

            # 综合相似度
            total_similarity = (
                preference_similarity * 0.5 +
                interest_similarity * 0.3 +
                expertise_similarity * 0.2
            )

            return total_similarity

        except Exception as e:
            logger.error(f"画像相似度计算失败: {str(e)}")
            return 0.0

    # 模拟方法（实际实现中应该连接真实数据源）
    async def _search_related_documents(self, interest: str, limit: int) -> List[Tuple[str, float]]:
        """搜索相关文档"""
        # 模拟实现
        import random
        return [(f"doc_{i}", random.uniform(0.5, 1.0)) for i in range(limit)]

    async def _search_category_documents(self, category: str, limit: int) -> List[Tuple[str, float]]:
        """搜索类别文档"""
        # 模拟实现
        import random
        return [(f"doc_{category}_{i}", random.uniform(0.4, 0.9)) for i in range(limit)]

    async def _search_expert_documents(self, limit: int) -> List[Tuple[str, float]]:
        """搜索专家文档"""
        # 模拟实现
        import random
        return [(f"expert_doc_{i}", random.uniform(0.6, 1.0)) for i in range(limit)]

    async def _search_intermediate_documents(self, limit: int) -> List[Tuple[str, float]]:
        """搜索中级文档"""
        # 模拟实现
        import random
        return [(f"intermediate_doc_{i}", random.uniform(0.5, 0.9)) for i in range(limit)]

    async def _search_beginner_documents(self, limit: int) -> List[Tuple[str, float]]:
        """搜索初级文档"""
        # 模拟实现
        import random
        return [(f"beginner_doc_{i}", random.uniform(0.4, 0.8)) for i in range(limit)]

    def _apply_recommendation_filters(
        self,
        recommendations: List[Recommendation],
        filters: Dict[str, Any]
    ) -> List[Recommendation]:
        """应用推荐过滤条件"""
        try:
            filtered_recs = []

            for rec in recommendations:
                include = True

                # 类别过滤
                if "category" in filters:
                    # 这里应该检查文档的实际类别
                    pass

                # 分数过滤
                if "min_score" in filters and rec.score < filters["min_score"]:
                    include = False

                # 策略过滤
                if "strategy" in filters and rec.strategy != filters["strategy"]:
                    include = False

                if include:
                    filtered_recs.append(rec)

            return filtered_recs

        except Exception as e:
            logger.error(f"推荐过滤失败: {str(e)}")
            return recommendations

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return self.user_profiles.get(user_id)

    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        try:
            user_actions = [
                action for action in self.user_actions
                if action.user_id == user_id
            ]

            if not user_actions:
                return {}

            # 基本统计
            stats = {
                "total_actions": len(user_actions),
                "action_types": {},
                "active_days": len(set(action.timestamp.date() for action in user_actions)),
                "avg_session_duration": 0
            }

            # 行为类型统计
            action_counter = Counter([action.action_type.value for action in user_actions])
            stats["action_types"] = dict(action_counter)

            # 会话时长统计
            view_actions = [action for action in user_actions if action.duration]
            if view_actions:
                stats["avg_session_duration"] = np.mean([action.duration for action in view_actions])

            return stats

        except Exception as e:
            logger.error(f"用户统计获取失败: {str(e)}")
            return {}

    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            total_users = len(self.user_profiles)
            total_actions = len(self.user_actions)

            # 行为类型分布
            action_types = Counter([action.action_type.value for action in self.user_actions])

            # 活跃用户统计
            active_users = len(set(
                action.user_id for action in self.user_actions
                if action.timestamp > datetime.now() - timedelta(days=7)
            ))

            return {
                "total_users": total_users,
                "total_actions": total_actions,
                "active_users_7d": active_users,
                "action_distribution": dict(action_types),
                "avg_actions_per_user": total_actions / total_users if total_users > 0 else 0
            }

        except Exception as e:
            logger.error(f"系统统计获取失败: {str(e)}")
            return {}


# 全局用户行为分析器实例
user_behavior_analytics = UserBehaviorAnalytics()