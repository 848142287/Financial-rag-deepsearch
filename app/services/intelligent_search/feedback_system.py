"""
用户反馈收集系统

收集、分析和管理用户反馈
支持多种反馈类型和智能分析
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """反馈类型"""
    RATING = "rating"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    COMPARISON = "comparison"
    DETAILED = "detailed"

class FeedbackCategory(Enum):
    """反馈类别"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    USEFULNESS = "usefulness"
    CLARITY = "clarity"
    TIMELINESS = "timeliness"

@dataclass
class FeedbackItem:
    """反馈项"""
    id: str
    user_id: Optional[str]
    session_id: str
    question: str
    selected_mode: str
    feedback_type: FeedbackType
    category: FeedbackCategory
    rating: float  # 1.0-5.0
    confidence: float  # 0.0-1.0
    comments: Optional[str]
    suggested_mode: Optional[str]
    context: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class FeedbackAnalysis:
    """反馈分析结果"""
    total_feedbacks: int
    average_rating: float
    category_scores: Dict[str, float]
    mode_performance: Dict[str, float]
    improvement_areas: List[str]
    strengths: List[str]
    trends: Dict[str, Any]
    recommendations: List[str]

class FeedbackCollector:
    """反馈收集器"""

    def __init__(self, storage_path: str = "feedback_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.feedback_db = self._load_feedback_database()
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1小时缓存

    def _load_feedback_database(self) -> Dict[str, List[FeedbackItem]]:
        """加载反馈数据库"""
        feedback_db = {}

        try:
            for file_path in self.storage_path.glob("*.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item_data in data:
                        feedback = FeedbackItem(**item_data)
                        date_key = feedback.timestamp.strftime("%Y-%m")
                        if date_key not in feedback_db:
                            feedback_db[date_key] = []
                        feedback_db[date_key].append(feedback)

        except Exception as e:
            logger.error(f"Error loading feedback database: {e}")

        return feedback_db

    def collect_feedback(self, question: str, selected_mode: str,
                        feedback_data: Dict[str, Any],
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> str:
        """收集用户反馈"""
        try:
            # 生成唯一ID
            feedback_id = str(uuid.uuid4())

            # 解析反馈数据
            feedback_type = FeedbackType(feedback_data.get("type", "rating"))
            category = FeedbackCategory(feedback_data.get("category", "usefulness"))
            rating = float(feedback_data.get("rating", 3.0))
            confidence = float(feedback_data.get("confidence", 0.5))
            comments = feedback_data.get("comments")
            suggested_mode = feedback_data.get("suggested_mode")
            context = feedback_data.get("context", {})
            metadata = feedback_data.get("metadata", {})

            # 创建反馈项
            feedback = FeedbackItem(
                id=feedback_id,
                user_id=user_id,
                session_id=session_id or f"session_{uuid.uuid4().hex[:8]}",
                question=question,
                selected_mode=selected_mode,
                feedback_type=feedback_type,
                category=category,
                rating=rating,
                confidence=confidence,
                comments=comments,
                suggested_mode=suggested_mode,
                context=context,
                timestamp=datetime.now(),
                metadata=metadata
            )

            # 保存反馈
            self._save_feedback(feedback)

            # 清除缓存
            self._clear_cache()

            logger.info(f"Feedback collected: {feedback_id}")
            return feedback_id

        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            raise

    def _save_feedback(self, feedback: FeedbackItem):
        """保存反馈"""
        date_key = feedback.timestamp.strftime("%Y-%m")

        if date_key not in self.feedback_db:
            self.feedback_db[date_key] = []

        self.feedback_db[date_key].append(feedback)

        # 保存到文件
        month_file = self.storage_path / f"{date_key}.json"
        existing_data = []

        if month_file.exists():
            try:
                with open(month_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading existing feedback file: {e}")

        existing_data.append(asdict(feedback))

        try:
            with open(month_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback to file: {e}")

    def analyze_feedback(self, days_back: int = 30) -> FeedbackAnalysis:
        """分析反馈数据"""
        cache_key = f"analysis_{days_back}"
        current_time = datetime.now()

        # 检查缓存
        if cache_key in self.analysis_cache:
            cached_data = self.analysis_cache[cache_key]
            if (current_time - cached_data["timestamp"]).seconds < self.cache_ttl:
                return cached_data["analysis"]

        # 收集指定时间范围内的反馈
        cutoff_date = current_time - timedelta(days=days_back)
        all_feedbacks = []

        for date_key, feedbacks in self.feedback_db.items():
            # 解析日期键
            try:
                year, month = map(int, date_key.split('-'))
                month_start = datetime(year, month, 1)
                month_end = month_start.replace(day=28) + timedelta(days=4)  # 确保到下个月
                month_end = month_end - timedelta(days=month_end.day - 1)

                if month_end >= cutoff_date:
                    for feedback in feedbacks:
                        if feedback.timestamp >= cutoff_date:
                            all_feedbacks.append(feedback)
            except:
                continue

        # 执行分析
        analysis = self._perform_analysis(all_feedbacks)

        # 缓存结果
        self.analysis_cache[cache_key] = {
            "analysis": analysis,
            "timestamp": current_time
        }

        return analysis

    def _perform_analysis(self, feedbacks: List[FeedbackItem]) -> FeedbackAnalysis:
        """执行反馈分析"""
        if not feedbacks:
            return FeedbackAnalysis(
                total_feedbacks=0,
                average_rating=0.0,
                category_scores={},
                mode_performance={},
                improvement_areas=[],
                strengths=[],
                trends={},
                recommendations=[]
            )

        # 基础统计
        total_feedbacks = len(feedbacks)
        ratings = [f.rating for f in feedbacks]
        average_rating = sum(ratings) / len(ratings)

        # 类别评分
        category_scores = {}
        category_groups = {}
        for feedback in feedbacks:
            category = feedback.category.value
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(feedback.rating)

        for category, ratings in category_groups.items():
            category_scores[category] = sum(ratings) / len(ratings)

        # 模式性能
        mode_performance = {}
        mode_groups = {}
        for feedback in feedbacks:
            mode = feedback.selected_mode
            if mode not in mode_groups:
                mode_groups[mode] = []
            mode_groups[mode].append(feedback.rating)

        for mode, ratings in mode_groups.items():
            mode_performance[mode] = sum(ratings) / len(ratings)

        # 识别改进领域
        improvement_areas = []
        for category, score in category_scores.items():
            if score < 3.0:
                improvement_areas.append(f"{category} (平均评分: {score:.2f})")

        # 识别优势
        strengths = []
        for category, score in category_scores.items():
            if score >= 4.0:
                strengths.append(f"{category} (平均评分: {score:.2f})")

        # 趋势分析
        trends = self._analyze_trends(feedbacks)

        # 生成建议
        recommendations = self._generate_recommendations(category_scores, mode_performance, improvement_areas)

        return FeedbackAnalysis(
            total_feedbacks=total_feedbacks,
            average_rating=average_rating,
            category_scores=category_scores,
            mode_performance=mode_performance,
            improvement_areas=improvement_areas,
            strengths=strengths,
            trends=trends,
            recommendations=recommendations
        )

    def _analyze_trends(self, feedbacks: List[FeedbackItem]) -> Dict[str, Any]:
        """分析趋势"""
        if len(feedbacks) < 10:
            return {"message": "数据不足，无法进行趋势分析"}

        # 按时间分组
        time_groups = {}
        for feedback in feedbacks:
            day_key = feedback.timestamp.strftime("%Y-%m-%d")
            if day_key not in time_groups:
                time_groups[day_key] = []
            time_groups[day_key].append(feedback.rating)

        # 计算趋势
        sorted_days = sorted(time_groups.keys())
        daily_ratings = [sum(time_groups[day]) / len(time_groups[day]) for day in sorted_days]

        # 简单的趋势计算
        if len(daily_ratings) >= 2:
            recent_avg = sum(daily_ratings[-7:]) / min(7, len(daily_ratings))
            earlier_avg = sum(daily_ratings[:-7]) / max(1, len(daily_ratings) - 7)

            trend = "stable"
            if recent_avg > earlier_avg + 0.2:
                trend = "improving"
            elif recent_avg < earlier_avg - 0.2:
                trend = "declining"

            return {
                "trend": trend,
                "recent_average": recent_avg,
                "earlier_average": earlier_avg,
                "daily_ratings": dict(zip(sorted_days, daily_ratings))
            }

        return {"message": "趋势数据不足"}

    def _generate_recommendations(self, category_scores: Dict[str, float],
                                mode_performance: Dict[str, float],
                                improvement_areas: List[str]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于类别的建议
        if category_scores.get("accuracy", 0) < 3.0:
            recommendations.append("提升答案准确性：加强知识库质量，优化检索算法")

        if category_scores.get("relevance", 0) < 3.0:
            recommendations.append("提高相关性：改进问题理解，优化搜索匹配")

        if category_scores.get("completeness", 0) < 3.0:
            recommendations.append("增强完整性：扩展答案覆盖范围，提供更全面的信息")

        if category_scores.get("usefulness", 0) < 3.0:
            recommendations.append("提升实用性：关注用户实际需求，提供更有价值的信息")

        # 基于模式性能的建议
        worst_mode = min(mode_performance.items(), key=lambda x: x[1]) if mode_performance else None
        if worst_mode and worst_mode[1] < 3.0:
            recommendations.append(f"优化{worst_mode[0]}模式：该模式用户满意度较低")

        best_mode = max(mode_performance.items(), key=lambda x: x[1]) if mode_performance else None
        if best_mode and best_mode[1] > 4.0:
            recommendations.append(f"保持{best_mode[0]}模式优势：该模式表现优秀，可作为学习范例")

        return recommendations

    def get_mode_success_rate(self, mode: str, days_back: int = 30) -> float:
        """获取模式成功率"""
        analysis = self.analyze_feedback(days_back)
        return analysis.mode_performance.get(mode, 0.0) / 5.0  # 转换为0-1范围

    def get_category_performance(self, category: str, days_back: int = 30) -> float:
        """获取类别性能"""
        analysis = self.analyze_feedback(days_back)
        return analysis.category_scores.get(category, 0.0) / 5.0  # 转换为0-1范围

    def get_feedback_stats(self, days_back: int = 30) -> Dict[str, Any]:
        """获取反馈统计"""
        analysis = self.analyze_feedback(days_back)

        return {
            "total_feedbacks": analysis.total_feedbacks,
            "average_rating": analysis.average_rating,
            "success_rate": analysis.average_rating / 5.0,
            "category_scores": analysis.category_scores,
            "mode_performance": {mode: score/5.0 for mode, score in analysis.mode_performance.items()},
            "improvement_areas": analysis.improvement_areas,
            "strengths": analysis.strengths,
            "recommendations": analysis.recommendations
        }

    def export_feedback_data(self, output_path: str, days_back: int = 30):
        """导出反馈数据"""
        feedbacks = []

        cutoff_date = datetime.now() - timedelta(days=days_back)
        for date_key, month_feedbacks in self.feedback_db.items():
            try:
                year, month = map(int, date_key.split('-'))
                month_start = datetime(year, month, 1)

                if month_start >= cutoff_date:
                    for feedback in month_feedbacks:
                        if feedback.timestamp >= cutoff_date:
                            feedbacks.append(asdict(feedback))
            except:
                continue

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(feedbacks, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported {len(feedbacks)} feedback records to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting feedback data: {e}")

    def _clear_cache(self):
        """清除缓存"""
        self.analysis_cache.clear()

    def cleanup_old_data(self, days_to_keep: int = 365):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        removed_count = 0

        for date_key in list(self.feedback_db.keys()):
            try:
                year, month = map(int, date_key.split('-'))
                month_start = datetime(year, month, 1)

                if month_start < cutoff_date:
                    # 删除内存中的数据
                    del self.feedback_db[date_key]

                    # 删除文件
                    file_path = self.storage_path / f"{date_key}.json"
                    if file_path.exists():
                        file_path.unlink()
                        removed_count += 1

            except:
                continue

        logger.info(f"Cleaned up {removed_count} old feedback files")


class FeedbackAnalytics:
    """反馈分析器"""

    def __init__(self, feedback_collector: FeedbackCollector):
        self.collector = feedback_collector

    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """生成日报"""
        if date is None:
            date = datetime.now().date()

        # 获取当天数据
        start_date = datetime.combine(date, datetime.min.time())
        end_date = start_date + timedelta(days=1)

        daily_feedbacks = []
        for date_key, feedbacks in self.collector.feedback_db.items():
            for feedback in feedbacks:
                if start_date <= feedback.timestamp < end_date:
                    daily_feedbacks.append(feedback)

        if not daily_feedbacks:
            return {"date": date.isoformat(), "message": "No feedback data for this date"}

        # 分析当天数据
        ratings = [f.rating for f in daily_feedbacks]
        modes = [f.selected_mode for f in daily_feedbacks]
        categories = [f.category.value for f in daily_feedbacks]

        return {
            "date": date.isoformat(),
            "total_feedbacks": len(daily_feedbacks),
            "average_rating": sum(ratings) / len(ratings),
            "rating_distribution": {
                str(int(r)): ratings.count(r) for r in set(ratings)
            },
            "mode_distribution": {
                mode: modes.count(mode) for mode in set(modes)
            },
            "category_distribution": {
                category: categories.count(category) for category in set(categories)
            },
            "top_issues": self._get_top_issues(daily_feedbacks),
            "improvement_suggestions": self._get_improvement_suggestions(daily_feedbacks)
        }

    def generate_weekly_report(self, week_start: Optional[datetime] = None) -> Dict[str, Any]:
        """生成周报"""
        if week_start is None:
            today = datetime.now().date()
            week_start = datetime.combine(today - timedelta(days=today.weekday()), datetime.min.time())

        week_end = week_start + timedelta(days=7)

        # 收集周数据
        weekly_feedbacks = []
        for date_key, feedbacks in self.collector.feedback_db.items():
            for feedback in feedbacks:
                if week_start <= feedback.timestamp < week_end:
                    weekly_feedbacks.append(feedback)

        if not weekly_feedbacks:
            return {"week_start": week_start.isoformat(), "message": "No feedback data for this week"}

        # 分析周数据
        analysis = self.collector._perform_analysis(weekly_feedbacks)

        return {
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "total_feedbacks": analysis.total_feedbacks,
            "average_rating": analysis.average_rating,
            "category_scores": analysis.category_scores,
            "mode_performance": analysis.mode_performance,
            "improvement_areas": analysis.improvement_areas,
            "strengths": analysis.strengths,
            "recommendations": analysis.recommendations,
            "daily_breakdown": self._get_daily_breakdown(weekly_feedbacks)
        }

    def _get_top_issues(self, feedbacks: List[FeedbackItem]) -> List[str]:
        """获取主要问题"""
        issues = []

        # 低评分反馈
        low_ratings = [f for f in feedbacks if f.rating <= 2.0]
        if low_ratings:
            issues.append(f"有{len(low_ratings)}个低评分反馈")

        # 高频改进建议
        categories = [f.category.value for f in feedbacks if f.rating <= 3.0]
        if categories:
            from collections import Counter
            category_counts = Counter(categories)
            top_category = category_counts.most_common(1)[0]
            issues.append(f"用户最不满意的是{top_category}问题")

        return issues

    def _get_improvement_suggestions(self, feedbacks: List[FeedbackItem]) -> List[str]:
        """获取改进建议"""
        suggestions = []

        # 基于评论的关键词分析
        comments = [f.comments for f in feedbacks if f.comments and f.rating <= 3.0]
        comment_text = " ".join(comments).lower()

        common_issues = {
            "不准确": ["错", "不对", "错误", "不准确"],
            "不相关": ["无关", "不相关", "偏题"],
            "不完整": ["不完整", "缺少", "不够"],
            "不清楚": ["不清楚", "模糊", "难理解"],
            "太慢": ["慢", "延迟", "时间长"]
        }

        for issue, keywords in common_issues.items():
            if any(keyword in comment_text for keyword in keywords):
                suggestions.append(f"改善答案{issue}问题")

        return list(set(suggestions))

    def _get_daily_breakdown(self, feedbacks: List[FeedbackItem]) -> Dict[str, Dict[str, float]]:
        """获取每日分解"""
        daily_data = {}

        for feedback in feedbacks:
            day_key = feedback.timestamp.strftime("%Y-%m-%d")
            if day_key not in daily_data:
                daily_data[day_key] = {"count": 0, "total_rating": 0.0}

            daily_data[day_key]["count"] += 1
            daily_data[day_key]["total_rating"] += feedback.rating

        # 计算平均分
        for day, data in daily_data.items():
            data["average_rating"] = data["total_rating"] / data["count"] if data["count"] > 0 else 0

        return daily_data


# 使用示例
if __name__ == "__main__":
    # 创建反馈收集器
    collector = FeedbackCollector()

    # 模拟收集反馈
    sample_feedbacks = [
        {
            "type": "rating",
            "category": "accuracy",
            "rating": 4.0,
            "confidence": 0.8,
            "comments": "答案很准确，帮助很大"
        },
        {
            "type": "correction",
            "category": "relevance",
            "rating": 2.0,
            "confidence": 0.6,
            "comments": "答案不够相关，有些偏离问题",
            "suggested_mode": "knowledge_graph"
        }
    ]

    questions = [
        "什么是量化投资？",
        "平安证券和证券业的关系"
    ]

    for i, (question, feedback_data) in enumerate(zip(questions, sample_feedbacks)):
        feedback_id = collector.collect_feedback(
            question=question,
            selected_mode="vector",
            feedback_data=feedback_data,
            user_id=f"user_{i}",
            session_id=f"session_{i}"
        )
        print(f"Collected feedback: {feedback_id}")

    # 分析反馈
    analysis = collector.analyze_feedback(days_back=30)
    print(f"Average rating: {analysis.average_rating:.2f}")
    print(f"Improvement areas: {analysis.improvement_areas}")
    print(f"Recommendations: {analysis.recommendations}")