"""
增强的智能搜索服务

整合所有优化组件：
1. 改进的搜索选择算法
2. 增强的问题分类器
3. 优化的模式匹配算法
4. 用户反馈收集系统
5. 学习机制和优化循环

目标：将搜索选择准确率从16%提升到60%+
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from .enhanced_search_selector import EnhancedSearchSelector, QuestionAnalysis, ModeScore
from .enhanced_question_classifier import EnhancedQuestionClassifier, ClassificationResult
from .optimized_pattern_matcher import OptimizedPatternMatcher, ModeRecommendation
from .feedback_system import FeedbackCollector, FeedbackType, FeedbackCategory
from .learning_system import LearningSystem, LearningResult

logger = logging.getLogger(__name__)

@dataclass
class SearchRequest:
    """搜索请求"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    force_mode: Optional[str] = None

@dataclass
class SearchResult:
    """搜索结果"""
    mode: str
    confidence: float
    reasoning: List[str]
    risk_factors: List[str]
    alternatives: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class EnhancedSearchResult:
    """增强搜索结果"""
    search_result: SearchResult
    classification: Dict[str, ClassificationResult]
    question_analysis: QuestionAnalysis
    feedback_id: Optional[str] = None
    learning_applied: bool = False

class EnhancedIntelligentSearchService:
    """增强的智能搜索服务"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        # 初始化组件
        self.search_selector = EnhancedSearchSelector()
        self.question_classifier = EnhancedQuestionClassifier()
        self.pattern_matcher = OptimizedPatternMatcher()
        self.feedback_collector = FeedbackCollector()
        self.learning_system = LearningSystem(config_path)

        # 性能监控
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "mode_distribution": {},
            "accuracy_metrics": {},
            "last_update": datetime.now()
        }

        # 启动学习系统
        if self.config.get("auto_learning", True):
            self.learning_system.start_learning()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            "auto_learning": True,
            "feedback_collection": True,
            "performance_monitoring": True,
            "cache_enabled": True,
            "max_alternatives": 3,
            "confidence_threshold": 0.6,
            "learning_enabled": True,
            "optimization_frequency": "daily"
        }

        if config_path:
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")

        return default_config

    async def search(self, request: SearchRequest) -> EnhancedSearchResult:
        """执行智能搜索"""
        start_time = datetime.now()

        try:
            # 1. 问题分析和分类
            try:
                question_analysis = self.search_selector.analyze_question(request.query)
            except Exception as e:
                logger.error(f"Error in question analysis: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise

            try:
                classification = self.question_classifier.classify_comprehensive(request.query)
            except Exception as e:
                logger.error(f"Error in question classification: {e}")
                raise

            # 2. 模式推荐
            if request.force_mode:
                # 强制使用指定模式
                selected_mode = request.force_mode
                confidence = 1.0
                reasoning = [f"用户指定使用{selected_mode}模式"]
                risk_factors = []
                alternatives = []
            else:
                # 智能模式选择
                try:
                    mode_recommendation = self.pattern_matcher.recommend_mode(request.query)
                    selected_mode = mode_recommendation.mode.value
                    confidence = mode_recommendation.confidence
                    reasoning = [mode_recommendation.primary_reason] + mode_recommendation.supporting_reasons
                    risk_factors = mode_recommendation.risk_factors
                    alternatives = [
                        {"mode": alt[0].value, "confidence": alt[1]}
                        for alt in mode_recommendation.alternative_modes
                    ]
                except Exception as e:
                    logger.error(f"Error in mode recommendation: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # 使用默认模式
                    selected_mode = "hybrid"
                    confidence = 0.5
                    reasoning = ["模式推荐失败，使用默认混合模式"]
                    risk_factors = ["可能不是最优选择"]
                    alternatives = []

            # 3. 应用学习优化
            learning_applied = False
            if self.config.get("learning_enabled", False):
                # 创建安全的context副本，避免不可哈希类型
                safe_context = {}
                if request.context:
                    safe_context.update(request.context)

                # 移除可能包含列表的键，避免unhashable type错误
                for key, value in safe_context.items():
                    if isinstance(value, list):
                        safe_context[key] = str(value)  # 转换为字符串
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        safe_context[key] = str(value)  # 转换复杂对象为字符串

                try:
                    learning_prediction = self.learning_system.predict_performance(
                        request.query, selected_mode, safe_context
                    )
                except Exception as e:
                    logger.error(f"Error in learning prediction: {e}")
                    learning_prediction = {"satisfaction": 0.7, "accuracy": 0.8, "response_time": 2.0}

                # 根据学习预测调整置信度
                predicted_satisfaction = learning_prediction.get("satisfaction", 0.7)
                if predicted_satisfaction < 0.5 and alternatives:
                    # 选择替代方案
                    best_alternative = max(alternatives, key=lambda x: x["confidence"])
                    if best_alternative["confidence"] > confidence:
                        selected_mode = best_alternative["mode"]
                        confidence = best_alternative["confidence"]
                        reasoning.append("基于学习历史调整模式选择")
                        learning_applied = True

            # 4. 创建搜索结果
            processing_time = (datetime.now() - start_time).total_seconds()

            # 创建安全的metadata，避免不可哈希类型
            safe_metadata = {
                "question_type": question_analysis.type.value if hasattr(question_analysis.type, 'value') else str(question_analysis.type),
                "question_complexity": question_analysis.complexity,
                "question_domain": question_analysis.domain,
                "question_keywords": ",".join(question_analysis.keywords) if question_analysis.keywords else "",
                "question_entities": ",".join(question_analysis.entities) if question_analysis.entities else "",
                "classification_intent": classification.get("intent", {}).get("intent", "unknown"),
                "classification_type": classification.get("type", {}).get("primary_type", "unknown"),
                "learning_applied": learning_applied,
                "timestamp": datetime.now().isoformat()
            }

            search_result = SearchResult(
                mode=selected_mode,
                confidence=confidence,
                reasoning=reasoning,
                risk_factors=risk_factors,
                alternatives=alternatives[:self.config.get("max_alternatives", 3)],
                processing_time=processing_time,
                metadata=safe_metadata
            )

            # 5. 更新性能指标
            self._update_performance_metrics(search_result)

            # 6. 创建增强结果
            enhanced_result = EnhancedSearchResult(
                search_result=search_result,
                classification=classification,
                question_analysis=question_analysis,
                learning_applied=learning_applied
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Error in intelligent search: {e}")
            # 返回默认结果
            processing_time = (datetime.now() - start_time).total_seconds()

            return EnhancedSearchResult(
                search_result=SearchResult(
                    mode="hybrid",
                    confidence=0.5,
                    reasoning=["系统错误，使用默认混合模式"],
                    risk_factors=["可能不是最优选择"],
                    alternatives=[],
                    processing_time=processing_time,
                    metadata={"error": str(e)}
                ),
                classification={},
                question_analysis=QuestionAnalysis(
                    text=request.query,
                    type=None,
                    complexity="unknown",
                    domain="general",
                    keywords=[],
                    entities=[],
                    intent="general",
                    temporal_aspect="none",
                    scope="general"
                ),
                learning_applied=False
            )

    def submit_feedback(self, query: str, selected_mode: str, search_result: SearchResult,
                        feedback_data: Dict[str, Any],
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> str:
        """提交反馈"""
        try:
            # 收集基础反馈
            feedback_id = self.feedback_collector.collect_feedback(
                question=query,
                selected_mode=selected_mode,
                feedback_data=feedback_data,
                user_id=user_id,
                session_id=session_id
            )

            # 提取特征用于学习
            features = self._extract_feedback_features(query, selected_mode, search_result, feedback_data)

            # 添加学习数据
            if self.config.get("learning_enabled", False):
                actual_performance = self._calculate_actual_performance(search_result, feedback_data)
                user_feedback = feedback_data.get("rating", 3.0) / 5.0  # 转换为0-1范围

                self.learning_system.add_learning_data(
                    question=query,
                    selected_mode=selected_mode,
                    actual_performance=actual_performance,
                    user_feedback=user_feedback,
                    context=feedback_data.get("context", {}),
                    features=features
                )

            logger.info(f"Feedback submitted: {feedback_id}")
            return feedback_id

        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return "error"

    def _extract_feedback_features(self, query: str, selected_mode: str,
                                  search_result: SearchResult, feedback_data: Dict[str, Any]) -> Dict[str, float]:
        """提取反馈特征"""
        features = {
            "query_length": len(query),
            "response_time": search_result.processing_time,
            "confidence": search_result.confidence,
            "rating": feedback_data.get("rating", 3.0),
            "has_alternatives": len(search_result.alternatives) > 0,
            "has_risk_factors": len(search_result.risk_factors) > 0,
            "reasoning_count": len(search_result.reasoning)
        }

        # 模式特征
        mode_features = {
            f"mode_{selected_mode}": 1.0,
            "mode_vector": 1.0 if selected_mode == "vector" else 0.0,
            "mode_knowledge_graph": 1.0 if selected_mode == "knowledge_graph" else 0.0,
            "mode_deep_search": 1.0 if selected_mode == "deep_search" else 0.0,
            "mode_hybrid": 1.0 if selected_mode == "hybrid" else 0.0
        }
        features.update(mode_features)

        return features

    def _calculate_actual_performance(self, search_result: SearchResult, feedback_data: Dict[str, Any]) -> float:
        """计算实际性能"""
        # 综合多个因素
        factors = []

        # 用户评分 (0-1范围)
        user_rating = feedback_data.get("rating", 3.0) / 5.0
        factors.append(user_rating)

        # 响应时间因子 (越快越好)
        response_time = search_result.processing_time
        time_factor = max(0, 1.0 - response_time / 5.0)  # 5秒为基准
        factors.append(time_factor)

        # 置信度因子
        factors.append(search_result.confidence)

        # 相关性评分
        relevance = feedback_data.get("relevance", 0.7)
        factors.append(relevance)

        # 加权平均
        weights = [0.4, 0.2, 0.2, 0.2]  # 用户评分权重最高
        performance = sum(f * w for f, w in zip(factors, weights))

        return max(0, min(1.0, performance))

    def _update_performance_metrics(self, search_result: SearchResult):
        """更新性能指标"""
        self.performance_metrics["total_requests"] += 1

        if search_result.confidence >= self.config.get("confidence_threshold", 0.6):
            self.performance_metrics["successful_requests"] += 1

        # 更新模式分布
        mode = search_result.mode
        if mode not in self.performance_metrics["mode_distribution"]:
            self.performance_metrics["mode_distribution"][mode] = 0
        self.performance_metrics["mode_distribution"][mode] += 1

        # 更新平均响应时间
        total_time = self.performance_metrics.get("total_response_time", 0.0)
        requests = self.performance_metrics["total_requests"]
        current_avg = total_time / max(1, requests - 1)
        new_avg = (current_avg * (requests - 1) + search_result.processing_time) / requests
        self.performance_metrics["average_response_time"] = new_avg
        self.performance_metrics["total_response_time"] = total_time + search_result.processing_time

        self.performance_metrics["last_update"] = datetime.now()

    def get_performance_report(self, days_back: int = 7) -> Dict[str, Any]:
        """获取性能报告"""
        # 基础指标
        total_requests = self.performance_metrics["total_requests"]
        successful_requests = self.performance_metrics["successful_requests"]
        success_rate = successful_requests / max(1, total_requests)
        avg_response_time = self.performance_metrics["average_response_time"]

        # 模式分布
        mode_distribution = self.performance_metrics["mode_distribution"]
        total_mode_requests = sum(mode_distribution.values()) if mode_distribution else 1
        mode_percentages = {
            mode: (count / total_mode_requests) * 100
            for mode, count in mode_distribution.items()
        }

        # 反馈分析
        feedback_stats = {}
        if self.config.get("feedback_collection", False):
            feedback_stats = self.feedback_collector.get_feedback_stats(days_back)

        # 学习系统状态
        learning_status = {}
        if self.config.get("learning_enabled", False):
            learning_status = self.learning_system.get_learning_status()

        # 准确性估算
        accuracy_estimates = self._estimate_accuracy()

        # 生成报告
        report = {
            "period_days": days_back,
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "average_response_time": avg_response_time
            },
            "mode_distribution": mode_percentages,
            "feedback_analysis": feedback_stats,
            "learning_status": learning_status,
            "accuracy_estimates": accuracy_estimates,
            "recommendations": self._generate_recommendations(success_rate, avg_response_time, feedback_stats),
            "last_update": self.performance_metrics["last_update"].isoformat()
        }

        return report

    def _estimate_accuracy(self) -> Dict[str, float]:
        """估算准确性"""
        # 基于模式性能和学习数据估算准确性
        estimates = {}

        # 获取各模式的成功率
        mode_success_rates = {}
        total_mode_requests = sum(self.performance_metrics["mode_distribution"].values())

        for mode, count in self.performance_metrics["mode_distribution"].items():
            if total_mode_requests > 0:
                mode_success_rates[mode] = count / total_mode_requests

        # 基于反馈数据调整
        if self.config.get("feedback_collection", False):
            feedback_stats = self.feedback_collector.get_feedback_stats(30)
            for mode, score in feedback_stats.get("mode_performance", {}).items():
                if mode in mode_success_rates:
                    # 结合成功率（权重0.3）和用户满意度（权重0.7）
                    estimates[mode] = (mode_success_rates[mode] * 0.3 + score * 0.7)
                else:
                    estimates[mode] = score

        return estimates

    def _generate_recommendations(self, success_rate: float, avg_response_time: float,
                                feedback_stats: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if success_rate < 0.7:
            recommendations.append(f"成功率较低({success_rate:.1%})，建议优化搜索选择算法")

        if avg_response_time > 3.0:
            recommendations.append(f"响应时间较慢({avg_response_time:.2f}s)，建议优化系统性能")

        if feedback_stats:
            avg_rating = feedback_stats.get("average_rating", 3.0)
            if avg_rating < 3.5:
                recommendations.append(f"用户评分较低({avg_rating:.1f})，需要改进答案质量")

            improvement_areas = feedback_stats.get("improvement_areas", [])
            if improvement_areas:
                recommendations.append(f"重点改进: {', '.join(improvement_areas[:2])}")

        return recommendations

    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            "components_status": {
                "search_selector": "active",
                "question_classifier": "active" if self.question_classifier.is_trained else "needs_training",
                "pattern_matcher": "active",
                "feedback_collector": "active",
                "learning_system": "active" if self.config.get("learning_enabled", False) else "disabled"
            },
            "performance_metrics": self.performance_metrics,
            "config": self.config,
            "recent_feedback": self.feedback_collector.analyze_feedback(1).__dict__ if self.config.get("feedback_collection", False) else {},
            "learning_system_status": self.learning_system.get_learning_status() if self.config.get("learning_enabled", False) else {}
        }

    def optimize_models(self, force: bool = False) -> Dict[str, Any]:
        """优化模型"""
        results = {}

        # 训练问题分类器
        if not self.question_classifier.is_trained or force:
            # 使用反馈数据训练分类器
            feedback_data = self.feedback_collector.analyze_feedback(30)

            # 生成训练数据
            training_data = []
            for date_key, feedbacks in self.feedback_collector.feedback_db.items():
                for feedback in feedbacks:
                    from .feedback_system import TrainingData
                    training_data.append(TrainingData(
                        text=feedback.question,
                        category=feedback.category.value,
                        timestamp=feedback.timestamp
                    ))

            if len(training_data) >= 50:
                try:
                    self.question_classifier.train_ml_models(training_data)
                    results["question_classifier"] = "training_completed"
                    logger.info("Question classifier training completed")
                except Exception as e:
                    results["question_classifier"] = f"training_failed: {str(e)}"
                    logger.error(f"Question classifier training failed: {e}")

        # 触发学习系统优化
        if self.config.get("learning_enabled", False):
            try:
                self.learning_system.save_state()
                results["learning_system"] = "state_saved"
            except Exception as e:
                results["learning_system"] = f"save_failed: {str(e)}"

        return results

    def shutdown(self):
        """关闭服务"""
        logger.info("Shutting down enhanced intelligent search service")

        # 停止学习系统
        if hasattr(self, 'learning_system'):
            self.learning_system.stop_learning_system()

        # 保存状态
        self.optimize_models(force=True)

        logger.info("Enhanced intelligent search service shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# 使用示例
async def example_usage():
    """使用示例"""

    # 创建服务
    service = EnhancedIntelligentSearchService()

    # 搜索测试
    request = SearchRequest(
        query="什么是量化投资策略？",
        user_id="user123",
        session_id="session456"
    )

    result = await service.search(request)
    print(f"Selected mode: {result.search_result.mode}")
    print(f"Confidence: {result.search_result.confidence:.2f}")
    print(f"Reasoning: {result.search_result.reasoning}")

    # 提交反馈
    feedback_id = service.submit_feedback(
        query=request.query,
        selected_mode=result.search_result.mode,
        search_result=result.search_result,
        feedback_data={
            "type": "rating",
            "category": "accuracy",
            "rating": 4.0,
            "relevance": 0.8,
            "comments": "答案很准确"
        },
        user_id=request.user_id,
        session_id=request.session_id
    )

    print(f"Feedback submitted: {feedback_id}")

    # 获取性能报告
    report = service.get_performance_report()
    print(f"Success rate: {report['summary']['success_rate']:.2%}")

    # 获取优化状态
    status = service.get_optimization_status()
    print(f"Components status: {status['components_status']}")

    # 关闭服务
    service.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())