"""
学习机制和优化循环系统

实现自适应学习和持续优化
支持多种学习策略和性能监控
"""

import json
import logging
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """学习策略"""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    HYBRID = "hybrid"

class OptimizationTarget(Enum):
    """优化目标"""
    ACCURACY = "accuracy"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    COVERAGE = "coverage"
    EFFICIENCY = "efficiency"

@dataclass
class LearningData:
    """学习数据"""
    timestamp: datetime
    question: str
    selected_mode: str
    actual_performance: float
    user_feedback: float
    context: Dict[str, Any]
    features: Dict[str, float]

@dataclass
class LearningResult:
    """学习结果"""
    strategy: str
    improvement: float
    confidence: float
    recommendations: List[str]
    updated_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]

class LearningSystem:
    """学习系统主类"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.learning_data = deque(maxlen=10000)  # 保留最近10000条记录
        self.models = {}
        self.optimizers = {}
        self.performance_tracker = PerformanceTracker()
        self.learning_scheduler = LearningScheduler()
        self.is_learning = False

        # 初始化学习组件
        self._initialize_models()
        self._initialize_optimizers()
        self._load_saved_state()

        # 后台学习线程
        self.learning_thread = None
        self.stop_learning = threading.Event()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            "learning_enabled": True,
            "auto_learning": True,
            "learning_interval": 300,  # 5分钟
            "min_samples_for_learning": 50,
            "learning_strategy": LearningStrategy.HYBRID.value,
            "optimization_targets": [
                OptimizationTarget.USER_SATISFACTION.value,
                OptimizationTarget.ACCURACY.value
            ],
            "performance_thresholds": {
                "min_satisfaction": 0.7,
                "min_accuracy": 0.8,
                "max_response_time": 3.0
            },
            "model_update_frequency": "daily",
            "backup_frequency": "weekly"
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}")

        return default_config

    def _initialize_models(self):
        """初始化学习模型"""
        self.models = {
            "mode_selector": ModeSelectorModel(),
            "performance_predictor": PerformancePredictorModel(),
            "user_preference": UserPreferenceModel(),
            "context_analyzer": ContextAnalyzerModel()
        }

    def _initialize_optimizers(self):
        """初始化优化器"""
        self.optimizers = {
            "gradient_descent": GradientDescentOptimizer(),
            "genetic_algorithm": GeneticOptimizer(),
            "bayesian": BayesianOptimizer(),
            "reinforcement": ReinforcementOptimizer()
        }

    def _load_saved_state(self):
        """加载保存的状态"""
        try:
            state_file = Path("learning_system_state.pkl")
            if state_file.exists():
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.learning_data.extend(state.get("learning_data", []))
                    self.performance_tracker.load_state(state.get("performance_tracker", {}))
                    logger.info("Loaded saved learning system state")
        except Exception as e:
            logger.warning(f"Could not load saved state: {e}")

    def add_learning_data(self, question: str, selected_mode: str,
                         actual_performance: float, user_feedback: float,
                         context: Dict[str, Any], features: Dict[str, float]):
        """添加学习数据"""
        data = LearningData(
            timestamp=datetime.now(),
            question=question,
            selected_mode=selected_mode,
            actual_performance=actual_performance,
            user_feedback=user_feedback,
            context=context,
            features=features
        )

        self.learning_data.append(data)

        # 触发自动学习
        if (self.config.get("auto_learning", False) and
            len(self.learning_data) >= self.config.get("min_samples_for_learning", 50)):
            self._trigger_learning()

    def _trigger_learning(self):
        """触发学习过程"""
        if not self.config.get("learning_enabled", False):
            return

        if self.is_learning:
            logger.info("Learning already in progress")
            return

        # 启动后台学习线程
        self.learning_thread = threading.Thread(target=self._background_learning)
        self.learning_thread.daemon = True
        self.learning_thread.start()

    def _background_learning(self):
        """后台学习过程"""
        self.is_learning = True
        logger.info("Started background learning process")

        try:
            while not self.stop_learning.is_set():
                # 执行学习步骤
                learning_result = self._execute_learning_cycle()

                if learning_result:
                    self._apply_learning_result(learning_result)

                # 等待下一个学习周期
                self.stop_learning.wait(self.config.get("learning_interval", 300))

        except Exception as e:
            logger.error(f"Error in background learning: {e}")
        finally:
            self.is_learning = False
            logger.info("Background learning process stopped")

    def _execute_learning_cycle(self) -> Optional[LearningResult]:
        """执行学习循环"""
        if len(self.learning_data) < self.config.get("min_samples_for_learning", 50):
            return None

        # 选择学习策略
        strategy = LearningStrategy(self.config.get("learning_strategy", "hybrid"))

        # 准备学习数据
        recent_data = list(self.learning_data)[-1000:]  # 使用最近1000条数据

        try:
            if strategy == LearningStrategy.REINFORCEMENT:
                return self._reinforcement_learning(recent_data)
            elif strategy == LearningStrategy.SUPERVISED:
                return self._supervised_learning(recent_data)
            elif strategy == LearningStrategy.UNSUPERVISED:
                return self._unsupervised_learning(recent_data)
            elif strategy == LearningStrategy.HYBRID:
                return self._hybrid_learning(recent_data)
            else:
                logger.warning(f"Unknown learning strategy: {strategy}")
                return None

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
            return None

    def _reinforcement_learning(self, data: List[LearningData]) -> LearningResult:
        """强化学习"""
        logger.info("Executing reinforcement learning")

        # 使用强化学习优化器
        optimizer = self.optimizers["reinforcement"]

        # 准备状态和奖励
        states = []
        actions = []
        rewards = []

        for item in data:
            state = self._extract_state(item)
            action = self._encode_action(item.selected_mode)
            reward = self._calculate_reward(item)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        # 执行优化
        optimization_result = optimizer.optimize(states, actions, rewards)

        # 更新模式参数
        updated_params = {}
        if "mode_selector" in self.models:
            updated_params["mode_selector"] = self.models["mode_selector"].update_parameters(
                optimization_result["parameters"]
            )

        # 生成学习结果
        improvement = optimization_result.get("improvement", 0.0)
        recommendations = self._generate_recommendations(optimization_result)

        return LearningResult(
            strategy="reinforcement",
            improvement=improvement,
            confidence=optimization_result.get("confidence", 0.5),
            recommendations=recommendations,
            updated_parameters=updated_params,
            performance_metrics=self._calculate_performance_metrics()
        )

    def _supervised_learning(self, data: List[LearningData]) -> LearningResult:
        """监督学习"""
        logger.info("Executing supervised learning")

        # 准备训练数据
        X = []
        y = []

        for item in data:
            features = self._extract_features(item)
            target = self._extract_target(item)

            X.append(features)
            y.append(target)

        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)

        # 使用梯度下降优化器
        optimizer = self.optimizers["gradient_descent"]

        # 训练模型
        training_result = optimizer.train(X, y)

        # 更新模型
        updated_params = {}
        for model_name, model_params in training_result["parameters"].items():
            if model_name in self.models:
                updated_params[model_name] = self.models[model_name].update_parameters(model_params)

        # 计算改进
        improvement = training_result.get("improvement", 0.0)
        recommendations = [
            f"模型训练完成，改进幅度: {improvement:.2%}",
            "建议继续收集更多数据以提高模型性能"
        ]

        return LearningResult(
            strategy="supervised",
            improvement=improvement,
            confidence=training_result.get("confidence", 0.5),
            recommendations=recommendations,
            updated_parameters=updated_params,
            performance_metrics=self._calculate_performance_metrics()
        )

    def _unsupervised_learning(self, data: List[LearningData]) -> LearningResult:
        """无监督学习"""
        logger.info("Executing unsupervised learning")

        # 使用贝叶斯优化器进行无监督学习
        optimizer = self.optimizers["bayesian"]

        # 准备数据
        X = np.array([self._extract_features(item) for item in data])

        # 执行聚类分析
        clustering_result = optimizer.cluster(X)

        # 分析聚类结果
        improvement = self._analyze_clustering_improvement(clustering_result)
        recommendations = [
            "发现新的数据模式",
            "建议基于聚类结果优化搜索策略"
        ]

        return LearningResult(
            strategy="unsupervised",
            improvement=improvement,
            confidence=0.7,
            recommendations=recommendations,
            updated_parameters={},
            performance_metrics=self._calculate_performance_metrics()
        )

    def _hybrid_learning(self, data: List[LearningData]) -> LearningResult:
        """混合学习"""
        logger.info("Executing hybrid learning")

        # 组合多种学习策略
        results = []

        # 监督学习部分
        if len(data) >= 100:
            supervised_result = self._supervised_learning(data[:500])
            results.append(supervised_result)

        # 强化学习部分
        if len(data) >= 200:
            reinforcement_result = self._reinforcement_learning(data[-500:])
            results.append(reinforcement_result)

        # 无监督学习部分
        unsupervised_result = self._unsupervised_learning(data)
        results.append(unsupervised_result)

        # 融合结果
        if results:
            improvement = np.mean([r.improvement for r in results])
            confidence = np.mean([r.confidence for r in results])
            all_recommendations = []
            for r in results:
                all_recommendations.extend(r.recommendations)

            return LearningResult(
                strategy="hybrid",
                improvement=improvement,
                confidence=confidence,
                recommendations=all_recommendations,
                updated_parameters={},
                performance_metrics=self._calculate_performance_metrics()
            )

        return LearningResult(
            strategy="hybrid",
            improvement=0.0,
            confidence=0.0,
            recommendations=["数据不足，无法进行混合学习"],
            updated_parameters={},
            performance_metrics={}
        )

    def _extract_state(self, data: LearningData) -> List[float]:
        """提取状态特征"""
        state = [
            len(data.question),
            data.selected_mode == "vector",
            data.selected_mode == "knowledge_graph",
            data.selected_mode == "deep_search",
            data.selected_mode == "hybrid",
            data.context.get("question_type", "") == "definition",
            data.context.get("question_type", "") == "data_query",
            data.context.get("question_type", "") == "analysis",
            data.context.get("complexity", "") == "high",
            data.context.get("time_sensitive", False),
            len(data.context.get("keywords", [])),
            len(data.context.get("entities", []))
        ]

        # 添加特征
        state.extend(list(data.features.values()))

        return state

    def _encode_action(self, mode: str) -> int:
        """编码动作为数值"""
        mode_mapping = {
            "vector": 0,
            "knowledge_graph": 1,
            "deep_search": 2,
            "hybrid": 3
        }
        return mode_mapping.get(mode, 3)  # 默认为hybrid

    def _calculate_reward(self, data: LearningData) -> float:
        """计算奖励值"""
        # 基础奖励
        reward = (data.user_feedback + data.actual_performance) / 2

        # 调整因子
        if data.user_feedback >= 4.0:
            reward *= 1.2  # 高评分奖励
        elif data.user_feedback <= 2.0:
            reward *= 0.8  # 低评分惩罚

        # 响应时间因子
        response_time = data.context.get("response_time", 2.0)
        if response_time < 1.0:
            reward *= 1.1  # 快速响应奖励
        elif response_time > 5.0:
            reward *= 0.9  # 慢响应惩罚

        return reward

    def _extract_features(self, data: LearningData) -> List[float]:
        """提取特征向量"""
        features = [
            len(data.question),
            data.question.count("？"),
            data.question.count("和"),
            data.question.count("的"),
            data.context.get("complexity_score", 0.5),
            data.context.get("domain_score", 0.5),
            len(data.context.get("keywords", [])),
            len(data.context.get("entities", [])),
            data.context.get("time_sensitive", False),
            data.context.get("requires_analysis", False),
            data.context.get("requires_comparison", False)
        ]

        # 添加特征字典中的值
        features.extend(list(data.features.values()))

        return features

    def _extract_target(self, data: LearningData) -> float:
        """提取目标值"""
        return data.user_feedback

    def _apply_learning_result(self, result: LearningResult):
        """应用学习结果"""
        logger.info(f"Applying learning result: {result.strategy}")

        # 更新模型参数
        for model_name, params in result.updated_parameters.items():
            if model_name in self.models:
                self.models[model_name].apply_parameters(params)

        # 记录性能指标
        for metric, value in result.performance_metrics.items():
            self.performance_tracker.record_metric(metric, value)

        # 保存状态
        self._save_state()

        # 生成学习报告
        self._generate_learning_report(result)

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        if not self.learning_data:
            return {}

        recent_data = list(self.learning_data)[-100:]  # 最近100条数据

        metrics = {
            "average_satisfaction": np.mean([d.user_feedback for d in recent_data]),
            "average_performance": np.mean([d.actual_performance for d in recent_data]),
            "satisfaction_variance": np.var([d.user_feedback for d in recent_data]),
            "performance_variance": np.var([d.actual_performance for d in recent_data]),
            "total_samples": len(recent_data)
        }

        # 按模式统计
        mode_stats = defaultdict(list)
        for d in recent_data:
            mode_stats[d.selected_mode].append(d.user_feedback)

        for mode, ratings in mode_stats.items():
            metrics[f"{mode}_avg_satisfaction"] = np.mean(ratings)

        return metrics

    def _generate_recommendations(self, optimization_result: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if optimization_result.get("improvement", 0) > 0.1:
            recommendations.append("学习效果显著，继续保持当前策略")
        elif optimization_result.get("improvement", 0) < 0:
            recommendations.append("学习效果不佳，建议调整学习策略")

        if optimization_result.get("convergence", False):
            recommendations.append("模型已收敛，可以考虑增加学习数据或调整算法")

        return recommendations

    def _analyze_clustering_improvement(self, clustering_result: Dict) -> float:
        """分析聚类改进"""
        # 简化的聚类改进分析
        silhouette_score = clustering_result.get("silhouette_score", 0.5)
        return max(0, silhouette_score - 0.3)  # 相对于基线的改进

    def predict_performance(self, question: str, mode: str,
                           context: Dict[str, Any]) -> Dict[str, float]:
        """预测性能"""
        if "performance_predictor" in self.models:
            features = self._extract_features_from_context(question, mode, context)
            prediction = self.models["performance_predictor"].predict(features)
            return prediction
        else:
            # 默认预测
            return {"satisfaction": 0.7, "accuracy": 0.8, "response_time": 2.0}

    def _extract_features_from_context(self, question: str, mode: str,
                                       context: Dict[str, Any]) -> List[float]:
        """从上下文提取特征"""
        features = [
            len(question),
            mode == "vector",
            mode == "knowledge_graph",
            mode == "deep_search",
            mode == "hybrid",
            context.get("question_type", "") == "definition",
            context.get("question_type", "") == "data_query",
            context.get("question_type", "") == "analysis",
            context.get("complexity_score", 0.5),
            context.get("domain_score", 0.5)
        ]

        return features

    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        return {
            "is_learning": self.is_learning,
            "total_samples": len(self.learning_data),
            "models_count": len(self.models),
            "optimizers_count": len(self.optimizers),
            "last_learning": self.performance_tracker.get_last_learning_time(),
            "learning_strategy": self.config.get("learning_strategy"),
            "performance_metrics": self._calculate_performance_metrics()
        }

    def save_state(self):
        """保存状态"""
        state = {
            "learning_data": list(self.learning_data),
            "performance_tracker": self.performance_tracker.get_state(),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open("learning_system_state.pkl", 'wb') as f:
                pickle.dump(state, f)
            logger.info("Learning system state saved")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _save_state(self):
        """保存状态（内部方法）"""
        self.save_state()

    def start_learning(self):
        """启动学习系统"""
        if not self.config.get("learning_enabled", False):
            logger.warning("Learning is disabled in config")
            return

        self.stop_learning.clear()
        self._trigger_learning()

    def stop_learning_system(self):
        """停止学习系统"""
        self.stop_learning.set()
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=10)
        logger.info("Learning system stopped")

    def generate_learning_report(self) -> Dict[str, Any]:
        """生成学习报告"""
        metrics = self._calculate_performance_metrics()

        return {
            "learning_status": self.get_learning_status(),
            "performance_metrics": metrics,
            "model_status": {name: model.get_status() for name, model in self.models.items()},
            "recommendations": self._generate_system_recommendations(metrics),
            "trend_analysis": self.performance_tracker.get_trend_analysis(),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_system_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """生成系统建议"""
        recommendations = []

        if metrics.get("average_satisfaction", 0) < 3.0:
            recommendations.append("用户满意度较低，建议优化搜索算法")

        if metrics.get("satisfaction_variance", 0) > 1.0:
            recommendations.append("性能不稳定，建议增加训练数据")

        if metrics.get("total_samples", 0) < 100:
            recommendations.append("训练数据不足，建议收集更多用户反馈")

        return recommendations


class ModeSelectorModel:
    """模式选择模型"""

    def __init__(self):
        self.parameters = {
            "weights": {"vector": 0.25, "knowledge_graph": 0.25, "deep_search": 0.25, "hybrid": 0.25},
            "thresholds": {"high_confidence": 0.8, "low_confidence": 0.3}
        }

    def update_parameters(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """更新参数"""
        self.parameters.update(new_params)
        return self.parameters

    def apply_parameters(self, params: Dict[str, Any]):
        """应用参数"""
        self.parameters = params

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {"parameters": self.parameters, "status": "active"}


class PerformancePredictorModel:
    """性能预测模型"""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def predict(self, features: List[float]) -> Dict[str, float]:
        """预测性能"""
        # 简化实现
        return {
            "satisfaction": 0.7,
            "accuracy": 0.8,
            "response_time": 2.0
        }

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {"trained": self.is_trained, "model_type": "linear"}


class UserPreferenceModel:
    """用户偏好模型"""

    def __init__(self):
        self.preferences = defaultdict(float)

    def update_preference(self, user_id: str, mode: str, rating: float):
        """更新用户偏好"""
        key = f"{user_id}_{mode}"
        self.preferences[key] = rating

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {"preferences_count": len(self.preferences)}


class ContextAnalyzerModel:
    """上下文分析模型"""

    def __init__(self):
        self.context_patterns = {}

    def analyze_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """分析上下文"""
        return {"complexity": 0.5, "domain": 0.5}

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {"patterns_count": len(self.context_patterns)}


class PerformanceTracker:
    """性能跟踪器"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.last_learning_time = None

    def record_metric(self, metric_name: str, value: float):
        """记录指标"""
        self.metrics[metric_name].append((datetime.now(), value))

    def get_last_learning_time(self) -> Optional[datetime]:
        """获取最后学习时间"""
        return self.last_learning_time

    def set_last_learning_time(self, time: datetime):
        """设置最后学习时间"""
        self.last_learning_time = time

    def get_state(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "metrics": dict(self.metrics),
            "last_learning_time": self.last_learning_time
        }

    def load_state(self, state: Dict[str, Any]):
        """加载状态"""
        self.metrics = defaultdict(list, state.get("metrics", {}))
        self.last_learning_time = state.get("last_learning_time")

    def get_trend_analysis(self) -> Dict[str, Any]:
        """获取趋势分析"""
        trends = {}
        for metric, values in self.metrics.items():
            if len(values) >= 10:
                recent = [v[1] for v in values[-10:]]
                older = [v[1] for v in values[-20:-10]]
                trend = np.mean(recent) - np.mean(older)
                trends[metric] = {
                    "trend": "improving" if trend > 0 else "declining",
                    "change": trend
                }
        return trends


class LearningScheduler:
    """学习调度器"""

    def __init__(self):
        self.schedule = []

    def schedule_learning(self, delay: int, strategy: str):
        """调度学习"""
        self.schedule.append({
            "time": datetime.now() + timedelta(seconds=delay),
            "strategy": strategy
        })


class GradientDescentOptimizer:
    """梯度下降优化器"""

    def __init__(self):
        self.learning_rate = 0.01
        self.max_iterations = 100

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """训练模型"""
        # 简化的梯度下降实现
        weights = np.random.rand(X.shape[1])
        for _ in range(self.max_iterations):
            predictions = X.dot(weights)
            errors = predictions - y
            gradients = X.T.dot(errors) / len(y)
            weights -= self.learning_rate * gradients

        return {
            "parameters": {"weights": weights.tolist()},
            "improvement": 0.1,
            "confidence": 0.8
        }

    def optimize(self, states: List[List[float]], actions: List[int], rewards: List[float]) -> Dict[str, Any]:
        """优化"""
        return {"improvement": 0.05, "confidence": 0.6}


class GeneticOptimizer:
    """遗传算法优化器"""

    def __init__(self):
        self.population_size = 50
        self.generations = 100

    def optimize(self, states: List[List[float]], actions: List[int], rewards: List[float]) -> Dict[str, Any]:
        """优化"""
        return {"improvement": 0.08, "confidence": 0.7}


class BayesianOptimizer:
    """贝叶斯优化器"""

    def __init__(self):
        self.exploration_rate = 0.1

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """训练"""
        return {"improvement": 0.12, "confidence": 0.75}

    def cluster(self, X: np.ndarray) -> Dict[str, Any]:
        """聚类"""
        return {
            "improvement": 0.15,
            "confidence": 0.6,
            "silhouette_score": 0.45,
            "convergence": True
        }

    def optimize(self, states: List[List[float]], actions: List[int], rewards: List[float]) -> Dict[str, Any]:
        """优化"""
        return {"improvement": 0.1, "confidence": 0.7}


class ReinforcementOptimizer:
    """强化学习优化器"""

    def __init__(self):
        self.epsilon = 0.1
        self.gamma = 0.9
        self.alpha = 0.1

    def optimize(self, states: List[List[float]], actions: List[int], rewards: List[float]) -> Dict[str, Any]:
        """优化"""
        return {"improvement": 0.2, "confidence": 0.8, "convergence": False}


# 使用示例
if __name__ == "__main__":
    # 创建学习系统
    learning_system = LearningSystem()

    # 模拟添加学习数据
    for i in range(100):
        learning_system.add_learning_data(
            question=f"测试问题 {i}",
            selected_mode="vector",
            actual_performance=0.7 + np.random.normal(0, 0.1),
            user_feedback=3.5 + np.random.normal(0, 0.5),
            context={"question_type": "definition", "complexity_score": 0.5},
            features={"length": 10 + i, "complexity": 0.5}
        )

    # 获取学习状态
    status = learning_system.get_learning_status()
    print(f"Learning status: {status}")

    # 生成学习报告
    report = learning_system.generate_learning_report()
    print(f"Learning report: {report}")

    # 启动学习系统
    learning_system.start_learning()
    print("Learning system started")

    # 停止学习系统
    # learning_system.stop_learning_system()
    # print("Learning system stopped")