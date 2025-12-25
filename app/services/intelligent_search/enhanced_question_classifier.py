"""
增强的问题分类器

使用机器学习和NLP技术提升问题分类准确性
支持多种分类维度和自定义训练
"""

import re
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, using rule-based classification only")

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba not available, using basic text processing")

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """分类结果"""
    category: str
    confidence: float
    probabilities: Dict[str, float]
    features: List[str]
    method: str  # 'rule_based', 'ml_based', 'ensemble'

@dataclass
class TrainingData:
    """训练数据"""
    text: str
    category: str
    subcategory: Optional[str] = None
    difficulty: Optional[str] = None
    domain: Optional[str] = None
    timestamp: Optional[str] = None

class EnhancedQuestionClassifier:
    """增强的问题分类器"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/question_classifier"
        self.vectorizer = None
        self.classifiers = {}
        self.feature_extractors = {}
        self.rules = self._load_classification_rules()
        self.training_data = []
        self.is_trained = False

        # 初始化jieba词典
        if JIEBA_AVAILABLE:
            self._initialize_jieba()

        # 加载预训练模型
        self.load_models()

    def _initialize_jieba(self):
        """初始化jieba金融词典"""
        financial_terms = [
            "量化投资", "择时策略", "动量效应", "反转效应", "价值投资", "成长投资",
            "技术分析", "基本面分析", "风险控制", "资产配置", "夏普比率", "最大回撤",
            "波动率", "相关性", "贝塔系数", "阿尔法收益", "信息比率", "跟踪误差",
            "市盈率", "市净率", "ROE", "ROA", "毛利率", "净利率", "负债率",
            "平安证券", "国信证券", "海通证券", "招商证券", "国泰君安", "中信证券",
            "MACD", "RSI", "KDJ", "布林带", "移动平均线", "成交量", "能量潮"
        ]

        for term in financial_terms:
            jieba.add_word(term)

    def _load_classification_rules(self) -> Dict[str, Dict]:
        """加载分类规则"""
        return {
            "category": {
                "definition": {
                    "patterns": [
                        r"什么是(.+?)\??",
                        r"(.+?)的定义",
                        r"解释(.+?)",
                        r"描述(.+?)",
                        r"(.+?)是什么"
                    ],
                    "keywords": ["定义", "是什么", "解释", "描述", "概念", "含义"],
                    "weight": 0.9
                },
                "data_query": {
                    "patterns": [
                        r"(.+?)的数据",
                        r"查询(.+?)",
                        r"(.+?)的指标",
                        r"(.+?)的最新",
                        r"(.+?)的统计"
                    ],
                    "keywords": ["数据", "查询", "指标", "统计", "最新", "当前", "实时"],
                    "weight": 0.8
                },
                "methodology": {
                    "patterns": [
                        r"如何(.+?)",
                        r"怎么(.+?)",
                        r"(.+?)的方法",
                        r"(.+?)的策略",
                        r"(.+?)的步骤"
                    ],
                    "keywords": ["如何", "怎么", "方法", "策略", "步骤", "流程", "做法"],
                    "weight": 0.8
                },
                "comparison": {
                    "patterns": [
                        r"(.+?)和(.+?)的区别",
                        r"比较(.+?)",
                        r"(.+?)与(.+?)",
                        r"(.+?)对比(.+?)"
                    ],
                    "keywords": ["区别", "比较", "对比", "差异", "优劣", "优缺点"],
                    "weight": 0.9
                },
                "analysis": {
                    "patterns": [
                        r"分析(.+?)",
                        r"评估(.+?)",
                        r"研究(.+?)",
                        r"探讨(.+?)",
                        r"(.+?)的影响"
                    ],
                    "keywords": ["分析", "评估", "研究", "探讨", "影响", "作用", "效果"],
                    "weight": 0.8
                },
                "prediction": {
                    "patterns": [
                        r"预测(.+?)",
                        r"(.+?)的趋势",
                        r"未来(.+?)",
                        r"(.+?)的前景",
                        r"预计(.+?)"
                    ],
                    "keywords": ["预测", "趋势", "未来", "前景", "预计", "展望", "推测"],
                    "weight": 0.9
                },
                "relationship": {
                    "patterns": [
                        r"(.+?)和(.+?)的关系",
                        r"(.+?)如何影响(.+?)",
                        r"(.+?)与(.+?)的关联",
                        r"(.+?)和(.+?)之间"
                    ],
                    "keywords": ["关系", "影响", "关联", "连接", "依赖", "相关"],
                    "weight": 0.9
                }
            },
            "difficulty": {
                "low": {
                    "indicators": ["简单", "基础", "概述", "简介"],
                    "max_length": 20,
                    "max_entities": 1
                },
                "medium": {
                    "indicators": ["详细", "具体", "分析", "说明"],
                    "max_length": 50,
                    "max_entities": 2
                },
                "high": {
                    "indicators": ["深入", "全面", "复杂", "综合"],
                    "max_length": 80,
                    "max_entities": 3
                },
                "very_high": {
                    "indicators": ["系统", "完整", "详细分析", "综合评估"],
                    "min_length": 80,
                    "min_entities": 3
                }
            },
            "domain": {
                "investment": ["投资", "股票", "基金", "债券", "资产", "收益"],
                "trading": ["交易", "买卖", "套利", "投机", "持仓", "开仓", "平仓"],
                "risk": ["风险", "回撤", "波动", "损失", "安全", "控制"],
                "analysis": ["分析", "研究", "评估", "诊断", "检验"],
                "strategy": ["策略", "方法", "技巧", "方案", "规划"],
                "data": ["数据", "统计", "指标", "比率", "数值"]
            }
        }

    def extract_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        features = {}

        # 基础特征
        features["length"] = len(text)
        features["word_count"] = len(text.split())
        features["sentence_count"] = len(re.split(r'[。！？；]', text))

        # 标点符号特征
        features["question_marks"] = text.count("？") + text.count("?")
        features["commas"] = text.count("，") + text.count(",")
        features["punctuation_ratio"] = (features["question_marks"] + features["commas"]) / max(features["word_count"], 1)

        # 关键词特征
        for category, rules in self.rules["category"].items():
            keywords = rules.get("keywords", [])
            features[f"{category}_keyword_count"] = sum(1 for kw in keywords if kw in text)
            features[f"{category}_keyword_ratio"] = features[f"{category}_keyword_count"] / max(features["word_count"], 1)

        # 模式匹配特征
        for category, rules in self.rules["category"].items():
            patterns = rules.get("patterns", [])
            features[f"{category}_pattern_match"] = any(re.search(pattern, text) for pattern in patterns)

        # 实体特征
        features["entity_count"] = self._count_entities(text)
        features["has_companies"] = self._has_companies(text)
        features["has_metrics"] = self._has_metrics(text)
        features["has_time_expressions"] = self._has_time_expressions(text)

        # 时间特征
        features["has_future_reference"] = any(word in text for word in ["未来", "预测", "趋势", "前景"])
        features["has_past_reference"] = any(word in text for word in ["历史", "过去", "以前", "曾经"])
        features["has_comparison"] = any(word in text for word in ["比较", "对比", "区别", "差异"])

        # 复杂度特征
        features["avg_word_length"] = sum(len(word) for word in text.split()) / max(features["word_count"], 1)
        features["unique_words"] = len(set(text.split()))
        features["lexical_diversity"] = features["unique_words"] / max(features["word_count"], 1)

        # TF-IDF特征（如果可用）
        if SKLEARN_AVAILABLE and self.vectorizer is not None:
            try:
                tfidf_vector = self.vectorizer.transform([text])
                features["tfidf_norm"] = np.linalg.norm(tfidf_vector.toarray())
            except:
                features["tfidf_norm"] = 0

        return features

    def _count_entities(self, text: str) -> int:
        """计算实体数量"""
        entities = ["平安证券", "国信证券", "海通证券", "招商证券", "国泰君安", "中信证券",
                   "华泰证券", "广发证券", "申万宏源", "银河证券"]
        return sum(1 for entity in entities if entity in text)

    def _has_companies(self, text: str) -> bool:
        """是否包含公司实体"""
        return self._count_entities(text) > 0

    def _has_metrics(self, text: str) -> bool:
        """是否包含金融指标"""
        metrics = ["ROE", "ROA", "P/E", "P/B", "EPS", "市盈率", "市净率", "毛利率", "净利率"]
        return any(metric in text for metric in metrics)

    def _has_time_expressions(self, text: str) -> bool:
        """是否包含时间表达"""
        time_expressions = ["2023年", "2024年", "最近", "目前", "未来", "过去", "第一季度", "第二季度"]
        return any(expr in text for expr in time_expressions)

    def classify_by_rules(self, text: str, category_type: str = "category") -> Tuple[str, float]:
        """基于规则的分类"""
        if category_type not in self.rules:
            return "unknown", 0.0

        rules = self.rules[category_type]
        scores = {}

        for category, rule_set in rules.items():
            score = 0

            # 模式匹配得分
            if "patterns" in rule_set:
                pattern_matches = sum(1 for pattern in rule_set["patterns"] if re.search(pattern, text))
                score += pattern_matches * 0.5

            # 关键词匹配得分
            if "keywords" in rule_set:
                keyword_matches = sum(1 for kw in rule_set["keywords"] if kw in text)
                score += keyword_matches * 0.3

            # 应用权重
            if "weight" in rule_set:
                score *= rule_set["weight"]

            scores[category] = score

        if not scores or max(scores.values()) == 0:
            return "unknown", 0.0

        best_category = max(scores, key=scores.get)
        confidence = scores[best_category] / sum(scores.values()) if sum(scores.values()) > 0 else 0

        return best_category, confidence

    def train_ml_models(self, training_data: List[TrainingData]):
        """训练机器学习模型"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, skipping ML training")
            return

        if len(training_data) < 10:
            logger.warning("Insufficient training data, need at least 10 samples")
            return

        self.training_data = training_data

        # 准备数据
        texts = [data.text for data in training_data]
        categories = [data.category for data in training_data]

        # 特征提取
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None if not JIEBA_AVAILABLE else None
        )

        try:
            X_text = self.vectorizer.fit_transform(texts)
        except Exception as e:
            logger.error(f"Error fitting TF-IDF vectorizer: {e}")
            return

        # 提取额外特征
        X_features = []
        for text in texts:
            features = self.extract_features(text)
            feature_vector = [
                features.get("length", 0),
                features.get("word_count", 0),
                features.get("question_marks", 0),
                features.get("entity_count", 0),
                features.get("has_companies", 0),
                features.get("has_metrics", 0),
                features.get("has_future_reference", 0),
                features.get("has_comparison", 0),
                features.get("lexical_diversity", 0)
            ]
            X_features.append(feature_vector)

        X_features = np.array(X_features)

        # 合并特征
        from scipy.sparse import hstack
        X = hstack([X_text, X_features])

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, categories, test_size=0.2, random_state=42, stratify=categories
        )

        # 训练多个分类器
        self.classifiers = {
            "naive_bayes": MultinomialNB(),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": SVC(probability=True, random_state=42)
        }

        for name, classifier in self.classifiers.items():
            try:
                classifier.fit(X_train, y_train)
                # 交叉验证
                cv_scores = cross_val_score(classifier, X_train, y_train, cv=3)
                logger.info(f"{name} CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")

        self.is_trained = True
        logger.info("ML models trained successfully")

    def classify_by_ml(self, text: str) -> ClassificationResult:
        """基于机器学习的分类"""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return self.classify_by_rules(text)

        try:
            # 特征提取
            X_text = self.vectorizer.transform([text])
            features = self.extract_features(text)
            feature_vector = [[
                features.get("length", 0),
                features.get("word_count", 0),
                features.get("question_marks", 0),
                features.get("entity_count", 0),
                features.get("has_companies", 0),
                features.get("has_metrics", 0),
                features.get("has_future_reference", 0),
                features.get("has_comparison", 0),
                features.get("lexical_diversity", 0)
            ]]

            from scipy.sparse import hstack
            X = hstack([X_text, feature_vector])

            # 集成预测
            predictions = {}
            probabilities = {}

            for name, classifier in self.classifiers.items():
                try:
                    pred = classifier.predict(X)[0]
                    if hasattr(classifier, "predict_proba"):
                        probs = classifier.predict_proba(X)[0]
                        classes = classifier.classes_
                        prob_dict = dict(zip(classes, probs))

                        if pred not in probabilities:
                            probabilities[pred] = []
                        probabilities[pred].append(prob_dict.get(pred, 0))

                        for cls, prob in prob_dict.items():
                            if cls not in probabilities:
                                probabilities[cls] = []
                            probabilities[cls].append(prob)

                    predictions[name] = pred
                except Exception as e:
                    logger.error(f"Error in {name} prediction: {e}")

            if not predictions:
                return self.classify_by_rules(text)

            # 投票决策
            from collections import Counter
            vote_counts = Counter(predictions.values())
            best_category = vote_counts.most_common(1)[0][0]

            # 计算平均概率
            avg_probabilities = {}
            for category, probs in probabilities.items():
                avg_probabilities[category] = np.mean(probs)

            confidence = avg_probabilities.get(best_category, 0)

            return ClassificationResult(
                category=best_category,
                confidence=confidence,
                probabilities=avg_probabilities,
                features=list(features.keys()),
                method="ml_based"
            )

        except Exception as e:
            logger.error(f"Error in ML classification: {e}")
            return self.classify_by_rules(text)

    def classify(self, text: str, use_ensemble: bool = True) -> ClassificationResult:
        """综合分类"""
        # 基于规则的分类
        rule_category, rule_confidence = self.classify_by_rules(text)
        rule_result = ClassificationResult(
            category=rule_category,
            confidence=rule_confidence,
            probabilities={rule_category: rule_confidence},
            features=["rule_based"],
            method="rule_based"
        )

        # 如果没有ML模型，直接返回规则结果
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return rule_result

        # 基于ML的分类
        ml_result = self.classify_by_ml(text)

        if not use_ensemble:
            # 选择置信度更高的结果
            return ml_result if ml_result.confidence > rule_result.confidence else rule_result

        # 集成结果
        if ml_result.confidence > rule_result.confidence * 1.2:
            # ML结果明显更好
            final_category = ml_result.category
            final_confidence = ml_result.confidence
        elif rule_result.confidence > ml_result.confidence * 1.2:
            # 规则结果明显更好
            final_category = rule_result.category
            final_confidence = rule_result.confidence
        else:
            # 加权平均
            categories = [rule_result.category, ml_result.category]
            weights = [rule_result.confidence, ml_result.confidence]
            final_category = categories[np.argmax(weights)]
            final_confidence = max(rule_result.confidence, ml_result.confidence)

        # 合并概率
        combined_probabilities = {}
        for prob_dict in [rule_result.probabilities, ml_result.probabilities]:
            for category, prob in prob_dict.items():
                if category not in combined_probabilities:
                    combined_probabilities[category] = []
                combined_probabilities[category].append(prob)

        avg_probabilities = {cat: np.mean(probs) for cat, probs in combined_probabilities.items()}

        return ClassificationResult(
            category=final_category,
            confidence=final_confidence,
            probabilities=avg_probabilities,
            features=list(set(rule_result.features + ml_result.features)),
            method="ensemble"
        )

    def classify_comprehensive(self, text: str) -> Dict[str, ClassificationResult]:
        """综合分类（多个维度）"""
        results = {}

        # 主分类
        results["category"] = self.classify(text)

        # 难度分类
        difficulty, diff_confidence = self.classify_by_rules(text, "difficulty")
        results["difficulty"] = ClassificationResult(
            category=difficulty,
            confidence=diff_confidence,
            probabilities={difficulty: diff_confidence},
            features=["rule_based"],
            method="rule_based"
        )

        # 领域分类
        domain, domain_confidence = self.classify_by_rules(text, "domain")
        results["domain"] = ClassificationResult(
            category=domain,
            confidence=domain_confidence,
            probabilities={domain: domain_confidence},
            features=["rule_based"],
            method="rule_based"
        )

        return results

    def add_training_data(self, text: str, category: str, **kwargs):
        """添加训练数据"""
        training_data = TrainingData(
            text=text,
            category=category,
            subcategory=kwargs.get("subcategory"),
            difficulty=kwargs.get("difficulty"),
            domain=kwargs.get("domain"),
            timestamp=datetime.now().isoformat()
        )

        self.training_data.append(training_data)

    def save_models(self):
        """保存模型"""
        if not self.model_path:
            return

        model_dir = Path(self.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存分类器
        if self.classifiers:
            for name, classifier in self.classifiers.items():
                model_file = model_dir / f"{name}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(classifier, f)

        # 保存向量化器
        if self.vectorizer:
            with open(model_dir / "vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)

        # 保存训练数据
        if self.training_data:
            with open(model_dir / "training_data.json", 'w', encoding='utf-8') as f:
                data_list = [asdict(data) for data in self.training_data]
                json.dump(data_list, f, ensure_ascii=False, indent=2)

        # 保存规则
        with open(model_dir / "rules.json", 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

        logger.info(f"Models saved to {self.model_path}")

    def load_models(self):
        """加载模型"""
        if not self.model_path:
            return

        model_dir = Path(self.model_path)
        if not model_dir.exists():
            logger.info(f"Model directory {self.model_path} does not exist")
            return

        # 加载分类器
        self.classifiers = {}
        for model_file in model_dir.glob("*.pkl"):
            if model_file.name != "vectorizer.pkl":
                try:
                    with open(model_file, 'rb') as f:
                        classifier = pickle.load(f)
                        self.classifiers[model_file.stem] = classifier
                        logger.info(f"Loaded classifier: {model_file.stem}")
                except Exception as e:
                    logger.error(f"Error loading {model_file}: {e}")

        # 加载向量化器
        vectorizer_file = model_dir / "vectorizer.pkl"
        if vectorizer_file.exists():
            try:
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                    logger.info("Loaded TF-IDF vectorizer")
            except Exception as e:
                logger.error(f"Error loading vectorizer: {e}")

        # 加载训练数据
        training_data_file = model_dir / "training_data.json"
        if training_data_file.exists():
            try:
                with open(training_data_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    self.training_data = [TrainingData(**data) for data in data_list]
                    logger.info(f"Loaded {len(self.training_data)} training samples")
            except Exception as e:
                logger.error(f"Error loading training data: {e}")

        if self.classifiers or self.vectorizer:
            self.is_trained = True
            logger.info("Models loaded successfully")

    def get_classification_stats(self) -> Dict[str, Any]:
        """获取分类统计信息"""
        stats = {
            "is_trained": self.is_trained,
            "training_data_size": len(self.training_data),
            "available_classifiers": list(self.classifiers.keys()) if self.classifiers else [],
            "has_vectorizer": self.vectorizer is not None,
            "sklearn_available": SKLEARN_AVAILABLE,
            "jieba_available": JIEBA_AVAILABLE
        }

        if self.training_data:
            # 训练数据统计
            categories = [data.category for data in self.training_data]
            from collections import Counter
            category_counts = Counter(categories)
            stats["training_data_distribution"] = dict(category_counts)

        return stats

# 使用示例和测试
if __name__ == "__main__":
    # 创建分类器
    classifier = EnhancedQuestionClassifier()

    # 示例训练数据
    sample_data = [
        TrainingData("什么是量化投资？", "definition"),
        TrainingData("查询平安证券的ROE数据", "data_query"),
        TrainingData("如何进行价值投资？", "methodology"),
        TrainingData("比较成长投资和价值投资", "comparison"),
        TrainingData("分析2024年股市表现", "analysis"),
        TrainingData("预测未来市场趋势", "prediction"),
        TrainingData("利率变化对股市的影响", "relationship")
    ]

    # 训练模型
    if SKLEARN_AVAILABLE:
        classifier.train_ml_models(sample_data)

    # 测试分类
    test_questions = [
        "解释夏普比率的定义",
        "获取最新的市场数据",
        "如何使用MACD指标",
        "平安证券和招商证券的区别",
        "评估投资组合风险",
        "预测明年股市走势",
        "通胀对债券市场的影响"
    ]

    for question in test_questions:
        result = classifier.classify(question)
        print(f"问题: {question}")
        print(f"分类: {result.category}")
        print(f"置信度: {result.confidence:.3f}")
        print(f"方法: {result.method}")
        print("-" * 50)