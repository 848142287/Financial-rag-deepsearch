"""
反馈优化机制
基于用户反馈和检索结果持续优化检索策略
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import create_engine, text


@dataclass
class FeedbackData:
    """反馈数据"""
    query_id: str
    query: str
    retrieval_method: str
    results: List[Dict[str, Any]]
    user_feedback: Dict[str, Any]  # {rating: int, clicked_docs: [], relevance: str}
    latency: float
    timestamp: str


@dataclass
class OptimizationMetrics:
    """优化指标"""
    timestamp: str
    avg_precision: float
    avg_recall: float
    avg_user_rating: float
    click_through_rate: float
    strategy_performance: Dict[str, float]


class FeedbackOptimizer:
    """反馈优化器"""

    def __init__(
        self,
        mysql_config: Dict[str, Any],
        feedback_file: str = "evaluation_results/feedback_data.json",
        metrics_file: str = "evaluation_results/optimization_metrics.json"
    ):
        """
        初始化反馈优化器

        Args:
            mysql_config: MySQL配置
            feedback_file: 反馈数据文件
            metrics_file: 优化指标文件
        """
        self.mysql_config = mysql_config
        self.feedback_file = feedback_file
        self.metrics_file = metrics_file
        self.engine = create_engine(
            f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@"
            f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
        )

        # 加载历史反馈数据
        self.feedback_history = self._load_feedback_history()

        # 当前检索权重
        self.current_weights = {
            "lightrag": 1.0,
            "graphrag": 0.8,
            "deepsearch": 1.2
        }

        # 加载优化指标
        self.optimization_metrics = self._load_optimization_metrics()

    def collect_feedback(
        self,
        query_id: str,
        query: str,
        retrieval_method: str,
        results: List[Dict[str, Any]],
        user_feedback: Dict[str, Any],
        latency: float
    ):
        """
        收集用户反馈

        Args:
            query_id: 查询ID
            query: 查询文本
            retrieval_method: 检索方法
            results: 检索结果
            user_feedback: 用户反馈
                - rating: 1-5评分
                - clicked_docs: 点击的文档ID列表
                - relevance: 相关性评价（"relevant", "partially_relevant", "not_relevant"）
            latency: 检索延迟（秒）
        """
        feedback = FeedbackData(
            query_id=query_id,
            query=query,
            retrieval_method=retrieval_method,
            results=results,
            user_feedback=user_feedback,
            latency=latency,
            timestamp=datetime.now().isoformat()
        )

        # 添加到历史记录
        self.feedback_history.append(asdict(feedback))

        # 保存到文件
        self._save_feedback_history()

        print(f"✅ 已收集反馈: {query_id}")

    def analyze_feedback(self) -> Dict[str, Any]:
        """
        分析反馈数据

        Returns:
            分析结果
        """
        if not self.feedback_history:
            return {"error": "No feedback data available"}

        # 统计分析
        total_feedbacks = len(self.feedback_history)

        # 按检索方法统计
        method_stats = defaultdict(lambda: {
            "count": 0,
            "total_rating": 0,
            "total_latency": 0,
            "clicks": 0,
            "relevant_count": 0
        })

        for feedback in self.feedback_history:
            method = feedback["retrieval_method"]
            rating = feedback["user_feedback"].get("rating", 0)
            latency = feedback["latency"]
            clicks = len(feedback["user_feedback"].get("clicked_docs", []))
            relevance = feedback["user_feedback"].get("relevance", "")

            method_stats[method]["count"] += 1
            method_stats[method]["total_rating"] += rating
            method_stats[method]["total_latency"] += latency
            method_stats[method]["clicks"] += clicks

            if relevance == "relevant":
                method_stats[method]["relevant_count"] += 1

        # 计算平均指标
        analysis = {}
        for method, stats in method_stats.items():
            count = stats["count"]

            analysis[method] = {
                "avg_rating": stats["total_rating"] / count if count > 0 else 0,
                "avg_latency": stats["total_latency"] / count if count > 0 else 0,
                "avg_clicks": stats["clicks"] / count if count > 0 else 0,
                "relevance_rate": stats["relevant_count"] / count if count > 0 else 0,
                "total_queries": count
            }

        return {
            "total_feedbacks": total_feedbacks,
            "method_performance": analysis,
            "timestamp": datetime.now().isoformat()
        }

    def update_weights(self) -> Dict[str, float]:
        """
        基于反馈更新检索权重

        Returns:
            更新后的权重
        """
        analysis = self.analyze_feedback()

        if "method_performance" not in analysis:
            return self.current_weights

        method_performance = analysis["method_performance"]

        # 基于平均评分调整权重
        max_rating = max(
            perf.get("avg_rating", 0)
            for perf in method_performance.values()
        ) if method_performance else 1.0

        new_weights = {}
        for method, weight in self.current_weights.items():
            if method in method_performance:
                perf = method_performance[method]
                # 基于评分调整权重：评分越高，权重越高
                rating_ratio = perf.get("avg_rating", 0) / max_rating if max_rating > 0 else 1.0

                # 平滑更新：70%旧权重 + 30%新权重
                new_weights[method] = weight * 0.7 + rating_ratio * 0.3
            else:
                new_weights[method] = weight

        # 归一化权重
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {
                method: weight / total_weight * 3  # 保持总权重为3
                for method, weight in new_weights.items()
            }

        self.current_weights = new_weights

        # 保存优化指标
        self._save_optimization_metrics(analysis)

        print(f"✅ 已更新权重: {new_weights}")

        return new_weights

    def get_optimization_suggestions(self) -> List[str]:
        """
        获取优化建议

        Returns:
            建议列表
        """
        suggestions = []

        analysis = self.analyze_feedback()

        if "method_performance" not in analysis:
            return ["暂无足够数据生成建议"]

        method_performance = analysis["method_performance"]

        # 检查各检索方法的性能
        for method, perf in method_performance.items():
            avg_rating = perf.get("avg_rating", 0)
            avg_latency = perf.get("avg_latency", 0)
            relevance_rate = perf.get("relevance_rate", 0)

            # 评分低
            if avg_rating < 3.0:
                suggestions.append(
                    f"⚠️ {method} 平均评分较低 ({avg_rating:.2f})，建议优化检索算法或调整参数"
                )

            # 延迟高
            if avg_latency > 2.0:
                suggestions.append(
                    f"⚠️ {method} 延迟较高 ({avg_latency:.2f}s)，建议优化查询性能或增加缓存"
                )

            # 相关性低
            if relevance_rate < 0.7:
                suggestions.append(
                    f"⚠️ {method} 相关性较低 ({relevance_rate:.2%})，建议改进检索策略或查询理解"
                )

        # 总体建议
        total_feedbacks = analysis.get("total_feedbacks", 0)
        if total_feedbacks < 100:
            suggestions.append(
                f"ℹ️ 反馈数据较少 ({total_feedbacks} 条)，建议收集更多反馈以获得更准确的优化建议"
            )

        if not suggestions:
            suggestions.append("✅ 各项指标表现良好，继续保持！")

        return suggestions

    def _load_feedback_history(self) -> List[Dict[str, Any]]:
        """加载历史反馈数据"""
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("feedbacks", [])
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Failed to load feedback history: {e}")
            return []

    def _save_feedback_history(self):
        """保存反馈历史"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)

        data = {
            "last_updated": datetime.now().isoformat(),
            "total_feedbacks": len(self.feedback_history),
            "feedbacks": self.feedback_history
        }

        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_optimization_metrics(self) -> List[Dict[str, Any]]:
        """加载优化指标"""
        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("metrics", [])
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Failed to load optimization metrics: {e}")
            return []

    def _save_optimization_metrics(self, analysis: Dict[str, Any]):
        """保存优化指标"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

        # 创建指标记录
        metric = OptimizationMetrics(
            timestamp=datetime.now().isoformat(),
            avg_precision=0.0,  # 需要从RAGAS评估获取
            avg_recall=0.0,
            avg_user_rating=sum(
                perf.get("avg_rating", 0) * perf.get("total_queries", 0)
                for perf in analysis.get("method_performance", {}).values()
            ) / sum(perf.get("total_queries", 1) for perf in analysis.get("method_performance", {}).values())
            if analysis.get("method_performance") else 0,
            click_through_rate=0.0,  # 需要计算
            strategy_performance=analysis.get("method_performance", {})
        )

        # 加载现有指标
        metrics = self._load_optimization_metrics()
        metrics.append(asdict(metric))

        # 保留最近100条记录
        metrics = metrics[-100:]

        data = {
            "last_updated": datetime.now().isoformat(),
            "total_records": len(metrics),
            "metrics": metrics
        }

        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def generate_feedback_report(
        self,
        output_file: str = "evaluation_results/reports/feedback_report.md"
    ) -> str:
        """
        生成反馈分析报告

        Args:
            output_file: 输出文件路径

        Returns:
            报告内容
        """
        analysis = self.analyze_feedback()
        suggestions = self.get_optimization_suggestions()

        report_lines = [
            "# 反馈分析与优化报告",
            f"",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            "=" * 80,
            "反馈数据概览",
            "=" * 80,
            f"",
            f"总反馈数: {analysis.get('total_feedbacks', 0)}",
            f"",
        ]

        # 各检索方法性能
        if "method_performance" in analysis:
            report_lines.extend([
                "=" * 80,
                "检索方法性能对比",
                "=" * 80,
                f""
            ])

            for method, perf in analysis["method_performance"].items():
                report_lines.extend([
                    f"## {method.upper()}",
                    f"",
                    f"查询次数: {perf.get('total_queries', 0)}",
                    f"平均评分: {perf.get('avg_rating', 0):.2f}/5.0",
                    f"平均延迟: {perf.get('avg_latency', 0):.2f}秒",
                    f"平均点击数: {perf.get('avg_clicks', 0):.2f}",
                    f"相关性率: {perf.get('relevance_rate', 0):.2%}",
                    f""
                ])

        # 优化建议
        report_lines.extend([
            "=" * 80,
            "优化建议",
            "=" * 80,
            f""
        ])

        for suggestion in suggestions:
            report_lines.append(f"- {suggestion}")

        report_lines.extend([
            f"",
            "=" * 80,
            "当前权重配置",
            "=" * 80,
            f""
        ])

        for method, weight in self.current_weights.items():
            report_lines.append(f"- {method}: {weight:.2f}")

        report_lines.append("")
        report_content = "\n".join(report_lines)

        # 保存报告
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✅ 反馈报告已保存到: {output_file}")

        return report_content

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前检索权重"""
        return self.current_weights.copy()

    def set_weights(self, weights: Dict[str, float]):
        """
        设置检索权重

        Args:
            weights: 权重字典
        """
        self.current_weights = weights
        print(f"✅ 已更新权重: {weights}")


import os
