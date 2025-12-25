"""
RAG策略路由配置系统
支持动态配置和调优参数管理
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """阈值配置"""
    lightrag_score: float = 8.0      # LightRAG最小得分
    graphrag_score: float = 12.0     # Graph RAG最小得分
    agentic_score: float = 18.0      # Agentic RAG最小得分
    hybrid_score: float = 10.0       # Hybrid RAG最小得分

    # 置信度阈值
    min_confidence: float = 0.6      # 最小置信度
    high_confidence: float = 0.8     # 高置信度阈值
    excellent_confidence: float = 0.9 # 优秀置信度阈值

    # 响应时间阈值（毫秒）
    response_time_warning: float = 5000.0   # 响应时间警告阈值
    response_time_critical: float = 10000.0 # 响应时间严重阈值
    response_time_timeout: float = 30000.0  # 响应时间超时阈值

    # 结果数量阈值
    min_results: int = 3             # 最小结果数量
    max_results: int = 50            # 最大结果数量
    ideal_results: int = 15          # 理想结果数量


@dataclass
class WeightConfig:
    """权重配置"""
    # 决策因子权重
    entity_count: float = 0.30           # 实体数量权重
    relation_complexity: float = 0.25    # 关系复杂度权重
    time_sensitivity: float = 0.15       # 时间敏感度权重
    answer_granularity: float = 0.15     # 答案粒度权重
    query_simplicity: float = 0.10       # 查询简洁性权重
    historical_success: float = 0.05     # 历史成功率权重

    # 策略权重
    lightrag_weight: float = 0.25
    graphrag_weight: float = 0.30
    agentic_weight: float = 0.35
    hybrid_weight: float = 0.10

    # 融合权重
    source_reliability: float = 0.4      # 源可靠性权重
    recency: float = 0.3                # 时效性权重
    citation_count: float = 0.2         # 引用次数权重
    cross_validation: float = 0.1        # 交叉验证权重


@dataclass
class ConditionConfig:
    """条件配置"""
    # 降级条件
    downgrade_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "response_time_exceeds": 5000,    # 毫秒
        "user_feedback_score": 2.0,       # 1-5分，低于此分触发
        "error_rate_threshold": 0.1,      # 错误率阈值
        "confidence_drop_threshold": 0.3, # 置信度下降阈值
        "result_count_insufficient": 3    # 结果数量不足阈值
    })

    # 升级条件
    upgrade_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "follow_up_queries": 3,           # 连续追问次数
        "query_modification": True,       # 用户修改原查询
        "complexity_increase": 0.2,       # 复杂度增加阈值
        "user_satisfaction_low": 2.5,     # 用户满意度低阈值
        "insufficient_results": True       # 结果不足条件
    })

    # 动态调整条件
    dynamic_adjustment: Dict[str, Any] = field(default_factory=lambda: {
        "enable_auto_adjustment": True,
        "adjustment_interval": 300,        # 秒
        "min_samples_for_adjustment": 10,  # 最小样本数
        "max_weight_change": 0.2,          # 最大权重变化
        "adjustment_sensitivity": 0.1      # 调整敏感度
    })


@dataclass
class PerformanceConfig:
    """性能配置"""
    # 并发控制
    max_concurrent_strategies: int = 3
    strategy_timeout: float = 30.0         # 策略执行超时（秒）
    fusion_timeout: float = 10.0          # 融合超时（秒）

    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600                 # 缓存TTL（秒）
    max_cache_size: int = 1000           # 最大缓存大小

    # 资源限制
    max_memory_usage: float = 0.8        # 最大内存使用率
    max_cpu_usage: float = 0.8           # 最大CPU使用率

    # 性能监控
    enable_monitoring: bool = True
    monitoring_interval: int = 60        # 监控间隔（秒）
    performance_history_size: int = 1000 # 性能历史记录大小


@dataclass
class RAGRoutingConfig:
    """RAG路由配置"""
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    conditions: ConditionConfig = field(default_factory=ConditionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # 通用配置
    enable_mixed_execution: bool = True
    default_execution_mode: str = "adaptive"  # parallel_fusion, pipeline, adaptive
    enable_dynamic_adjustment: bool = True
    config_auto_reload: bool = True
    config_reload_interval: int = 60      # 秒

    # 调试和日志
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_execution_trace: bool = False
    max_trace_records: int = 500


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_file()
        self.config = RAGRoutingConfig()
        self.last_modified = 0
        self._load_config()
        self._setup_auto_reload()

    def _get_default_config_file(self) -> str:
        """获取默认配置文件路径"""
        # 尝试多个可能的配置文件位置
        possible_paths = [
            "config/rag_routing.yaml",
            "config/rag_routing.yml",
            "rag_routing.yaml",
            "rag_routing.yml"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 如果都不存在，使用默认路径
        return "config/rag_routing.yaml"

    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                if config_data:
                    self._update_config_from_dict(config_data)
                    logger.info(f"Configuration loaded from {self.config_file}")
                else:
                    logger.warning(f"Empty config file: {self.config_file}")
            else:
                logger.info(f"Config file not found: {self.config_file}, using defaults")
                self._create_default_config_file()

        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")

    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """从字典更新配置"""
        try:
            # 更新阈值配置
            if "thresholds" in config_data:
                thresholds_config = config_data["thresholds"]
                for key, value in thresholds_config.items():
                    if hasattr(self.config.thresholds, key):
                        setattr(self.config.thresholds, key, value)

            # 更新权重配置
            if "weights" in config_data:
                weights_config = config_data["weights"]
                for key, value in weights_config.items():
                    if hasattr(self.config.weights, key):
                        setattr(self.config.weights, key, value)

            # 更新条件配置
            if "conditions" in config_data:
                conditions_config = config_data["conditions"]
                if "downgrade_conditions" in conditions_config:
                    self.config.conditions.downgrade_conditions.update(
                        conditions_config["downgrade_conditions"]
                    )
                if "upgrade_conditions" in conditions_config:
                    self.config.conditions.upgrade_conditions.update(
                        conditions_config["upgrade_conditions"]
                    )
                if "dynamic_adjustment" in conditions_config:
                    self.config.conditions.dynamic_adjustment.update(
                        conditions_config["dynamic_adjustment"]
                    )

            # 更新性能配置
            if "performance" in config_data:
                performance_config = config_data["performance"]
                for key, value in performance_config.items():
                    if hasattr(self.config.performance, key):
                        setattr(self.config.performance, key, value)

            # 更新通用配置
            for key in ["enable_mixed_execution", "default_execution_mode",
                       "enable_dynamic_adjustment", "config_auto_reload",
                       "debug_mode", "log_level", "enable_execution_trace"]:
                if key in config_data:
                    setattr(self.config, key, config_data[key])

        except Exception as e:
            logger.error(f"Failed to update config from dict: {e}")

    def _create_default_config_file(self) -> None:
        """创建默认配置文件"""
        try:
            # 确保目录存在
            config_dir = os.path.dirname(self.config_file)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            # 生成默认配置
            default_config = asdict(self.config)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Default config file created: {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to create default config file: {e}")

    def _setup_auto_reload(self) -> None:
        """设置自动重载"""
        if self.config.config_auto_reload:
            # 这里可以设置一个定时任务来检查配置文件变化
            pass

    def get_config(self) -> RAGRoutingConfig:
        """获取当前配置"""
        return self.config

    def get_threshold_config(self) -> ThresholdConfig:
        """获取阈值配置"""
        return self.config.thresholds

    def get_weight_config(self) -> WeightConfig:
        """获取权重配置"""
        return self.config.weights

    def get_condition_config(self) -> ConditionConfig:
        """获取条件配置"""
        return self.config.conditions

    def get_performance_config(self) -> PerformanceConfig:
        """获取性能配置"""
        return self.config.performance

    def update_threshold(self, key: str, value: Union[float, int]) -> bool:
        """更新阈值配置"""
        try:
            if hasattr(self.config.thresholds, key):
                setattr(self.config.thresholds, key, value)
                self._save_config()
                logger.info(f"Threshold {key} updated to {value}")
                return True
            else:
                logger.warning(f"Unknown threshold key: {key}")
                return False
        except Exception as e:
            logger.error(f"Failed to update threshold {key}: {e}")
            return False

    def update_weight(self, key: str, value: float) -> bool:
        """更新权重配置"""
        try:
            if hasattr(self.config.weights, key):
                setattr(self.config.weights, key, value)
                self._save_config()
                logger.info(f"Weight {key} updated to {value}")
                return True
            else:
                logger.warning(f"Unknown weight key: {key}")
                return False
        except Exception as e:
            logger.error(f"Failed to update weight {key}: {e}")
            return False

    def update_condition(self, category: str, key: str, value: Any) -> bool:
        """更新条件配置"""
        try:
            if category == "downgrade_conditions":
                self.config.conditions.downgrade_conditions[key] = value
            elif category == "upgrade_conditions":
                self.config.conditions.upgrade_conditions[key] = value
            elif category == "dynamic_adjustment":
                self.config.conditions.dynamic_adjustment[key] = value
            else:
                logger.warning(f"Unknown condition category: {category}")
                return False

            self._save_config()
            logger.info(f"Condition {category}.{key} updated to {value}")
            return True
        except Exception as e:
            logger.error(f"Failed to update condition {category}.{key}: {e}")
            return False

    def _save_config(self) -> None:
        """保存配置到文件"""
        try:
            config_dict = asdict(self.config)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Configuration saved to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            old_config = asdict(self.config)
            self._load_config()
            new_config = asdict(self.config)

            if old_config != new_config:
                logger.info("Configuration reloaded with changes")
                return True
            else:
                logger.info("Configuration reloaded (no changes)")
                return True

        except Exception as e:
            logger.error(f"Failed to reload config: {e}")
            return False

    def validate_config(self) -> Dict[str, Any]:
        """验证配置有效性"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        try:
            # 验证阈值
            thresholds = self.config.thresholds
            if thresholds.lightrag_score < 0 or thresholds.lightrag_score > 20:
                validation_result["errors"].append("lightrag_score should be between 0 and 20")
                validation_result["valid"] = False

            if thresholds.min_confidence < 0 or thresholds.min_confidence > 1:
                validation_result["errors"].append("min_confidence should be between 0 and 1")
                validation_result["valid"] = False

            # 验证权重
            weights = self.config.weights
            total_weight = (
                weights.entity_count + weights.relation_complexity +
                weights.time_sensitivity + weights.answer_granularity +
                weights.query_simplicity + weights.historical_success
            )

            if abs(total_weight - 1.0) > 0.01:
                validation_result["warnings"].append(
                    f"Decision factor weights sum to {total_weight:.3f}, expected 1.0"
                )

            # 验证策略权重
            strategy_weight_sum = (
                weights.lightrag_weight + weights.graphrag_weight +
                weights.agentic_weight + weights.hybrid_weight
            )

            if abs(strategy_weight_sum - 1.0) > 0.01:
                validation_result["warnings"].append(
                    f"Strategy weights sum to {strategy_weight_sum:.3f}, expected 1.0"
                )

            # 验证性能配置
            performance = self.config.performance
            if performance.max_concurrent_strategies < 1 or performance.max_concurrent_strategies > 10:
                validation_result["warnings"].append(
                    "max_concurrent_strategies should be between 1 and 10"
                )

        except Exception as e:
            validation_result["errors"].append(f"Configuration validation error: {e}")
            validation_result["valid"] = False

        return validation_result

    def export_config(self, file_path: str) -> bool:
        """导出配置到文件"""
        try:
            config_dict = asdict(self.config)

            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Configuration exported to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            return False

    def import_config(self, file_path: str) -> bool:
        """从文件导入配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if config_data:
                self._update_config_from_dict(config_data)
                self._save_config()
                logger.info(f"Configuration imported from {file_path}")
                return True
            else:
                logger.error(f"Empty config file: {file_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "thresholds": {
                "lightrag_score": self.config.thresholds.lightrag_score,
                "graphrag_score": self.config.thresholds.graphrag_score,
                "agentic_score": self.config.thresholds.agentic_score,
                "min_confidence": self.config.thresholds.min_confidence,
                "response_time_warning": self.config.thresholds.response_time_warning
            },
            "weights": {
                "entity_count": self.config.weights.entity_count,
                "relation_complexity": self.config.weights.relation_complexity,
                "time_sensitivity": self.config.weights.time_sensitivity,
                "answer_granularity": self.config.weights.answer_granularity,
                "lightrag_weight": self.config.weights.lightrag_weight,
                "graphrag_weight": self.config.weights.graphrag_weight,
                "agentic_weight": self.config.weights.agentic_weight,
                "hybrid_weight": self.config.weights.hybrid_weight
            },
            "conditions": {
                "downgrade_response_time": self.config.conditions.downgrade_conditions["response_time_exceeds"],
                "downgrade_feedback_score": self.config.conditions.downgrade_conditions["user_feedback_score"],
                "upgrade_follow_up_queries": self.config.conditions.upgrade_conditions["follow_up_queries"],
                "enable_auto_adjustment": self.config.conditions.dynamic_adjustment["enable_auto_adjustment"]
            },
            "performance": {
                "max_concurrent_strategies": self.config.performance.max_concurrent_strategies,
                "strategy_timeout": self.config.performance.strategy_timeout,
                "enable_cache": self.config.performance.enable_cache,
                "enable_monitoring": self.config.performance.enable_monitoring
            },
            "general": {
                "enable_mixed_execution": self.config.enable_mixed_execution,
                "default_execution_mode": self.config.default_execution_mode,
                "enable_dynamic_adjustment": self.config.enable_dynamic_adjustment,
                "debug_mode": self.config.debug_mode
            }
        }


# 全局配置管理器实例
config_manager = ConfigManager()