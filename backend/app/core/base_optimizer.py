"""
优化器基类
为所有优化器提供统一的接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from app.core.structured_logging import get_structured_logger
import time
from enum import Enum

logger = get_structured_logger(__name__)


class OptimizationStatus(Enum):
    """优化状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OptimizationConfig:
    """优化配置基类"""
    enabled: bool = True
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    timeout_seconds: Optional[float] = None
    verbose: bool = False
    save_intermediate_results: bool = False
    config_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    success: bool
    status: OptimizationStatus
    objective_value: float
    iterations: int
    execution_time: float
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'status': self.status.value,
            'objective_value': self.objective_value,
            'iterations': self.iterations,
            'execution_time': self.execution_time,
            'best_parameters': self.best_parameters,
            'optimization_history': self.optimization_history,
            'metadata': self.metadata,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class OptimizationMetrics:
    """优化指标"""
    timestamp: datetime
    iteration: int
    objective_value: float
    current_parameters: Dict[str, Any]
    improvement: float = 0.0
    convergence_score: float = 0.0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """
    优化器基类

    所有优化器都应该继承这个基类并实现其抽象方法
    """

    # 类变量：优化器类型
    optimizer_type: str = "base"
    optimizer_version: str = "1.0.0"

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化优化器

        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._optimization_history: List[OptimizationMetrics] = []
        self._current_iteration = 0
        self._start_time: Optional[float] = None
        self._is_running = False

    @property
    def optimizer_name(self) -> str:
        """获取优化器名称"""
        return self.__class__.__name__

    @abstractmethod
    async def optimize(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        initial_parameters: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        执行优化

        Args:
            objective_func: 目标函数，接收参数字典，返回优化目标值
            initial_parameters: 初始参数
            constraints: 约束条件

        Returns:
            OptimizationResult: 优化结果
        """
        pass

    @abstractmethod
    def should_stop(self, metrics: OptimizationMetrics) -> bool:
        """
        判断是否应该停止优化

        Args:
            metrics: 当前优化指标

        Returns:
            bool: True表示应该停止
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        验证参数有效性

        Args:
            parameters: 待验证的参数

        Returns:
            (is_valid, error_message): 是否有效和错误消息
        """
        if not parameters:
            return False, "参数不能为空"

        return True, None

    def evaluate_objective(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        parameters: Dict[str, Any]
    ) -> float:
        """
        评估目标函数

        Args:
            objective_func: 目标函数
            parameters: 参数

        Returns:
            float: 目标值
        """
        try:
            return objective_func(parameters)
        except Exception as e:
            self.logger.error(f"评估目标函数失败: {e}")
            raise

    def record_metrics(self, metrics: OptimizationMetrics):
        """
        记录优化指标

        Args:
            metrics: 优化指标
        """
        self._optimization_history.append(metrics)
        self._current_iteration = metrics.iteration

        if self.config.verbose:
            self.logger.info(
                f"Iteration {metrics.iteration}: "
                f"Objective={metrics.objective_value:.6f}, "
                f"Improvement={metrics.improvement:.6f}"
            )

    def get_optimization_history(self) -> List[OptimizationMetrics]:
        """获取优化历史"""
        return self._optimization_history.copy()

    def reset(self):
        """重置优化器状态"""
        self._optimization_history.clear()
        self._current_iteration = 0
        self._start_time = None
        self._is_running = False

    def get_optimizer_info(self) -> Dict[str, Any]:
        """获取优化器信息"""
        return {
            'name': self.optimizer_name,
            'type': self.optimizer_type,
            'version': self.optimizer_version,
            'config': {
                'enabled': self.config.enabled,
                'max_iterations': self.config.max_iterations,
                'convergence_threshold': self.config.convergence_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'verbose': self.config.verbose
            },
            'current_iteration': self._current_iteration,
            'is_running': self._is_running,
            'history_length': len(self._optimization_history)
        }

    def _create_result(
        self,
        success: bool,
        status: OptimizationStatus,
        objective_value: float,
        best_parameters: Dict[str, Any],
        error_message: Optional[str] = None
    ) -> OptimizationResult:
        """
        创建优化结果

        Args:
            success: 是否成功
            status: 优化状态
            objective_value: 目标值
            best_parameters: 最佳参数
            error_message: 错误消息

        Returns:
            OptimizationResult
        """
        execution_time = 0.0
        started_at = None
        completed_at = None

        if self._start_time:
            started_at = datetime.fromtimestamp(self._start_time)
            completed_at = datetime.now()
            execution_time = completed_at.timestamp() - self._start_time

        return OptimizationResult(
            success=success,
            status=status,
            objective_value=objective_value,
            iterations=self._current_iteration,
            execution_time=execution_time,
            best_parameters=best_parameters,
            optimization_history=[
                {
                    'iteration': m.iteration,
                    'objective_value': m.objective_value,
                    'improvement': m.improvement,
                    'convergence_score': m.convergence_score,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self._optimization_history
            ],
            metadata={
                'optimizer_name': self.optimizer_name,
                'optimizer_type': self.optimizer_type,
                'optimizer_version': self.optimizer_version
            },
            error_message=error_message,
            started_at=started_at,
            completed_at=completed_at
        )

    def _check_timeout(self) -> bool:
        """检查是否超时"""
        if not self.config.timeout_seconds or not self._start_time:
            return False

        elapsed = time.time() - self._start_time
        return elapsed >= self.config.timeout_seconds

    async def _execute_optimization(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        initial_parameters: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        optimization_logic: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        执行优化的通用逻辑

        Args:
            objective_func: 目标函数
            initial_parameters: 初始参数
            constraints: 约束条件
            optimization_logic: 具体的优化逻辑函数

        Returns:
            OptimizationResult
        """
        if not self.config.enabled:
            return self._create_result(
                success=False,
                status=OptimizationStatus.CANCELLED,
                objective_value=0.0,
                best_parameters=initial_parameters,
                error_message="优化器未启用"
            )

        # 验证参数
        is_valid, error_msg = self.validate_parameters(initial_parameters)
        if not is_valid:
            return self._create_result(
                success=False,
                status=OptimizationStatus.FAILED,
                objective_value=0.0,
                best_parameters=initial_parameters,
                error_message=error_msg
            )

        # 初始化
        self._is_running = True
        self._start_time = time.time()
        current_parameters = initial_parameters.copy()

        try:
            # 评估初始目标值
            current_value = self.evaluate_objective(objective_func, current_parameters)

            # 记录初始指标
            initial_metrics = OptimizationMetrics(
                timestamp=datetime.now(),
                iteration=0,
                objective_value=current_value,
                current_parameters=current_parameters.copy()
            )
            self.record_metrics(initial_metrics)

            # 执行优化逻辑
            if optimization_logic:
                result = await optimization_logic(
                    objective_func,
                    current_parameters,
                    constraints
                )
            else:
                result = await self.optimize(
                    objective_func,
                    current_parameters,
                    constraints
                )

            return result

        except Exception as e:
            self.logger.error(f"优化过程出错: {e}", exc_info=True)
            return self._create_result(
                success=False,
                status=OptimizationStatus.FAILED,
                objective_value=0.0,
                best_parameters=current_parameters,
                error_message=str(e)
            )

        finally:
            self._is_running = False


class OptimizerError(Exception):
    """优化器错误基类"""
    pass


class OptimizationTimeoutError(OptimizerError):
    """优化超时错误"""
    pass


class OptimizationConvergenceError(OptimizerError):
    """优化收敛错误"""
    pass


class ParameterValidationError(OptimizerError):
    """参数验证错误"""
    pass
