"""
服务基类模块
提供所有服务类的公共功能，减少代码重复
"""

from app.core.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)
from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseService(ABC):
    """
    通用服务基类

    提供所有服务类的公共功能：
    - 自动初始化logger
    - 标准化的health_check方法
    - 通用的to_dict方法

    使用示例：
        class MyService(BaseService):
            def __init__(self):
                super().__init__()
                # self.logger已经自动初始化
    """

    def __init__(self):
        """初始化服务"""
        self._setup_logger()

    def _setup_logger(self):
        """设置logger"""
        self.logger = logging.getLogger(self.__class__.__name__)

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        返回服务的健康状态，子类可以重写此方法以提供更详细的信息

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "status": "healthy",
            "service": self.__class__.__name__,
            "type": "base"
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        将服务对象转换为字典

        返回服务对象的非私有属性字典

        Returns:
            Dict[str, Any]: 服务对象的字典表示
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }

    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}()"


class BaseModelService(BaseService):
    """
    模型服务基类

    为需要加载/卸载模型的服务提供通用接口

    使用示例：
        class MyModelService(BaseModelService):
            async def load_model(self, model_path: str):
                # 实现模型加载逻辑
                pass
    """

    def __init__(self):
        """初始化模型服务"""
        super().__init__()
        self._model = None
        self._model_loaded = False

    @abstractmethod
    async def load_model(self, model_path: str, **kwargs) -> None:
        """
        加载模型

        Args:
            model_path: 模型路径
            **kwargs: 其他模型参数
        """
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """卸载模型并释放资源"""
        pass

    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            bool: 模型是否已加载
        """
        return self._model_loaded

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查（包含模型状态）

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        base_health = super().health_check()
        base_health.update({
            "type": "model_service",
            "model_loaded": self._model_loaded
        })
        return base_health


class BaseStorageService(BaseService):
    """
    存储服务基类

    为需要连接管理的服务提供通用接口

    使用示例：
        class MyStorageService(BaseStorageService):
            async def connect(self):
                # 实现连接逻辑
                pass
    """

    def __init__(self):
        """初始化存储服务"""
        super().__init__()
        self._connected = False
        self._connection = None

    @abstractmethod
    async def connect(self, **kwargs) -> None:
        """
        建立连接

        Args:
            **kwargs: 连接参数
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接并释放资源"""
        pass

    def is_connected(self) -> bool:
        """
        检查是否已连接

        Returns:
            bool: 是否已连接
        """
        return self._connected

    async def ensure_connected(self) -> None:
        """确保已连接，如果未连接则自动连接"""
        if not self._connected:
            await self.connect()

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查（包含连接状态）

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        base_health = super().health_check()
        base_health.update({
            "type": "storage_service",
            "connected": self._connected
        })
        return base_health


class BaseAnalysisService(BaseService):
    """
    分析服务基类

    为数据分析类服务提供通用统计和转换方法

    使用示例：
        class MyAnalysisService(BaseAnalysisService):
            def analyze(self, data):
                # 实现分析逻辑
                pass
    """

    def __init__(self):
        """初始化分析服务"""
        super().__init__()

    def calculate_statistics(self, data: list) -> Dict[str, Any]:
        """
        计算基本统计信息

        Args:
            data: 数值列表

        Returns:
            Dict[str, Any]: 统计信息（count, sum, avg, min, max）
        """
        if not data:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

        return {
            "count": len(data),
            "sum": sum(data),
            "avg": sum(data) / len(data),
            "min": min(data),
            "max": max(data)
        }

    def calculate_trend(self, data: list) -> str:
        """
        计算数据趋势

        Args:
            data: 数值列表

        Returns:
            str: 趋势方向（"up", "down", "stable"）
        """
        if len(data) < 2:
            return "stable"

        first = data[0]
        last = data[-1]
        change = (last - first) / first if first != 0 else 0

        if change > 0.05:
            return "up"
        elif change < -0.05:
            return "down"
        else:
            return "stable"

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        base_health = super().health_check()
        base_health.update({
            "type": "analysis_service"
        })
        return base_health
