"""
统一基础接口 - 定义系统核心抽象
减少代码冗余，提供一致的接口规范
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import asyncio


# ============================================================================
# 通用类型定义
# ============================================================================

T = TypeVar('T')


class ServiceStatus(str, Enum):
    """服务状态"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ProcessingResult(str, Enum):
    """处理结果"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ServiceResponse(Generic[T]):
    """统一服务响应"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "metadata": self.metadata
        }


@dataclass
class ProcessingMetrics:
    """处理指标"""
    processing_time: float
    items_processed: int
    items_succeeded: int
    items_failed: int
    memory_used_mb: Optional[float] = None
    cache_hit_rate: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.items_processed == 0:
            return 0.0
        return self.items_succeeded / self.items_processed


# ============================================================================
# 服务基础接口
# ============================================================================

class IService(ABC):
    """服务基础接口"""

    @abstractmethod
    async def start(self) -> None:
        """启动服务"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止服务"""
        pass

    @abstractmethod
    def get_status(self) -> ServiceStatus:
        """获取服务状态"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class IConfigurable(ABC):
    """可配置接口"""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """配置服务"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        pass


class IMonitorable(ABC):
    """可监控接口"""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        """重置指标"""
        pass


class ICachable(ABC):
    """可缓存接口"""

    @abstractmethod
    async def get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取"""
        pass

    @abstractmethod
    async def set_to_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存"""
        pass

    @abstractmethod
    async def invalidate_cache(self, key: Optional[str] = None) -> None:
        """失效缓存"""
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        pass


# ============================================================================
# 处理器接口
# ============================================================================

class IProcessor(ABC, Generic[T]):
    """处理器基础接口"""

    @abstractmethod
    async def process(self, item: T, **kwargs) -> ServiceResponse[Any]:
        """处理单个项目"""
        pass

    @abstractmethod
    async def process_batch(self, items: List[T], **kwargs) -> ServiceResponse[List[Any]]:
        """批量处理"""
        pass


class IParser(ABC):
    """解析器接口"""

    @abstractmethod
    async def parse(self, source: Any, **kwargs) -> ServiceResponse[Any]:
        """解析数据源"""
        pass

    @abstractmethod
    def supports(self, source: Any) -> bool:
        """检查是否支持该数据源"""
        pass


class IRetriever(ABC):
    """检索器接口"""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> ServiceResponse[List[Dict[str, Any]]]:
        """检索相关内容"""
        pass

    @abstractmethod
    async def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> ServiceResponse[List[List[Dict[str, Any]]]]:
        """批量检索"""
        pass


class IEmbedder(ABC):
    """嵌入器接口"""

    @abstractmethod
    async def embed(self, text: str) -> ServiceResponse[List[float]]:
        """生成单个文本的嵌入"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> ServiceResponse[List[List[float]]]:
        """批量生成嵌入"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        pass


class IReranker(ABC):
    """重排序器接口"""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10
    ) -> ServiceResponse[List[Dict[str, Any]]]:
        """重新排序文档"""
        pass


class IExtractor(ABC):
    """抽取器接口"""

    @abstractmethod
    async def extract(self, content: str, **kwargs) -> ServiceResponse[List[Dict[str, Any]]]:
        """抽取信息"""
        pass

    @abstractmethod
    async def extract_batch(self, contents: List[str], **kwargs) -> ServiceResponse[List[List[Dict[str, Any]]]]:
        """批量抽取"""
        pass


# ============================================================================
# 存储接口
# ============================================================================

class IStorage(ABC):
    """存储接口"""

    @abstractmethod
    async def save(self, key: str, value: Any) -> ServiceResponse[bool]:
        """保存数据"""
        pass

    @abstractmethod
    async def load(self, key: str) -> ServiceResponse[Any]:
        """加载数据"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> ServiceResponse[bool]:
        """删除数据"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> ServiceResponse[bool]:
        """检查数据是否存在"""
        pass


class IVectorStorage(IStorage):
    """向量存储接口"""

    @abstractmethod
    async def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> ServiceResponse[List[Dict[str, Any]]]:
        """向量搜索"""
        pass

    @abstractmethod
    async def insert(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> ServiceResponse[bool]:
        """插入向量"""
        pass


class IGraphStorage(ABC):
    """图存储接口"""

    @abstractmethod
    async def add_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> ServiceResponse[bool]:
        """添加节点"""
        pass

    @abstractmethod
    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: str,
        properties: Dict[str, Any]
    ) -> ServiceResponse[bool]:
        """添加边"""
        pass

    @abstractmethod
    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> ServiceResponse[List[Any]]:
        """查询图"""
        pass


# ============================================================================
# 复合接口
# ============================================================================

class BaseService(IService, IConfigurable, IMonitorable):
    """基础服务类 - 实现通用服务功能"""

    def __init__(self):
        self._status = ServiceStatus.STOPPED
        self._config = {}
        self._metrics = {
            "start_time": None,
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_processing_time": 0.0
        }

    async def start(self) -> None:
        """启动服务"""
        self._status = ServiceStatus.STARTING
        # 子类实现具体启动逻辑
        self._status = ServiceStatus.RUNNING
        self._metrics["start_time"] = asyncio.get_event_loop().time()

    async def stop(self) -> None:
        """停止服务"""
        self._status = ServiceStatus.STOPPING
        # 子类实现具体停止逻辑
        self._status = ServiceStatus.STOPPED

    def get_status(self) -> ServiceStatus:
        """获取服务状态"""
        return self._status

    async def health_check(self) -> bool:
        """健康检查"""
        return self._status == ServiceStatus.RUNNING

    def configure(self, config: Dict[str, Any]) -> None:
        """配置服务"""
        self._config.update(config)

    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return self._config.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """重置指标"""
        self._metrics = {
            "start_time": None,
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_processing_time": 0.0
        }

    def _record_request(self, success: bool, processing_time: float):
        """记录请求"""
        self._metrics["requests_total"] += 1
        if success:
            self._metrics["requests_success"] += 1
        else:
            self._metrics["requests_failed"] += 1

        # 更新平均处理时间
        total = self._metrics["requests_total"]
        current_avg = self._metrics["avg_processing_time"]
        self._metrics["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )


class CacheDecorator(ICachable):
    """缓存装饰器 - 为任何服务添加缓存功能"""

    def __init__(self, cache_backend):
        self._cache = cache_backend

    async def get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取"""
        return await self._cache.get(key)

    async def set_to_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存"""
        await self._cache.set(key, value, expire=ttl)

    async def invalidate_cache(self, key: Optional[str] = None) -> None:
        """失效缓存"""
        if key:
            await self._cache.delete(key)
        else:
            await self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return getattr(self._cache, 'get_stats', lambda: {})()


# ============================================================================
# 工厂接口
# ============================================================================

class IFactory(ABC, Generic[T]):
    """工厂接口"""

    @abstractmethod
    def create(self, **kwargs) -> T:
        """创建实例"""
        pass

    @abstractmethod
    def create_batch(self, count: int, **kwargs) -> List[T]:
        """批量创建实例"""
        pass


# ============================================================================
# 适配器接口
# ============================================================================

class IAdapter(ABC, Generic[T]):
    """适配器接口"""

    @abstractmethod
    def adapt(self, source: Any) -> T:
        """适配源数据到目标格式"""
        pass

    @abstractmethod
    def adapt_batch(self, sources: List[Any]) -> List[T]:
        """批量适配"""
        pass
