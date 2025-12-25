"""
统一服务注册表和依赖注入系统

管理所有服务的生命周期、依赖注入和统一接口
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import logging
from functools import wraps
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import get_db

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceLifetime(str, Enum):
    """服务生命周期"""
    SINGLETON = "singleton"  # 单例
    TRANSIENT = "transient"  # 每次创建新实例
    SCOPED = "scoped"        # 作用域内单例

class ServiceStatus(str, Enum):
    """服务状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class IService(ABC):
    """服务接口基类"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化服务"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止服务"""
        pass

    @abstractmethod
    def get_status(self) -> ServiceStatus:
        """获取服务状态"""
        pass

class ServiceDescriptor:
    """服务描述符"""

    def __init__(
        self,
        service_type: Type[T],
        implementation: Type[T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        dependencies: Optional[List[Type]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.dependencies = dependencies or []
        self.config = config or {}
        self.instance: Optional[T] = None
        self.status = ServiceStatus.STOPPED

class ServiceRegistry:
    """服务注册表"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._initialization_order: List[Type] = []

    def register_singleton(
        self,
        service_type: Type[T],
        implementation: Type[T],
        dependencies: Optional[List[Type]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册单例服务"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            dependencies=dependencies,
            config=config
        )
        self._services[service_type] = descriptor
        self._update_initialization_order()

    def register_transient(
        self,
        service_type: Type[T],
        implementation: Type[T],
        dependencies: Optional[List[Type]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册瞬态服务"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            dependencies=dependencies,
            config=config
        )
        self._services[service_type] = descriptor
        self._update_initialization_order()

    def register_scoped(
        self,
        service_type: Type[T],
        implementation: Type[T],
        dependencies: Optional[List[Type]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册作用域服务"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED,
            dependencies=dependencies,
            config=config
        )
        self._services[service_type] = descriptor
        self._update_initialization_order()

    def get(self, service_type: Type[T], scope_id: Optional[str] = None) -> T:
        """获取服务实例"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type} not registered")

        descriptor = self._services[service_type]

        # 处理单例
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.instance is None:
                descriptor.instance = self._create_instance(descriptor)
                descriptor.status = ServiceStatus.RUNNING
            return descriptor.instance

        # 处理作用域
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if scope_id is None:
                scope_id = "default"

            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}

            if service_type not in self._scoped_instances[scope_id]:
                self._scoped_instances[scope_id][service_type] = self._create_instance(descriptor)

            return self._scoped_instances[scope_id][service_type]

        # 处理瞬态
        else:
            return self._create_instance(descriptor)

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        # 解析依赖
        dependencies = {}
        for dep_type in descriptor.dependencies:
            dependencies[dep_type] = self.get(dep_type)

        # 创建实例
        instance = descriptor.implementation(**dependencies)

        # 应用配置
        if descriptor.config:
            for key, value in descriptor.config.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

        return instance

    def _update_initialization_order(self) -> None:
        """更新初始化顺序（基于依赖关系）"""
        visited = set()
        order = []

        def visit(service_type: Type):
            if service_type in visited:
                return

            if service_type in self._services:
                descriptor = self._services[service_type]
                for dep in descriptor.dependencies:
                    visit(dep)

            visited.add(service_type)
            order.append(service_type)

        for service_type in self._services:
            visit(service_type)

        self._initialization_order = order

    async def initialize_all(self) -> None:
        """初始化所有服务"""
        logger.info("Initializing all services...")

        for service_type in self._initialization_order:
            descriptor = self._services.get(service_type)
            if descriptor:
                try:
                    descriptor.status = ServiceStatus.INITIALIZING
                    instance = self.get(service_type)

                    if isinstance(instance, IService):
                        await instance.initialize()

                    descriptor.status = ServiceStatus.RUNNING
                    logger.info(f"Service {service_type.__name__} initialized successfully")

                except Exception as e:
                    descriptor.status = ServiceStatus.ERROR
                    logger.error(f"Failed to initialize service {service_type.__name__}: {e}")
                    raise

    async def stop_all(self) -> None:
        """停止所有服务"""
        logger.info("Stopping all services...")

        # 按相反顺序停止服务
        for service_type in reversed(self._initialization_order):
            descriptor = self._services.get(service_type)
            if descriptor and descriptor.instance:
                try:
                    descriptor.status = ServiceStatus.STOPPING

                    if isinstance(descriptor.instance, IService):
                        await descriptor.instance.stop()

                    descriptor.status = ServiceStatus.STOPPED
                    logger.info(f"Service {service_type.__name__} stopped successfully")

                except Exception as e:
                    descriptor.status = ServiceStatus.ERROR
                    logger.error(f"Failed to stop service {service_type.__name__}: {e}")

        # 清理作用域实例
        self._scoped_instances.clear()

    def clear_scope(self, scope_id: str) -> None:
        """清理作用域实例"""
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]

    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务状态"""
        status = {}
        for service_type, descriptor in self._services.items():
            status[service_type.__name__] = {
                "status": descriptor.status.value,
                "lifetime": descriptor.lifetime.value,
                "dependencies": [dep.__name__ for dep in descriptor.dependencies],
                "has_instance": descriptor.instance is not None
            }
        return status

# 全局服务注册表实例
service_registry = ServiceRegistry()

# 依赖注入装饰器
def inject(service_type: Type[T], scope_id: Optional[str] = None):
    """依赖注入装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            service = service_registry.get(service_type, scope_id)
            return await func(*args, service, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            service = service_registry.get(service_type, scope_id)
            return func(*args, service, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# 作用域管理器
class ScopeManager:
    """作用域管理器"""

    def __init__(self, scope_id: Optional[str] = None):
        self.scope_id = scope_id or f"scope_{id(self)}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        service_registry.clear_scope(self.scope_id)

    def get_service(self, service_type: Type[T]) -> T:
        """在当前作用域中获取服务"""
        return service_registry.get(service_type, self.scope_id)

# 服务健康检查接口
class HealthChecker:
    """服务健康检查器"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry

    async def check_all_services(self) -> Dict[str, Any]:
        """检查所有服务健康状态"""
        results = {}
        overall_status = "healthy"

        for service_type, descriptor in self.registry._services.items():
            service_name = service_type.__name__

            try:
                if descriptor.instance and hasattr(descriptor.instance, 'health_check'):
                    health_result = await descriptor.instance.health_check()
                    results[service_name] = {
                        "status": health_result.get("status", "unknown"),
                        "details": health_result.get("details", {}),
                        "response_time": health_result.get("response_time")
                    }
                else:
                    results[service_name] = {
                        "status": "healthy" if descriptor.status == ServiceStatus.RUNNING else "error",
                        "details": {"service_status": descriptor.status.value}
                    }

                if results[service_name]["status"] not in ["healthy", "degraded"]:
                    overall_status = "unhealthy"
                elif overall_status == "healthy" and results[service_name]["status"] == "degraded":
                    overall_status = "degraded"

            except Exception as e:
                results[service_name] = {
                    "status": "error",
                    "details": {"error": str(e)}
                }
                overall_status = "unhealthy"

        return {
            "overall_status": overall_status,
            "services": results,
            "timestamp": "2024-01-01T00:00:00Z"
        }

# 应用生命周期管理
class ApplicationLifecycle:
    """应用生命周期管理"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.health_checker = HealthChecker(registry)

    async def startup(self) -> None:
        """应用启动"""
        logger.info("Starting application...")
        await self.registry.initialize_all()
        logger.info("Application started successfully")

    async def shutdown(self) -> None:
        """应用关闭"""
        logger.info("Shutting down application...")
        await self.registry.stop_all()
        logger.info("Application shutdown completed")

# 全局应用生命周期实例
app_lifecycle = ApplicationLifecycle(service_registry)