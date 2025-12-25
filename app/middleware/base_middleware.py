"""
基础中间件抽象类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from langchain_core.callbacks import BaseCallbackHandler
import time

class BaseMiddleware(BaseCallbackHandler, ABC):
    """
    RAG系统基础中间件抽象类

    继承自LangChain的BaseCallbackHandler，提供统一的中间件接口
    """

    def __init__(self, name: str = "BaseMiddleware", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.enabled = True

    def enable(self):
        """启用中间件"""
        self.enabled = True

    def disable(self):
        """禁用中间件"""
        self.enabled = False

    @abstractmethod
    async def before_tool_run(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具执行前的处理

        Args:
            tool_name: 工具名称
            inputs: 工具输入参数

        Returns:
            处理后的输入参数
        """
        pass

    @abstractmethod
    async def after_tool_run(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        工具执行后的处理

        Args:
            tool_name: 工具名称
            inputs: 工具输入参数
            outputs: 工具输出结果
            execution_time: 执行时间

        Returns:
            处理后的输出结果
        """
        pass

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """LangChain工具开始回调"""
        if not self.enabled:
            return

        tool_name = serialized.get("name", "unknown")
        # 这里可以添加工具开始时的处理逻辑

    async def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        """LangChain工具结束回调"""
        if not self.enabled:
            return

        # 这里可以添加工具结束时的处理逻辑

    async def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """LangChain工具错误回调"""
        if not self.enabled:
            return

        # 这里可以添加工具错误时的处理逻辑

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """LangChain链开始回调"""
        if not self.enabled:
            return

        # 这里可以添加链开始时的处理逻辑

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """LangChain链结束回调"""
        if not self.enabled:
            return

        # 这里可以添加链结束时的处理逻辑

    async def on_chain_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """LangChain链错误回调"""
        if not self.enabled:
            return

        # 这里可以添加链错误时的处理逻辑


class MiddlewareManager:
    """
    中间件管理器

    负责管理多个中间件的执行顺序和生命周期
    """

    def __init__(self):
        self.middlewares: List[BaseMiddleware] = []
        self.global_context: Dict[str, Any] = {}

    def add_middleware(self, middleware: BaseMiddleware):
        """添加中间件"""
        self.middlewares.append(middleware)

    def remove_middleware(self, middleware_name: str):
        """移除中间件"""
        self.middlewares = [
            m for m in self.middlewares
            if m.name != middleware_name
        ]

    def get_middleware(self, middleware_name: str) -> Optional[BaseMiddleware]:
        """获取中间件"""
        for middleware in self.middlewares:
            if middleware.name == middleware_name:
                return middleware
        return None

    async def execute_before_tool_run(
        self,
        tool_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行所有中间件的工具前处理"""
        current_inputs = inputs

        for middleware in self.middlewares:
            if middleware.enabled:
                try:
                    current_inputs = await middleware.before_tool_run(
                        tool_name, current_inputs
                    )
                except Exception as e:
                    # 中间件错误不应影响主要流程
                    print(f"Middleware {middleware.name} before_tool_run error: {e}")

        return current_inputs

    async def execute_after_tool_run(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """执行所有中间件的工具后处理"""
        current_outputs = outputs

        for middleware in self.middlewares:
            if middleware.enabled:
                try:
                    current_outputs = await middleware.after_tool_run(
                        tool_name, inputs, current_outputs, execution_time
                    )
                except Exception as e:
                    # 中间件错误不应影响主要流程
                    print(f"Middleware {middleware.name} after_tool_run error: {e}")

        return current_outputs

    def set_global_context(self, key: str, value: Any):
        """设置全局上下文"""
        self.global_context[key] = value

    def get_global_context(self, key: str, default: Any = None) -> Any:
        """获取全局上下文"""
        return self.global_context.get(key, default)