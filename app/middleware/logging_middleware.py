"""
日志中间件
"""

import json
import time
from typing import Dict, Any
from app.middleware.base_middleware import BaseMiddleware
from app.core.logging import logger
from app.core.config import get_settings

class LoggingMiddleware(BaseMiddleware):
    """
    日志中间件

    记录RAG系统的执行日志，包括工具调用、参数、结果、性能等
    """

    def __init__(self, name: str = "LoggingMiddleware", **kwargs):
        super().__init__(name=name, **kwargs)
        self.settings = get_settings()
        self.log_level = kwargs.get("log_level", "INFO")
        self.log_inputs = kwargs.get("log_inputs", True)
        self.log_outputs = kwargs.get("log_outputs", True)

    async def before_tool_run(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行前的日志记录"""
        start_time = time.time()

        # 记录工具开始执行
        log_data = {
            "event": "tool_start",
            "tool_name": tool_name,
            "timestamp": start_time,
            "inputs": inputs if self.log_inputs else {"logged": False},
            "middleware": self.name
        }

        logger.info(f"[{self.name}] Tool '{tool_name}' started", extra=log_data)

        # 添加开始时间到输入中，供后续使用
        inputs["_middleware_start_time"] = start_time
        return inputs

    async def after_tool_run(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """工具执行后的日志记录"""
        end_time = time.time()

        # 计算执行时间
        start_time = inputs.pop("_middleware_start_time", end_time)
        actual_execution_time = end_time - start_time

        # 准备日志数据
        log_data = {
            "event": "tool_end",
            "tool_name": tool_name,
            "timestamp": end_time,
            "execution_time": actual_execution_time,
            "provided_execution_time": execution_time,
            "success": outputs.get("success", False),
            "output_size": len(str(outputs)),
            "middleware": self.name
        }

        if self.log_outputs:
            # 记录输出摘要而非完整输出，避免日志过大
            output_summary = {
                "success": outputs.get("success"),
                "result_count": len(outputs.get("documents", [])) if "documents" in outputs else 0,
                "has_answer": bool(outputs.get("answer")),
                "error": outputs.get("error")
            }
            log_data["outputs_summary"] = output_summary

        # 根据执行结果选择日志级别
        if outputs.get("success", False):
            logger.info(
                f"[{self.name}] Tool '{tool_name}' completed successfully in {actual_execution_time:.3f}s",
                extra=log_data
            )
        else:
            logger.warning(
                f"[{self.name}] Tool '{tool_name}' failed in {actual_execution_time:.3f}s",
                extra=log_data
            )

        return outputs

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """LangChain工具开始回调"""
        if not self.enabled:
            return

        tool_name = serialized.get("name", "unknown")
        logger.debug(
            f"[{self.name}] LangChain tool '{tool_name}' started",
            extra={
                "event": "langchain_tool_start",
                "tool_name": tool_name,
                "input_str": input_str[:200] + "..." if len(input_str) > 200 else input_str
            }
        )

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """LangChain工具结束回调"""
        if not self.enabled:
            return

        logger.debug(
            f"[{self.name}] LangChain tool completed",
            extra={
                "event": "langchain_tool_end",
                "output_length": len(output),
                "output_preview": output[:100] + "..." if len(output) > 100 else output
            }
        )

    async def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """LangChain工具错误回调"""
        if not self.enabled:
            return

        logger.error(
            f"[{self.name}] LangChain tool error: {str(error)}",
            extra={
                "event": "langchain_tool_error",
                "error_type": type(error).__name__,
                "error_message": str(error)
            },
            exc_info=True
        )

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """LangChain链开始回调"""
        if not self.enabled:
            return

        chain_name = serialized.get("name", "unknown")
        logger.debug(
            f"[{self.name}] LangChain chain '{chain_name}' started",
            extra={
                "event": "langchain_chain_start",
                "chain_name": chain_name,
                "input_keys": list(inputs.keys())
            }
        )

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """LangChain链结束回调"""
        if not self.enabled:
            return

        logger.debug(
            f"[{self.name}] LangChain chain completed",
            extra={
                "event": "langchain_chain_end",
                "output_keys": list(outputs.keys())
            }
        )

    async def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """LangChain链错误回调"""
        if not self.enabled:
            return

        logger.error(
            f"[{self.name}] LangChain chain error: {str(error)}",
            extra={
                "event": "langchain_chain_error",
                "error_type": type(error).__name__,
                "error_message": str(error)
            },
            exc_info=True
        )