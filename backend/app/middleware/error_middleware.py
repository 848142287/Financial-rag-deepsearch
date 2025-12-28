"""
错误处理中间件
"""

import traceback
import time
from typing import Dict, Any, Optional, Callable, List
from app.middleware.base_middleware import BaseMiddleware
from app.core.logging import logger
from app.core.config import get_settings

class ErrorMiddleware(BaseMiddleware):
    """
    错误处理中间件

    提供统一的错误处理、重试机制、降级策略和错误恢复功能
    """

    def __init__(
        self,
        name: str = "ErrorMiddleware",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        enable_fallback: bool = True,
        error_handlers: Dict[str, Callable] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.settings = get_settings()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.enable_fallback = enable_fallback
        self.error_handlers = error_handlers or {}

        # 错误统计
        self.error_stats = {
            "total_errors": 0,
            "retries_attempted": 0,
            "retries_successful": 0,
            "fallback_used": 0,
            "error_types": {}
        }

    async def before_tool_run(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行前设置错误处理上下文"""
        # 设置重试上下文
        inputs["_error_retry_count"] = 0
        inputs["_error_last_exception"] = None
        inputs["_error_context"] = {
            "tool_name": tool_name,
            "start_time": time.time(),
            "original_inputs": inputs.copy()
        }

        return inputs

    async def after_tool_run(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """工具执行后处理错误和记录统计"""
        error_context = inputs.get("_error_context", {})
        retry_count = inputs.get("_error_retry_count", 0)

        # 如果输出包含错误，尝试处理
        if not outputs.get("success", False):
            error = outputs.get("error", "Unknown error")
            await self._handle_error(tool_name, error, inputs, outputs)
        elif retry_count > 0:
            # 成功但经过了重试
            self.error_stats["retries_successful"] += 1
            logger.info(
                f"[{self.name}] Tool '{tool_name}' succeeded after {retry_count} retries",
                extra={
                    "tool_name": tool_name,
                    "retry_count": retry_count,
                    "execution_time": execution_time
                }
            )

        return outputs

    async def _handle_error(
        self,
        tool_name: str,
        error: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理错误"""
        retry_count = inputs.get("_error_retry_count", 0)
        last_exception = inputs.get("_error_last_exception")

        # 更新错误统计
        self.error_stats["total_errors"] += 1
        error_type = type(last_exception).__name__ if last_exception else "Unknown"
        self.error_stats["error_types"][error_type] = self.error_stats["error_types"].get(error_type, 0) + 1

        # 记录错误
        logger.error(
            f"[{self.name}] Error in tool '{tool_name}': {error}",
            extra={
                "tool_name": tool_name,
                "error": error,
                "error_type": error_type,
                "retry_count": retry_count,
                "traceback": traceback.format_exc() if last_exception else None
            }
        )

        # 检查是否应该重试
        if await self._should_retry(tool_name, error, retry_count, last_exception):
            return await self._retry_tool(tool_name, inputs)

        # 检查是否使用降级策略
        if self.enable_fallback:
            fallback_result = await self._try_fallback(tool_name, inputs, error)
            if fallback_result:
                self.error_stats["fallback_used"] += 1
                logger.info(
                    f"[{self.name}] Used fallback for tool '{tool_name}'",
                    extra={"tool_name": tool_name, "fallback_success": True}
                )
                return fallback_result

        # 无法恢复，返回错误结果
        return {
            "success": False,
            "error": error,
            "error_type": error_type,
            "retry_count": retry_count,
            "fallback_used": False,
            "tool_name": tool_name
        }

    async def _should_retry(
        self,
        tool_name: str,
        error: str,
        retry_count: int,
        last_exception: Optional[Exception]
    ) -> bool:
        """判断是否应该重试"""
        if retry_count >= self.max_retries:
            return False

        # 检查错误类型是否可重试
        non_retryable_errors = [
            "AuthenticationError",
            "AuthorizationError",
            "ValidationError",
            "ValueError"
        ]

        if last_exception and type(last_exception).__name__ in non_retryable_errors:
            return False

        # 检查错误消息中是否包含不可重试的关键词
        non_retryable_keywords = [
            "invalid api key",
            "unauthorized",
            "forbidden",
            "invalid parameter",
            "malformed request"
        ]

        error_lower = error.lower()
        for keyword in non_retryable_keywords:
            if keyword in error_lower:
                return False

        return True

    async def _retry_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """重试工具执行"""
        retry_count = inputs.get("_error_retry_count", 0) + 1
        inputs["_error_retry_count"] = retry_count

        # 计算延迟时间（指数退避）
        delay = self.retry_delay * (self.retry_backoff_factor ** (retry_count - 1))

        logger.info(
            f"[{self.name}] Retrying tool '{tool_name}' (attempt {retry_count}/{self.max_retries})",
            extra={
                "tool_name": tool_name,
                "retry_count": retry_count,
                "delay": delay
            }
        )

        self.error_stats["retries_attempted"] += 1

        # 等待后重试
        await asyncio.sleep(delay)

        # 这里应该触发工具重新执行
        # 实际实现中需要与工具系统集成
        # 这里返回一个标记，表示需要重试
        return {
            "success": False,
            "error": "Retry requested",
            "retry_count": retry_count,
            "retry_requested": True,
            "tool_name": tool_name
        }

    async def _try_fallback(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        error: str
    ) -> Optional[Dict[str, Any]]:
        """尝试降级策略"""
        # 根据工具类型提供不同的降级策略
        fallback_handlers = {
            "document_retrieval": self._fallback_retrieval,
            "rag_generation": self._fallback_generation,
            "multimodal_processing": self._fallback_multimodal,
            "knowledge_graph": self._fallback_knowledge_graph,
            "rag_evaluation": self._fallback_evaluation
        }

        handler = fallback_handlers.get(tool_name)
        if handler:
            try:
                return await handler(inputs, error)
            except Exception as e:
                logger.error(
                    f"[{self.name}] Fallback handler failed for tool '{tool_name}': {e}",
                    extra={"tool_name": tool_name, "fallback_error": str(e)}
                )

        return None

    async def _fallback_retrieval(self, inputs: Dict[str, Any], error: str) -> Dict[str, Any]:
        """文档检索降级策略"""
        return {
            "success": True,
            "documents": [],
            "query": inputs.get("query", ""),
            "fallback_used": True,
            "fallback_reason": f"Original retrieval failed: {error}",
            "retrieval_mode": "fallback_empty"
        }

    async def _fallback_generation(self, inputs: Dict[str, Any], error: str) -> Dict[str, Any]:
        """生成降级策略"""
        return {
            "success": True,
            "answer": f"I apologize, but I encountered an error while processing your request: {error}",
            "citations": [],
            "fallback_used": True,
            "fallback_reason": f"Original generation failed: {error}"
        }

    async def _fallback_multimodal(self, inputs: Dict[str, Any], error: str) -> Dict[str, Any]:
        """多模态处理降级策略"""
        return {
            "success": True,
            "analysis": {
                "description": "Unable to process the image/document due to a system error.",
                "error": error,
                "fallback": True
            },
            "fallback_used": True,
            "fallback_reason": f"Original multimodal processing failed: {error}"
        }

    async def _fallback_knowledge_graph(self, inputs: Dict[str, Any], error: str) -> Dict[str, Any]:
        """知识图谱降级策略"""
        return {
            "success": True,
            "result": {},
            "fallback_used": True,
            "fallback_reason": f"Original graph operation failed: {error}",
            "error": error
        }

    async def _fallback_evaluation(self, inputs: Dict[str, Any], error: str) -> Dict[str, Any]:
        """评估降级策略"""
        return {
            "success": True,
            "scores": {
                "overall_score": 0.0,
                "context_precision": 0.0,
                "answer_relevancy": 0.0,
                "faithfulness": 0.0
            },
            "fallback_used": True,
            "fallback_reason": f"Original evaluation failed: {error}",
            "error": error
        }

    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "retries_attempted": self.error_stats["retries_attempted"],
            "retries_successful": self.error_stats["retries_successful"],
            "fallback_used": self.error_stats["fallback_used"],
            "retry_success_rate": (
                self.error_stats["retries_successful"] / self.error_stats["retries_attempted"]
                if self.error_stats["retries_attempted"] > 0 else 0.0
            ),
            "error_types": self.error_stats["error_types"].copy()
        }

    def reset_error_stats(self):
        """重置错误统计"""
        self.error_stats = {
            "total_errors": 0,
            "retries_attempted": 0,
            "retries_successful": 0,
            "fallback_used": 0,
            "error_types": {}
        }