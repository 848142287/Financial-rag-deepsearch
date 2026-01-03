"""
流式响应增强服务
支持思考过程(Reasoning)输出，适用于DeepSeek-R1等推理模型
从 swxy/backend 移植并优化
使用现有系统的 llm_service (DeepSeek/Qwen)
"""

from app.core.structured_logging import get_structured_logger
import json
from typing import AsyncGenerator, Dict, Any, Optional, List

logger = get_structured_logger(__name__)


class StreamingResponseService:
    """流式响应服务 - 支持思考过程和正常回答的分离输出"""

    def __init__(self):
        self.thinking_tag = "thinking"
        self.content_tag = "content"

    async def stream_completion_with_thinking(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        include_thinking: bool = True,
        use_qwen: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式生成回答，支持思考过程输出

        Args:
            prompt: 用户提示词
            model: 模型名称（默认使用deepseek-chat或qwen-vl-plus）
            temperature: 温度参数
            max_tokens: 最大token数
            include_thinking: 是否包含思考过程
            use_qwen: 是否使用Qwen模型（False表示使用DeepSeek）

        Yields:
            流式响应数据块，格式：
            {
                "type": "thinking" | "content" | "error" | "end",
                "content": str,
                "metadata": dict
            }
        """
        try:
            logger.info(f"开始流式生成，include_thinking: {include_thinking}, use_qwen: {use_qwen}")

            # 使用现有系统的LLM服务
            from app.services.llm.unified_llm_service import llm_service

            messages = [{"role": "user", "content": prompt}]

            # 如果是DeepSeek-R1或其他支持推理的模型
            if model and "deepseek" in model.lower() and "r1" in model.lower():
                async for chunk in self._stream_deepseek_r1(
                    llm_service, messages, model, temperature, max_tokens
                ):
                    yield chunk
            else:
                # 普通模型，直接流式输出内容
                async for chunk in self._stream_standard(
                    llm_service, messages, model, temperature, max_tokens, use_qwen
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield {
                "type": "error",
                "content": str(e),
                "metadata": {}
            }

    async def _stream_deepseek_r1(
        self,
        llm_service,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """DeepSeek-R1模型的流式输出（支持reasoning_content）"""
        try:
            # 调用LLM服务获取流式响应
            stream_response = await llm_service.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                use_qwen=False  # DeepSeek-R1
            )

            # 处理流式响应
            async for chunk in stream_response:
                # OpenAI流式响应格式
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    # 检查是否有推理过程（DeepSeek-R1特有）
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        yield {
                            "type": "thinking",
                            "content": delta.reasoning_content,
                            "metadata": {
                                "model": model,
                                "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None
                            }
                        }

                    # 检查是否有正常内容
                    elif hasattr(delta, 'content') and delta.content:
                        yield {
                            "type": "content",
                            "content": delta.content,
                            "metadata": {
                                "model": model,
                                "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None
                            }
                        }

                    # 检查是否完成
                    if hasattr(choice, 'finish_reason') and choice.finish_reason == 'stop':
                        yield {
                            "type": "end",
                            "content": "",
                            "metadata": {
                                "model": model,
                                "finish_reason": "stop"
                            }
                        }

        except Exception as e:
            logger.error(f"DeepSeek-R1流式输出失败: {e}")
            raise

    async def _stream_standard(
        self,
        llm_service,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        use_qwen: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """标准模型的流式输出（无推理过程）"""
        try:
            stream_response = await llm_service.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                use_qwen=use_qwen
            )

            async for chunk in stream_response:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    if hasattr(delta, 'content') and delta.content:
                        yield {
                            "type": "content",
                            "content": delta.content,
                            "metadata": {
                                "model": model or ("qwen-vl-plus" if use_qwen else "deepseek-chat"),
                                "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None
                            }
                        }

                    # 检查是否完成
                    if hasattr(choice, 'finish_reason') and choice.finish_reason == 'stop':
                        yield {
                            "type": "end",
                            "content": "",
                            "metadata": {
                                "model": model or ("qwen-vl-plus" if use_qwen else "deepseek-chat"),
                                "finish_reason": "stop"
                            }
                        }

        except Exception as e:
            logger.error(f"标准流式输出失败: {e}")
            raise

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        include_thinking: bool = True,
        use_qwen: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        基于消息列表的流式对话

        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度
            max_tokens: 最大token
            include_thinking: 是否包含思考过程
            use_qwen: 是否使用Qwen模型

        Yields:
            流式响应数据块
        """
        # 将消息转换为提示词
        prompt = self._messages_to_prompt(messages)

        async for chunk in self.stream_completion_with_thinking(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            include_thinking=include_thinking,
            use_qwen=use_qwen
        ):
            yield chunk

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表转换为提示词"""
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)

    async def format_as_sse(
        self,
        stream_generator: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[str, None]:
        """
        将流式响应格式化为SSE（Server-Sent Events）格式

        Args:
            stream_generator: 流式响应生成器

        Yields:
            SSE格式的字符串
        """
        async for chunk in stream_generator:
            sse_data = json.dumps(chunk, ensure_ascii=False)
            yield f"event: message\ndata: {sse_data}\n\n"

        # 发送结束信号
        yield "event: end\ndata: [DONE]\n\n"


# 创建全局服务实例
streaming_response_service = StreamingResponseService()


def get_streaming_response_service() -> StreamingResponseService:
    """获取流式响应服务实例"""
    return streaming_response_service
