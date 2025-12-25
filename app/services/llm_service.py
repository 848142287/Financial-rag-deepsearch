"""
大语言模型服务
支持DeepSeek和Qwen等多种模型
"""

from typing import List, Dict, Any, Optional
import openai
import logging
# from tenacity import retry, stop_after_attempt, wait_exponential  # TODO: 安装tenacity依赖

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """大语言模型服务"""

    def __init__(self):
        self.deepseek_client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )

        # Qwen客户端配置
        self.qwen_client = openai.OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url
        )

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))  # TODO: 安装tenacity依赖
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
        use_qwen: bool = False
    ) -> Dict[str, Any]:
        """
        聊天完成接口

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式返回
            use_qwen: 是否使用Qwen模型

        Returns:
            模型响应结果
        """
        try:
            # 选择客户端和模型
            client = self.qwen_client if use_qwen else self.deepseek_client
            model = model or (settings.multimodal_model if use_qwen else settings.llm_model)

            logger.info(f"调用LLM模型: {model}, 消息数量: {len(messages)}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                return response  # 流式响应
            else:
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "model": model,
                    "finish_reason": response.choices[0].finish_reason
                }

        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise

    async def simple_chat(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = 0.7,
        use_qwen: bool = False
    ) -> str:
        """
        简单聊天接口

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            model: 模型名称
            temperature: 温度参数
            use_qwen: 是否使用Qwen模型

        Returns:
            模型回复文本
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            use_qwen=use_qwen
        )

        return response["content"]

    async def structured_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: str = None,
        model: str = None,
        use_qwen: bool = False
    ) -> Dict[str, Any]:
        """
        结构化输出完成

        Args:
            prompt: 输入提示
            schema: 输出结构schema
            system_prompt: 系统提示
            model: 模型名称
            use_qwen: 是否使用Qwen模型

        Returns:
            结构化输出结果
        """
        # 添加JSON格式化指令
        if system_prompt:
            system_prompt += f"\n\n请严格按照以下JSON格式返回结果：\n{schema}"
        else:
            system_prompt = f"请严格按照以下JSON格式返回结果：\n{schema}"

        response = await self.simple_chat(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            use_qwen=use_qwen
        )

        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"结构化输出解析失败: {response}")
            raise ValueError("模型返回的JSON格式不正确")


# 全局LLM服务实例
llm_service = LLMService()