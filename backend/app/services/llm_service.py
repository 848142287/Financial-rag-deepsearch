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
        # DeepSeek客户端配置 - 优先使用环境变量中的配置
        api_key = getattr(settings, 'deepseek_api_key', None) or settings.openai_api_key
        base_url = getattr(settings, 'deepseek_base_url', None) or settings.openai_base_url

        logger.info(f"初始化DeepSeek客户端:")
        logger.info(f"  API Key: {api_key[:20]}...{api_key[-5:] if api_key else 'None'}")
        logger.info(f"  Base URL: {base_url}")
        logger.info(f"  Model: {settings.llm_model}")

        self.deepseek_client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # Qwen客户端配置
        logger.info(f"初始化Qwen客户端:")
        if settings.qwen_api_key:
            logger.info(f"  API Key: {settings.qwen_api_key[:20]}...{settings.qwen_api_key[-5:]}")
        else:
            logger.warning("  API Key: Not configured (Qwen API will not be available)")
        logger.info(f"  Base URL: {settings.qwen_base_url}")

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
            model = model or (settings.qwen_multimodal_model if use_qwen else settings.llm_model)

            logger.info("="*60)
            logger.info(f"🚀 调用LLM模型")
            logger.info(f"  模型: {model}")
            logger.info(f"  使用Qwen: {use_qwen}")
            logger.info(f"  Temperature: {temperature}")
            logger.info(f"  Max Tokens: {max_tokens}")
            logger.info(f"  消息数量: {len(messages)}")
            logger.info(f"  Stream: {stream}")

            # 打印消息内容（前200字符）
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]
                logger.info(f"  消息{i+1} [{role}]: {content}...")

            logger.info("="*60)

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                logger.info("✓ 返回流式响应")
                return response  # 流式响应
            else:
                result = {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "model": model,
                    "finish_reason": response.choices[0].finish_reason
                }

                logger.info("="*60)
                logger.info(f"✓ LLM调用成功")
                logger.info(f"  模型: {result['model']}")
                logger.info(f"  Token使用: {result['usage']['total_tokens']}")
                logger.info(f"    - Prompt: {result['usage']['prompt_tokens']}")
                logger.info(f"    - Completion: {result['usage']['completion_tokens']}")
                logger.info(f"  完成原因: {result['finish_reason']}")
                logger.info(f"  响应内容: {result['content'][:200]}...")
                logger.info("="*60)

                return result

        except Exception as e:
            logger.error("="*60)
            logger.error(f"✗ LLM调用失败")
            logger.error(f"  错误类型: {type(e).__name__}")
            logger.error(f"  错误信息: {str(e)}")
            logger.error("="*60)
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