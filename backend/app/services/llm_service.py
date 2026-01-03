"""
大语言模型服务
支持主模型和备份模型的自动切换

配置说明：
- 主模型：Deepseek（deepseek-chat）
- 备份模型：GLM-4.7（智谱AI）
"""

import openai
from typing import List, Dict, Any, Optional
from app.core.structured_logging import get_structured_logger

from app.core.config import settings

logger = get_structured_logger(__name__)

class LLMService:
    """大语言模型服务 - 支持主模型和备份模型自动切换

    配置说明：
    - 主模型：Deepseek（deepseek-chat）
    - 备份模型：GLM-4.7（智谱AI）

    自动切换逻辑：
    1. 优先使用主模型（Deepseek）
    2. 主模型失败时，自动切换到备份模型（GLM-4.7）
    3. 可通过配置禁用自动切换
    """

    def __init__(self):
        # 从settings中读取配置
        self.primary_model = settings.primary_llm_model  # "deepseek"
        self.fallback_model = settings.fallback_llm_model  # "glm"
        self.fallback_enabled = True  # 默认启用自动切换

        # 初始化Deepseek客户端（主模型）
        deepseek_api_key = getattr(settings, 'deepseek_api_key', None)
        deepseek_base_url = getattr(settings, 'deepseek_base_url', None)

        if deepseek_api_key:
            self.deepseek_client = openai.OpenAI(
                api_key=deepseek_api_key,
                base_url=deepseek_base_url
            )
            logger.info("✓ Deepseek (主模型) 客户端初始化成功")
        else:
            self.deepseek_client = None
            logger.warning("✗ Deepseek API Key未配置")

        # 初始化GLM-4.7客户端（备份模型）
        glm_api_key = getattr(settings, 'glm_api_key', None)
        glm_base_url = getattr(settings, 'glm_base_url', None)

        if glm_api_key:
            self.glm_client = openai.OpenAI(
                api_key=glm_api_key,
                base_url=glm_base_url
            )
            logger.info("✓ GLM-4.7 (备份模型) 客户端初始化成功")
        else:
            self.glm_client = None
            logger.warning("✗ GLM-4.7 API Key未配置")

        logger.info("="*80)
        logger.info(f"🎯 LLM服务初始化完成")
        logger.info(f"  - 主模型: {self._get_model_name(self.primary_model)}")
        logger.info(f"  - 备份模型: {self._get_model_name(self.fallback_model)}")
        logger.info(f"  - 自动切换: {'启用' if self.fallback_enabled else '禁用'}")
        logger.info("="*80)

        self.current_model = self.primary_model  # 当前使用的模型

    def _get_model_name(self, model_key: str) -> str:
        """获取模型的显示名称"""
        model_names = {
            "deepseek": "Deepseek (deepseek-chat)",
            "glm": "GLM-4.7 (智谱AI)",
            "qwen": "Qwen (通义千问)"
        }
        return model_names.get(model_key, model_key)

    def _get_client_and_model(self, model_key: str):
        """获取模型对应的客户端和模型名称"""
        if model_key == "deepseek":
            return self.deepseek_client, settings.deepseek_model
        elif model_key == "glm":
            return self.glm_client, settings.glm_model
        else:
            raise ValueError(f"不支持的模型: {model_key}")

    # 使用统一的重试机制 (app.core.retry)
    # 如需重试功能，使用 @retry_on_failure 装饰器
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        聊天完成接口 - 支持自动切换到备份模型

        Args:
            messages: 消息列表
            model: 模型名称（可选），不指定则使用主模型
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式返回

        Returns:
            模型响应结果
        """
        # 确定使用的模型
        model_key = model or self.current_model

        # 验证模型是否可用
        if model_key not in [self.primary_model, self.fallback_model]:
            logger.warning(f"⚠ 指定的模型 {model_key} 不可用，使用主模型")
            model_key = self.primary_model

        try:
            # 获取客户端和模型名称
            client, model_name = self._get_client_and_model(model_key)

            if client is None:
                raise ValueError(f"模型 {model_key} 的客户端未初始化")

            logger.info("="*80)
            logger.info(f"🚀 调用LLM模型: {self._get_model_name(model_key)}")
            logger.info(f"  模型: {model_name}")
            logger.info(f"  Temperature: {temperature}")
            logger.info(f"  Max Tokens: {max_tokens}")
            logger.info(f"  消息数量: {len(messages)}")
            logger.info(f"  Stream: {stream}")

            # 打印消息内容（前200字符）
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]
                logger.info(f"  消息{i+1} [{role}]: {content}...")

            logger.info("="*80)

            response = client.chat.completions.create(
                model=model_name,
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
                    "model": model_name,
                    "model_key": model_key,
                    "finish_reason": response.choices[0].finish_reason
                }

                logger.info("="*80)
                logger.info(f"✓ LLM调用成功: {self._get_model_name(model_key)}")
                logger.info(f"  模型: {result['model']}")
                logger.info(f"  Token使用: {result['usage']['total_tokens']}")
                logger.info(f"    - Prompt: {result['usage']['prompt_tokens']}")
                logger.info(f"    - Completion: {result['usage']['completion_tokens']}")
                logger.info(f"  完成原因: {result['finish_reason']}")
                logger.info(f"  响应内容: {result['content'][:200]}...")
                logger.info("="*80)

                return result

        except Exception as e:
            logger.error("="*80)
            logger.error(f"✗ 模型 {self._get_model_name(model_key)} 调用失败")
            logger.error(f"  错误类型: {type(e).__name__}")
            logger.error(f"  错误信息: {str(e)}")
            logger.error("="*80)

            # 尝试使用备份模型
            if self.fallback_enabled and model_key != self.fallback_model:
                logger.info(f"🔄 自动切换到备份模型: {self._get_model_name(self.fallback_model)}")
                self.current_model = self.fallback_model
                return await self.chat_completion(
                    messages=messages,
                    model=self.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )

            raise

    async def simple_chat(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = 0.7
    ) -> str:
        """
        简单聊天接口

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            model: 模型名称
            temperature: 温度参数

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
            temperature=temperature
        )

        return response["content"]

    async def structured_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: str = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        结构化输出完成

        Args:
            prompt: 输入提示
            schema: 输出结构schema
            system_prompt: 系统提示
            model: 模型名称

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
            model=model
        )

        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"结构化输出解析失败: {response}")
            raise ValueError("模型返回的JSON格式不正确")

# 全局LLM服务实例
llm_service = LLMService()