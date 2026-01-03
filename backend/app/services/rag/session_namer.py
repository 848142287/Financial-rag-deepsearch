"""
会话自动命名服务
使用现有系统的 DeepSeek 模型生成会话标题
"""

from app.core.structured_logging import get_structured_logger
import json
import re
from typing import Optional

logger = get_structured_logger(__name__)


class SessionNamer:
    """会话命名器 - 基于用户问题生成简洁的会话标题"""

    def __init__(self):
        self.temperature = 0.3
        self.max_tokens = 50
        self.max_length = 20

    async def generate_session_name(
        self,
        user_question: str,
        fallback: Optional[str] = None
    ) -> str:
        """
        生成会话名称

        使用现有系统的 DeepSeek 模型
        """
        if not user_question or not user_question.strip():
            return fallback or "新对话"

        # 构建提示词
        prompt = f"""请根据以下用户提问，生成一个简洁且具有代表性的会话名称。

用户提问：{user_question}

要求：
1. 会话名称应简洁明了，能够概括用户提问的主题
2. 长度控制在10-20个字符
3. 使用专业、正式的表达方式
4. 突出金融/业务主题
5. 返回一个 JSON 对象，包含一个字段 "session_name"

输出格式示例：
{{
  "session_name": "股票市盈率分析"
}}

请严格按照上述格式返回 JSON 对象，不要包含其他文字。"""

        try:
            # 使用现有系统的 LLM 服务
            from app.services.llm.unified_llm_service import llm_service

            logger.info(f"开始生成会话名称，问题: {user_question[:50]}...")

            messages = [{"role": "user", "content": prompt}]

            # 调用 DeepSeek 模型
            response = await llm_service.chat_completion(
                messages=messages,
                model=None,  # 使用默认的 deepseek-chat
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
                use_qwen=False  # 使用 DeepSeek
            )

            # 解析响应 - llm_service 返回 {"content": "...", "usage": {...}, ...}
            response_text = response.get('content', '')
            session_name = self._parse_response(response_text)

            if session_name:
                # 截断过长的标题
                if len(session_name) > self.max_length:
                    session_name = session_name[:self.max_length] + "..."

                logger.info(f"成功生成会话名称: {session_name}")
                return session_name
            else:
                return fallback or self._extract_fallback_name(user_question)

        except Exception as e:
            logger.error(f"生成会话名称失败: {e}")
            return fallback or self._extract_fallback_name(user_question)

    def _parse_response(self, response: str) -> Optional[str]:
        """解析LLM响应"""
        try:
            # 清理响应
            cleaned_response = response.strip()

            # 去掉 markdown 代码块
            json_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
            match = re.search(json_pattern, cleaned_response, re.DOTALL | re.IGNORECASE)

            if match:
                cleaned_response = match.group(1).strip()

            # 解析 JSON
            response_json = json.loads(cleaned_response)
            session_name = response_json.get("session_name", "").strip()

            return session_name if session_name else None

        except Exception as e:
            logger.error(f"解析会话名称JSON失败: {e}")
            return None

    def _extract_fallback_name(self, user_question: str) -> str:
        """从用户问题中提取回退名称"""
        question = user_question.strip()

        if len(question) <= 15:
            return question
        else:
            # 尝试在问号处截断
            if '？' in question:
                idx = question.index('？')
                if idx <= 15:
                    return question[:idx + 1]
            elif '?' in question:
                idx = question.index('?')
                if idx <= 15:
                    return question[:idx + 1]

            # 直接截断
            return question[:15] + "..."


# 创建全局服务实例
session_namer = SessionNamer()


def get_session_namer() -> SessionNamer:
    """获取会话命名服务实例"""
    return session_namer
