"""
推荐问题生成服务
从 swxy/backend 移植并优化，使用现有系统的模型（DeepSeek）
"""

from app.core.structured_logging import get_structured_logger
import json
import re
from typing import List, Optional, Dict, Any

logger = get_structured_logger(__name__)


class QuestionRecommender:
    """推荐问题生成器 - 基于用户提问和文档上下文生成相关问题"""

    def __init__(self):
        self.default_count = 3
        self.temperature = 0.7
        self.max_tokens = 500

    async def generate_recommendations(
        self,
        user_question: str,
        retrieved_content: Optional[List[Dict[str, Any]]] = None,
        count: int = 3,
        session_id: Optional[str] = None
    ) -> List[str]:
        """
        生成相关问题推荐

        使用现有系统的 DeepSeek 模型
        """
        if not user_question or not user_question.strip():
            logger.warning("用户问题为空，无法生成推荐")
            return []

        # 提取文档主题
        document_topics = []
        if retrieved_content and len(retrieved_content) > 0:
            document_names = list(set([
                ref.get('document_name', '')
                for ref in retrieved_content
                if ref.get('document_name')
            ]))
            document_topics = document_names[:3]

        # 构建上下文
        context_info = ""
        if document_topics:
            context_info = f"当前对话基于这些文档：{', '.join(document_topics)}"

        # 构建提示词
        prompt = f"""你是一个专业的金融领域智能助手，请基于用户的问题生成{count}个相关的推荐问题。

用户问题：{user_question}
{context_info}

要求：
1. 生成的问题应该与用户问题相关，但从不同角度深入
2. 问题要具体、有价值
3. 如果有文档上下文，可以围绕文档主题生成相关问题
4. 返回JSON格式

输出格式：
{{
  "recommended_questions": [
    "具体问题1",
    "具体问题2",
    "具体问题3"
  ]
}}

请直接返回JSON，不要包含其他文字。"""

        try:
            # 使用现有系统的 LLM 服务
            from app.services.llm.unified_llm_service import llm_service

            logger.info(f"开始生成推荐问题，session_id: {session_id}")

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
            questions = self._parse_response(response_text, count)

            if questions:
                logger.info(f"成功生成 {len(questions)} 个推荐问题，session_id: {session_id}")
                return questions
            else:
                logger.warning("推荐问题解析为空")
                return []

        except Exception as e:
            logger.error(f"生成推荐问题失败: {e}, session_id: {session_id}")
            return []

    def _parse_response(self, response: str, count: int) -> List[str]:
        """解析LLM响应"""
        try:
            # 清理响应内容
            cleaned_response = response.strip()

            # 去掉 markdown 代码块
            json_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
            match = re.search(json_pattern, cleaned_response, re.DOTALL | re.IGNORECASE)

            if match:
                cleaned_response = match.group(1).strip()

            # 解析 JSON
            response_json = json.loads(cleaned_response)
            recommended_questions = response_json.get("recommended_questions", [])

            if isinstance(recommended_questions, list) and len(recommended_questions) > 0:
                questions = [q for q in recommended_questions if q and q.strip()]
                return questions[:count]
            else:
                return []

        except Exception as e:
            logger.error(f"解析推荐问题JSON失败: {e}")
            return []


# 创建全局服务实例
question_recommender = QuestionRecommender()


def get_question_recommender() -> QuestionRecommender:
    """获取推荐问题服务实例"""
    return question_recommender
