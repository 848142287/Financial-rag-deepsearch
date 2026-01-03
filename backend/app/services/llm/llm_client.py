"""
LLM客户端
简化版本，用于快速启动
"""

from typing import List, Dict, Any
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class LLMClient:
    """简化的LLM客户端"""

    def __init__(self):
        self.models = {
            "glm-4.7": {"available": True, "type": "chat"},
            "glm-4.6v": {"available": True, "type": "multimodal"},
            "gpt-4": {"available": True, "type": "chat"},
            "text-embedding-v3": {"available": True, "type": "embedding"}
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "glm-4.7",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """聊天完成"""
        # 简化实现，返回模拟响应
        return {
            "id": "chat_" + str(hash(str(messages))),
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "这是一个简化的LLM响应。系统正在运行，但LLM功能被简化了。"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

    async def embedding(
        self,
        text: str,
        model: str = "text-embedding-v3",
        **kwargs
    ) -> List[float]:
        """生成文本嵌入"""
        # 简化实现，返回模拟嵌入向量
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()

        # 将哈希值转换为向量
        vector = []
        for i in range(0, min(len(hash_hex), 64), 2):
            hex_pair = hash_hex[i:i+2]
            if len(hex_pair) == 2:
                vector.append(int(hex_pair, 16) / 255.0 - 0.5)

        # 填充到标准维度
        while len(vector) < 1536:  # OpenAI embedding维度
            vector.append(0.0)

        return vector[:1536]

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        model: str = "glm-4.6v",
        **kwargs
    ) -> Dict[str, Any]:
        """图像分析"""
        # 简化实现，返回模拟响应
        return {
            "success": True,
            "analysis": "这是简化的图像分析结果。系统可以处理图像，但需要配置真实的视觉模型。",
            "model": model,
            "prompt": prompt
        }

    def is_model_available(self, model: str) -> bool:
        """检查模型是否可用"""
        return model in self.models and self.models[model]["available"]

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return [model for model, info in self.models.items() if info["available"]]


# 创建全局实例
llm_client = LLMClient()

# 兼容性函数
async def chat_with_llm(prompt: str, model: str = "glm-4.7") -> str:
    """简化版聊天函数"""
    messages = [{"role": "user", "content": prompt}]
    response = await llm_client.chat_completion(messages, model)
    return response["choices"][0]["message"]["content"]

async def generate_embedding(text: str) -> List[float]:
    """简化版嵌入生成"""
    return await llm_client.embedding(text)