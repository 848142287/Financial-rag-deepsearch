"""
统一LLM服务层 - 解决LLM服务重复和分散问题

整合所有LLM调用到一个统一的服务层，提供：
1. 统一的调用接口
2. 统一的错误处理
3. 统一的日志记录
4. 批量调用优化
5. Prompt模板管理
"""

import asyncio
from enum import Enum
from dataclasses import dataclass
from app.core.structured_logging import get_structured_logger
from datetime import datetime

logger = get_structured_logger(__name__)

class LLMModel(str, Enum):
    """支持的LLM模型"""
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"
    GLM_4_PLUS = "glm-4-plus"
    GLM_4_05_PLUS = "glm-4-0528-plus"  # GLM-4.7
    GLM_4V = "glm-4v"  # GLM-4.6V
    QWEN_VL_MAX = "qwen-vl-max"
    QWEN_VL_PLUS = "qwen-vl-plus"
    QWEN_VL_OCR = "qwen-vl-ocr"

@dataclass
class LLMResponse:
    """LLM响应统一格式"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[int] = None
    success: bool = True
    error: Optional[str] = None

@dataclass
class LLMRequest:
    """LLM请求统一格式"""
    prompt: str
    model: LLMModel = LLMModel.DEEPSEEK_CHAT
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.9
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None

class LLMError(Exception):
    """LLM调用错误基类"""
    def __init__(self, message: str, model: str, cause: Optional[Exception] = None):
        self.message = message
        self.model = model
        self.cause = cause
        super().__init__(f"[{model}] {message}")

class UnifiedLLMService:
    """
    统一LLM服务

    整合所有分散的LLM调用，包括：
    - llm_service.py (DeepSeek)
    - financial_llm_service.py (金融专用)
    - rag_glm_service.py (GLM)
    - ocr_service.py (Qwen-VL)
    """

    def __init__(self):
        self._service_cache = {}
        self._prompt_templates = {}
        self._initialized = False

    async def initialize(self):
        """初始化所有LLM服务"""
        if self._initialized:
            return

        logger.info("初始化统一LLM服务...")

        try:
            # 延迟导入，避免循环依赖
            from app.services.llm.unified_llm_service import llm_service
            self._service_cache['deepseek'] = llm_service
            logger.info("✅ DeepSeek服务已加载")

        except Exception as e:
            logger.warning(f"⚠️ DeepSeek服务加载失败: {e}")
            self._service_cache['deepseek'] = None

        try:
            from app.services.rag_glm_service import RAGGLMService
            self._service_cache['glm'] = RAGGLMService()
            await self._service_cache['glm'].initialize()
            logger.info("✅ GLM服务已加载")

        except Exception as e:
            logger.warning(f"⚠️ GLM服务加载失败: {e}")
            self._service_cache['glm'] = None

        try:
            from app.services.ocr_service import OCRService
            self._service_cache['qwen_vl'] = OCRService()
            logger.info("✅ Qwen-VL服务已加载")

        except Exception as e:
            logger.warning(f"⚠️ Qwen-VL服务加载失败: {e}")
            self._service_cache['qwen_vl'] = None

        # 加载Prompt模板
        self._load_prompt_templates()

        self._initialized = True
        logger.info("✅ 统一LLM服务初始化完成")

    def _load_prompt_templates(self):
        """加载Prompt模板"""
        self._prompt_templates = {
            'entity_extraction': """请从以下金融文本中抽取重要实体，包括：
1. 公司名称
2. 股票代码
3. 财务指标（收入、利润、资产、负债等）
4. 关键人物
5. 重要日期
6. 金额和百分比

文本：
{content}

请以JSON格式返回，只返回高置信度的实体（confidence > 0.7）：
{{
    "entities": [
        {{"text": "实体文本", "type": "实体类型", "confidence": 0.9}}
    ]
}}""",

            'document_summary': """请为以下金融文档生成摘要，包括：
1. 主要观点（3-5条）
2. 核心数据
3. 投资建议（如有）

文档内容：
{content}

请以Markdown格式返回摘要。""",

            'qa_rag': """基于以下检索到的上下文信息回答问题：

上下文：
{context}

问题：{question}

请提供准确、详细的回答，如果上下文中没有相关信息，请明确说明。""",

            'markdown_fusion': """请融合以下两部分内容，生成统一的文档：

1. PDF解析内容：
{pdf_content}

2. Markdown补充内容：
{markdown_content}

请生成一个完整的、结构化的Markdown文档。"""
        }

    # ========================================================================
    # 核心API
    # ========================================================================

    async def chat(
        self,
        prompt: str,
        model: LLMModel = LLMModel.DEEPSEEK_CHAT,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        统一的聊天接口

        Args:
            prompt: 提示词
            model: LLM模型
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            LLMResponse
        """
        start_time = datetime.now()

        try:
            # 路由到对应的服务
            if model in [LLMModel.DEEPSEEK_CHAT, LLMModel.DEEPSEEK_REASONER]:
                response = await self._chat_deepseek(prompt, model, temperature, max_tokens, **kwargs)
            elif model in [LLMModel.GLM_4_PLUS, LLMModel.GLM_4_05_PLUS, LLMModel.GLM_4V]:
                response = await self._chat_glm(prompt, model, temperature, max_tokens, **kwargs)
            elif model in [LLMModel.QWEN_VL_MAX, LLMModel.QWEN_VL_PLUS, LLMModel.QWEN_VL_OCR]:
                response = await self._chat_qwen_vl(prompt, model, temperature, max_tokens, **kwargs)
            else:
                raise LLMError(f"不支持的模型: {model}", model.value)

            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            logger.info(f"✅ LLM调用成功: model={model.value}, latency={latency_ms}ms, "
                       f"tokens={response.usage.get('total_tokens', 'N/A') if response.usage else 'N/A'}")

            return LLMResponse(
                content=response.content,
                model=model.value,
                usage=response.usage,
                latency_ms=latency_ms,
                success=True
            )

        except Exception as e:
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            logger.error(f"❌ LLM调用失败: model={model.value}, error={str(e)}, latency={latency_ms}ms")

            return LLMResponse(
                content="",
                model=model.value,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )

    async def chat_with_template(
        self,
        template_name: str,
        template_vars: Dict[str, Any],
        model: LLMModel = LLMModel.DEEPSEEK_CHAT,
        **kwargs
    ) -> LLMResponse:
        """
        使用Prompt模板进行对话

        Args:
            template_name: 模板名称
            template_vars: 模板变量
            model: LLM模型
            **kwargs: 其他参数

        Returns:
            LLMResponse
        """
        if template_name not in self._prompt_templates:
            raise ValueError(f"未找到模板: {template_name}")

        prompt = self._prompt_templates[template_name].format(**template_vars)
        return await self.chat(prompt, model, **kwargs)

    async def batch_chat(
        self,
        prompts: List[str],
        model: LLMModel = LLMModel.DEEPSEEK_CHAT,
        temperature: float = 0.7,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[LLMResponse]:
        """
        批量聊天接口（并发执行，减少总耗时）

        Args:
            prompts: 提示词列表
            model: LLM模型
            temperature: 温度参数
            max_concurrent: 最大并发数
            **kwargs: 其他参数

        Returns:
            List[LLMResponse]
        """
        logger.info(f"🚀 批量LLM调用: count={len(prompts)}, model={model.value}, max_concurrent={max_concurrent}")

        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def chat_with_limit(prompt: str, index: int):
            async with semaphore:
                response = await self.chat(prompt, model, temperature, **kwargs)
                return index, response

        # 并发执行所有请求
        tasks = [chat_with_limit(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 按原始顺序排序
        sorted_responses = [None] * len(prompts)
        success_count = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量调用异常: {result}")
                continue

            index, response = result
            sorted_responses[index] = response
            if response.success:
                success_count += 1

        logger.info(f"✅ 批量LLM调用完成: success={success_count}/{len(prompts)}")

        return sorted_responses

    # ========================================================================
    # 私有方法 - 路由到具体服务
    # ========================================================================

    async def _chat_deepseek(
        self,
        prompt: str,
        model: LLMModel,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """DeepSeek聊天"""
        service = self._service_cache.get('deepseek')
        if not service:
            raise LLMError("DeepSeek服务未初始化", model.value)

        # 调用原始服务
        response = await service.simple_chat(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return LLMResponse(
            content=response,
            model=model.value
        )

    async def _chat_glm(
        self,
        prompt: str,
        model: LLMModel,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """GLM聊天"""
        service = self._service_cache.get('glm')
        if not service:
            raise LLMError("GLM服务未初始化", model.value)

        # 根据模型类型选择方法
        if model == LLMModel.GLM_4V:
            # 视觉模型
            response = await service.analyze_image(
                prompt,
                kwargs.get('image')
            )
        else:
            # 文本模型
            response = await service.generate(
                prompt,
                temperature=temperature
            )

        return LLMResponse(
            content=response,
            model=model.value
        )

    async def _chat_qwen_vl(
        self,
        prompt: str,
        model: LLMModel,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Qwen-VL聊天"""
        service = self._service_cache.get('qwen_vl')
        if not service:
            raise LLMError("Qwen-VL服务未初始化", model.value)

        # 调用OCR/视觉服务
        response = await service.analyze_document(
            kwargs.get('file_content'),
            kwargs.get('filename')
        )

        return LLMResponse(
            content=response.get('text', ''),
            model=model.value
        )

# 全局单例
_unified_llm_service: Optional[UnifiedLLMService] = None

def get_unified_llm_service() -> UnifiedLLMService:
    """获取统一LLM服务单例"""
    global _unified_llm_service
    if _unified_llm_service is None:
        _unified_llm_service = UnifiedLLMService()
    return _unified_llm_service

async def get_unified_llm_service_initialized() -> UnifiedLLMService:
    """获取已初始化的统一LLM服务"""
    service = get_unified_llm_service()
    if not service._initialized:
        await service.initialize()
    return service
