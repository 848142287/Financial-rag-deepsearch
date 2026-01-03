"""
增强查询处理器
借鉴DocMind项目的查询处理策略，包括：
1. 上下文重写（解决指代问题）
2. 双轨查询重写（语义+关键词）
3. HyDE生成（假设文档）
4. 元问题识别
"""

import re
from dataclasses import dataclass
from app.core.structured_logging import get_structured_logger
from app.services.llm.unified_llm_service import get_unified_llm_service_initialized, LLMModel

logger = get_structured_logger(__name__)

@dataclass
class ProcessedQuery:
    """处理后的查询"""
    original_query: str
    standalone_query: str  # 独立查询（上下文重写后）
    vector_query: str  # 向量检索查询
    keywords: List[str]  # BM25关键词
    hyde_doc: str  # HyDE生成的假设文档
    is_meta_question: bool  # 是否是元问题
    direct_answer: Optional[str]  # 元问题的直接回答

class EnhancedQueryProcessor:
    """
    增强查询处理器

    功能：
    1. 元问题识别与直接回答
    2. 上下文重写（处理多轮对话的指代）
    3. 双轨查询重写（语义+关键词）
    4. HyDE生成（假设性文档）
    """

    def __init__(
        self,
        enable_context_rewrite: bool = True,
        enable_query_rewrite: bool = True,
        enable_hyde: bool = True,
        max_history_turns: int = 3
    ):
        """
        初始化查询处理器

        Args:
            enable_context_rewrite: 是否启用上下文重写
            enable_query_rewrite: 是否启用查询重写
            enable_hyde: 是否启用HyDE
            max_history_turns: 最大历史轮数
        """
        self.enable_context_rewrite = enable_context_rewrite
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_hyde = enable_hyde
        self.max_history_turns = max_history_turns

        logger.info(
            f"EnhancedQueryProcessor初始化: "
            f"context_rewrite={enable_context_rewrite}, "
            f"query_rewrite={enable_query_rewrite}, "
            f"hyde={enable_hyde}"
        )

    async def process(
        self,
        query: str,
        history: List[Dict[str, str]] = None
    ) -> ProcessedQuery:
        """
        处理查询

        Args:
            query: 用户查询
            history: 对话历史 [{"role": "user", "content": "...}, ...]

        Returns:
            ProcessedQuery
        """
        history = history or []

        # 1. 元问题识别
        is_meta, direct_answer = self._identify_meta_question(query, history)

        if is_meta:
            logger.info(f"✅ 识别为元问题，直接回答")
            return ProcessedQuery(
                original_query=query,
                standalone_query=query,
                vector_query=query,
                keywords=[],
                hyde_doc="",
                is_meta_question=True,
                direct_answer=direct_answer
            )

        # 2. 上下文重写
        standalone_query = query
        if self.enable_context_rewrite and history:
            standalone_query = await self._rewrite_context(query, history)
            logger.debug(f"上下文重写: {query} -> {standalone_query}")

        # 3. 双轨查询重写
        vector_query, keywords = standalone_query, []
        if self.enable_query_rewrite:
            vector_query, keywords = await self._rewrite_query(standalone_query)
            logger.debug(f"查询重写: vector={vector_query}, keywords={keywords}")

        # 4. HyDE生成
        hyde_doc = ""
        if self.enable_hyde:
            hyde_doc = await self._generate_hyde(standalone_query)
            logger.debug(f"HyDE生成: {hyde_doc[:100]}...")

        return ProcessedQuery(
            original_query=query,
            standalone_query=standalone_query,
            vector_query=vector_query,
            keywords=keywords,
            hyde_doc=hyde_doc,
            is_meta_question=False,
            direct_answer=None
        )

    def _identify_meta_question(
        self,
        query: str,
        history: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str]]:
        """
        识别元问题

        元问题包括：
        - 身份询问："你是谁"、"你叫什么"
        - 历史询问："我刚刚问了什么"
        - 系统询问："你能做什么"
        """
        query_lower = query.lower()

        # 身份询问
        identity_patterns = [
            "你是谁", "你叫什么", "你的名字", "self intro"
        ]
        if any(p in query_lower for p in identity_patterns):
            return True, (
                "我是金融RAG智能助手，专注于金融文档分析和问答。\n"
                "我可以帮您：\n"
                "• 分析财务报告和财报\n"
                "• 搜索金融知识\n"
                "• 回答投资相关问题\n"
                "• 提供市场数据分析"
            )

        # 历史询问
        history_patterns = ["我刚刚问", "上一个问题", "刚才说的"]
        if any(p in query_lower for p in history_patterns):
            if history:
                last_question = next(
                    (h["content"] for h in reversed(history) if h["role"] == "user"),
                    "没有找到之前的问题"
                )
                return True, f"你刚刚问的是：{last_question}"
            else:
                return True, "这是我们的第一次对话"

        # 能力询问
        capability_patterns = ["你能做什么", "有什么功能", "help"]
        if any(p in query_lower for p in capability_patterns):
            return True, (
                "我可以帮助您：\n"
                "1. 📊 分析财务报告和财报数据\n"
                "2. 🔍 搜索金融知识和监管文件\n"
                "3. 📈 提供市场趋势分析\n"
                "4. 💡 回答投资相关问题\n"
                "5. 📝 生成金融摘要和报告\n\n"
                "请上传您的金融文档或直接提问！"
            )

        return False, None

    async def _rewrite_context(
        self,
        query: str,
        history: List[Dict[str, str]]
    ) -> str:
        """
        上下文重写

        将包含指代的查询重写为独立完整的查询
        例如："它的增长率是多少？" -> "苹果公司的增长率是多少？"
        """
        # 获取最近N轮对话
        recent_history = history[-self.max_history_turns:]

        # 构建历史文本
        history_text = "\n".join([
            f"{h['role']}: {h['content']}"
            for h in recent_history
        ])

        prompt = f"""你是一个查询重写专家。请将用户的最新问题重写为独立完整的查询。

**要求**：
1. 如果问题中包含代词（"它"、"这个"、"那个"），请根据对话历史替换为具体的名词
2. 保持问题的原意
3. 使问题可以独立理解，不依赖历史对话

**对话历史**：
{history_text}

**用户最新问题**：{query}

**重写后的独立查询**："""

        try:
            llm = await get_unified_llm_service_initialized()
            response = await llm.chat(
                prompt=prompt,
                model=LLMModel.DEEPSEEK_CHAT,
                temperature=0.3,
                max_tokens=200
            )

            rewritten = response.content.strip()
            # 如果重写失败或为空，返回原查询
            if not rewritten or len(rewritten) < len(query) // 2:
                return query

            return rewritten

        except Exception as e:
            logger.warning(f"上下文重写失败: {e}，使用原查询")
            return query

    async def _rewrite_query(self, query: str) -> Tuple[str, List[str]]:
        """
        双轨查询重写

        生成两部分内容：
        1. Vector Query: 适合向量检索的查询（逻辑完整、去口语化）
        2. Keywords: 适合BM25的关键词（3-5个核心词）
        """
        prompt = f"""你是一个金融领域的查询优化专家。请将用户的查询重写为两部分。

**要求**：
1. [Vector] 逻辑完整、去口语化的专业陈述句
2. [Keywords] 提取3-5个核心关键词，用于精确匹配（关键词之间用逗号分隔）

**用户查询**：{query}

**输出格式**：
[Vector] 营业收入、净利润的同比增长率
[Keywords] 营收, 净利润, 同比增长

**请开始重写**："""

        try:
            llm = await get_unified_llm_service_initialized()
            response = await llm.chat(
                prompt=prompt,
                model=LLMModel.DEEPSEEK_CHAT,
                temperature=0.3,
                max_tokens=200
            )

            result = response.content.strip()

            # 解析输出
            vector_query = query  # 默认值
            keywords = []

            # 提取 [Vector] 部分
            vector_match = re.search(r'\[Vector\]\s*(.+?)(?:\[Keywords\]|$)', result, re.DOTALL)
            if vector_match:
                vector_query = vector_match.group(1).strip()

            # 提取 [Keywords] 部分
            keywords_match = re.search(r'\[Keywords\]\s*(.+)', result)
            if keywords_match:
                keywords_str = keywords_match.group(1).strip()
                keywords = [k.strip() for k in re.split(r'[,，、]', keywords_str) if k.strip()]

            # 如果没有提取到关键词，使用原查询的分词
            if not keywords:
                keywords = self._extract_keywords(query)

            logger.debug(f"查询重写结果: vector={vector_query}, keywords={keywords}")

            return vector_query, keywords

        except Exception as e:
            logger.warning(f"查询重写失败: {e}，使用原查询")
            return query, self._extract_keywords(query)

    def _extract_keywords(self, query: str) -> List[str]:
        """
        简单的关键词提取

        使用jieba分词并过滤停用词
        """
        try:
            import jieba

            # 金融领域停用词
            stopwords = {
                "如何", "怎么", "什么", "为什么", "哪些", "多少",
                "的", "了", "吗", "呢", "是", "在", "有", "和", "与",
                "我", "你", "他", "它", "我们", "你们"
            }

            # 分词
            words = jieba.cut(query)

            # 过滤停用词和短词
            keywords = [
                w for w in words
                if len(w) >= 2 and w not in stopwords
            ]

            # 取前5个
            return keywords[:5]

        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
            return []

    async def _generate_hyde(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)

        生成一个假设性的回答，用于向量检索
        """
        prompt = f"""你是一个金融知识专家。请针对以下问题，写一段简短、专业的假设性回答。

**要求**：
1. 回答要包含相关的金融术语和概念
2. 回答要专业、准确
3. 长度控制在150-300字
4. 这个回答将用于检索相关文档

**问题**：{query}

**假设性回答**："""

        try:
            llm = await get_unified_llm_service_initialized()
            response = await llm.chat(
                prompt=prompt,
                model=LLMModel.DEEPSEEK_CHAT,
                temperature=0.5,
                max_tokens=500
            )

            hyde_doc = response.content.strip()

            # 限制长度
            if len(hyde_doc) > 500:
                hyde_doc = hyde_doc[:500] + "..."

            return hyde_doc

        except Exception as e:
            logger.warning(f"HyDE生成失败: {e}")
            return ""  # 返回空，不影响其他检索路径

# 全局实例
_query_processor = None

def get_query_processor() -> EnhancedQueryProcessor:
    """获取查询处理器单例"""
    global _query_processor
    if _query_processor is None:
        _query_processor = EnhancedQueryProcessor()
    return _query_processor
