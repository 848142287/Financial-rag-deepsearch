"""
RAG工具包集成器
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .retrieval_tool import DocumentRetrievalTool
from .generation_tool import RAGGenerationTool
from .multi_modal_tool import MultiModalProcessingTool
from .knowledge_graph_tool import KnowledgeGraphTool
from .evaluation_tool import RAGEvaluationTool
from app.middleware.base_middleware import MiddlewareManager
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.cache_middleware import CacheMiddleware
from app.middleware.metrics_middleware import MetricsMiddleware
from app.middleware.error_middleware import ErrorMiddleware
from app.core.logging import logger
from app.core.config import get_settings

class RAGToolkit:
    """
    RAG工具包集成器

    集成所有RAG工具和中间件，提供统一的访问接口
    """

    def __init__(
        self,
        enable_middleware: bool = True,
        cache_ttl: int = 3600,
        log_level: str = "INFO",
        **middleware_kwargs
    ):
        self.settings = get_settings()
        self.enable_middleware = enable_middleware

        # 初始化工具
        self.tools = self._init_tools()

        # 初始化中间件
        self.middleware_manager = None
        if enable_middleware:
            self.middleware_manager = self._init_middleware(
                cache_ttl=cache_ttl,
                log_level=log_level,
                **middleware_kwargs
            )

    def _init_tools(self) -> List[BaseTool]:
        """初始化所有RAG工具"""
        tools = []

        try:
            # 文档检索工具
            retrieval_tool = DocumentRetrievalTool()
            tools.append(retrieval_tool)
            logger.info("DocumentRetrievalTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentRetrievalTool: {e}")

        try:
            # RAG生成工具
            generation_tool = RAGGenerationTool()
            tools.append(generation_tool)
            logger.info("RAGGenerationTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAGGenerationTool: {e}")

        try:
            # 多模态处理工具
            multimodal_tool = MultiModalProcessingTool()
            tools.append(multimodal_tool)
            logger.info("MultiModalProcessingTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MultiModalProcessingTool: {e}")

        try:
            # 知识图谱工具
            knowledge_graph_tool = KnowledgeGraphTool()
            tools.append(knowledge_graph_tool)
            logger.info("KnowledgeGraphTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphTool: {e}")

        try:
            # 评估工具
            evaluation_tool = RAGEvaluationTool()
            tools.append(evaluation_tool)
            logger.info("RAGEvaluationTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAGEvaluationTool: {e}")

        return tools

    def _init_middleware(
        self,
        cache_ttl: int = 3600,
        log_level: str = "INFO",
        **kwargs
    ) -> MiddlewareManager:
        """初始化中间件管理器"""
        manager = MiddlewareManager()

        # 日志中间件
        try:
            logging_middleware = LoggingMiddleware(log_level=log_level)
            manager.add_middleware(logging_middleware)
            logger.info("LoggingMiddleware initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LoggingMiddleware: {e}")

        # 缓存中间件
        try:
            cache_middleware = CacheMiddleware(cache_ttl=cache_ttl, **kwargs.get("cache", {}))
            manager.add_middleware(cache_middleware)
            logger.info("CacheMiddleware initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CacheMiddleware: {e}")

        # 指标中间件
        try:
            metrics_middleware = MetricsMiddleware(**kwargs.get("metrics", {}))
            manager.add_middleware(metrics_middleware)
            logger.info("MetricsMiddleware initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MetricsMiddleware: {e}")

        # 错误处理中间件
        try:
            error_middleware = ErrorMiddleware(**kwargs.get("error", {}))
            manager.add_middleware(error_middleware)
            logger.info("ErrorMiddleware initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ErrorMiddleware: {e}")

        return manager

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return [tool.name for tool in self.tools]

    async def execute_tool(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        apply_middleware: bool = True
    ) -> Dict[str, Any]:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            inputs: 输入参数
            apply_middleware: 是否应用中间件

        Returns:
            工具执行结果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "tool_name": tool_name
            }

        start_time = asyncio.get_event_loop().time()

        try:
            # 应用中间件前置处理
            if apply_middleware and self.middleware_manager:
                inputs = await self.middleware_manager.execute_before_tool_run(
                    tool_name, inputs
                )

            # 检查是否有缓存命中
            if inputs.get("_cache_hit", False):
                return inputs.get("_cached_result", {})

            # 执行工具
            if hasattr(tool, '_arun'):
                result = await tool._arun(**inputs)
            else:
                result = await tool._arun(inputs)

            execution_time = asyncio.get_event_loop().time() - start_time

            # 应用中间件后置处理
            if apply_middleware and self.middleware_manager:
                result = await self.middleware_manager.execute_after_tool_run(
                    tool_name, inputs, result, execution_time
                )

            return result

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            # 创建错误结果
            error_result = {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "execution_time": execution_time
            }

            # 应用中间件错误处理
            if apply_middleware and self.middleware_manager:
                error_result = await self.middleware_manager.execute_after_tool_run(
                    tool_name, inputs, error_result, execution_time
                )

            return error_result

    def create_agent_prompt(self) -> ChatPromptTemplate:
        """创建智能体提示模板"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""你是一个智能RAG助手，可以使用以下工具来回答用户的问题：

可用工具：
{tool_descriptions}

使用指南：
1. 对于需要查找信息的问题，使用 document_retrieval 工具
2. 对于需要生成回答的问题，使用 rag_generation 工具
3. 对于图片或文档分析，使用 multimodal_processing 工具
4. 对于实体关系查询，使用 knowledge_graph 工具
5. 对于系统评估，使用 rag_evaluation 工具

请根据用户问题选择合适的工具，并提供有用的回答。"""),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        return prompt_template

    def create_agent_executor(self) -> Optional[AgentExecutor]:
        """创建智能体执行器（如果安装了LangChain agent组件）"""
        try:
            # 尝试导入agent相关组件
            from langchain.agents import AgentExecutor, create_openai_tools_agent
            from langchain_core.language_models import BaseLanguageModel

            # 这里需要实际的LLM实例
            # llm = BaseLanguageModel()  # 需要根据实际情况配置

            # prompt = self.create_agent_prompt()
            # agent = create_openai_tools_agent(llm, self.tools, prompt)
            # executor = AgentExecutor(agent=agent, tools=self.tools)

            # 暂时返回None，需要实际配置LLM后才能创建
            logger.warning("AgentExecutor creation requires LLM configuration")
            return None

        except ImportError:
            logger.warning("LangChain agent components not available")
            return None
        except Exception as e:
            logger.error(f"Failed to create AgentExecutor: {e}")
            return None

    def get_middleware_stats(self) -> Dict[str, Any]:
        """获取中间件统计信息"""
        if not self.middleware_manager:
            return {"middleware_enabled": False}

        stats = {"middleware_enabled": True, "components": {}}

        # 获取各个中间件的统计信息
        for middleware in self.middleware_manager.middlewares:
            if hasattr(middleware, "get_cache_stats"):
                stats["components"][middleware.name] = middleware.get_cache_stats()
            elif hasattr(middleware, "get_system_metrics"):
                stats["components"][middleware.name] = middleware.get_system_metrics()
            elif hasattr(middleware, "get_error_stats"):
                stats["components"][middleware.name] = middleware.get_error_stats()

        return stats

    async def cleanup(self):
        """清理资源"""
        # 清理中间件资源
        if self.middleware_manager:
            for middleware in self.middleware_manager.middlewares:
                if hasattr(middleware, "cleanup"):
                    try:
                        await middleware.cleanup()
                    except Exception as e:
                        logger.error(f"Middleware cleanup failed: {e}")

        logger.info("RAGToolkit cleanup completed")

    def __str__(self) -> str:
        return f"RAGToolkit(tools={len(self.tools)}, middleware={self.enable_middleware})"

    def __repr__(self) -> str:
        return self.__str__()