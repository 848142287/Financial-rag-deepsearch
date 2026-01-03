"""
统一文档处理编排器（重构版）
整合orchestrator和core_service_integrator的功能

优化点：
- 清晰的职责分离
- 插件化的服务架构
- 统一的流程控制
- 完善的错误处理
- 详细的进度跟踪
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class ProcessingStage(Enum):
    """处理阶段"""
    VALIDATION = "validation"           # 文件验证
    PARSING = "parsing"                 # 文档解析
    CHUNKING = "chunking"               # 文档分块
    ENTITY_EXTRACTION = "entity_extraction"  # 实体提取
    EMBEDDING = "embedding"             # 向量生成
    STORAGE = "storage"                 # 存储入库
    INDEXING = "indexing"               # 索引构建


@dataclass
class StageResult:
    """阶段结果"""
    stage: ProcessingStage
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stage': self.stage.value,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'data': self.data,
            'error': self.error
        }


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    document_id: str
    filename: str
    stages: List[StageResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'document_id': self.document_id,
            'filename': self.filename,
            'stages': [stage.to_dict() for stage in self.stages],
            'metrics': self.metrics,
            'error': self.error
        }


class ProcessingPipeline:
    """
    处理流水线

    特点：
    - 插件化的阶段处理器
    - 可配置的流程控制
    - 完善的错误处理
    """

    def __init__(self):
        self.handlers: Dict[ProcessingStage, Callable] = {}
        self.middleware: List[Callable] = []

    def register_handler(
        self,
        stage: ProcessingStage,
        handler: Callable
    ):
        """注册阶段处理器"""
        self.handlers[stage] = handler
        logger.debug(f"注册处理器: {stage.value}")

    def register_middleware(self, middleware: Callable):
        """注册中间件"""
        self.middleware.append(middleware)
        logger.debug(f"注册中间件: {middleware.__name__}")

    async def execute(
        self,
        context: Dict[str, Any],
        stages: List[ProcessingStage]
    ) -> ProcessingResult:
        """
        执行流水线

        Args:
            context: 处理上下文
            stages: 要执行的阶段列表

        Returns:
            ProcessingResult
        """
        document_id = context.get('document_id', 'unknown')
        filename = context.get('filename', 'unknown')

        result = ProcessingResult(
            success=True,
            document_id=document_id,
            filename=filename
        )

        start_time = datetime.now()

        try:
            # 执行前置中间件
            for middleware in self.middleware:
                await middleware(context, 'before')

            # 执行各个阶段
            for stage in stages:
                stage_result = await self._execute_stage(stage, context)
                result.stages.append(stage_result)

                if stage_result.status == 'failed':
                    # 阶段失败，停止后续处理
                    result.success = False
                    result.error = f"阶段 {stage.value} 失败: {stage_result.error}"
                    logger.error(result.error)
                    break

                # 将阶段结果传递给下一个阶段
                context[f'{stage.value}_result'] = stage_result

            # 执行后置中间件
            for middleware in self.middleware:
                await middleware(context, 'after')

            # 计算总耗时
            total_duration = (datetime.now() - start_time).total_seconds()
            result.metrics['total_duration'] = total_duration

            logger.info(f"✅ 文档处理完成: {filename} (耗时: {total_duration:.2f}秒)")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"❌ 文档处理异常: {e}", exc_info=True)

        return result

    async def _execute_stage(
        self,
        stage: ProcessingStage,
        context: Dict[str, Any]
    ) -> StageResult:
        """
        执行单个阶段

        Args:
            stage: 处理阶段
            context: 处理上下文

        Returns:
            StageResult
        """
        stage_result = StageResult(
            stage=stage,
            status='pending'
        )

        start_time = datetime.now()

        try:
            logger.info(f"🔄 阶段开始: {stage.value}")
            stage_result.status = 'running'
            stage_result.start_time = start_time

            # 检查是否有注册的处理器
            if stage not in self.handlers:
                logger.warning(f"⚠️  阶段 {stage.value} 没有注册处理器，跳过")
                stage_result.status = 'completed'
                return stage_result

            # 执行处理器
            handler = self.handlers[stage]
            result_data = await handler(context)

            stage_result.status = 'completed'
            stage_result.data = result_data

            # 记录耗时
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            stage_result.end_time = end_time
            stage_result.duration = duration

            logger.info(f"✅ 阶段完成: {stage.value} (耗时: {duration:.2f}秒)")

        except Exception as e:
            stage_result.status = 'failed'
            stage_result.error = str(e)
            stage_result.end_time = datetime.now()
            stage_result.duration = (stage_result.end_time - start_time).total_seconds()

            logger.error(f"❌ 阶段失败: {stage.value} - {e}")

        return stage_result


class UnifiedOrchestrator:
    """
    统一文档处理编排器（重构版）

    整合了orchestrator和core_service_integrator的功能

    特点：
    - 清晰的职责分离（只负责编排，不负责业务逻辑）
    - 插件化的服务架构
    - 统一的流程控制
    - 完善的错误处理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pipeline = ProcessingPipeline()
        self.services: Dict[str, Any] = {}

        # 流水线配置
        self.enable_validation = self.config.get('enable_validation', True)
        self.enable_parsing = self.config.get('enable_parsing', True)
        self.enable_chunking = self.config.get('enable_chunking', True)
        self.enable_entity_extraction = self.config.get('enable_entity_extraction', False)
        self.enable_embedding = self.config.get('enable_embedding', True)
        self.enable_storage = self.config.get('enable_storage', True)

        self._initialized = False

    async def initialize(self):
        """初始化编排器和服务"""
        if self._initialized:
            return

        logger.info("🔧 初始化统一编排器...")

        # 初始化服务
        await self._initialize_services()

        # 注册阶段处理器
        self._register_handlers()

        # 注册中间件
        self._register_middleware()

        self._initialized = True
        logger.info("✅ 统一编排器初始化完成")

    async def _initialize_services(self):
        """初始化服务"""
        # 延迟导入，避免循环依赖
        from app.services.parsers.parser_factory import get_parser_factory
        from app.services.embeddings.unified_embedding_service import get_embedding_service
        from app.services.unified_chunker import UnifiedChunker

        # 解析器工厂
        self.services['parser_factory'] = get_parser_factory()
        logger.info("✅ 解析器工厂已加载")

        # Embedding服务
        self.services['embedding'] = get_embedding_service()
        logger.info("✅ Embedding服务已加载")

        # Chunker
        self.services['chunker'] = UnifiedChunker(config=self.config)
        logger.info("✅ Chunker已加载")

        # 可选服务
        if self.enable_entity_extraction:
            try:
                from app.services.financial_entity_extractor import get_financial_entity_extractor
                self.services['entity_extractor'] = get_financial_entity_extractor()
                logger.info("✅ 实体提取器已加载")
            except ImportError as e:
                logger.warning(f"⚠️  实体提取器加载失败: {e}")

    def _register_handlers(self):
        """注册阶段处理器"""
        # 文档解析
        if self.enable_parsing:
            self.pipeline.register_handler(
                ProcessingStage.PARSING,
                self._handle_parsing
            )

        # 文档分块
        if self.enable_chunking:
            self.pipeline.register_handler(
                ProcessingStage.CHUNKING,
                self._handle_chunking
            )

        # 向量生成
        if self.enable_embedding:
            self.pipeline.register_handler(
                ProcessingStage.EMBEDDING,
                self._handle_embedding
            )

        # 存储
        if self.enable_storage:
            self.pipeline.register_handler(
                ProcessingStage.STORAGE,
                self._handle_storage
            )

    def _register_middleware(self):
        """注册中间件"""
        # 日志中间件
        async def logging_middleware(context, position):
            if position == 'before':
                logger.info(f"📄 开始处理文档: {context.get('filename')}")
            elif position == 'after':
                logger.info(f"📊 处理统计: {context.get('metrics', {})}")

        self.pipeline.register_middleware(logging_middleware)

    # ========================================================================
    # 阶段处理器
    # ========================================================================

    async def _handle_parsing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文档解析

        Args:
            context: 处理上下文

        Returns:
            解析结果
        """
        file_path = context['file_path']

        # 使用解析器工厂自动解析
        parse_result = await self.services['parser_factory'].parse_document(file_path)

        return {
            'success': parse_result.success,
            'content': parse_result.content,
            'markdown': parse_result.markdown,
            'metadata': parse_result.metadata.to_dict() if parse_result.metadata else {},
            'sections_count': len(parse_result.sections),
            'tables_count': len(parse_result.tables),
            'images_count': len(parse_result.images)
        }

    async def _handle_chunking(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文档分块

        Args:
            context: 处理上下文

        Returns:
            分块结果
        """
        parsing_result = context['parsing_result']
        content = parsing_result['content']
        metadata = parsing_result['metadata']

        # 使用chunker进行分块
        chunks = await self.services['chunker'].chunk([content], metadata)

        return {
            'chunks_count': len(chunks),
            'avg_chunk_size': sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
        }

    async def _handle_embedding(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理向量生成

        Args:
            context: 处理上下文

        Returns:
            Embedding结果
        """
        parsing_result = context['parsing_result']
        content = parsing_result['content']

        # 生成embedding
        embedding = await self.services['embedding'].embed(content)

        return {
            'embedding_dimension': len(embedding),
            'embedding_norm': float(abs(sum(embedding)))  # L1范数
        }

    async def _handle_storage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理存储

        Args:
            context: 处理上下文

        Returns:
            存储结果
        """
        # TODO: 实现实际的存储逻辑
        document_id = context.get('document_id')

        return {
            'stored': True,
            'document_id': document_id
        }

    # ========================================================================
    # 公共接口
    # ========================================================================

    async def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        **kwargs
    ) -> ProcessingResult:
        """
        处理文档（主入口）

        Args:
            file_path: 文件路径
            document_id: 文档ID
            **kwargs: 额外参数

        Returns:
            ProcessingResult
        """
        if not self._initialized:
            await self.initialize()

        # 准备处理上下文
        context = {
            'file_path': file_path,
            'filename': kwargs.get('filename', file_path),
            'document_id': document_id,
            'config': kwargs
        }

        # 定义处理流程
        stages = [
            ProcessingStage.PARSING,
            ProcessingStage.CHUNKING,
            ProcessingStage.EMBEDDING,
            ProcessingStage.STORAGE,
        ]

        # 执行流水线
        result = await self.pipeline.execute(context, stages)

        return result

    async def batch_process_documents(
        self,
        file_paths: List[str],
        **kwargs
    ) -> List[ProcessingResult]:
        """
        批量处理文档

        Args:
            file_paths: 文件路径列表
            **kwargs: 额外参数

        Returns:
            ProcessingResult列表
        """
        tasks = []

        for i, file_path in enumerate(file_paths):
            document_id = kwargs.get(f'document_id_{i}') or f'doc_{i}'

            task = self.process_document(
                file_path=file_path,
                document_id=document_id,
                **kwargs
            )

            tasks.append(task)

        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ProcessingResult(
                        success=False,
                        document_id=f'doc_{i}',
                        filename=file_paths[i],
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取编排器指标

        Returns:
            指标字典
        """
        return {
            'initialized': self._initialized,
            'registered_handlers': list(self.pipeline.handlers.keys()),
            'middleware_count': len(self.pipeline.middleware),
            'services': list(self.services.keys())
        }


# ============================================================================
# 全局实例
# ============================================================================

_global_orchestrator: Optional[UnifiedOrchestrator] = None


def get_orchestrator(config: Optional[Dict[str, Any]] = None) -> UnifiedOrchestrator:
    """
    获取全局编排器实例

    Args:
        config: 配置参数

    Returns:
        UnifiedOrchestrator实例
    """
    global _global_orchestrator

    if _global_orchestrator is None:
        _global_orchestrator = UnifiedOrchestrator(config)
        logger.info("全局编排器已创建")

    return _global_orchestrator


# ============================================================================
# 便捷函数
# ============================================================================

async def process_document(
    file_path: str,
    config: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """
    处理文档（便捷函数）

    Args:
        file_path: 文件路径
        config: 配置参数

    Returns:
        ProcessingResult
    """
    orchestrator = get_orchestrator(config)
    return await orchestrator.process_document(file_path)


async def batch_process_documents(
    file_paths: List[str],
    config: Optional[Dict[str, Any]] = None
) -> List[ProcessingResult]:
    """
    批量处理文档（便捷函数）

    Args:
        file_paths: 文件路径列表
        config: 配置参数

    Returns:
        ProcessingResult列表
    """
    orchestrator = get_orchestrator(config)
    return await orchestrator.batch_process_documents(file_paths)


__all__ = [
    'UnifiedOrchestrator',
    'get_orchestrator',
    'process_document',
    'batch_process_documents',
    'ProcessingStage',
    'StageResult',
    'ProcessingResult',
    'ProcessingPipeline'
]
