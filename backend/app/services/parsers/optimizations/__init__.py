"""
文档解析优化模块

提供性能优化和资源管理功能:
1. FileDownloadManager - 统一文件下载和缓存管理
2. TempFileManager - 临时文件生命周期管理
3. ConcurrentProcessor - 高效并发处理工具
4. OptimizedDocumentOrchestratorCore - 集成的优化编排器核心

使用示例:
    from app.services.parsers.optimizations import (
        FileDownloadManager,
        TempFileManager,
        ConcurrentProcessor,
        OptimizedDocumentOrchestratorCore
    )

    # 使用优化的编排器核心
    core = OptimizedDocumentOrchestratorCore(minio_service=...)
    result = await core.parse_document_optimized(document, config, parser_factory)
"""

__all__ = [
    # File Download Manager
    'FileDownloadManager',
    'CachedFile',
    'managed_file_download',

    # Temp File Manager
    'TempFileManager',
    'TempFileType',
    'TempFileRecord',
    'get_temp_manager',

    # Concurrent Processor
    'ConcurrentProcessor',
    'ProcessResult',
    'BatchProcessStats',
    'process_images_concurrent',
    'process_elements_concurrent',

    # Optimized Orchestrator Core
    'OptimizedDocumentOrchestratorCore',
    'create_progress_callback',
]

__version__ = '1.0.0'
__author__ = 'Claude Code Optimization Assistant'
