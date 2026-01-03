"""
优化的文档编排器核心组件
整合文件下载管理、临时文件管理、并发处理等优化
"""

import logging
from app.core.structured_logging import get_structured_logger
from typing import Dict, List, Any, Optional
from pathlib import Path

from app.services.parsers.optimizations.file_download_manager import FileDownloadManager
from app.services.parsers.optimizations.concurrent_processor import ConcurrentProcessor

logger = get_structured_logger(__name__)

class OptimizedDocumentOrchestratorCore:
    """
    优化的文档编排器核心

    优化点:
    1. 使用FileDownloadManager避免重复下载
    2. 使用TempFileManager统一管理临时文件
    3. 使用ConcurrentProcessor并发处理图片等任务
    4. 修复增量处理逻辑
    5. 添加进度跟踪
    """

    def __init__(
        self,
        minio_service=None,
        max_concurrency: int = 5,
        cache_ttl_seconds: int = 3600
    ):
        """
        初始化优化的编排器核心

        Args:
            minio_service: MinIO服务
            max_concurrency: 最大并发数
            cache_ttl_seconds: 文件缓存生存时间
        """
        # 文件下载管理器
        self.file_manager = FileDownloadManager(
            minio_service=minio_service,
            cache_ttl_seconds=cache_ttl_seconds
        )

        # 临时文件管理器
        self.temp_manager = get_temp_manager()

        # 并发处理器
        self.concurrent_processor = ConcurrentProcessor(
            max_concurrency=max_concurrency
        )

        logger.info("OptimizedDocumentOrchestratorCore initialized")

    async def get_document_for_processing(
        self,
        document_file_path: str
    ) -> Optional[str]:
        """
        获取文档本地路径用于处理(自动下载并缓存)

        Args:
            document_file_path: MinIO中的文档路径

        Returns:
            本地文件路径,失败返回None
        """
        try:
            local_path = await self.file_manager.get_file(document_file_path)
            logger.info(f"Document ready for processing: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to get document for processing: {e}")
            return None

    async def check_document_changed(
        self,
        document_file_path: str,
        known_hash: Optional[str] = None
    ) -> bool:
        """
        检查文档是否已更改(修复增量处理)

        Args:
            document_file_path: MinIO中的文档路径
            known_hash: 已知的哈希值

        Returns:
            True表示文档已更改或无法确定,False表示未更改
        """
        try:
            # 获取当前文件哈希
            current_hash = self.file_manager.get_file_hash(document_file_path)

            if current_hash is None:
                # 文件未缓存,需要下载来计算哈希
                await self.file_manager.get_file(document_file_path)
                current_hash = self.file_manager.get_file_hash(document_file_path)

            if known_hash is None:
                # 没有已知哈希,认为文档已更改
                return True

            # 比较哈希
            changed = (current_hash != known_hash)

            if changed:
                logger.info(f"Document has changed: {document_file_path}")
            else:
                logger.info(f"Document unchanged: {document_file_path}")

            return changed

        except Exception as e:
            logger.error(f"Failed to check document changes: {e}")
            # 出错时认为文档已更改,确保不会跳过处理
            return True

    async def parse_document_optimized(
        self,
        document,
        config,
        parser_factory: callable
    ) -> Dict[str, Any]:
        """
        优化的文档解析流程

        Args:
            document: 文档对象
            config: 处理配置
            parser_factory: 解析器工厂函数

        Returns:
            解析结果
        """
        import time
        start_time = time.time()

        try:
            # 1. 获取本地文件(自动处理下载和缓存)
            local_path = await self.get_document_for_processing(document.file_path)
            if not local_path:
                return {
                    "success": False,
                    "error": "Failed to get document for processing"
                }

            logger.info(f"Document downloaded to: {local_path}")

            # 2. 创建临时目录用于提取图片等
            with self.temp_manager.temp_directory(
                prefix=f'doc_{document.id}_',
                file_type=TempFileType.EXTRACTED_IMAGES
            ) as temp_dir:
                logger.info(f"Created temp directory: {temp_dir}")

                # 3. 获取解析器
                parser = parser_factory(config)

                # 4. 解析文档
                parse_result = await parser.parse(
                    local_path,
                    temp_dir=temp_dir,
                    extract_images=config.extract_images,
                    enable_ocr=config.use_ocr
                )

                parse_time = time.time() - start_time
                logger.info(f"Document parsed in {parse_time:.2f}s")

                return {
                    "success": parse_result.success,
                    "content": parse_result.content,
                    "metadata": parse_result.metadata,
                    "parse_time": parse_time
                }

        except Exception as e:
            logger.error(f"Optimized document parsing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def analyze_images_concurrent(
        self,
        image_elements: List[Dict[str, Any]],
        multimodal_analyzer,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        并发分析图片(优化多模态分析)

        Args:
            image_elements: 图片元素列表
            multimodal_analyzer: 多模态分析器
            progress_callback: 进度回调

        Returns:
            分析结果列表
        """
        if not image_elements:
            return []

        logger.info(f"Starting concurrent analysis of {len(image_elements)} images")

        async def analyze_single_image(image_info):
            """分析单张图片"""
            image_path = image_info.get('temp_path')
            if not image_path or not Path(image_path).exists():
                return {
                    'error': 'Image file not found',
                    'original_info': image_info
                }

            try:
                result = await multimodal_analyzer.analyze_image(
                    image_path,
                    image_type='auto'
                )

                return {
                    'success': True,
                    'description': result.description,
                    'ocr_text': result.ocr_text,
                    'image_type': result.image_type,
                    'chart_info': result.chart_info,
                    'formula_info': result.formula_info,
                    'confidence': result.confidence,
                    'original_info': image_info
                }

            except Exception as e:
                logger.error(f"Failed to analyze image: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'original_info': image_info
                }

        try:
            # 使用并发处理器
            results, stats = await self.concurrent_processor.process_items_async(
                items=image_elements,
                process_func=analyze_single_image,
                progress_callback=progress_callback
            )

            logger.info(
                f"Concurrent image analysis completed: "
                f"{stats.successful}/{stats.total_items} successful, "
                f"{stats.total_time:.2f}s"
            )

            return [r.result for r in results]

        except Exception as e:
            logger.error(f"Concurrent image analysis failed: {e}")
            # 降级到串行处理
            logger.warning("Falling back to sequential processing")
            return await self._analyze_images_sequential(
                image_elements, multimodal_analyzer
            )

    async def _analyze_images_sequential(
        self,
        image_elements: List[Dict[str, Any]],
        multimodal_analyzer
    ) -> List[Dict[str, Any]]:
        """串行分析图片(降级方案)"""
        results = []

        for i, image_info in enumerate(image_elements):
            logger.info(f"Analyzing image {i+1}/{len(image_elements)}")

            image_path = image_info.get('temp_path')
            if not image_path or not Path(image_path).exists():
                results.append({
                    'error': 'Image file not found',
                    'original_info': image_info
                })
                continue

            try:
                result = await multimodal_analyzer.analyze_image(
                    image_path,
                    image_type='auto'
                )

                results.append({
                    'success': True,
                    'description': result.description,
                    'ocr_text': result.ocr_text,
                    'image_type': result.image_type,
                    'chart_info': result.chart_info,
                    'formula_info': result.formula_info,
                    'confidence': result.confidence,
                    'original_info': image_info
                })

            except Exception as e:
                logger.error(f"Failed to analyze image {i+1}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'original_info': image_info
                })

        return results

    def get_document_hash(self, document_file_path: str) -> Optional[str]:
        """
        获取文档哈希值(用于增量处理)

        Args:
            document_file_path: MinIO中的文档路径

        Returns:
            文档哈希值,失败返回None
        """
        return self.file_manager.get_file_hash(document_file_path)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            "file_cache": self.file_manager.get_cache_stats(),
            "temp_files": self.temp_manager.get_stats(),
            "concurrent_processor": {
                "max_concurrency": self.concurrent_processor.max_concurrency,
                "max_retries": self.concurrent_processor.max_retries
            }
        }

    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up optimized orchestrator resources...")

        # 清理文件缓存
        await self.file_manager.cleanup()

        # 关闭并发处理器
        self.concurrent_processor.shutdown()

        logger.info("Optimized orchestrator cleanup completed")

# 便捷函数
def create_progress_callback(logger: logging.Logger, operation: str):
    """
    创建进度回调函数

    Args:
        logger: 日志记录器
        operation: 操作名称

    Returns:
        进度回调函数
    """
    def callback(completed: int, total: int):
        percentage = (completed / total) * 100 if total > 0 else 0
        logger.info(f"{operation}: {completed}/{total} ({percentage:.1f}%)")

    return callback
