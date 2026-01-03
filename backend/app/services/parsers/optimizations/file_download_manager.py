"""
统一文件下载管理器
解决重复下载MinIO文件的问题,提供缓存机制
"""

import hashlib
from app.core.structured_logging import get_structured_logger
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = get_structured_logger(__name__)


@dataclass
class CachedFile:
    """缓存的文件信息"""
    local_path: str
    original_path: str
    size: int
    hash: str
    downloaded_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class FileDownloadManager:
    """
    统一文件下载管理器

    功能:
    1. 避免重复下载同一文件
    2. 提供文件哈希校验
    3. 自动管理临时文件
    4. 支持缓存过期
    """

    def __init__(
        self,
        minio_service=None,
        cache_ttl_seconds: int = 3600,  # 缓存1小时
        max_cache_size_mb: int = 1024,  # 最大缓存1GB
        temp_dir: Optional[str] = None
    ):
        """
        初始化文件下载管理器

        Args:
            minio_service: MinIO服务实例
            cache_ttl_seconds: 缓存生存时间(秒)
            max_cache_size_mb: 最大缓存大小(MB)
            temp_dir: 临时文件目录
        """
        self.minio_service = minio_service
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # 转换为字节

        # 创建临时目录
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix='file_download_cache_')
        else:
            self.temp_dir = temp_dir
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # 文件缓存: original_path -> CachedFile
        self.file_cache: Dict[str, CachedFile] = {}

        # 当前缓存大小
        self.current_cache_size = 0

        logger.info(
            f"FileDownloadManager initialized: "
            f"cache_ttl={cache_ttl_seconds}s, "
            f"max_cache_size={max_cache_size_mb}MB"
        )

    async def get_file(
        self,
        file_path: str,
        force_refresh: bool = False
    ) -> Optional[str]:
        """
        获取文件本地路径(自动下载并缓存)

        Args:
            file_path: MinIO中的文件路径
            force_refresh: 是否强制重新下载

        Returns:
            本地文件路径,如果失败返回None
        """
        try:
            # 检查缓存
            if not force_refresh and file_path in self.file_cache:
                cached = self.file_cache[file_path]

                # 检查缓存是否过期
                if datetime.now() - cached.downloaded_at < self.cache_ttl:
                    # 验证文件是否仍然存在
                    if Path(cached.local_path).exists():
                        cached.access_count += 1
                        cached.last_accessed = datetime.now()
                        logger.debug(
                            f"Using cached file: {file_path} "
                            f"(accessed {cached.access_count} times)"
                        )
                        return cached.local_path
                    else:
                        # 文件已被删除,从缓存中移除
                        logger.warning(f"Cached file missing: {cached.local_path}")
                        del self.file_cache[file_path]
                        self.current_cache_size -= cached.size

            # 下载文件
            return await self._download_and_cache(file_path)

        except Exception as e:
            logger.error(f"Failed to get file {file_path}: {e}")
            return None

    async def _download_and_cache(self, file_path: str) -> Optional[str]:
        """下载文件并缓存"""
        if self.minio_service is None:
            raise Exception("MinIO service not available")

        # 下载文件数据
        file_data = await self.minio_service.download_file(file_path)
        if not file_data:
            raise FileNotFoundError(f"File not found in MinIO: {file_path}")

        # 计算文件哈希
        file_hash = hashlib.sha256(file_data).hexdigest()
        file_size = len(file_data)

        # 检查缓存空间,必要时清理
        await self._ensure_cache_space(file_size)

        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(
            dir=self.temp_dir,
            delete=False,
            suffix=Path(file_path).suffix
        )
        temp_file.write(file_data)
        temp_file.close()

        local_path = temp_file.name

        # 添加到缓存
        cached_file = CachedFile(
            local_path=local_path,
            original_path=file_path,
            size=file_size,
            hash=file_hash,
            downloaded_at=datetime.now(),
            access_count=1,
            last_accessed=datetime.now()
        )

        self.file_cache[file_path] = cached_file
        self.current_cache_size += file_size

        logger.info(
            f"Downloaded and cached: {file_path} "
            f"({file_size / 1024 / 1024:.2f} MB, "
            f"cache size: {self.current_cache_size / 1024 / 1024:.2f} MB)"
        )

        return local_path

    async def _ensure_cache_space(self, required_size: int):
        """确保有足够的缓存空间"""
        if self.current_cache_size + required_size <= self.max_cache_size:
            return

        logger.info("Cache full, cleaning up old files...")

        # 按最后访问时间排序(最久未访问的先删除)
        cached_files = sorted(
            self.file_cache.values(),
            key=lambda x: x.last_accessed or x.downloaded_at
        )

        freed_space = 0
        for cached in cached_files:
            # 删除文件
            try:
                Path(cached.local_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete cached file: {e}")

            # 更新缓存
            self.current_cache_size -= cached.size
            del self.file_cache[cached.original_path]
            freed_space += cached.size

            logger.debug(
                f"Removed from cache: {cached.original_path} "
                f"({cached.size / 1024 / 1024:.2f} MB)"
            )

            # 检查是否释放了足够空间
            if self.current_cache_size + required_size <= self.max_cache_size:
                break

        logger.info(
            f"Freed {freed_space / 1024 / 1024:.2f} MB, "
            f"current cache size: {self.current_cache_size / 1024 / 1024:.2f} MB"
        )

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """获取文件哈希值(用于增量处理)"""
        if file_path in self.file_cache:
            return self.file_cache[file_path].hash
        return None

    async def cleanup(self):
        """清理所有缓存文件"""
        logger.info(f"Cleaning up {len(self.file_cache)} cached files...")

        for cached in self.file_cache.values():
            try:
                Path(cached.local_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete cached file: {e}")

        # 清空缓存
        self.file_cache.clear()
        self.current_cache_size = 0

        # 删除临时目录
        try:
            Path(self.temp_dir).rmdir()
        except Exception as e:
            logger.warning(f"Failed to delete temp directory: {e}")

        logger.info("FileDownloadManager cleanup completed")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cached_files": len(self.file_cache),
            "cache_size_mb": round(self.current_cache_size / 1024 / 1024, 2),
            "max_cache_size_mb": round(self.max_cache_size / 1024 / 1024, 2),
            "cache_usage_percent": round(
                self.current_cache_size / self.max_cache_size * 100, 2
            ),
            "temp_dir": self.temp_dir,
            "cache_ttl_seconds": self.cache_ttl.total_seconds(),
            "total_access_count": sum(
                cached.access_count for cached in self.file_cache.values()
            )
        }

    def invalidate_cache(self, file_path: Optional[str] = None):
        """
        使缓存失效

        Args:
            file_path: 指定文件路径,如果为None则清空所有缓存
        """
        if file_path is None:
            # 清空所有缓存
            for cached in self.file_cache.values():
                try:
                    Path(cached.local_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete cached file: {e}")

            self.file_cache.clear()
            self.current_cache_size = 0
            logger.info("All cache invalidated")
        else:
            # 使指定文件缓存失效
            if file_path in self.file_cache:
                cached = self.file_cache[file_path]
                try:
                    Path(cached.local_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete cached file: {e}")

                self.current_cache_size -= cached.size
                del self.file_cache[file_path]
                logger.info(f"Cache invalidated for: {file_path}")
            else:
                logger.warning(f"File not in cache: {file_path}")


# 上下文管理器支持
from contextlib import asynccontextmanager


@asynccontextmanager
async def managed_file_download(
    download_manager: FileDownloadManager,
    file_path: str
):
    """
    文件下载上下文管理器

    用法:
        async with managed_file_download(manager, path) as local_path:
            # 使用本地文件路径
            process_file(local_path)
        # 自动清理
    """
    local_path = None
    try:
        local_path = await download_manager.get_file(file_path)
        yield local_path
    finally:
        # 可选: 立即释放缓存
        # download_manager.invalidate_cache(file_path)
        pass
