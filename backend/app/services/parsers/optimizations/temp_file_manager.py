"""
临时文件管理器
统一管理临时文件的创建、追踪和清理,防止临时文件泄漏
"""

from app.core.structured_logging import get_structured_logger
import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from enum import Enum

logger = get_structured_logger(__name__)

class TempFileType(Enum):
    """临时文件类型"""
    EXTRACTED_IMAGES = "extracted_images"
    CONVERTED_DOCUMENT = "converted_document"
    DOWNLOADED_CONTENT = "downloaded_content"
    PROCESSED_DATA = "processed_data"
    GENERAL = "general"

@dataclass
class TempFileRecord:
    """临时文件记录"""
    file_path: str
    file_type: TempFileType
    created_at: datetime
    size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    persistent: bool = False  # 是否持久化(不自动删除)

class TempFileManager:
    """
    临时文件管理器

    功能:
    1. 统一创建临时文件和目录
    2. 自动追踪所有临时文件
    3. 支持上下文管理器自动清理
    4. 提供批量清理和按需清理
    5. 防止临时文件泄漏
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        auto_cleanup: bool = True,
        max_age_seconds: int = 3600  # 1小时后自动清理
    ):
        """
        初始化临时文件管理器

        Args:
            base_dir: 基础临时目录
            auto_cleanup: 是否自动清理
            max_age_seconds: 文件最大保存时间
        """
        self.auto_cleanup = auto_cleanup
        self.max_age_seconds = max_age_seconds

        # 创建基础临时目录
        if base_dir is None:
            self.base_dir = tempfile.mkdtemp(prefix='temp_file_manager_')
        else:
            self.base_dir = base_dir
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)

        # 文件记录: file_path -> TempFileRecord
        self.files: Dict[str, TempFileRecord] = {}

        # 目录记录: dir_path -> creation_time
        self.directories: Dict[str, datetime] = {}

        logger.info(
            f"TempFileManager initialized: "
            f"base_dir={self.base_dir}, "
            f"auto_cleanup={auto_cleanup}"
        )

    def create_temp_dir(
        self,
        prefix: str = "",
        file_type: TempFileType = TempFileType.GENERAL
    ) -> str:
        """
        创建临时目录

        Args:
            prefix: 目录前缀
            file_type: 文件类型

        Returns:
            临时目录路径
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self.base_dir)

        self.directories[temp_dir] = datetime.now()

        logger.debug(f"Created temp directory: {temp_dir}")
        return temp_dir

    def create_temp_file(
        self,
        suffix: str = "",
        prefix: str = "",
        content: Optional[bytes] = None,
        file_type: TempFileType = TempFileType.GENERAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建临时文件

        Args:
            suffix: 文件后缀
            prefix: 文件前缀
            content: 文件内容
            file_type: 文件类型
            metadata: 元数据

        Returns:
            临时文件路径
        """
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=self.base_dir,
            delete=False  # 手动管理删除
        )

        if content:
            temp_file.write(content)

        temp_file.close()

        file_path = temp_file.name
        file_size = os.path.getsize(file_path)

        # 记录文件
        record = TempFileRecord(
            file_path=file_path,
            file_type=file_type,
            created_at=datetime.now(),
            size=file_size,
            metadata=metadata or {}
        )

        self.files[file_path] = record

        logger.debug(
            f"Created temp file: {file_path} "
            f"(type={file_type.value}, size={file_size})"
        )

        return file_path

    def register_existing_file(
        self,
        file_path: str,
        file_type: TempFileType = TempFileType.GENERAL,
        metadata: Optional[Dict[str, Any]] = None,
        persistent: bool = False
    ):
        """
        注册已存在的文件

        Args:
            file_path: 文件路径
            file_type: 文件类型
            metadata: 元数据
            persistent: 是否持久化
        """
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return

        file_size = os.path.getsize(file_path)

        record = TempFileRecord(
            file_path=file_path,
            file_type=file_type,
            created_at=datetime.now(),
            size=file_size,
            metadata=metadata or {},
            persistent=persistent
        )

        self.files[file_path] = record

        logger.debug(f"Registered existing file: {file_path}")

    def mark_persistent(self, file_path: str):
        """标记文件为持久化(不自动删除)"""
        if file_path in self.files:
            self.files[file_path].persistent = True
            logger.debug(f"Marked as persistent: {file_path}")
        else:
            logger.warning(f"File not in registry: {file_path}")

    def cleanup_file(self, file_path: str, ignore_errors: bool = True):
        """
        清理指定文件

        Args:
            file_path: 文件路径
            ignore_errors: 是否忽略错误
        """
        if file_path in self.files:
            # 检查是否持久化
            if self.files[file_path].persistent:
                logger.debug(f"Skipping persistent file: {file_path}")
                return

            # 删除文件
            try:
                os.unlink(file_path)
                logger.debug(f"Deleted temp file: {file_path}")
            except Exception as e:
                if not ignore_errors:
                    raise
                logger.warning(f"Failed to delete file {file_path}: {e}")

            # 从记录中移除
            del self.files[file_path]
        else:
            # 文件未注册,尝试直接删除
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Deleted unregistered file: {file_path}")
            except Exception as e:
                if not ignore_errors:
                    raise
                logger.warning(f"Failed to delete unregistered file {file_path}: {e}")

    def cleanup_dir(self, dir_path: str, ignore_errors: bool = True):
        """
        清理指定目录

        Args:
            dir_path: 目录路径
            ignore_errors: 是否忽略错误
        """
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=ignore_errors)
                logger.debug(f"Deleted temp directory: {dir_path}")

            if dir_path in self.directories:
                del self.directories[dir_path]

        except Exception as e:
            if not ignore_errors:
                raise
            logger.warning(f"Failed to delete directory {dir_path}: {e}")

    def cleanup_by_type(self, file_type: TempFileType):
        """按类型清理文件"""
        to_remove = [
            file_path
            for file_path, record in self.files.items()
            if record.file_type == file_type and not record.persistent
        ]

        for file_path in to_remove:
            self.cleanup_file(file_path)

        logger.info(f"Cleaned up {len(to_remove)} files of type {file_type.value}")

    def cleanup_old_files(self, max_age_seconds: Optional[int] = None):
        """
        清理旧文件

        Args:
            max_age_seconds: 最大年龄(秒),None则使用默认值
        """
        max_age = max_age_seconds or self.max_age_seconds
        cutoff_time = datetime.now().timestamp() - max_age

        to_remove = [
            file_path
            for file_path, record in self.files.items()
            if record.created_at.timestamp() < cutoff_time
            and not record.persistent
        ]

        for file_path in to_remove:
            self.cleanup_file(file_path)

        logger.info(f"Cleaned up {len(to_remove)} old files")

    def cleanup_all(self):
        """清理所有临时文件和目录"""
        logger.info(f"Cleaning up all temp files ({len(self.files)} files, {len(self.directories)} dirs)...")

        # 清理所有文件
        for file_path in list(self.files.keys()):
            self.cleanup_file(file_path)

        # 清理所有目录
        for dir_path in list(self.directories.keys()):
            self.cleanup_dir(dir_path)

        # 清理基础目录
        try:
            if os.path.exists(self.base_dir):
                os.rmdir(self.base_dir)
        except Exception as e:
            logger.warning(f"Failed to delete base directory: {e}")

        logger.info("All temp files cleaned up")

    def __del__(self):
        """析构函数 - 自动清理"""
        if self.auto_cleanup:
            self.cleanup_all()

    @contextmanager
    def temp_directory(
        self,
        prefix: str = "",
        file_type: TempFileType = TempFileType.GENERAL
    ):
        """
        临时目录上下文管理器

        用法:
            with manager.temp_directory(prefix='test_') as temp_dir:
                # 使用temp_dir
                process_files(temp_dir)
            # 自动清理
        """
        temp_dir = self.create_temp_dir(prefix=prefix, file_type=file_type)
        try:
            yield temp_dir
        finally:
            self.cleanup_dir(temp_dir)

    @contextmanager
    def temp_file(
        self,
        suffix: str = "",
        prefix: str = "",
        content: Optional[bytes] = None,
        file_type: TempFileType = TempFileType.GENERAL
    ):
        """
        临时文件上下文管理器

        用法:
            with manager.temp_file(suffix='.pdf') as temp_path:
                # 使用temp_path
                process_file(temp_path)
            # 自动清理
        """
        temp_path = self.create_temp_file(
            suffix=suffix,
            prefix=prefix,
            content=content,
            file_type=file_type
        )
        try:
            yield temp_path
        finally:
            self.cleanup_file(temp_path)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_size = sum(record.size for record in self.files.values())

        # 按类型统计
        type_counts = {}
        type_sizes = {}
        for record in self.files.values():
            type_name = record.file_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            type_sizes[type_name] = type_sizes.get(type_name, 0) + record.size

        return {
            "total_files": len(self.files),
            "total_directories": len(self.directories),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "base_dir": self.base_dir,
            "auto_cleanup": self.auto_cleanup,
            "type_counts": type_counts,
            "type_sizes_mb": {
                k: round(v / 1024 / 1024, 2)
                for k, v in type_sizes.items()
            }
        }

# 全局单例
_global_temp_manager: Optional[TempFileManager] = None

def get_temp_manager() -> TempFileManager:
    """获取全局临时文件管理器"""
    global _global_temp_manager
    if _global_temp_manager is None:
        _global_temp_manager = TempFileManager()
    return _global_temp_manager
