"""
文件验证工具
"""

from typing import Tuple, Optional
from pathlib import Path


def validate_file(
    file_path: str,
    max_size: int = 100 * 1024 * 1024
) -> Tuple[bool, Optional[str]]:
    """验证文件是否可解析"""
    path = Path(file_path)

    # 检查文件是否存在
    if not path.exists():
        return False, f"文件不存在: {file_path}"

    # 检查文件是否可读
    if not path.is_file():
        return False, f"不是有效文件: {file_path}"

    # 检查文件大小
    file_size = path.stat().st_size
    if file_size > max_size:
        return False, f"文件过大: {file_size} bytes (最大: {max_size})"

    if file_size == 0:
        return False, "文件为空"

    return True, None
