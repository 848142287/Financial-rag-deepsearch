#!/usr/bin/env python3
"""
日志系统迁移工具
自动将项目中的日志初始化代码迁移到统一框架 (structured_logging)

使用方法:
    python tools/migrate_logging.py --dry-run  # 预览更改
    python tools/migrate_logging.py --execute   # 执行迁移
    python tools/migrate_logging.py --path app/services/agentic_rag  # 指定路径
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class LoggingMigrationTool:
    """日志迁移工具 - 迁移到 structured_logging"""

    # 需要替换的模式
    PATTERNS = [
        # 模式1: import logging; logger = logging.getLogger(__name__)
        (
            r'import logging\nlogger = logging\.getLogger\(__name__\)',
            'from app.core.structured_logging import get_structured_logger\nlogger = get_structured_logger(__name__)'
        ),
        # 模式2: import logging; logger = logging.getLogger(module_name)
        (
            r'import logging\nlogger = logging\.getLogger\("([^"]+)"\)',
            r'from app.core.structured_logging import get_structured_logger\nlogger = get_structured_logger("\1")'
        ),
        # 模式3: import logging 后面跟的 logger 初始化
        (
            r'import logging\n(.+?)logger = logging\.getLogger\(__name__\)',
            r'from app.core.structured_logging import get_structured_logger\n\1logger = get_structured_logger(__name__)'
        ),
        # 模式4: 已经有 logging.getLogger 但在类中
        (
            r'self\.logger = logging\.getLogger\(__name__\)',
            'self.logger = get_structured_logger(__name__)  # 使用统一日志框架'
        ),
        # 模式5: 已经有 logging.getLogger 但在类中（使用类名）
        (
            r'self\.logger = logging\.getLogger\(self\.__class\.__name__\)',
            'self.logger = get_structured_logger(self.__class__.__name__)  # 使用统一日志框架'
        ),
    ]

    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.stats = {
            'files_scanned': 0,
            'files_modified': 0,
            'replacements_made': 0,
            'errors': []
        }

    def find_python_files(self, path: Path) -> List[Path]:
        """查找所有 Python 文件"""
        if path.is_file() and path.suffix == '.py':
            return [path]

        python_files = []
        for root, dirs, files in os.walk(path):
            # 跳过虚拟环境和缓存目录
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', 'venv', 'env', '.venv',
                'node_modules', '.git', 'dist', 'build'
            }]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        return python_files

    def should_migrate_file(self, file_path: Path) -> bool:
        """判断文件是否需要迁移"""
        # 跳过已经迁移的文件
        content = file_path.read_text(encoding='utf-8', errors='ignore')

        # 如果已经使用了统一框架，跳过
        if 'from app.core.structured_logging import get_structured_logger' in content:
            return False

        # 如果使用了旧的 logging 模式，需要迁移
        if 'import logging' in content and 'logging.getLogger' in content:
            return True

        return False

    def migrate_file(self, file_path: Path) -> Tuple[bool, int]:
        """迁移单个文件"""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            replacements = 0

            # 应用替换模式
            for pattern, replacement in self.PATTERNS:
                new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
                if new_content != content:
                    replacements += 1
                    content = new_content

            # 如果没有替换，尝试简单的模式
            if replacements == 0:
                # 简单的 import logging 替换
                if 'import logging' in content and 'logging.getLogger' in content:
                    # 提取模块名
                    module_match = re.search(r'logger = logging\.getLogger\("([^"]+)"\)', content)
                    if module_match:
                        module_name = module_match.group(1)
                        import_replacement = f'from app.core.structured_logging import get_structured_logger\nlogger = get_structured_logger("{module_name}")'
                    else:
                        import_replacement = 'from app.core.structured_logging import get_structured_logger\nlogger = get_structured_logger(__name__)'

                    content = re.sub(
                        r'import logging',
                        import_replacement,
                        content,
                        count=1
                    )
                    replacements += 1

            if content != original_content:
                if not self.dry_run:
                    # 备份原文件
                    backup_path = file_path.with_suffix(f'{file_path.suffix}.bak')
                    backup_path.write_text(original_content, encoding='utf-8')

                    # 写入新内容
                    file_path.write_text(content, encoding='utf-8')

                return True, replacements

            return False, 0

        except Exception as e:
            self.stats['errors'].append(f"{file_path}: {str(e)}")
            return False, 0

    def migrate(self, target_path: Path = None):
        """执行迁移"""
        path = target_path or self.project_root / 'app'
        if not path.is_absolute():
            path = self.project_root / path

        python_files = self.find_python_files(path)

        print(f"🔍 扫描 {len(python_files)} 个 Python 文件...")

        for file_path in python_files:
            self.stats['files_scanned'] += 1

            if self.should_migrate_file(file_path):
                modified, replacements = self.migrate_file(file_path)

                if modified:
                    self.stats['files_modified'] += 1
                    self.stats['replacements_made'] += replacements
                    try:
                        rel_path = file_path.relative_to(self.project_root)
                    except ValueError:
                        rel_path = file_path
                    print(f"✅ 迁移: {rel_path} ({replacements} 处替换)")

        self.print_summary()

    def print_summary(self):
        """打印迁移摘要"""
        print("\n" + "="*60)
        print("迁移完成统计")
        print("="*60)
        print(f"扫描文件数: {self.stats['files_scanned']}")
        print(f"修改文件数: {self.stats['files_modified']}")
        print(f"替换次数: {self.stats['replacements_made']}")

        if self.stats['errors']:
            print(f"\n❌ 错误 ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                print(f"  - {error}")

        if self.dry_run:
            print("\n⚠️  这是预览模式，没有实际修改文件")
            print("   使用 --execute 参数执行实际迁移")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='日志系统迁移工具')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='预览模式，不实际修改文件（默认）'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='执行实际迁移'
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=None,
        help='指定要迁移的路径（默认: app 目录）'
    )

    args = parser.parse_args()

    # 如果指定了 --execute，禁用 dry-run
    dry_run = not args.execute

    project_root = Path(__file__).parent.parent
    tool = LoggingMigrationTool(project_root, dry_run=dry_run)

    print(f"🚀 日志系统迁移工具 (structured_logging)")
    print(f"模式: {'预览' if dry_run else '执行'}")
    print(f"目标路径: {args.path or 'app'}")
    print()

    tool.migrate(args.path)


if __name__ == '__main__':
    main()
