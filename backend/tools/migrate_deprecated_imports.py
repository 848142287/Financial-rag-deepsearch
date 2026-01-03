#!/usr/bin/env python3
"""
迁移已删除模块的导入
自动更新代码中对已删除模块的引用
"""

import re
from pathlib import Path
from typing import Dict, Tuple

# 迁移规则：(旧模式, 新模式)
MIGRATIONS: Dict[str, str] = {
    # === rag_cache 相关 ===
    r'from \.rag_cache import': 'from app.core.cache.migration_adapter import',
    r'from app\.services\.agentic_rag\.rag_cache import': 'from app.core.cache.migration_adapter import',

    # === consolidated_rag_service 相关 ===
    r'from app\.services\.consolidated_rag_service import': 'from app.services.rag.unified_rag_entry import',
    'ConsolidatedRAGService': 'UnifiedRAGService',
    'RetrievalMode': 'RAGMode',
    'RetrievalLevel': 'RAGQuery',  # 注意：RAGLevel 已改为 RAGQuery

    # === legacy_doc_parser 相关 ===
    r'from \.legacy_doc_parser import': 'from app.services.parsers.unified_parser import',
    r'from app\.services\.parsers\.legacy_doc_parser import': 'from app.services.parsers.unified_parser import',
    'LegacyDocParser': 'UnifiedDocumentParser',

    # === 其他常见替换 ===
    'agentic_rag_cache': 'rag_cache',  # 使用适配器中的全局实例
}

def migrate_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    迁移单个文件

    Args:
        file_path: 文件路径
        dry_run: 是否预览模式

    Returns:
        (是否修改, 替换数量)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        replacements = 0

        # 应用迁移规则
        for pattern, replacement in MIGRATIONS.items():
            # 使用正则替换
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                replacements += count

        if content != original:
            if not dry_run:
                # 备份原文件
                backup_path = file_path.with_suffix(f'{file_path.suffix}.bak')
                backup_path.write_text(original, encoding='utf-8')

                # 写入新内容
                file_path.write_text(content, encoding='utf-8')

            return True, replacements

        return False, 0

    except Exception as e:
        print(f"❌ 处理文件失败 {file_path}: {e}")
        return False, 0

def find_files_to_migrate(project_root: Path) -> list[Path]:
    """查找需要迁移的文件"""
    backend_dir = project_root / 'backend' / 'app'
    python_files = []

    # 查找所有包含已删除模块导入的文件
    deprecated_imports = [
        'from .rag_cache import',
        'from app.services.agentic_rag.rag_cache import',
        'from app.services.consolidated_rag_service import',
        'from .legacy_doc_parser import',
        'from app.services.parsers.legacy_doc_parser import',
    ]

    for py_file in backend_dir.rglob('*.py'):
        # 跳过 __pycache__ 和虚拟环境
        if '__pycache__' in str(py_file) or 'venv' in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')
            for pattern in deprecated_imports:
                if pattern in content:
                    python_files.append(py_file)
                    break
        except Exception:
            continue

    return python_files

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='迁移已删除模块的导入')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，不实际修改文件'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='执行实际迁移'
    )

    args = parser.parse_args()

    # 确定执行模式
    dry_run = not args.execute

    # 获取项目根目录
    script_path = Path(__file__).absolute()
    project_root = script_path.parent.parent.parent

    print("🔄 已删除模块导入迁移工具")
    print(f"模式: {'预览' if dry_run else '执行'}")
    print(f"项目根目录: {project_root}")
    print()

    # 查找需要迁移的文件
    print("🔍 扫描需要迁移的文件...")
    files_to_migrate = find_files_to_migrate(project_root)

    if not files_to_migrate:
        print("✅ 没有需要迁移的文件")
        return

    print(f"找到 {len(files_to_migrate)} 个需要迁移的文件")
    print()

    # 迁移文件
    total_replacements = 0
    migrated_files = []

    for file_path in files_to_migrate:
        modified, replacements = migrate_file(file_path, dry_run)

        if modified:
            migrated_files.append(file_path)
            total_replacements += replacements
            rel_path = file_path.relative_to(project_root)
            print(f"✓ {'[预览] ' if dry_run else ''}迁移: {rel_path} ({replacements} 处替换)")

    # 打印摘要
    print()
    print("=" * 60)
    print("迁移完成统计")
    print("=" * 60)
    print(f"扫描文件数: {len(files_to_migrate)}")
    print(f"迁移文件数: {len(migrated_files)}")
    print(f"替换次数: {total_replacements}")

    if dry_run:
        print()
        print("⚠️  这是预览模式，没有实际修改文件")
        print("   使用 --execute 参数执行实际迁移")

    print("=" * 60)

    # 输出需要手动检查的文件
    if migrated_files:
        print()
        print("📝 建议手动检查以下文件:")
        for file_path in migrated_files:
            print(f"   - {file_path.relative_to(project_root)}")

if __name__ == '__main__':
    main()
