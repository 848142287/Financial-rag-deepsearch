"""
TODO清理和处理工具
自动扫描、分类和处理代码中的TODO标记
"""

import re
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class TodoCategory(Enum):
    """TODO类别"""
    IMPLEMENT_FEATURE = "implement_feature"  # 需要实现的功能
    OPTIMIZATION = "optimization"            # 性能优化
    REFACTOR = "refactor"                    # 代码重构
    REMOVE = "remove"                        # 可以移除
    DEPENDENCY = "dependency"                # 依赖问题
    BUG = "bug"                              # 需要修复的bug

@dataclass
class TodoItem:
    """TODO项"""
    file_path: str
    line_number: int
    content: str
    category: TodoCategory
    priority: str  # high, medium, low
    action: str    # keep, remove, implement, defer

def categorize_todo(content: str) -> TodoCategory:
    """
    根据内容判断TODO类别

    Args:
        content: TODO内容

    Returns:
        TODO类别
    """
    content_lower = content.lower()

    # 功能实现相关
    keywords_feature = [
        '实现', 'implement', '集成', 'integrate',
        '添加', 'add', '创建', 'create'
    ]
    if any(kw in content_lower for kw in keywords_feature):
        return TodoCategory.IMPLEMENT_FEATURE

    # 优化相关
    keywords_opt = [
        '优化', 'optimize', '更精细', '更智能',
        '性能', 'performance', 'efficient'
    ]
    if any(kw in content_lower for kw in keywords_opt):
        return TodoCategory.OPTIMIZATION

    # 重构相关
    keywords_refactor = [
        '重构', 'refactor', '调整', 'adjust'
    ]
    if any(kw in content_lower for kw in keywords_refactor):
        return TodoCategory.REFACTOR

    # 依赖相关
    keywords_dep = [
        '安装', 'install', '依赖', 'dependency',
        'pip install', 'requirements'
    ]
    if any(kw in content_lower for kw in keywords_dep):
        return TodoCategory.DEPENDENCY

    # 移除相关
    keywords_remove = [
        '删除', 'delete', '移除', 'remove',
        '实际删除', '实际操作'
    ]
    if any(kw in content_lower for kw in keywords_remove):
        return TodoCategory.REMOVE

    # 默认为实现
    return TodoCategory.IMPLEMENT_FEATURE

def get_priority(content: str) -> str:
    """
    根据内容判断优先级

    Args:
        content: TODO内容

    Returns:
        优先级 (high, medium, low)
    """
    content_lower = content.lower()

    # 高优先级关键词
    high_keywords = [
        '同步', 'sync', '安全', 'security',
        '备份', 'backup', '恢复', 'recover'
    ]
    if any(kw in content_lower for kw in high_keywords):
        return "high"

    # 低优先级关键词
    low_keywords = [
        '可以', 'could', '可能', 'maybe',
        '可选', 'optional'
    ]
    if any(kw in content_lower for kw in low_keywords):
        return "low"

    return "medium"

def get_action(content: str, category: TodoCategory) -> str:
    """
    根据类别和内容判断处理动作

    Args:
        content: TODO内容
        category: TODO类别

    Returns:
        处理动作 (keep, remove, implement, defer)
    """
    content_stripped = content.strip()

    # 可以直接移除的
    if category == TodoCategory.DEPENDENCY:
        # 检查是否已经解决
        if 'tenacity' in content_stripped.lower():
            return "resolve"  # 已有重试机制
        return "defer"

    # 简单的删除操作可以移除TODO
    if category == TodoCategory.REMOVE:
        if '实际删除' in content_stripped:
            return "replace"  # 替换为实际代码

    # 实现功能
    if category == TodoCategory.IMPLEMENT_FEATURE:
        # 检查是否已有实现
        if '同步' in content_stripped and '向量库' in content_stripped:
            return "defer"  # 已有向量同步代码
        if '知识图谱' in content_stripped:
            return "defer"  # 知识图谱是大功能
        return "implement"

    return "keep"

def find_todos(directory: str) -> List[TodoItem]:
    """
    查找目录中的所有TODO

    Args:
        directory: 扫描目录

    Returns:
        TODO列表
    """
    todos = []
    root_path = Path(directory)

    # TODO模式
    patterns = [
        r'#\s*TODO\s*:?\s*(.+)',
        r'#\s*FIXME\s*:?\s*(.+)',
        r'#\s*HACK\s*:?\s*(.+)',
        r'#\s*XXX\s*:?\s*(.+)',
        r'\'\'\'\s*TODO\s*:?\s*(.+)\'\'\'',
        r'"""\s*TODO\s*:?\s*(.+)"""',
    ]

    for py_file in root_path.rglob('*.py'):
        # 跳过某些目录
        if 'node_modules' in str(py_file) or '.git' in str(py_file):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern in patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            content = match.group(1).strip()
                            category = categorize_todo(content)
                            priority = get_priority(content)
                            action = get_action(content, category)

                            todos.append(TodoItem(
                                file_path=str(py_file.relative_to(root_path)),
                                line_number=line_num,
                                content=content,
                                category=category,
                                priority=priority,
                                action=action
                            ))
                            break  # 避免重复匹配同一行
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return todos

def analyze_todos(todos: List[TodoItem]) -> Dict[str, Any]:
    """
    分析TODO统计信息

    Args:
        todos: TODO列表

    Returns:
        统计信息字典
    """
    # 按类别统计
    by_category = {}
    for todo in todos:
        cat = todo.category.value
        by_category[cat] = by_category.get(cat, 0) + 1

    # 按优先级统计
    by_priority = {}
    for todo in todos:
        by_priority[todo.priority] = by_priority.get(todo.priority, 0) + 1

    # 按动作统计
    by_action = {}
    for todo in todos:
        by_action[todo.action] = by_action.get(todo.action, 0) + 1

    # 按文件统计
    by_file = {}
    for todo in todos:
        by_file[todo.file_path] = by_file.get(todo.file_path, 0) + 1

    return {
        'total': len(todos),
        'by_category': by_category,
        'by_priority': by_priority,
        'by_action': by_action,
        'by_file': dict(sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10])
    }

def generate_todo_report(todos: List[TodoItem], output_file: str = "TODO_REPORT.md"):
    """
    生成TODO报告

    Args:
        todos: TODO列表
        output_file: 输出文件路径
    """
    stats = analyze_todos(todos)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# TODO清理报告\n\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        f.write(f"- **总TODO数**: {stats['total']}\n\n")

        # 按类别统计
        f.write("### 按类别\n\n")
        f.write("| 类别 | 数量 |\n")
        f.write("|------|------|\n")
        for cat, count in stats['by_category'].items():
            f.write(f"| {cat} | {count} |\n")
        f.write("\n")

        # 按优先级统计
        f.write("### 按优先级\n\n")
        f.write("| 优先级 | 数量 |\n")
        f.write("|--------|------|\n")
        for pri, count in stats['by_priority'].items():
            f.write(f"| {pri} | {count} |\n")
        f.write("\n")

        # 按动作统计
        f.write("### 建议处理动作\n\n")
        f.write("| 动作 | 数量 | 说明 |\n")
        f.write("|------|------|------|\n")
        action_desc = {
            'keep': '保留，需要关注',
            'remove': '可以移除',
            'implement': '需要实现',
            'defer': '延后处理',
            'resolve': '已解决，可移除',
            'replace': '替换为实际代码'
        }
        for act, count in stats['by_action'].items():
            f.write(f"| {act} | {count} | {action_desc.get(act, '')} |\n")
        f.write("\n")

        # TODO最多的文件
        f.write("### TODO最多的文件 (Top 10)\n\n")
        f.write("| 文件 | TODO数 |\n")
        f.write("|------|--------|\n")
        for file, count in stats['by_file'].items():
            f.write(f"| {file} | {count} |\n")
        f.write("\n")

        # 详细列表
        f.write("## 详细TODO列表\n\n")

        # 按类别分组
        by_category = {}
        for todo in todos:
            cat = todo.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(todo)

        for category, items in sorted(by_category.items()):
            f.write(f"### {category}\n\n")

            # 按优先级排序
            items_sorted = sorted(items, key=lambda x: {
                'high': 0, 'medium': 1, 'low': 2
            }.get(x.priority, 3))

            for todo in items_sorted[:20]:  # 每个类别最多显示20个
                f.write(f"#### {todo.file_path}:{todo.line_number}\n\n")
                f.write(f"**内容**: `{todo.content}`\n\n")
                f.write(f"**优先级**: {todo.priority}\n\n")
                f.write(f"**建议动作**: {todo.action}\n\n")
                f.write("---\n\n")

    print(f"TODO报告已生成: {output_file}")

if __name__ == "__main__":
    import datetime

    # 查找所有TODO
    todos = find_todos("backend/app")

    # 生成报告
    generate_todo_report(todos)

    # 打印统计
    stats = analyze_todos(todos)
    print("\n=== TODO统计 ===")
    print(f"总数: {stats['total']}")
    print(f"\n按类别: {stats['by_category']}")
    print(f"按优先级: {stats['by_priority']}")
    print(f"按动作: {stats['by_action']}")
