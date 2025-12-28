"""
Markdown和文本解析器快速开始脚本

演示如何使用新增的Markdown解析器和文本解析器
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.parsers import MarkdownParser, TextParser
from app.services.parsers.register_parsers import get_default_registry


async def test_markdown_parser():
    """测试Markdown解析器"""
    print("="*60)
    print("测试 Markdown 解析器")
    print("="*60)

    # 创建一个示例Markdown文件
    sample_md = """---
title: 无线充电技术发展趋势
author: 技术研究部
date: 2024-12-28
tags: [无线充电, 技术趋势]
categories: [技术分析]
---

# 无线充电技术发展趋势

## 概述

无线充电技术正在快速发展，预计未来几年将迎来大规模商业化应用。

## 技术对比

### 磁感应充电

- **优点**: 技术成熟，成本较低
- **缺点**: 传输距离短

### 磁共振充电

- **优点**: 传输距离较远
- **缺点**: 成本较高

## 技术参数

| 参数 | 磁感应 | 磁共振 |
|------|--------|--------|
| 传输距离 | < 5mm | < 50mm |
| 效率 | 85% | 75% |
| 成本 | 低 | 中 |

```python
# 示例代码
def wireless_charge():
    return "charging..."
```

## 总结

无线充电技术将在未来智能家居领域发挥重要作用。
"""

    # 保存示例文件
    test_file = Path("test_document.md")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(sample_md)

    # 创建解析器
    parser = MarkdownParser({
        'extract_metadata': True,
        'preserve_html': False
    })

    # 解析文件
    result = await parser.parse(str(test_file))

    print(f"\n✓ 解析成功: {result.success}")
    print(f"✓ 解析时间: {result.parse_time:.3f}秒")
    print(f"✓ 内容长度: {len(result.content)} 字符")

    # 显示提取的元数据
    metadata = result.metadata.get('metadata', {})
    print(f"\n📋 文档元数据:")
    print(f"  标题: {metadata.get('title')}")
    print(f"  作者: {metadata.get('author')}")
    print(f"  日期: {metadata.get('date')}")
    print(f"  标签: {metadata.get('tags')}")
    print(f"  分类: {metadata.get('categories')}")

    # 显示统计信息
    stats = result.metadata.get('statistics', {})
    print(f"\n📊 统计信息:")
    print(f"  标题数: {stats.get('heading_count')}")
    print(f"  代码块数: {stats.get('code_block_count')}")
    print(f"  表格数: {stats.get('table_count')}")
    print(f"  链接数: {stats.get('link_count')}")
    print(f"  行数: {stats.get('line_count')}")

    # 显示标题结构
    heading_structure = result.metadata.get('heading_structure', [])
    print(f"\n📑 标题结构:")
    for heading in heading_structure:
        indent = "  " * heading['level']
        print(f"{indent}- {heading['title']}")

    # 提取表格
    tables = parser.extract_tables(result.content)
    print(f"\n📊 提取到 {len(tables)} 个表格")
    for i, table in enumerate(tables):
        print(f"  表格 {i+1}: {table['column_count']} 列 x {table['row_count']} 行")
        print(f"    表头: {table['headers']}")

    # 提取代码块
    code_blocks = parser.extract_code_blocks(result.content)
    print(f"\n💻 提取到 {len(code_blocks)} 个代码块")
    for i, block in enumerate(code_blocks):
        print(f"  代码块 {i+1}: {block['language']} ({block['length']} 字符)")

    # 智能分块
    chunks = parser.chunk_content(
        result.content,
        chunk_size=500,
        chunk_overlap=50
    )

    print(f"\n🔪 智能分块: 分为 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        title_path = chunk.metadata.get('title_path', '根级别')
        print(f"  块 {i+1}: {title_path[:40]}... ({len(chunk.content)} 字符)")

    # 清理测试文件
    test_file.unlink()


async def test_text_parser():
    """测试文本解析器"""
    print("\n" + "="*60)
    print("测试文本解析器")
    print("="*60)

    # 创建示例文本文件
    sample_txt = """无线充电技术发展趋势

无线充电技术正在快速发展，预计未来几年将迎来大规模商业化应用。

技术对比：

磁感应充电技术成熟，成本较低，但传输距离短。磁共振充电传输距离较远，但成本较高。

市场预测：
2024年市场规模预计达到100亿美元
2025年预计增长到150亿美元
2026年预计突破200亿美元

主要应用领域包括：智能手机、电动汽车、智能家居设备等。
"""

    # 保存示例文件
    test_file = Path("test_document.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(sample_txt)

    # 创建解析器
    parser = TextParser({
        'detect_language': True,
        'chunk_by_paragraph': True
    })

    # 解析文件
    result = await parser.parse(str(test_file))

    print(f"\n✓ 解析成功: {result.success}")
    print(f"✓ 编码: {result.encoding}")
    print(f"✓ 解析时间: {result.parse_time:.3f}秒")
    print(f"✓ 内容长度: {len(result.content)} 字符")

    # 显示统计信息
    stats = result.metadata.get('statistics', {})
    print(f"\n📊 统计信息:")
    print(f"  行数: {stats.get('line_count')}")
    print(f"  词数: {stats.get('word_count')}")
    print(f"  段落数: {stats.get('paragraph_count')}")
    print(f"  字符数: {stats.get('char_count')}")

    # 显示检测信息
    print(f"\n🔍 自动检测:")
    print(f"  语言: {result.metadata.get('detected_language', 'N/A')}")
    print(f"  内容类型: {result.metadata.get('content_type_hint', 'N/A')}")

    # 按段落分块
    chunks = parser.chunk_content(
        result.content,
        chunk_size=200,
        chunk_overlap=20
    )

    print(f"\n🔪 段落分块: 分为 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        para_count = chunk.metadata.get('paragraph_count', 0)
        method = chunk.metadata.get('chunking_method', 'N/A')
        print(f"  块 {i+1}: {para_count} 个段落 ({method})")
        print(f"    预览: {chunk.content[:60]}...")

    # 清理测试文件
    test_file.unlink()


async def test_parser_registry():
    """测试解析器注册表"""
    print("\n" + "="*60)
    print("测试解析器注册表")
    print("="*60)

    # 获取注册表
    registry = get_default_registry()

    print(f"\n📊 注册表统计:")
    print(f"  总解析器数: {registry.get_parser_count()}")
    print(f"  支持的扩展名数: {registry.get_extension_count()}")

    print(f"\n📝 支持的文件类型:")
    for ext_info in registry.list_extensions():
        parsers = ", ".join(ext_info['parser_names'])
        print(f"  {ext_info['extension']:15} -> {parsers}")

    # 测试自动选择解析器
    print(f"\n🔍 测试自动选择解析器:")
    test_files = [
        "test.md",
        "test.txt",
        "test.docx",
        "test.xlsx",
        "test.pdf"
    ]

    for file_path in test_files:
        ext = Path(file_path).suffix
        parser = registry.get_parser_by_extension(ext)
        if parser:
            print(f"  {file_path:15} -> {parser.parser_name}")
        else:
            print(f"  {file_path:15} -> 未找到解析器")


async def main():
    """运行所有测试"""
    try:
        await test_markdown_parser()
        await test_text_parser()
        await test_parser_registry()

        print("\n" + "="*60)
        print("✓ 所有测试完成！")
        print("="*60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
