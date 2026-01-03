#!/usr/bin/env python3
"""
代码冗余自动检测工具
功能：
1. 检测重复的类定义
2. 检测相似的函数实现
3. 检测重复的导入语句
4. 生成冗余报告
"""

import ast
import logging
from pathlib import Path
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)

class RedundancyDetector:
    """代码冗余检测器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.python_files = []
        self.class_definitions = defaultdict(list)
        self.function_definitions = defaultdict(list)
        self.import_statements = defaultdict(list)
        self.redundancy_report = {
            "duplicate_classes": [],
            "similar_functions": [],
            "duplicate_imports": [],
            "unused_files": []
        }

    def scan_directory(self):
        """扫描目录获取所有Python文件"""
        logger.info(f"扫描目录: {self.root_dir}")
        self.python_files = list(self.root_dir.rglob("*.py"))
        logger.info(f"找到 {len(self.python_files)} 个Python文件")
        return self.python_files

    def analyze_file(self, file_path: Path) -> dict:
        """分析单个Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))

            analysis = {
                "classes": [],
                "functions": [],
                "imports": [],
                "lines": len(content.splitlines()),
                "file_size": len(content)
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)
                    self.class_definitions[node.name].append(str(file_path))

                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)
                    self.function_definitions[node.name].append(str(file_path))

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                        self.import_statements[alias.name].append(str(file_path))

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        full_import = f"{module}.{alias.name}" if module else alias.name
                        analysis["imports"].append(full_import)
                        self.import_statements[full_import].append(str(file_path))

            return analysis

        except Exception as e:
            logger.error(f"分析文件失败 {file_path}: {e}")
            return None

    def detect_duplicate_classes(self) -> List[dict]:
        """检测重复的类定义"""
        duplicates = []
        for class_name, files in self.class_definitions.items():
            if len(files) > 1:
                duplicates.append({
                    "class_name": class_name,
                    "files": files,
                    "count": len(files)
                })
        return sorted(duplicates, key=lambda x: x["count"], reverse=True)

    def detect_duplicate_imports(self) -> List[dict]:
        """检测重复的导入语句"""
        duplicates = []
        for import_name, files in self.import_statements.items():
            if len(files) > 5:  # 超过5个文件使用相同导入
                duplicates.append({
                    "import_name": import_name,
                    "files": files,
                    "count": len(files)
                })
        return sorted(duplicates, key=lambda x: x["count"], reverse=True)

    def detect_similar_functions(self, threshold: float = 0.8) -> List[dict]:
        """检测相似的函数（基于名称相似度）"""
        similar = []
        function_names = list(self.function_definitions.keys())

        for i, name1 in enumerate(function_names):
            for name2 in function_names[i+1:]:
                similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
                if similarity >= threshold:
                    files1 = self.function_definitions[name1]
                    files2 = self.function_definitions[name2]
                    similar.append({
                        "func1": name1,
                        "func2": name2,
                        "similarity": similarity,
                        "files1": files1,
                        "files2": files2
                    })

        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

    def find_potential_unused_files(self) -> List[dict]:
        """查找可能未使用的文件（基于导入引用）"""
        all_imported_files = set()
        for files in self.import_statements.values():
            all_imported_files.update(files)

        all_files = set(str(f) for f in self.python_files)
        unused = all_files - all_imported_files

        # 排除 __init__.py 和主入口文件
        potentially_unused = [
            f for f in unused
            if "__init__.py" not in f and "main.py" not in f
        ]

        return potentially_unused

    def generate_report(self) -> dict:
        """生成冗余检测报告"""
        logger.info("开始生成冗余检测报告...")

        # 扫描所有文件
        for py_file in self.python_files:
            self.analyze_file(py_file)

        # 检测各种冗余
        self.redundancy_report["duplicate_classes"] = self.detect_duplicate_classes()
        self.redundancy_report["duplicate_imports"] = self.detect_duplicate_imports()
        self.redundancy_report["similar_functions"] = self.detect_similar_functions()
        self.redundancy_report["unused_files"] = self.find_potential_unused_files()

        return self.redundancy_report

    def print_report(self):
        """打印报告到控制台"""
        report = self.redundancy_report

        print("\n" + "="*80)
        print("代码冗余检测报告".center(80))
        print("="*80 + "\n")

        # 重复的类
        print(f"📌 发现 {len(report['duplicate_classes'])} 个重复的类定义:")
        for item in report["duplicate_classes"][:10]:
            print(f"  - {item['class_name']}: {item['count']} 个文件")

        # 重复的导入
        print(f"\n📌 发现 {len(report['duplicate_imports'])} 个广泛使用的导入:")
        for item in report["duplicate_imports"][:10]:
            print(f"  - {item['import_name']}: {item['count']} 个文件")

        # 相似的函数
        print(f"\n📌 发现 {len(report['similar_functions'])} 个相似的函数名:")
        for item in report["similar_functions"][:10]:
            print(f"  - {item['func1']} <-> {item['func2']} ({item['similarity']:.2%})")

        # 可能未使用的文件
        print(f"\n📌 发现 {len(report['unused_files'])} 个可能未使用的文件:")
        for file_path in report["unused_files"][:10]:
            rel_path = Path(file_path).relative_to(self.root_dir)
            print(f"  - {rel_path}")

        print("\n" + "="*80 + "\n")

    def save_report(self, output_file: str):
        """保存报告到文件"""
        import json
        from datetime import datetime

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "scan_directory": str(self.root_dir),
            "total_files": len(self.python_files),
            "results": self.redundancy_report
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"报告已保存到: {output_file}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="代码冗余检测工具")
    parser.add_argument("directory", help="要扫描的目录")
    parser.add_argument("-o", "--output", help="输出报告文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 创建检测器
    detector = RedundancyDetector(args.directory)

    # 扫描目录
    detector.scan_directory()

    # 生成报告
    detector.generate_report()

    # 打印报告
    detector.print_report()

    # 保存报告
    if args.output:
        detector.save_report(args.output)

if __name__ == "__main__":
    main()
