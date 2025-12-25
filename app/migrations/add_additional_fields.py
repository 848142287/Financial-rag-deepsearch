"""
数据库迁移脚本：添加额外的文档管理字段

此脚本添加以下字段到documents表：
- mime_type: MIME类型
- storage_path: 存储路径
- parsed_content: 解析后的内容
"""

import sys
from pathlib import Path

# 添加backend目录到Python路径
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from sqlalchemy import text
from app.core.database import engine

def add_additional_fields():
    """添加额外的文档管理字段到documents表"""

    # 先检查列是否存在
    check_queries = [
        "SELECT COUNT(*) as count FROM information_schema.columns WHERE table_name='documents' AND column_name='mime_type'",
        "SELECT COUNT(*) as count FROM information_schema.columns WHERE table_name='documents' AND column_name='storage_path'",
        "SELECT COUNT(*) as count FROM information_schema.columns WHERE table_name='documents' AND column_name='parsed_content'"
    ]

    # 要执行的ALTER语句
    alter_queries = [
        ("ALTER TABLE documents ADD COLUMN mime_type VARCHAR(100)", "mime_type"),
        ("ALTER TABLE documents ADD COLUMN storage_path VARCHAR(1000)", "storage_path"),
        ("ALTER TABLE documents ADD COLUMN parsed_content JSON", "parsed_content")
    ]

    with engine.connect() as connection:
        try:
            # 检查列是否存在
            results = []
            for check_query in check_queries:
                result = connection.execute(text(check_query)).fetchone()
                results.append(result[0] > 0)

            # 添加不存在的列
            for i, (alter_query, column_name) in enumerate(alter_queries):
                if not results[i]:
                    connection.execute(text(alter_query))
                    print(f"成功添加列: {column_name}")
                else:
                    print(f"列已存在，跳过: {column_name}")

            connection.commit()
            print("数据库迁移完成！")

        except Exception as e:
            connection.rollback()
            print(f"数据库迁移失败: {e}")
            raise

if __name__ == "__main__":
    add_additional_fields()